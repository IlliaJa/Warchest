# Parallel rollout collection (multiprocessing)

**Status:** Phases 1–5 implemented.
- 1–3: `rollout_core.play_episode` (shared core), `rollout_collector.ParallelRolloutCollector`
  (spawn worker pool), `RolloutBuffer.ingest_chunks`, incremental pool sync, `n_workers=6`.
- 4 (P11a): dynamic load balancing — workers claim episodes from a shared atomic counter
  instead of a fixed per-worker share (submit/gather API in the collector).
- 5 (P11b): `overlap_collection=True` — batch N+1's rollout runs on the CPU workers while the
  GPU updates batch N (1-step-stale behavior weights).

Timing log reports rollout (critical-path = slowest worker's wall), env and model_play
(aggregate sums across workers), value_pass, actor_gradient, critic_gradient, IPC (gather wall
not explained by the slowest worker), and total wall. Correctness verified for all three modes
(serial, parallel barrier, parallel overlap); in overlap the rollout is fully hidden (IPC→0
steady state). **Real-config speed + learning-quality A/B still pending** (overlap adds
off-policy staleness — compare elo/wr, not just wall-clock).

## Motivation

After the CPU-rollout / deferred-critic / opponent-caching fixes (`docs/history.md`,
2026-07-04), per-batch wall time is dominated by rollout collection: `env` + `model_play`
together are ~80% of a batch. The gradient/critic/GAE tail is ~20%. To cut the 80% we must
parallelize **both** the Python game engine (`env`) and the single-obs policy forwards
(`model_play`).

### Why multiprocessing (not the alternatives)

| Approach | Speeds up | Blocker |
|---|---|---|
| Batched lockstep (K games/step, one batch=K forward) | only `model_play` | `env` stays serial Python → ceiling ~2× at env=20-30%; complex loop rewrite (mixed active players / opponents / episode lengths). |
| Threading | nothing | GIL — both env and CPU-torch are GIL-bound. |
| **Episode-level multiprocessing** ✅ | **both `env` and `model_play`** | IPC + weight broadcast. Episodes are independent → embarrassingly parallel. |

## Key enabler: workers need only the policy

Since critic values are computed in one batched GPU pass in the main process *after*
collection (deferred-critic optimization), rollout workers run **only** `policy.act()` +
the opponent move + `env.step()`. The critic (now 192-wide) never leaves the main process
and is never broadcast. Each batch the main process broadcasts only the small policy
(`hidden_dim=64`, ~1-2 MB) plus, incrementally, any new pool snapshot.

## Architecture

```
Main process (GPU)
  ├── N persistent workers (spawn), CPU-only, torch.set_num_threads(1)
  ├── per batch: broadcast(policy.state_dict + new pool snapshot(s) + phase config)
  ├── workers claim episodes from a shared counter → return (numpy transitions + ep metrics)
  ├── buffer.ingest_chunks(...)  (concatenate, shift episode_ends)
  ├── stack() → _compute_values_batched() [GPU] → compute_gae() → PPO update [GPU]
  └── maybe_snapshot / eval / log   (unchanged)
```

### Worker (persistent process)

- **init (once):** `matplotlib.use('Agg')`; `torch.set_num_threads(1)` **(critical — else
  N workers × intra-op threads oversubscribe the 12 cores and slow everything down)**;
  construct `WarChestEnv`, a local empty `Policy`, a local `OpponentPool`; seed
  `np`/`torch`/env RNG from `(base_seed, worker_id)`.
- **loop:** block on this worker's task queue →
  1. `policy.load_state_dict(task['policy_sd'])`;
  2. if `task['new_snapshots']`: append to local pool (same `maxlen` deque → worker pools stay
     in sync without shipping the whole pool);
  3. apply `task['weights']` (p_random/greedy/pool) + `shaping_anneal`/`holding_reward_rate`;
  4. **claim episodes from the shared atomic counter** until it hits 0 (dynamic balancing —
     not a fixed per-worker share), playing each via `rollout_core.play_episode` (shared with
     the single-process path);
  5. pre-stack its transitions into numpy and put `(worker_id, status, payload)` on the shared
     result queue (`payload` = the numpy arrays + `episode_dicts` + `t_env`/`t_model_play`/
     `t_worker_wall` timing).
- **shutdown:** `None` sentinel → clean exit; main `terminate()`+`join()` in `finally`.

### IPC protocol

- **Per-worker task queues** (list of N) + **one shared result queue** + **one shared atomic
  counter** (`mp.Value('i')`, remaining episodes this batch). Per-worker queues guarantee every
  worker receives every batch → incremental pool sync is correct. (`ProcessPoolExecutor` gives
  no such guarantee about *which* worker runs a task, which breaks incremental pool
  replication — hence manual `Process`+`Queue`.) The episode *count* is handed out via the
  counter, not the task, so a fast worker naturally claims more (dynamic balancing).
- **Task** (small, pickled per batch, identical to every worker): `policy_sd` (CPU tensors),
  `new_snapshots` (0-1 usually), `weights` (p_random/greedy/pool), `gamma`, `shaping_anneal`,
  `holding_reward_rate`, `max_t`. The number of episodes is *not* in the task — workers claim
  from the shared counter until it drains.
- **Result:** worker pre-stacks its transitions into numpy: `boards [n,C,7,7] f32`,
  `globals`, `masks`, `actions i64`, `log_probs f32`, `rewards f32`, `opp_onehots`,
  `privileged`, `episode_ends`. Whole-batch total ~55 MB → trivial at ~12 s/batch. If it
  ever bottlenecks, move board arrays to `multiprocessing.shared_memory`.

### Pool snapshot sync (incremental)

A snapshot is appended by `maybe_snapshot` every `pool_snapshot_every` batches, *after* the
update. Main owns the authoritative pool and ships only snapshots added since the last task
(0 or 1). Workers append into their own `deque(maxlen=pool_max_size)`; identical
append+eviction order keeps contents identical. The snapshot index is chosen locally in each
worker (`np.random.randint`) — no need to sync the choice. Snapshots are stored as **CPU**
tensors (`.cpu()` at snapshot time) so broadcast is cheap and never drags CUDA into workers.

### Load balancing

Episodes vary 60-200 turns.
- **Implemented (P11a, IDEAS #11):** dynamic balancing via a shared atomic counter
  (`mp.Value('i')` = remaining episodes). Every worker loops "atomically decrement the counter
  → play one episode" until it reads ≤0, so a worker that draws short episodes claims more and
  the slowest-episode tail is removed — no fixed per-worker share.
- The earlier static split (`collect_episodes // N` + remainder) was superseded by the counter
  and is no longer used.

## File changes

- **`src/services/environment/rollout_core.py`** (new): pure
  `play_episode(env, policy, opp, main_pid, opp_type, *, gamma, shaping_anneal,
  holding_reward_rate, max_t) -> (steps, episode_dict)`, extracted from the old
  `PPOTrainer._collect_episode` with no `self`/critic dependency. `steps` is a dict of
  parallel per-decision lists (obs, actions, log_probs, rewards, opp_onehots, privileged);
  the terminal/truncation reward is folded into `rewards[-1]`. `episode_dict` also carries
  `t_env`/`t_model_play` so the caller accumulates timing. **Single source of truth** used by
  both the worker and the single-process path.
- **`src/utils/rollout_buffer.py`:** `ingest_chunks(worker_arrays)` — concatenate arrays +
  shift `episode_ends`. `add_step` retained for the single-process fallback.
- **`src/services/rollout_collector.py`** (new): `ParallelRolloutCollector` — owns workers,
  broadcast, gather; encapsulates all multiprocessing.
- **`src/app/ppo.py`:** the training loop picks one of three collection paths per batch:
  `n_workers ≤ 1` → in-process serial through `play_episode`; `n_workers > 1` and
  `overlap_collection=False` → `_collect_parallel_barrier` (submit+gather in one step);
  `n_workers > 1` and `overlap_collection=True` → pipelined: `_submit_parallel(N+1)` is
  launched right after the value/GAE pass so the workers run batch N+1 while the GPU updates
  batch N, and `_gather_and_ingest` collects it at the top of the next iteration. Everything
  downstream unchanged.
- **`src/services/opponent_pool.py`:** `new_snapshots_since(seen_count) -> (new_snapshots,
  total_count)` for incremental broadcast; `weights` property.
- **hp:** `'n_workers': 6` (≤1 = in-process path), `'overlap_collection': True` (P11b),
  `'rollout_seed': 0`. `n_workers` is capped at `collect_episodes` so every worker gets ≥1
  episode.

## Correctness

- **Critic values:** unchanged path — workers compute no values; main runs the same batched
  pass. Bit-identical to single-process for the same states.
- **RNG:** each worker seeds its `np`/`torch`/env RNG once from `(base_seed, worker_id)` →
  parallel streams produce *different* concrete episodes, statistically equivalent. Everything
  downstream (GAE/update) is identical in form. **Not bit-reproducible** — with dynamic
  counter balancing, *which* worker plays *which* episode depends on timing, so even a fixed
  `(rollout_seed, n_workers)` does not pin the episode sample (documented tradeoff of P11a).
- **Equivalence:** the serial path (`n_workers ≤ 1`) and the workers call the same
  `play_episode`, so they are behaviourally/statistically equivalent — but the serial path
  runs in-process and does **not** go through the collector, so this is form-equivalence, not
  a bit-identical reproduction.

## Pitfalls (accounted for)

- `torch.set_num_threads(1)` per worker — mandatory.
- **spawn**, not fork (fork+torch can deadlock on Linux/WSL); persistent workers pay
  torch+matplotlib import once.
- `matplotlib.use('Agg')` before importing env in the worker (headless).
- **Worker death:** worker catches exceptions and sends `(worker_id, ERROR, traceback)`; main
  raises and cleanly shuts all workers down (`try/finally` + `terminate()`/`join()`) — no
  zombie processes.
- **Invalid-action path** (`make_random_step`, `log_prob=0`) moves into `play_episode` as-is.
- **No oversubscription:** `n_workers` is set manually (default **6** on this 12-core box) to
  leave cores for the GPU update, IPC and the OS, and each worker pins `torch.set_num_threads(1)`
  so N workers don't spawn N×intra-op threads. The only automatic cap is
  `min(n_workers, collect_episodes)` (so every worker gets ≥1 episode) — there is no auto-cap
  against core count, so raising `n_workers` past ~8 on 12 cores is the caller's responsibility.

## Phases

All five are implemented; listed here as the delivery order.

1. **Refactor:** extract `play_episode` into `rollout_core.py`; route `_collect_batch` through
   it. *Zero-risk groundwork.*
2. **Collector + workers:** `ParallelRolloutCollector` owning the spawn pool, broadcast, gather.
3. **Scale to N=6:** timing / balance; error handling + clean shutdown.
4. **(P11a, IDEAS #11) Dynamic balancing:** shared atomic counter instead of a static split.
5. **(P11b, IDEAS #11) Overlap:** collect batch N+1 with the GPU update of batch N (policy is
   frozen during collection anyway). Adds 1-step off-policy staleness + a second in-flight
   buffer; gated by `overlap_collection`.

Remaining work is validation/tuning, not implementation — see IDEAS #11 (P11c real-config
speed + learning-quality A/B; P11d shared-memory IPC only if profiling demands it).

## Expected gain (Amdahl)

`overall = 1 / (0.2 + 0.8 / S_collect)`. Realistic `S_collect ≈ 4.5-5×` (6 workers minus
IPC/imbalance/broadcast) → **~2.8× overall**, on top of the prior 2× → **~5×** vs the original
(5+ h → ~1 h). Unlike lockstep, `env` is parallelized too, so the ceiling is higher.
