# Parallel rollout collection (multiprocessing)

**Status:** Phases 1–3 implemented — `rollout_core.play_episode` (shared core),
`rollout_collector.ParallelRolloutCollector` (spawn worker pool), `RolloutBuffer.ingest_chunks`,
incremental pool sync, `n_workers=6` default. Correctness verified (serial + parallel run clean,
pool sync + worker-failure path tested). **Speed benchmark on the real config still pending**
(run `n_workers=1` vs `6` head-to-head on an otherwise-idle box). Phases 4–5 tracked in
`docs/IDEAS.md` #11.

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
  ├── workers each play a share of collect_episodes → return (numpy transitions + ep metrics)
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
  1. `policy.load_state_dict(task.policy_sd)`;
  2. if `task.new_snapshots`: append to local pool (same `maxlen` deque → worker pools stay
     in sync without shipping the whole pool);
  3. `pool.set_weights(task.weights)`; set `shaping_anneal`;
  4. play `task.n_episodes` via `rollout_core.play_episode` (shared with single-process path);
  5. put `(worker_id, stacked_arrays, episode_dicts)` on the shared result queue.
- **shutdown:** `None` sentinel → clean exit; main `terminate()`+`join()` in `finally`.

### IPC protocol

- **Per-worker task queues** (list of N) + **one shared result queue**. Per-worker queues
  guarantee every worker receives every batch → incremental pool sync is correct.
  (`ProcessPoolExecutor` gives no such guarantee about *which* worker runs a task, which
  breaks incremental pool replication — hence manual `Process`+`Queue`.)
- **Task** (small, pickled per batch): `policy_sd` (CPU tensors), `new_snapshots` (0-1
  usually), `weights` (p_random/greedy/pool), `n_episodes`, `anneal`, `batch_num`.
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
- **v1 (simple):** static split `collect_episodes // N` (+ remainder to first workers).
- **v2 (better, IDEAS #11):** dynamic task queue — workers pull episodes until the batch
  quota is met, removing the slowest-episode tail. Start with v1; move to v2 if profiling
  shows imbalance.

## File changes

- **`src/services/environment/rollout_core.py`** (new): pure
  `play_episode(env, policy, opp, opp_type, main_pid, *, gamma, shaping_anneal,
  holding_reward_rate, ...) -> (transitions, episode_dict)`, extracted from the old
  `PPOTrainer._collect_episode` with no `self`/critic dependency. **Single source of truth**
  used by both the worker and the single-process path.
- **`src/utils/rollout_buffer.py`:** `ingest_chunks(worker_arrays)` — concatenate arrays +
  shift `episode_ends`. `add_step` retained for the single-process fallback.
- **`src/services/rollout_collector.py`** (new): `ParallelRolloutCollector` — owns workers,
  broadcast, gather; encapsulates all multiprocessing.
- **`src/app/ppo.py`:** `_collect_batch` branches on `n_workers` (`1` = current serial path
  through `play_episode`; `>1` = collector). Everything downstream unchanged.
- **`src/services/opponent_pool.py`:** version counter / `snapshots_since(idx)` for
  incremental broadcast.
- **hp:** `'n_workers': 6` (1 = current behavior), `'rollout_seed': <int>`.

## Correctness

- **Critic values:** unchanged path — workers compute no values; main runs the same batched
  pass. Bit-identical to single-process for the same states.
- **RNG:** parallel streams with distinct seeds → *different* concrete episodes, statistically
  equivalent. Everything downstream (GAE/update) is identical in form. Not bit-reproducible
  vs single-process; **is** reproducible for fixed `(rollout_seed, n_workers)`. Changing
  `n_workers` changes the episode sample.
- **Equivalence test:** `n_workers=1` through the collector should reproduce the serial path
  exactly (same `play_episode`, same RNG order) — the safety net before scaling.

## Pitfalls (accounted for)

- `torch.set_num_threads(1)` per worker — mandatory.
- **spawn**, not fork (fork+torch can deadlock on Linux/WSL); persistent workers pay
  torch+matplotlib import once.
- `matplotlib.use('Agg')` before importing env in the worker (headless).
- **Worker death:** worker catches exceptions and sends `(worker_id, ERROR, traceback)`; main
  raises and cleanly shuts all workers down (`try/finally` + `terminate()`/`join()`) — no
  zombie processes.
- **Invalid-action path** (`make_random_step`, `log_prob=0`) moves into `play_episode` as-is.
- **No oversubscription:** `N = min(n_workers, nproc-2)`; leave cores for the GPU update, IPC
  and the OS. On 12 cores, default **6**, ceiling 8.

## Phases

1. **Refactor (this doc's implemented part):** extract `play_episode` into `rollout_core.py`;
   route the current `_collect_batch` through it. Prove metrics unchanged vs the pre-refactor
   run. *Zero-risk groundwork.*
2. **Collector + workers, `n_workers=1`:** prove metric equivalence against Phase 1.
3. **Scale to N=6:** measure `t=` / balance; wire error handling + shutdown.
4. **(IDEAS #11) Dynamic work-queue** if a tail imbalance shows up.
5. **(IDEAS #11) Overlap** collection of batch N+1 with the GPU update of batch N (policy is
   frozen during collection anyway) — another ~15-20%, more complexity; not for v1.

## Expected gain (Amdahl)

`overall = 1 / (0.2 + 0.8 / S_collect)`. Realistic `S_collect ≈ 4.5-5×` (6 workers minus
IPC/imbalance/broadcast) → **~2.8× overall**, on top of the prior 2× → **~5×** vs the original
(5+ h → ~1 h). Unlike lockstep, `env` is parallelized too, so the ceiling is higher.
