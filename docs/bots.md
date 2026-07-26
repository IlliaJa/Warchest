# Bots

Non-learned/search-based opponents and yardsticks, plus the trained-policy
gauntlet entrant. See `src/services/bots/` for code, `docs/lookahead_bot_plan.md`
for `LookaheadBot`'s original design rationale.

| Bot | File | Interface | Summary |
|---|---|---|---|
| `RandomBot` | `random_bot.py` | `act(obs)` | Uniform random legal action. |
| `GreedyBot` (a.k.a. `greedy_fast`) | `greedy_bot.py` | `act(obs)` | Obs-only, hand-blind priority ladder: attack → control → move toward nearest base → deploy → pass. First action in each bucket, no search. **Cannot** recruit/bolster/claim-initiative/initiate a tactic, so it can't use half the roster (Archer/Lancer only attack via a tactic). Kept as the cheap training-loop opponent (`OPP_TYPE_IDX['greedy']`) and as the `greedy_fast` gauntlet entrant. |
| `SimGreedyBot` (the gauntlet's `greedy_sim`) | `greedy_sim_bot.py` | `act(env)` | Shallow (2-ply-in-turns) forward-simulation greedy scored by the shared `HeuristicEvaluator`. Uses the whole game (every verb, via simulated consequence), opponent-aware. See below. |
| `LookaheadBot` | `lookahead_bot.py` | `act(env)` | Alpha-beta search; leaf delegated to the shared `HeuristicEvaluator` (`evaluation.py`), cheap ordering-key pruning (`_ordering_key`). |
| `LookaheadCriticBot` | `lookahead_critic_bot.py` | `act(env)` | Beam search scored by a trained `Critic` network instead of a hand-tuned heuristic. See below. |
| `PolicyCriticBot` | `policy_critic_bot.py` | `act(env)` | `LookaheadCriticBot` whose `max_branching` candidate cut is a trained policy's move prior instead of the heuristic ordering key. The prior is used *once* (to prune), then discarded — the beam still ranks by the critic. |
| `RoundCriticBot` | `round_critic_bot.py` | `act(env)` | `PolicyCriticBot` that searches to the end of the current round rather than a fixed ply depth. |
| `PuctBot` | `puct_bot.py` | `act(env)` | Full PUCT/MCTS: policy priors + critic value over a visit-counted tree, so the prior steers the *whole* search (not just a one-shot prune). See below. |
| trained `Policy` | `policy/policy.py` | `act(obs)` | The PPO-trained actor, no search — wrapped as `PolicyAgent` in `gauntlet.py`. |

Gauntlet name resolution: `greedy_sim` builds `SimGreedyBot`; `greedy_fast` builds the
obs-only `GreedyBot` wrapped in a `HeuristicAgent`. Training is unchanged — the
opponent pool still uses the obs-only `GreedyBot` (as `'greedy'` in `opponent_pool.py`
and `OPP_TYPE_IDX` — a separate naming scheme from the gauntlet CLI's `--bots`, not
renamed here since it never referred to `SimGreedyBot` in the first place).

---

## Shared evaluation & the SimGreedyBot rebuild (2026-07-24)

### Why: the old GreedyBot and LookaheadBot chose alike and ignored features

`LookaheadBot._ordering_key`'s buckets (attack > control > tactic > move > deploy >
bolster > face-down) are exactly `GreedyBot`'s obs-only priority ladder, and on the
many turns with no reward-bearing move in the search horizon the leaf potential is
nearly flat, so minimax falls back to that ordering — i.e. LookaheadBot often
degenerated to GreedyBot's pick. Both were also written before the full game
existed: `GreedyBot` never recruits/bolsters/claims-initiative/initiates a tactic
(it can't simulate, and it operates on the ego-centric obs), and `LookaheadBot`
buckets recruit/bolster/initiative below its `max_branching=8` cap so they are
pruned before evaluation.

### Shared `HeuristicEvaluator` (`evaluation.py`)

Single source of truth for "how good is this state for `root_player`", in
`rollout_core`'s reward-scale units. `LookaheadBot._leaf_potential`,
`LookaheadCriticBot` (inherited), and `SimGreedyBot` all delegate to it, so they
agree on state value. With `enable_new_terms=False` (the default) it reproduces the
exact old base/material/positional/risk `_leaf_potential` formula byte-for-byte
(`tests/test_evaluation.py`), which is what keeps `LookaheadCriticBot`'s value-scale
calibration — moment-matched to that distribution — valid.

### SimGreedyBot: shallow but plays the real game

For each legal root action it plays out the bot's whole turn (the action plus any
pending tactic continuation, resolved greedily to a quiescent state), then lets the
opponent play their single best whole turn against it (`reply_branching`-capped),
and scores the result — root maximizes, the reply minimizes. This:

- covers **every** verb for free (recruit/bolster/claim_initiative/tactic are just
  legal actions scored on their simulated consequence), so it uses the units the
  obs `GreedyBot` can't (Archer/Lancer/Cavalry/Ensign/Marshall …);
- picks the **best** attack/control/deploy (not the first), and gets Pikeman
  counter-coins, the Knight bolster-gate and recaptures right automatically because
  it scores the *resulting* state;
- is opponent-aware, which a pure 1-ply greedy is not — that 2nd ply is what took
  it from 95% to **100% vs RandomBot** and stopped it walking into free recaptures.

**Terminal dominance:** the base-PBRS leaf term is a *shaping* quantity, not a
bounded value — with a big base lead it can numerically exceed `WIN_REWARD`, so a
plain argmax would prefer holding a 5-base position (~1.16) to actually winning
(1.0). Game-ending outcomes are therefore returned as a dominating ±`_TERMINAL_VALUE`.

### Negative result: the "new" eval terms hurt — kept OFF

Step 0 of the plan added durability/economy/tempo/progress terms to nudge the bots
toward the ignored features. **Measurement showed them net-harmful and they default
OFF (`rich_eval=False`) everywhere:**

| Bot / config | Result |
|---|---|
| SimGreedyBot rich=False vs obs `GreedyBot` | ~47–48% (stable), healthy profile (move 43%, pass 14%) |
| SimGreedyBot rich=True vs obs `GreedyBot` | ~40–43%; `economy` made it spam `recruit` (~1/3 of moves) |
| LookaheadBot rich=False vs obs `GreedyBot` | **79%** |
| LookaheadBot rich=True vs obs `GreedyBot` | 50% |
| LookaheadBot rich=True **vs** rich=False | **20%** |

The terms reward long-horizon assets (a deeper deck, initiative, a bolstered stack)
that a depth-bounded search leaf can't cash in, so the bot trades away tempo for
them; `economy` was the worst. The features are already used *correctly* without
these terms, because every consumer scores an already-simulated state (a tactic's
kill, a bolster that saves a hanging unit via the risk term, etc. all show up in
`boxed`/stacks/at-risk on their own). The terms are retained (off) as a documented
negative result and a tunable for a possible trained-value or much deeper context —
do not enable without re-measuring. Corollary: the planned "unprune
recruit/bolster/initiative in `LookaheadBot._ordering_key`" change was **dropped** —
forcing those low-value moves into the search would hurt for the same reason.

### Validation (full-field gauntlet, k=24, seed=7)

Fully transitive, sensible tiers — greedy and lookahead are now clearly distinct:

```
BT ranking:  lookahead 1318  >  greedy 1124  >  greedy_fast 1084  >  random 474
WR: lookahead vs greedy 0.79 | greedy vs greedy_fast 0.58 | all bots vs random 1.00
```

`SimGreedyBot` self-reports a `usage` Counter (verb class committed per move) for the
"does it use the whole game now?" check. Six self-play games at the default
`rich_eval=False` exercise claim_initiative (~10%), tactic (~7%), recruit (~3%),
plus control/deploy/attack — all things the obs `GreedyBot` can never play. `bolster`
stays ~0%: bolstering is usually a tempo loss a shallow search won't pick (the obs
`GreedyBot` never bolsters either), and the durability term that would nudge it is
part of the net-harmful `rich_eval` bundle left off by default.

---

## LookaheadCriticBot

Beam search: at every node, every legal move (capped by `max_branching`,
pre-filtered by `LookaheadBot`'s cheap ordering key) is actually applied, the
resulting states are scored in one batched `Critic` forward pass, and only the
`beam_width` best survive to be recursed into. See the file's module docstring
for the full design (per-node pruning scope, root-perspective/negamax sign
convention, reward-path accounting reused from `LookaheadBot`).

### Baseline problem (2026-07-11)

`uv run python -m src.app.gauntlet --bots lookahead lookahead_critic` gave the
critic bot only **~30-35% WR vs `LookaheadBot`** (default `time_budget=0.5s` for
the critic bot, `0.1s` for `LookaheadBot` — both fixed by `gauntlet.py`'s CLI
defaults, out of `lookahead_critic_bot.py`'s scope to change). Sanity checks
against weaker baselines were the real red flag: **25% vs `GreedyBot`** and
**50% vs `RandomBot`** — barely better than a coin flip against an opponent
with zero search or learning. The same run's underlying `Policy` (paired with
the critic, same 2026-07-07 1500-episode training run) beat both `GreedyBot`
and `RandomBot` **100%** with no search at all, so the critic itself was not
fundamentally useless — something in how the search *used* it was broken.

### Bug found: missing critic denormalization

`Critic.value_batch()` returns a **normalized** value. `ppo.py`'s
`ReturnNormalizer` (an EMA of return mean/std, kept only for the duration of
training to stabilize the critic's loss scale) is always denormalized
(`value * std + mean`) before the critic's output is treated as a real value
anywhere in training (rollout bootstrapping, GAE). That EMA is training-loop
state and is **never written to the checkpoint** (`checkpoint.py` only saves
`state_dict`/`obs_version`/`arch`/`hidden_dim`) — so the exact denormalization
used when a given checkpoint was saved can't be recovered from the checkpoint
alone.

`_critic_root_values` was feeding the network's raw (still-normalized) output
straight into `_beam_value`, which sums it with real, reward-scale path
returns (`Action.reward`, holding, PBRS) at every node. This gives the
critic's contribution an arbitrary, depth-dependent weight relative to the
real rewards it's added to — nodes reached via more/fewer real-reward-bearing
steps end up comparing on incommensurable scales.

**Confirmed by isolating it:** swapped the critic call for `_leaf_potential`
(the same value `LookaheadBot`'s own heuristic search uses), *same beam-search
shape otherwise* — win rate vs `GreedyBot` went from 25% to **83% (5/6)**. The
beam-search structure itself was fine; the critic integration was the problem.

**Fix — `_calibrate_value_scale()`:** since the training-time normalizer stats
are gone, the checkpoint has no ground truth to denormalize against. Instead,
at bot construction, sample a batch of quick self-play states (mostly
`_ordering_key`'s pick, some uniform-random for diversity — realistic states,
not random-vs-random wandering) and fit an affine map (`scale`, `shift`)
matching the raw critic output's mean/std to `_leaf_potential`'s mean/std over
those same states. `_leaf_potential` is already reward-scale-correct (the
exact quantity `_minimax` sums real path rewards against), so this recovers a
substitute denormalization without needing the lost EMA.

**Honest limitation of this fix — it's an approximation, not a recovery.**
Moment-matching only fits 2 degrees of freedom (an affine map) and only
assumes the critic's output is *linearly* related to the true value — it
can't fix any non-linear miscalibration in the network itself, and its target
(`_leaf_potential`) is itself a hand-tuned heuristic, not the ground-truth
value function, so a "perfect" moment-match to it still isn't guaranteed to
recover the network's *actual* trained scale. It's the best available
substitute given a checkpoint with no recorded normalizer stats — not a
guarantee of exact denormalization.

**Follow-up (2026-07-11, later same day) — exact recovery for *new* checkpoints.**
`save_critic_checkpoint`/`load_critic_checkpoint` (`checkpoint.py`) now carry
optional `return_mean`/`return_std` floats — `ppo.py`'s `ReturnNormalizer` EMA
(exposed via new `.mean`/`.std` properties) at save time, passed at the one
`save_critic_checkpoint` call site in `ppo.py`. `LookaheadCriticBot.__init__`
uses these *exactly* (`self._value_scale, self._value_shift =
return_std, return_mean`) whenever a checkpoint has them, falling back to
`_calibrate_value_scale()`'s approximation only for older checkpoints that
don't (verified: both paths round-trip correctly, and the existing
`lookahead_critic_v2.pth` — saved before this change — still loads fine via
the fallback, 80% WR vs `lookahead` on a fresh k=20 run, consistent with the
established range). **This does not retroactively improve any existing
checkpoint** — `lookahead_critic_v2.pth` (or `_v1.pth`) was already saved
without these stats and there's no way to recover them after the fact; the
exact-recovery path only ever activates for a checkpoint produced by a
training run *after* this change, i.e. it needs a fresh/continued PPO run to
actually pay off. Whether the tuned `0.7/0.3` critic/heuristic blend weight
(above) is still optimal once a checkpoint with exact stats exists is
unverified — that weight was tuned against the *approximated* scale.

### Structural improvements

1. **Ply-dependent beam/branching narrowing** (`_beam_width_at`,
   `_max_branching_at`). Per-node cost is critic-forward-dominated (see module
   docstring profiling: ~0.7ms/action encoding + ~0.3-0.4ms/action critic
   forward, vs ~0.07ms/action clone+apply), and that cost multiplies across
   recursion levels — without narrowing, the search rarely got past depth 2-3
   in the 0.5s budget, vs `LookaheadBot`'s alpha-beta reaching depth 4-6 in a
   fifth of the time. Root (`ply == 0`) and the opponent's immediate reply
   (`ply == 1`) keep the full configured `beam_width`/`max_branching`; deeper
   plies narrow (`max(2, beam_width - (ply - 1))`, `max(3, max_branching - 2 *
   (ply - 1))`) since they only exist to sanity-check the root decision against
   a real reply, not to be the decision itself.
2. **Cross-iteration survivor caching** (`_survivor_cache`, keyed by the tuple
   of action ids taken from root). Iterative deepening re-enters the *same*
   tree at depth=0,1,2,... every outer-loop pass — root_state/root_queues are
   fixed for the whole `act()` call, so a shallower pass's fully-expanded,
   critic-scored, pruned node gets identically re-expanded from scratch by
   every deeper pass. Caching turns each new iteration into "extend the
   previous one" instead of "redo it plus one more ply."
3. **Blending the calibrated critic value with `_leaf_potential` at the leaf**
   (`0.7 * critic + 0.3 * heuristic`). The critic is only ever calibrated to a
   moment-matched scale (no ground truth available — see above), so blending
   in the proven heuristic hedges against the critic's own directional
   accuracy being noisier than a fully-trained value function's would be (this
   checkpoint is a 1500-episode run). This was the single biggest lever of all
   the changes — see the weight sweep below.

### Experiment log

All runs: `uv run python -m src.app.gauntlet --bots lookahead lookahead_critic
[--k-games N] [--seed S]`, `time_budget` fixed at the CLI defaults (0.5s
critic / 0.1s lookahead — not adjustable without touching `gauntlet.py`,
out of scope). Both bots use single-determinization future-draw sampling
(unseeded per `act()` call, by design — see `lookahead_bot.py`), so results
have real run-to-run variance on top of binomial sampling noise; treat
single 12-20-game numbers as noisy and larger-sample (≥40 game) numbers as
the reliable read.

| Change | WR vs `lookahead` | Notes |
|---|---|---|
| Baseline (before any fix) | 30-35% (k=20) | Also 25% vs greedy, 50% vs random — the real tell |
| + `_leaf_potential` instead of critic (diagnostic only, not shipped) | — (83% vs greedy, not tested vs lookahead) | Isolated: beam-search structure is fine, critic integration is the bug |
| + calibration fix alone (random-vs-random calibration rollout) | 17% (k=12) | Regression on this small sample — high variance, see next row |
| + ply-dependent narrowing, aggressive (`beam-2*ply`, `branch-3*ply`) | 17-50% (k=12) | Too aggressive — starves deep plies of width, reverted |
| + ply-dependent narrowing, milder (`beam-ply`, `branch-2*ply` beyond ply 1) | 58-70% (k=12-20) | Kept |
| + epsilon-greedy calibration rollout (0.8 ordering-key / 0.2 random, instead of pure random) | 65-70% (k=12-20) | More realistic calibration states; kept |
| widen root `max_branching` by +4 (one-time cost, tried as a lever) | 45% (k=20, seed=1) | Regression — reverted, not worth the per-move time it ate into deeper plies |
| + cross-iteration survivor caching | 65% (k=20, seed=1) | No clear win/loss on its own, but more nodes visited for the same budget with no downside — kept |
| blend weight sweep (critic vs `_leaf_potential`) — all on top of the above, default seed, k=20 | 1.0/0.0: 35%, 0.6/0.4: 60%, 0.5/0.5: 60%, 0.8/0.2: 70%, **0.7/0.3: 75%** | Both pure critic and pure heuristic clearly worse than the blend — **0.7/0.3 shipped** |
| `n_determinizations` (vote across N independent searches under fresh future-draw samples, `time_budget` split across them) — equal split, N=2 (0.25s/0.25s) | 60% (k=40, seed=7) | Regression vs. this seed's single-search baseline (68-70%, see below) |
| `n_determinizations`, weighted split, N=2 (0.8×/0.2× — primary keeps most of the budget, second is a cheap hedge) | 65% (k=40, seed=7) | Still below the single-search baseline at this seed — closer to 1.0 weight tracks closer to baseline |
| `n_determinizations=1` (reverted/confirmed) | 78% (k=40, seed=7) | Same code path, weight `[1.0]` — matches (exceeds, this run) the pre-experiment baseline; **shipped default** |
| **Final (all fixes + 0.7/0.3 blend, `n_determinizations=1`), large samples** | **68-78%** (k=40 seed=7 × 2 runs, k=60 seed=99) | 60-75% on individual k=20 runs (seed-dependent noise) |

### `max_branching`: root breadth vs. depth, default changed 8 → 5 (2026-07-26)

Hypothesis going in: narrowing `max_branching` should buy more search depth for
the same time budget, and more depth should mean stronger play. The first half
is true; the second isn't — quality is **not monotonic** in depth.

`--bots lookahead lookahead_critic [greedy_fast]`, `beam_width=5` and
`time_budget=0.1s` fixed throughout, only `max_branching` varied:

| `max_branching` | avg depth reached | avg nodes visited | WR vs `lookahead` (k=20) | WR vs `lookahead` (k=40 confirm) |
|---|---|---|---|---|
| 1 | ~5.9 | ~110 | 15% | — |
| 2 | ~3.8 | ~90 | 55% | — |
| 3 | ~3.0 | ~85 | 65% | — |
| 4 | — | — | — | 60% |
| **5** | ~2.0-2.15 | ~55-72 | **95%** | **88%** |
| 6 | — | — | — | 68% |
| 8 (old default) | ~1.3-1.7 | ~37-45 | 80% | 66% |

`max_branching=1` reaches the *deepest* search of the sweep (~6 plies) and gives
the *worst* result. The reason is structural: root and the opponent's immediate
reply (ply 0/1) always keep the full configured `max_branching`/`beam_width`
(see "Structural improvements" above). When `max_branching < beam_width`, the
beam-keep step is a no-op — there's no real shortlist for the critic to choose
from at the decision that matters most, so the bot just commits to the cheap
ordering-key's top pick and searches deep *along that one already-decided
line*. Depth without root breadth doesn't compare alternatives, it just
double-checks one. Above `beam_width` (mb=6, 8), root breadth is real but the
extra critic-forward cost per node collapses depth back to ~1.3-1.7 plies —
too shallow to sanity-check the root pick against a real reply. The optimum
sits at `max_branching == beam_width`, confirmed at k=40 (88% vs. 60-68% for
the neighbors) — not simply "as narrow as possible."

**Shipped:** `LookaheadCriticBot`'s own default (`lookahead_critic_bot.py`) and
the gauntlet CLI's `--lookahead-critic-max-branching` default both changed
8 → 5. This also changes `opponent_pool.py`'s training-time `LookaheadCriticBot`
(built via `_get_lookahead_bot()`, which never overrides `max_branching`) — not
just gauntlet eval. It happens to now match `PolicyCriticBot`'s own
already-5 default (untested here, but same `beam_width=5`, so plausibly the
same optimum). It does *not* touch `RoundCriticBot`'s own default of
`max_branching=3` — its search is round-bounded rather than
budget-iterative-deepened, so the same trade-off doesn't obviously transfer,
and it hasn't been swept. Note the CLI flag is shared across
`lookahead_critic`/`policy_critic`/`round_critic`, so running any of the
latter two from the gauntlet CLI now defaults to 5 regardless of their own
class defaults — pre-existing coupling, unchanged by this investigation.

**Why the determinization vote didn't help:** at this bot's fixed 0.5s budget it's already depth-starved relative to `LookaheadBot`'s alpha-beta (see "structural improvements" above) — every split tested, even a lopsided 80/20 two-way one, took real depth away from the primary search, and the resulting hedge against a single unlucky future-draw sample never recovered what that lost depth cost. This is the classic Perfect-Information-Monte-Carlo determinization-averaging technique, and it's a legitimate lever *in general* — it just isn't a net win at a budget this tight. `LookaheadCriticBot.n_determinizations` is left in the code (default `1`, i.e. off) rather than removed, in case a future eval run uses a meaningfully larger `--lookahead-critic-time-budget`, where the primary search would stop being the bottleneck and a hedge could pay for itself.

### Known limitation, not fixed

`opp_onehot` conditioning (the critic's 3-way opponent-identity one-hot,
trained as "identity of whoever is *not* the currently-encoded mover") is fed
a constant configured `opp_type` (default `'pool'`) at every search node
regardless of who's about to move. At nodes where `mover == root_player` this
matches the trained semantics (the passive side is root's real opponent). At
nodes where the opponent is about to move, the passive side is *root_player
itself*, which `OPP_TYPE_IDX` (`random`/`greedy`/`pool`) has no label for —
there's no "self" identity to feed. Tried both `opp_type='pool'` (default) and
`opp_type='greedy'` head-to-head vs `lookahead`: 65-70% either way, within the
run-to-run noise band, so this doesn't appear to be a large practical effect
given the taxonomy's limits — left as-is rather than guessing at a fix with no
clean correct answer.

### Diagnostics logging

`act()` logs at two levels (`logging.getLogger('warchest')` — same logger
name `ppo.py`'s `setup_run_logger` configures):
- DEBUG, one line per real move: `depth_reached`/`nodes_visited`/`elapsed` vs.
  the budget/`legal_at_root`/`best_value` (also `self.last_stats`, a plain
  dict, for programmatic access — unchanged, just now also logged). Fires
  every move, so DEBUG not INFO — would spam a console shown at INFO. Good
  for inspecting one specific decision, too granular to read across a whole
  game.
- INFO, a rolling aggregate every `stats_log_every` moves (constructor arg,
  default 20; 0 disables it): avg/min/max `depth_reached` (the number that
  answers "is the search actually looking ahead, or stuck at the leaf"),
  avg `nodes_visited`, avg `elapsed` vs. budget, avg `legal_at_root` — then
  resets the window. Added after per-move DEBUG traces turned out too
  granular to read at a glance; this is the one meant for "is the search
  behaving reasonably" during a normal run.

**Follow-up: the gauntlet CLI now configures this itself.** `gauntlet.py`'s
`main()` calls `logging.basicConfig(level=logging.INFO, ...)`, so running
e.g. `uv run python -m src.app.gauntlet --bots lookahead lookahead_critic`
shows the INFO aggregate lines with no extra setup. That call alone only
covers the CLI's own process, though — the default path (`n_workers > 1`,
i.e. not `--sequential`) runs every actual game inside `spawn`-context worker
processes (`gauntlet_parallel.py`), which don't inherit it (a `spawn` child is
a fresh interpreter, unlike `fork`), so `_worker_loop` configures its own copy
too — tagged `[worker N]` in the format string, since several workers' output
interleaves on one console. Bump to DEBUG (either call site) for the per-move
trace too, or rely on `ppo.py`'s `setup_run_logger` if this bot is used as a
training-time opponent instead (its file handler is already at DEBUG).

### Operational note: checkpoint path resolution changed mid-development

`gauntlet.py` no longer imports this file's old `DEFAULT_CRITIC_PATH`
constant — it now has its own `_latest_critic_path()`, globbing
`data/lookahead_critic/lookahead_critic_v*.pth` and always picking the
highest-numbered version, so the gauntlet CLI always plays whatever the
newest critic checkpoint is. This changed (and `data/lookahead_critic/`
gained a `_v2.pth` alongside `_v1.pth`, with `_v1.pth`'s content ending up
mismatched — a policy checkpoint, not a critic one) from external work on
this repo during the same development window as the fixes above, not from
anything in this file.

**Follow-up:** confirmed the versioning convention was intentional — removed
`DEFAULT_CRITIC_PATH` entirely and gave `LookaheadCriticBot` its own
`_latest_critic_path()` (a small separate copy of the same glob logic;
services/ shouldn't import from app/gauntlet.py, so not shared code, just the
same idea in both places). `critic_path=None` now resolves to the
highest-numbered checkpoint at construction time instead of a hardcoded,
eventually-stale literal path; raises `FileNotFoundError` if none exists and
none was passed explicitly. `tests/test_lookahead_critic_bot.py` (which
constructs the bot with no `critic_path`) still passes unchanged.

### Why not 100%

Both `LookaheadBot` and `LookaheadCriticBot` sample one determinization of
future draws per `act()` call from the unseeded global RNG (by design — see
`lookahead_bot.py`'s module docstring), so *both* bots' move quality is
stochastic even on an identical board. That's a property shared by both bots,
not a bug in either, and out of scope to change (`lookahead_bot.py` isn't this
bot's responsibility) — tried hedging *this* bot's own share of that variance
via multi-determinization voting instead (see the experiment log above); it
didn't pay off at this budget. A literal, deterministic 100% win rate isn't a
realistic target against an opponent that also gets to be lucky sometimes;
68-80% on large samples (vs the original 30-35%) is the ceiling reached so far
by fixing the confirmed bug and iterating on the structural levers above.
Further gains would most likely require a better-trained (or just better-
calibrated, now that a checkpoint *can* carry exact `return_mean`/`return_std`
— see the follow-up above) critic checkpoint — which needs a fresh/continued
training run to produce, not a code change here — or a meaningfully larger
engineering effort (e.g. batching critic evaluation across whole tree levels
rather than per-node, or a smarter search shape than beam + iterative
deepening).

---

## PuctBot — full PUCT/MCTS (2026-07-26)

`PolicyCriticBot`/`LookaheadCriticBot` are not MCTS: they run an alpha-beta-shaped
*beam*, and the policy prior there only cuts each node's raw legal moves down to
`max_branching` candidates before the beam ranks the survivors purely by the
critic. The prior is used once and thrown away; it never enters a selection
formula. `PuctBot` (`puct_bot.py`) is the real AlphaZero decomposition — "policy
proposes, value evaluates" over an actual visit-counted tree:

    argmax_a  [ ±Q(s,a) + c_puct · P(s,a) · sqrt(ΣN) / (1 + N(s,a)) ]

so the prior keeps steering exploration for the *whole* search.

### Design

- **Subclass of `PolicyCriticBot`.** That class already loads both nets this
  search needs (policy for priors, critic for leaf values), the encoders, the
  value-scale calibration, the single-determinization forward-sim harness, the
  real reward plumbing, and the `act()` wrapper that votes across
  `n_determinizations`. All inherited verbatim; only `_act_once` is overridden —
  it builds and runs a PUCT tree instead of a beam.
- **Root-perspective min/max, not textbook negamax.** Turns don't strictly
  alternate (tactic continuations, empty-hand skips) and this repo's reward
  accounting is written in *root_player* perspective, so Q/W/rewards are stored
  in root perspective and the perspective split lives only in *selection*: `+Q`
  at root_player's nodes (maximize), `-Q` at the opponent's (minimize). The
  exploration bonus is added the same way at both — it's about visit counts, not
  sides. This reuses `LookaheadCriticBot`'s reward decomposition unchanged.
- **One net eval per *node*, not per *child*.** `LookaheadCriticBot` critic-
  scores every child at every node; PUCT expansion evaluates a node once (one
  policy forward for all its priors, one critic forward for its leaf value), and
  revisiting an expanded node costs nothing. This is what lets a real tree fit a
  0.1s budget where the per-child beam is depth-starved. Measured: ~45–55
  simulations/expansions per move at 0.1s, reaching avg tree depth ~8–11 (max
  ~21), vs. the beam's typical depth 2–3.
- **Leaf value = 0.7·critic + 0.3·`_leaf_potential`** (`critic_weight`), the same
  blend and rationale as the beam bots: the critic is only moment-matched-
  calibrated, so the reward-scale-correct heuristic hedges its directional noise.
- **Final move by visit count** (AlphaZero's choice; more stable than argmax-Q at
  low sim counts). The root visit distribution is exactly the policy target
  expert iteration would later use (docs/next_steps.md — "search moves become new
  training targets"); optional root Dirichlet noise (`dirichlet_alpha`, off by
  default) is provided for that self-play use.

### Interfaces & wiring

Speaks the gauntlet `act(env)` contract (`--bots puct`, knobs `--puct-c`,
`--puct-max-branching`, `--puct-time-budget`; needs both a critic *and* a policy
checkpoint, like `policy_critic`). Also wired as a training opponent: it shares
the `pool` opponent-onehot slot (`rollout_core.OPP_ONEHOT_SLOT`, like
`lookahead_critic` — the critic's one-hot has no free slot), is routed through
`act(env)` by `_opponent_env_action` (`_SEARCH_OPP_TYPES`), and is sampled by
`OpponentPool` via `p_puct` / `puct_time_budget`. **Off by default**
(`p_puct_initial = p_puct_finetune = 0.0`): unlike `lookahead_critic` it also
needs a policy checkpoint on disk for its priors, which it loads frozen on first
sample — so a fresh run with no `data/warchest_ppo_*.pth` yet must keep it at 0.
Train-time win rate is logged as `wr_vs_puct_train`.

### First result (parallel gauntlet, k=2, seed 0)

A quick, small-sample sanity field had `puct` beating both `policy_critic` (2/0)
and `greedy_sim` (2/0), topping the Bradley-Terry ranking (~1199 vs. 1000 for
`policy_critic`) — i.e. the full tree search is stronger than the beam that reuses
the same policy+critic, the effect the AlphaZero direction predicts. A later k≥8
field confirmed it: `puct` first at ~1117 Elo over `lookahead_critic` (~1017),
`policy_critic` (~982), `lookahead` (~882), fully transitive.

### Expert iteration (ExIt / AlphaZero loop)

Because `puct` is the strongest agent, its *own* search output is a better teacher
than the raw policy that seeds it. `src/services/expert_iteration.py` +
`src/app/expert_iteration.py` close that loop (docs/next_steps.md — "search moves
become new training targets"):

1. **gen** — `puct` self-plays (root Dirichlet noise, temperature-sampled moves);
   per move it records the ego-frame obs, the ego-frame **root visit distribution**
   (policy target), the critic's privileged inputs, and — after each game — the
   outcome `z ∈ {+1,0,-1}` from the mover's perspective (critic target).
2. **distill** — warm-starts from the current nets and minimises `CE(policy, visits)`
   + `MSE(critic_raw, z)` in two independent Adam passes (mirroring PPO's separate
   actor/critic optimisers). New nets save via the existing
   `save_policy_checkpoint`/`save_critic_checkpoint`.
3. **loop** — gen → distil → re-seed `puct` → repeat.

Two design points that make it correct:
- **`PuctBot(value_mode='outcome')`.** The distilled critic predicts the outcome `z`
  (scale `[-1,1]`), not the shaped PPO return, so the search must not blend it with
  `_leaf_potential` (shaped scale) or accumulate shaped edge rewards. `value_mode`
  gates exactly that: `'outcome'` = leaf is critic-only + no intermediate shaped
  rewards (terminals are already ±1); `'shaped'` (default) is the unchanged gauntlet
  bot. The z-critic saves with `return_mean=0`/`return_std=1` so `PuctBot`'s
  denormalisation is identity.
- **ExIt artifacts live under `data/exit/`, never `data/lookahead_critic/`.** A z-scale
  critic must never become the "latest" critic the *shaped* bots (`lookahead_critic`,
  `policy_critic`, default `puct`) resolve to — that would be a scale mismatch. The
  loop pairs the z-critic only with outcome-mode search, passing paths explicitly.
  Round 0 bootstraps from the standing shaped checkpoints with `value_mode='shaped'`
  (the proven strong teacher); every later round uses the z-critic in `'outcome'` mode.

Frame note: visit counts are absolute-frame, the policy/mask are ego-frame — targets
are remapped absolute→ego at record time (`WarChestEnv.remap_action`) so they line up
index-for-index with the masked policy logits (verified: zero target mass on illegal
ids). Policy and critic must share one `obs_version` (they do when saved from one PPO
run); asserted by the CLI. Verified end-to-end on a small run: dataset shapes/frame
correct, distillation drives held-out `CE` and critic `MSE` down, and outcome-mode
`puct` loads and plays with the distilled nets. First real measurement owed: a
multi-round loop read against the gauntlet (distilled policy vs base; `puct(distilled)`
vs `puct(base)`).
