# Bots

Non-learned/search-based opponents and yardsticks, plus the trained-policy
gauntlet entrant. See `src/services/bots/` for code, `docs/lookahead_bot_plan.md`
for `LookaheadBot`'s original design rationale.

| Bot | File | Interface | Summary |
|---|---|---|---|
| `RandomBot` | `random_bot.py` | `act(obs)` | Uniform random legal action. |
| `GreedyBot` | `greedy_bot.py` | `act(obs)` | Priority: attack → control → move toward nearest base → deploy → pass. No search. |
| `LookaheadBot` | `lookahead_bot.py` | `act(env)` | Alpha-beta search, hand-tuned leaf heuristic (`_leaf_potential`), cheap ordering-key pruning (`_ordering_key`). |
| `LookaheadCriticBot` | `lookahead_critic_bot.py` | `act(env)` | Beam search scored by a trained `Critic` network instead of a hand-tuned heuristic. See below. |
| trained `Policy` | `policy/policy.py` | `act(obs)` | The PPO-trained actor, no search — wrapped as `PolicyAgent` in `gauntlet.py`. |

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
| **Final (all fixes + 0.7/0.3 blend), large samples** | **68-70%** (k=40 seed=7, k=60 seed=99) | 60-75% on individual k=20 runs (seed-dependent noise) |

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

### Why not 100%

Both `LookaheadBot` and `LookaheadCriticBot` sample one determinization of
future draws per `act()` call from the unseeded global RNG (by design — see
`lookahead_bot.py`'s module docstring), so *both* bots' move quality is
stochastic even on an identical board. That's a property shared by both bots,
not a bug in either, and out of scope to change (`lookahead_bot.py` isn't this
bot's responsibility). A literal, deterministic 100% win rate isn't a
realistic target against an opponent that also gets to be lucky sometimes;
68-70% on large samples (vs the original 30-35%) is the ceiling reached so far
by fixing the confirmed bug and iterating on the structural levers above.
Further gains would most likely require a better-trained critic checkpoint
(out of scope — would mean touching the training pipeline, not this bot) or a
meaningfully larger engineering effort (e.g. batching critic evaluation across
whole tree levels rather than per-node, or a smarter search shape than beam +
iterative deepening).
