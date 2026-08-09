# LookaheadBot — design plan

Written 2026-07-05, decisions on open questions added 2026-07-06. Design for a new bot that
searches 5-10 moves (plies) ahead, intended as a stronger yardstick than `GreedyBot` (see
`docs/IDEAS.md` — WR vs greedy is saturated at ~100% and no longer measures anything).

## Implementation status (2026-07-06)

M0-M2 built: `src/services/bots/lookahead_bot.py` (`LookaheadBot`), plus a small
`WarChestEnv._apply_action()` extraction from `step()` (same logic, no observation
encoding, so search can replay actions fast) and a `lookahead_agent()` factory in
`gauntlet.py`. Tests in `tests/test_lookahead_bot.py`; full suite still green.

Initial smoke benchmark via `gauntlet.play_game` (`time_budget=0.15`, `max_branching=6`)
against `GreedyBot`: strong vs `RandomBot` in both seats (6/6 as P1, 4/6+2 draws as P2), but
**not yet reliably beating `GreedyBot`** — 1/10 as P1, 4/10 as P2. No bug found on
inspection (no invalid-action fallbacks triggered, iterative deepening reaches depth 4+
within the 0.15s budget, the eval formula matches `rollout_core.py`'s weights). Most likely
cause: the material+base-only leaf heuristic is too coarse and/or the budget/branching
params need tuning — exactly the calibration **M3 already deferred**, now with a concrete
reason to prioritize it. The `see_opponent_hand`/no-draw-crossing pieces work as designed;
what's unproven is whether the current heuristic + params are good enough to win with them.
Sample sizes above are small (composition is redrafted randomly every game, so per-seat
comparisons are noisy) — worth re-checking with more games once M3 lands.

### M3 calibration — done, in two steps (2026-07-06)

10-game-per-seat benchmarks vs `GreedyBot` (`time_budget=0.3`, `max_branching=8`) across
the fixes below, all vs the same opponent:

| Version | WR vs GreedyBot |
|---|---|
| Base+material leaf potential only (original) | 25% (5/20) |
| + full `Action.reward` reuse (attack/win/loss/holding), incl. the move-step penalty | 10% (2/20) — regression |
| + same, with the move-step penalty excluded from accumulation | 40% (8/20) |
| + positional BFS-to-nearest-base term added to the leaf potential | **75% (15/20)** |

Two real bugs found and fixed along the way, not just tuning:

1. **Reward reuse regressed before it helped.** The env's reward is split across two
   places — `Action.reward` (win/loss/attack/invalid/`TURN_TEMPO_REWARD`) and
   `rollout_core.py`'s PBRS/holding terms. The original leaf potential only replayed the
   PBRS half. Accumulating the *full* `Action.reward` along the search path (gamma-
   discounted, added to the PBRS potential at the leaf) is a more faithful reuse of what
   the policy is actually trained to maximize — but reusing the tempo term
   verbatim made things worse, not better (25%→10%), because that term only makes sense
   paired with a long horizon and a bootstrapped critic that can see the eventual payoff
   of advancing; a short, critic-free search only sees the immediate -0.002 cost and
   learns to prefer standing still. Excluding just that one term from the accumulation
   (keeping attack/win/loss/holding) recovered past the original baseline (40%).
2. **No positional pull for the common case of "nothing capturable within the search
   horizon."** Profiling (`last_stats`) showed most turns evaluate to a perfectly flat
   0.0 — no attack, base claim, or material change is reachable within 4-5 plies during
   normal maneuvering, which is most of the game. `GreedyBot` doesn't need lookahead to
   handle this: it always BFS's toward the nearest capturable base. LookaheadBot had no
   equivalent, so ties fell back to move-priority order (attack > control > move > ...),
   which has zero spatial sense within the "move" bucket. Adding `_nearest_dist` (the
   same BFS idea as GreedyBot's `_best_move_toward_base`, on the real `Board`) as a leaf
   potential term was the single biggest jump (40%→75%). `docs/rewards.md` rejected this
   exact idea for the *trained policy's* reward (BFS-per-training-step cost, farming-
   exploit risk for a *learned* policy) — neither concern applies to a per-decision
   search heuristic with no gradient to exploit, so the rejection doesn't transfer here.

Not yet done: heuristic weight tuning proper (the values above are the pre-existing PBRS
constants plus one hand-picked `POS_COEFF = 0.01`, not calibrated against each other).
75% vs `GreedyBot` is a large improvement over the 25% starting point but still short of
dominant; the remaining gap is more likely additional missing heuristic terms (per-unit
threat/safety awareness, tactic value) than another outright bug, but that's not proven.

### Threat awareness + real-world time budget (2026-07-06, continued)

Added `_material_at_risk` to the leaf potential — `opp_at_risk - own_at_risk` (stack
coins a unit stands to lose next turn), reusing the engine's own `_threat_grids`/
`unit_threat_footprint` (the same machinery behind the obs encoder's material-at-risk
feature), computed *exactly* rather than the obs encoder's worst-case-availability
guess, since the search already knows both hands. `RISK_COEFF = 0.5 * C_MAT` (half of
material, since it's predictive, not realized loss). Result at `time_budget=0.3`: 75%
(15/20) — statistically indistinguishable from the positional-term-only result at this
sample size (N=20 has ~±10pp noise); general tactical-awareness improvement kept
regardless, since a hanging-piece check is correct play, not something a larger sample
would show as neutral-to-negative.

**The real constraint turned out to be speed, not win rate.** `time_budget=0.5` (and
even 0.3) is too slow to actually use — this bot is meant to be queried every decision
during rollout collection, not once per careful eval move. Cut the default to
`time_budget=0.1`. At 0.1s the *original* `deepcopy`-based clone only reached depth 2-3
(vs 4-5 at 0.5s) and WR vs `GreedyBot` dropped to 40% (12/30) — the win-rate gains above
were partly bought with a budget too slow to ship.

Fixed the actual bottleneck instead of just accepting the tradeoff: `_clone_state`
(hand-rolled) replaces `deepcopy(GameState)` on the hot path (`docs` in the function's
own docstring in `lookahead_bot.py`). `deepcopy`'s generic memoised walk was the
dominant per-node cost; `GameState`'s mutable pieces are actually shallow (a handful of
`Counter`s, a `Board` = numpy array + flat list of `BaseUnit`, each holding only
immutable fields plus a `.board` back-reference that needs repointing after cloning,
otherwise unused). Result: ~1.5x more nodes at the same 0.1s budget, and WR vs
`GreedyBot` recovered to **63.3% (19/30)** — most of the way back from the 40% the
tighter budget alone had cost, without slowing back down.

Tried `max_branching=4` (trade width for depth) at `time_budget=0.1`: reached depth 3-5
(vs 2-4 at `max_branching=8`) but WR *dropped* to 46.7% (14/30) — narrower branching
misses real replies often enough to outweigh the extra depth. Kept `max_branching=8`.

| Config | WR vs GreedyBot |
|---|---|
| `time_budget=0.5`, `max_branching=8`, `deepcopy` (all terms) | 75% (15/20) |
| `time_budget=0.1`, `max_branching=8`, `deepcopy` | 40% (12/30) |
| `time_budget=0.1`, `max_branching=8`, `_clone_state` | 63.3% (19/30) |
| `time_budget=0.1`, `max_branching=4`, `_clone_state` | 46.7% (14/30) — reverted |
| + `Board.get_adjacent_cells` adjacency cache + grid-free `_material_at_risk` | **66.7% (20/30)** |

### Profiling pass 2 — `_leaf_potential` was ~59% of runtime (2026-07-06, continued)

`cProfile` on 10 `act()` calls at `time_budget=0.1` found `_leaf_potential` (i.e.
`_material_at_risk` → `threat_grids`) at **59% of total wall time**, vs. `_clone_state`
at only 13% — the "is the leaf eval itself expensive" question from the previous
session's open item, answered: yes, badly so. Two fixes, both shipped:

1. **`Board.get_adjacent_cells` — global fix, not bot-specific.** The single hottest
   line by `tottime` (39276 calls, 0.32s) was numpy scalar indexing on every neighbour
   check. The hex grid's adjacency is fixed geometry — identical for every `Board`
   ever constructed — so it's now precomputed once into a class-level dict
   (`Board._adjacency_cache`) and looked up instead of recomputed. This is in
   `board.py`, used everywhere (training rollout, rendering, this search), not
   lookahead-specific; the full test suite got measurably faster too (33s → 26s), and
   all 105 tests still pass unchanged.
2. **`_material_at_risk` rewritten to skip `threat_grids` entirely.** The obs
   encoder's `threat_grids` builds full `(BOARD_DIM, BOARD_DIM)` numpy grids for both
   sides × every threat kind, needed for its observation *planes* — this search only
   ever reads two scalars (risk on own vs. enemy unit cells), so building whole grids
   was pure waste. Rewrote it to call the same public per-unit primitives
   (`unit_threat_footprint`, `attack_enabler_coins`) directly and accumulate into a
   plain dict keyed by the handful of cells that actually matter. Same result, no
   numpy array allocation/summation.

Net: `_leaf_potential`'s share of runtime dropped from ~59% to ~43%, and WR at
`time_budget=0.1` moved 63.3%→66.7% (within this sample's noise, but directionally
right, and the underlying speedup is real and reused elsewhere).

### Threat-footprint caching — audited and ruled out (2026-07-06, continued)

Audited `_threat_ranged_cells`/`_threat_charge_cells`/`_threat_cavalry_cells`/
`_threat_berserker_reach` for whether a per-`act()`-call cache keyed by `(unit.id,
unit.loc, unit.stack)` would be safe. **It is not, and the idea is a dead end, not
just unproven:**

- `_threat_charge_cells`/`_threat_ranged_cells` (straight-line) check
  `board.get_unit_at(...)` along the path for blockers — depends on every other
  unit's position, not just the acting unit's own state.
- `_threat_cavalry_cells`/`_threat_berserker_reach` both go through
  `get_free_adjacent_cells`/`_reachable`, which check board-wide occupancy the same way.
- Only the plain-melee footprint (`get_adjacent_cells`, geometry-only) is actually
  occupancy-independent.

So a correct cache key would need to fold in the *entire* board's unit layout, not
just the acting unit's own (id, loc, stack). At that point the key is effectively
"this exact node," and since this search has no transposition table, essentially
every node's board state is unique within one `act()` call — cache hit rate ≈ 0,
all cost, no benefit. Not implemented; not worth revisiting without a transposition
table (which is its own, larger project, and not currently planned).

### Two more cheap, safe wins found instead (2026-07-06, continued)

- **`Board.all_cells_list` cached** the same way as adjacency — it's an `np.where`
  full-array scan over an invariant (which cells are `INVALID`), called on every
  `_threat_berserker_reach` invocation. Same safety argument as the adjacency cache:
  pure geometry, never changes after `Board.__init__`.
- **`Board.get_free_adjacent_cells`** now checks membership against a `set` of unit
  locations instead of a `list` (was O(n) per check, six times per call).
- **`_fast_counter_copy`** in `lookahead_bot.py`: `Counter(other_counter)` pays real
  per-call overhead for generic argument-type dispatch (iterable vs. mapping vs.
  kwargs) that's pure waste when the input is always already a `Counter` — bypassing
  it via `Counter.__new__` + `dict.update` measured **3.5x faster** for `_clone_state`'s
  small counters (`timeit`, 200k iterations). `_clone_state` builds 12 of these per
  node, so this was a real, measurable share of total time (`Counter.__init__`/
  `update` were ~30% of wall time in one profile before this fix).

| Config | WR vs GreedyBot |
|---|---|
| `time_budget=0.5`, `max_branching=8`, `deepcopy` (all terms) | 75% (15/20) |
| `time_budget=0.1`, `max_branching=8`, `deepcopy` | 40% (12/30) |
| `time_budget=0.1`, `max_branching=8`, `_clone_state` | 63.3% (19/30) |
| `time_budget=0.1`, `max_branching=4`, `_clone_state` | 46.7% (14/30) — reverted |
| + adjacency cache + grid-free `_material_at_risk` | 66.7% (20/30) |
| + `all_cells_list` cache + set-based free-adjacency + `_fast_counter_copy` | **73.3% (22/30)** |

At `time_budget=0.1` — 5x less time than the original 0.5s benchmark — WR is now
within noise of the original 0.5s result (75%), purely from engineering speedups with
*zero* heuristic changes in this last round. This is the headline result of this
session: the bot got fast enough to actually ship at training-loop speed without
giving back the quality gains.

### Architectural review: the branching cap, not tuning, was the ceiling (2026-07-06, continued)

73% felt like a plateau worth stepping back from rather than continuing to tune
coefficients. Measured the actual legal-action count over a real game: **mean 18,
median 18, up to 37 — and 78.6% of turns have more legal actions than
`max_branching=8`.** Combined with `_move_priority`'s flat, state-blind buckets
(attack > control > tactic > move > deploy > bolster), this meant: whenever ≥8
attack/control/tactic actions exist, every "move" option was excluded from
consideration *everywhere in the tree, including the root* — and even when it
wasn't, the up-to-8 survivors from the (usually largest) "move" bucket were picked
by arbitrary unit/direction iteration order, not any quality measure. No amount of
leaf-evaluation quality matters if the actual best move is never a candidate.

Two candidate fixes, tested **separately** (the previous session's move-penalty bug
was exactly a lesson in not bundling changes before measuring):

1. **Never truncate the root's candidates** (only cap deeper plies) — the intuitive
   fix, since the root is the decision actually returned. **Regressed to 56.7%
   (17/30).** Root branching is high on exactly the turns this was meant to fix
   (mean 18), so leaving it uncapped starves those turns of *depth* instead —
   profiling caught a case reaching only depth 1 at `legal_root=27`, barely better
   than a single greedy heuristic pass, and worse than GreedyBot's own tuned one.
   The capped root's survivors (attack/control/tactic first, a reasonable default
   already) plus real depth beat an uncapped root with only 1-2 plies of lookahead.
2. **Keep the cap everywhere, but replace the "move" bucket's arbitrary tie-break
   with a distance-to-nearest-target ordering** — reusing the positional term's own
   notion of "target," but computed once per node as a multi-source BFS distance
   grid (`_dist_grid_to_targets`, O(board size) per node) rather than once per
   candidate. **86.7% (26/30)** — the single biggest jump of the whole session, on
   top of everything already found.

Kept (2), reverted (1). The lesson generalizes: the fix wasn't "give the search more
candidates," it was "make sure the candidates it already keeps are the *right* ones."

| Config | WR vs GreedyBot |
|---|---|
| ... (prior rows above) | |
| `time_budget=0.1`, root uncapped, distance-aware ordering | 56.7% (17/30) — reverted |
| `time_budget=0.1`, `max_branching=8` everywhere, distance-aware ordering | **86.7% (26/30)** |

### Attack/control tie-break — the same fix, applied twice (2026-07-06, continued)

Extended `_ordering_key` with tie-breaks for the other two buckets that had none:

- **attack**: a kill (target's stack would hit 0) before mere damage, then lowest
  remaining stack first (focus fire on whatever's closest to dying).
- **control**: stealing an enemy-held base before claiming a neutral one (a steal
  swings `base_diff` by 2 in one move; a neutral claim only by 1).

**93.3% (28/30)** — 14/15 in both seats. Confirms the pattern from the "move" bucket
fix generalizes: an uncalibrated but *state-aware* tie-break inside a capped
branching list beats a state-blind one, consistently, more than any single leaf-eval
term has so far.

| Config | WR vs GreedyBot |
|---|---|
| `time_budget=0.1`, `max_branching=8`, distance-aware "move" ordering only | 86.7% (26/30) |
| + kill/steal-aware "attack"/"control" ordering | **93.3% (28/30)** |

### Weight calibration (2026-07-06, continued)

`POS_COEFF` and `RISK_COEFF` were hand-picked (§ "Threat awareness" above) and never
actually tuned against each other or against win rate. With ordering fixes now
dominating the last two jumps, revisited whether the coefficients matter as much as
they seemed to.

A coordinate-descent sweep (vary `POS_COEFF` with `RISK_COEFF` fixed, then vice versa,
N=10 games/config) found "best" values (`POS_COEFF=0.005` vs. the current 0.01) — but
a confirmation run at N=30 for just that one change told a different story:

| Run | POS_COEFF=0.01 (current) | POS_COEFF=0.005 (candidate) |
|---|---|---|
| N=30, seeds 100000s (same batch as the ordering-fix result above) | 93.3% | — |
| N=30, seeds 400000s/500000s (calibration confirmation batch) | 80.0% | 83.3% |

**The same, unchanged config scored 93.3% and 80.0% on two different N=30 seed
batches** — a 13pp swing from nothing but which random unit compositions got
drafted. Against that, the candidate's 83.3% vs. baseline's 80.0% (both same batch,
3.3pp apart) is not a real signal — expected binomial noise at n=30 is already ~±7pp.
**Kept the current defaults** (`POS_COEFF=0.01`, `RISK_COEFF=0.5*C_MAT`); changing
them on this evidence would be fitting noise, not calibrating.

This result matters beyond just these two coefficients: LookaheadBot's WR vs
`GreedyBot` is now high enough (~80-93%, batch-dependent) that `GreedyBot` is
approaching the same "saturated yardstick" problem that motivated building
LookaheadBot in the first place (`docs/IDEAS.md`'s guiding principle, made about
the trained policy). Further fine-grained calibration will need either a much larger
per-config sample (expensive at real time-per-game) or a tougher/more discriminating
opponent than `GreedyBot` to measure against — e.g. the trained checkpoint, or
LookaheadBot-vs-itself with different configs.

### Move-ordering danger check (2026-07-06, continued)

Explicit ask: stop the search from *offering* obviously bad candidates in the first
place, not just eventually scoring them low. The "move" bucket's distance tie-break
had a real gap — it only asks "does this get closer to the objective", never "is the
destination safe" — so a move that's the single closest approach could also walk a
unit into a free capture, and nothing before this ranked it any worse than a safe
move of similar distance.

Added `_melee_threatened_cells(mover)`: cells adjacent to an opponent unit that could
make a normal attack this turn — deliberately melee-only (no BFS/blocking, unlike
`_material_at_risk`'s full accounting) so it's cheap enough to compute once per node
alongside `dist_grid`. The "move" ordering key becomes `(danger, distance)` instead
of just `distance` — a move into a threatened cell now sorts behind every safe move,
regardless of how much closer it gets.

Benchmarked on the trusted seed batch from the attack/control fix (where the current
code without this change scored 93.3%): **93.3%, identical win/loss pattern.** On a
different batch: 80.0%, inside the already-established 80-93% noise band, not a
regression. Net: this specific change is unproven by benchmark (GreedyBot win-rate
has stopped being able to resolve further changes — see above), but it's cheap and
strictly more correct than what it replaced (never *prefer* a demonstrably unsafe
move over a safe one of similar value), so it's kept on principle, not on measured
win-rate. This is the practical shape of "further calibration needs a better
opponent than GreedyBot" from the previous section, immediately in effect.

### A real benchmark: the trained checkpoint (2026-07-06, continued)

`GreedyBot` just stopped being able to resolve further changes — exactly the
"saturated yardstick" problem `docs/IDEAS.md` describes for the trained policy.
`app/gauntlet.py` already supports a checkpoint field (`--lookahead --no-greedy`, 20
games/pair via `round_robin`), so ran LookaheadBot directly against
`data/warchest_ppo_20260704-1243.pth` — a checkpoint that itself beats `GreedyBot`
85%, i.e. a genuinely tougher, more informative opponent than anything used to
measure this bot so far:

```
ckpt_20260704-1243[v10] vs lookahead: ckpt wins 75%, lookahead wins 25%
Bradley-Terry: ckpt 1090.0, lookahead 910.0
```

25% is a real, non-collapsed result (not 0%) — LookaheadBot is a genuine opponent for
the trained policy, not noise — but there's a large, honestly-measured gap left.
Unlike the `GreedyBot` numbers, this benchmark isn't saturated yet, so it's the
right one to use for validating any further change (heuristic terms, ordering,
depth/branching tuning) instead of continuing to read tea leaves from `GreedyBot`
batch-to-batch noise.

## Scope

- Search depth: 5-10 plies, which will span 2-3 hand refills (draws) per side.
- Reads the opponent's current hand directly (`state.hands[opponent]`) rather than the
  `E_opp_hand` estimate the observation encoder exposes to normal agents — a "cheat" used
  deliberately as a stress-test / exploitability-probe knob, not baked into the bot's name.
  This connects to `docs/IDEAS.md` #4 (exploitability probe): a hand-seeing
  heuristic search is a cheap proxy for "train a best-response against the frozen agent,
  see how badly it loses," without training a separate network.
- Also reads both players' bag `Counter`s directly (exact composition, not an estimate) —
  needed for the future-draws approach below, and cheap since it's the same kind of direct
  state access as the hand.
- Name: `LookaheadBot`, not `OracleBot` — hand visibility is a constructor flag
  (`see_opponent_hand=True/False`), so the same class can run in a fair mode (estimate via
  the existing `E_opp_hand` distribution) or a stress-test mode (exact hand). Keeps one
  implementation instead of two, and avoids accidentally using the cheating mode as the
  "fair" benchmark for measuring the trained policy's own progress.

## Architecture note: two existing call conventions

- `GauntletAgent.act(env)` (`src/services/gauntlet.py`) already passes the **full env**, not
  a sanitized obs — a forward-simulating bot fits this contract with no interface changes.
- `Bot.act(obs)` (`src/services/bots/base.py`), used by `ppo.py` / `rollout_core.py` /
  `opponent_pool.py` during training, only receives obs — no env/state access, so a
  lookahead bot can't drop into the training rollout loop as-is.
- Decision: build for evaluation only for now, against whichever contract is convenient
  internally. `Bot` and `GauntletAgent` are expected to be merged into one interface soon
  regardless, so this isn't worth an adapter — lives in `src/services/bots/lookahead_bot.py`
  next to `greedy_bot.py`.

## Components

1. **Forward-simulation harness** — apply an `action_id` to a `deepcopy` of `env.state` (or
   equivalent) without mutating the live env or its history; needs verification that the
   internal apply-action path has no side effects that leak when run on a copy.
2. **Ply unit** — decided: each `action_id` (including a `pending` tactic's continuation
   clicks) counts as one ply. Simpler than auto-resolving tactic chains; accepted that a
   tactic-heavy line "spends" more of the ply budget than a plain move.
3. **Depth control** — decided: no fixed ply target. Iterative deepening bounded by a
   wall-clock budget (~0.5s, see future-draws section) — return the best move found when
   time runs out. 5-10 plies is the expected *typical* depth this reaches, not a hard target.
4. **Branching reduction** — move ordering (reuse `GreedyBot`'s priority: attack → control →
   move → deploy → face-down), top-K expansion per node, alpha-beta pruning (valid because
   the search is deterministic within one hand-cycle once both hands are known). Deploy-verb
   branching: the existing legal-action mask is enough, no extra filtering planned.
5. **Leaf evaluation heuristic** — material diff, base control diff, material-at-risk diff,
   tempo/initiative. Prefer reusing the existing PBRS material term from `ppo.py` over
   inventing new weights from scratch. Weight tuning is explicitly deferred — ship with
   reasonable hand-set weights now, revisit calibration as a later phase once the search
   itself works (open, see below).
6. **Opponent hand + bag visibility** — constructor flag as described in Scope.
7. **Mid-round chance events** beyond hand-refill draws (e.g. Warrior Priest's bonus-coin
   draw) — decided: approximate with an expected outcome rather than modeling as explicit
   chance nodes. Not worth the complexity relative to how often they matter.
8. **Future draws** — decided, see dedicated section below.

## Phased delivery

- **M0** — forward-simulation harness; verify it reproduces real `env.step()` results
  exactly on the same action.
- **M1** — alpha-beta bounded to the current hands (no draw crossing), hand/bag visibility
  flag, leaf heuristic, move ordering + top-K, wall-clock iterative deepening. Benchmark vs
  `GreedyBot` via `gauntlet.py`.
- **M2** — add single-determinization future-draw handling (below), measure quality and
  wall-clock cost against the 0.5s budget.
- **M3 (deferred, separate phase)** — heuristic weight calibration (hand-tune vs. e.g.
  grid-search against `GreedyBot`); revisit only once M0-M2 show the search itself works.
- **M4 (later, not blocking)** — `Bot`/`GauntletAgent` interfaces are expected to merge; once
  that happens, wire this bot into `opponent_pool.py` for training, not just evaluation via
  `gauntlet.py`.

## Future draws: decided approach

The bag is a `Counter` and `_draw_one()` samples via `np.random.choice` at draw time — there
is no predetermined future sequence to "peek" at, only a known probability distribution. Two
facts pinned down the choice:

- Both bags' exact composition is now visible to the bot (see Scope) — the only remaining
  randomness is *draw order*, not *draw contents*. This removes the epistemic-uncertainty
  argument for modeling a probability distribution at all: any single sampled order is a
  fully legitimate possible future, not an approximation of hidden information.
- Time budget is tight (~0.5s) and the bot is meant to be called repeatedly during training
  rollout collection, not once per careful eval move. Exact reproducibility is not needed.

**Decision: single determinization per call (Monte Carlo determinization with N=1), no
expectimax, no multi-sample averaging.** At the start of a search call, sample one concrete
realization of both players' upcoming draws from the known bag `Counter`s, then run a plain
deterministic alpha-beta with iterative deepening against that fixed world. The next call (next
move) samples a fresh realization independently.

Why this beats the alternatives given the constraints above:
- Exact expectimax over the hypergeometric distribution is too expensive — 3 coins from ~5-6
  types at 2-3 draw points, stacked on existing per-ply action branching, won't fit 0.5s.
  Not worth doing now that bag contents aren't actually hidden — there's no accuracy gain to
  justify the cost.
- Multi-sample determinization (N=8-16) is also ruled out purely on budget — each sample
  costs a full depth-10 alpha-beta pass.
- Substituting a fixed "expected" hand at each draw boundary is no cheaper than sampling one
  real realization (both cost one deterministic pass), but a sampled realization is an actual
  possible world while an "expected" hand may not correspond to any real hand.
- Common-random-numbers only matters when comparing multiple samples against each other;
  with N=1 there's nothing to compare, so it's dropped.
- Single-call variance-blindness (the bot can't see "what if I get unlucky here") is accepted
  as a cost — it's queried fresh on every move during training, so the noise averages out
  across a game/run rather than biasing any one decision systematically.
