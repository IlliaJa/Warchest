# Training history — implemented improvements

Consolidated record of all applied fixes and architectural changes, oldest first. For the reasoning behind each item see the source doc noted in each section header.

---

## REINFORCE era: correctness fixes (~2026-05-22 to 05-25, pre-PPO)

*Source: `docs/IDEAS.md`. These unblocked training from producing a policy worse than random.*

| # | What was fixed | Effect |
|---|---|---|
| 1 | **Egocentric observation.** `generate_observation` now always puts the active player in slot 0; swaps base counts and board channels. Dropped the `active_player` scalar. | Eliminated directly contradictory gradients in self-play. Single biggest unblocking fix — went from WR < 0.5 to meaningful learning. |
| 2 | **`observation_space` shape.** Declared `global_features=3` but emitted 4. Fixed to match. | Removed latent shape mismatch. |
| 3 | **Reward shaping reset.** `MOVE_ON_BASE_REWARD` and `MOVE_NEAR_BASE_REWARD` were gated by flags that were never reset — silent after the first visit per episode. Replaced with a live distance-based signal. | Gave the critic a signal in the middle 80 turns of each game. |
| 4 | **`make_random_step` credit assignment.** When the policy picked an invalid action the env ran a random step but paired the random reward with the policy's log-prob. Now treated as a non-learning step. | Latent footgun removed (zero-count in logs, but wrong in principle). |
| 5 | **Random player's GAE.** The GAE loop ran for both players even when one was the random bot, producing meaningless advantages and a spurious `critic_loss≈0.038` in every log line. Now skipped for the random side. | Log clarity; minor compute saving. |
| 6 | **Sorted units ordering.** `generate_observation` sorted by class name; `get_active_player_units` used raw `board.units` order. Now both use `board.units` directly. | Prevented silent unit-ID mismatch when a second unit type is added. |
| 7 | **`gamma=0.99`.** Previous `gamma=0.9` made `0.9^99 ≈ 3×10⁻⁵` — win reward invisible to the critic for the first 50 turns. | Propagated win signal meaningfully through long episodes. |
| 8 | **`CLAIM_BASE_REWARD` balance.** Was 0.03 vs WIN_REWARD=1.0; cumulative claim reward (0.12) was negligible relative to the win. Raised to give claims comparable signal to the win in truncated episodes. | Gave the policy a reason to claim bases instead of just not losing. |
| 9 | **Return normalisation for critic.** Advantages were normalised but returns (the critic target) were not. With `gamma=0.99` and dense rewards the return scale grew unbounded. Added running mean/std on returns. | Stabilised critic targets across varying episode lengths. |

---

## Architecture migration: REINFORCE → PPO (2026-05-26 to 05-29)

*Source: `docs/IDEAS.md`. Large structural changes that moved the project from REINFORCE to PPO.*

**Separate actor/critic encoders (#13).** Gave `Policy` and `Critic` independent `board_encoder + unit_encoder` instances. Critic now takes a privileged `opp_onehot` input the actor never sees; separate `Adam` instances (`lr_actor=1e-4`, `lr_critic=3e-4`) honour different learning rates without interference. See `docs/rl_algorithms.md` → *Architecture: shared vs separate actor-critic encoders* for the full rationale.

**PPO migration (#14).** Replaced REINFORCE+GAE with PPO (`src/app/ppo.py`). Same actor-critic network; added rollout buffer, clipped surrogate loss, and inner epoch loop. `src/app/reinforce.py` retained as legacy reference.

**Opponent pool (#16).** `src/services/opponent_pool.py` — weighted sampler over random / greedy / past policy snapshots. Breaks the co-adaptation cycle that caused WR to *decline* over training under pure self-play.

**Hex-aware encoder (#15).** Replaced the standard 2D `Conv2d` with `HexConv2d` (custom kernel using the 6 hex-direction offsets). Removed the false-diagonal / missing-neighbour bias of the 3×3 square kernel on a hex grid. See `docs/decision.md`.

---

## PPO stability and tuning fixes (2026-05-27 to 05-28)

*Source: `docs/improvement_ideas.md` run `ppo_20260527-191432`. Starting point: WR vs greedy ~5–15%, critic MAE growing, entropy collapsing. Ending point: WR ~49%.*

| ID | What was fixed | Effect |
|---|---|---|
| C1 | **`ppo_epochs=4`.** Was 1 — PPO was silently A2C, clipping never active. Set to 4 (canonical default). | ~4× more gradient updates per episode. Biggest single fix. |
| C2 | **Normalise `action_count` in globals.** Was emitted raw (0–100), 100× larger than all other features. Divided by `max_actions`; also normalised `my_bases/6`, `opp_bases/6`. | Critic MAE started trending down instead of up. |
| C3 | **Opponent pool diversity.** `snapshot_every=1` meant the pool contained only the last 20 policies by batch 50. Set `snapshot_every=3`. | Stabilised the return distribution; older, weaker opponents stayed in the pool. |
| C4 | **PBR reward shaping + steal incentive.** Two bugs: (1) `phi_after` was measured after own move, not after opponent — breaking the potential-based telescoping; (2) stepping onto an enemy-controlled base triggered `MOVE_NEG_REWARD_PER_TURN`, punishing the very action that leads to stealing. | Unblocked the "policy never steals" failure mode. |
| C5 | **Clipped value loss.** Plain `F.mse_loss` on the critic caused two NaN events (batches 133, 143, `grad_c=46`). Added PPO-style clipped value loss with `clip_range=0.2`. | Eliminated NaN events and late-training critic instability. |
| C6 | **180° board rotation for P2.** The spatial encoder saw absolute board orientation — player 2's units always in the bottom-right corner. Added `np.rot90(board, 2)` (and action remapping) when `active_player==2`. | Lifted WR vs greedy from ~33% to ~49%. Effectively doubled spatial data efficiency. |
| C7 | **Entropy coefficient.** Was 0.005 — quickly outscaled by actor loss. Raised to 0.025. | Maintained exploration deeper into training; prevented premature commitment to the first "good enough" move. |
| C11 | **`hidden_dim=128`.** Was 64 — a 49× spatial compression bottleneck. Doubled to 128. | Modest convergence speed improvement; meaningful with the separate-encoder design. |
| C13 | **`HexConv2d` board encoder.** Already done as part of the PPO architecture migration above, confirmed in `src/services/policy/policy.py`. | — |

---

## 16-unit roster + per-game disjoint drafting (2026-06-01 to 06-02)

*Originally a later milestone on the (now-retired) full-game roadmap; the drafting mechanic (planned even later on that roadmap) was pulled forward into this same change, so the old `full_game_plan.md` is retired. Schema-breaking — a new obs/action generation (`OBS_VERSION=4`); prior saved models retired.*

| What changed | Detail |
|---|---|
| **Full 16-unit vanilla roster** | `environment/roster.py` is the single source of truth (id/icon/colour/total-coins); unit classes are generated from it in `units/__init__.py`. All units share move/attack/control/deploy/bolster (tactics/attributes land in "Tactics, attributes & restrictions" below). |
| **Per-game disjoint draft** | `set_init_state` samples 8 distinct types and gives 4 to each player, disjoint, + the shared Royal coin. Per-player `build_bag`/`build_supply` replace the old global `INITIAL_BAG/SUPPLY/INITIAL_OWNED`; `GameState.owned()` is per-composition. |
| **Full-roster sizing** | Action space 796→1776 (16 deploy verbs → `N_VERBS=30`; claim/pass over 17 coins; recruit 16×17). Board planes 10→38 (6 terrain + 16 own + 16 opp, stack-valued). `GLOBAL_DIM` 28→174, `PRIV_DIM` 9→51. |
| **No policy/critic edits** | The factored head groups by verb (8), not by type, and reads sizes from env constants — Policy/Critic/RolloutBuffer adapted automatically. |
| **Renderer + bots** | Coins use per-unit colours/glyphs (2-letter codes) from the roster; GreedyBot verb-offset constants updated; RandomBot unchanged (mask-driven). |
| **Tests** | `tests/test_phase3.py` (23 total green): disjoint-draft, roster totals, encode/decode round-trips, per-type legality/planes, recruit accounting, and a coin-conservation invariant across full random games. Old `test_phase1*` retired (dead schema); `test_phase2` (factored head) still passes. |

---

## Tactics, attributes & restrictions — full base game (2026-06-28 to 06-29)

*The last milestone of the (now-retired) full-game roadmap. Implemented incrementally (Cavalry → SELECT/Archer →
clusters 1–4) over ~4 weeks until all 16 units had their real abilities. Schema-breaking; prior models retired.*

**Why a new state machine instead of new action ids.** Tactics are multi-step and heterogeneous
(move-then-attack, ranged, line charges, friendly-unit grants, stack-paid repeats). Encoding each as
an atomic action would have exploded the action space and needed a bespoke mask per tactic. Instead a
tactic resolves as a *sequence of masked clicks* through a `GameState.pending` continuation: while
`pending` is set the turn does **not** pass, `get_possible_actions` returns only the legal next clicks,
and those clicks reuse the existing move/attack/SELECT verbs. The policy tells "normal maneuver" apart
from "tactic follow-up" via a **pending-context one-hot** appended to the globals. Net action-space
cost across all 16 units: one spatial `TACTIC` verb (initiate), one `SELECT` verb (arbitrary target/
destination cell), and one non-spatial `DECLINE` slot (end an optional continuation) — everything else
is masking.

### Incremental slices (the order it was actually built)

| Date | Slice | Schema move | Proof |
|---|---|---|---|
| 2026-06-02 | **Scaffolding + Cavalry** (`move_then_attack`) — the pending sub-turn machine, `TACTIC` verb, `DECLINE` slot, context one-hot | `N_VERBS` 30→31, `N_FACTORED_VERBS` 8→10, `GLOBAL_DIM` 174→177, `OBS_VERSION`→5 | `TACTIC@unit → move-dir (mandatory) → attack-dir (optional)`, coin paid once at initiation; optional attack ⇒ no softlock when no enemy adjacent |
| 2026-06-28 | **SELECT primitive + Archer** (`ranged_attack` any@2) + restriction `can_normal_attack=False` | `N_VERBS` 31→32, `N_FACTORED_VERBS` 10→11, `PENDING_CTX_DIM` 3→4, `GLOBAL_DIM`→178, `OBS_VERSION`→6 | tactics renamed by **mechanic not unit** so they reuse (`ranged_attack` covers Archer any-target and Crossbowman straight-line); test suite reorganized phase-named files → domain-named files |
| 2026-06-28 | **Clusters 1–4** — the remaining movement/ranged tactics, grant-flavor SELECT (friendly → nested granted maneuver), and all passive/triggered attributes | → `PENDING_CTX_DIM=15`, `GLOBAL_DIM=189`, `N_VERBS=32`, `ACTION_SPACE_SIZE=1875`, `OBS_VERSION=8` | 400-game random-play stress across every drafted composition: zero crashes / softlocks / conservation violations |

Per-unit mechanics (card text in `docs/UNITS.md`):

| Unit | Mechanic | Notes |
|---|---|---|
| Cavalry | `move_then_attack` | move (mandatory) → attack (optional) |
| Light Cavalry | `move_to` (≤2) | SELECT a reachable destination |
| Lancer | `line_charge` (≤2) | SELECT an in-line enemy → move + strike; no normal attack |
| Archer | `ranged_attack` any@2 | SELECT a distance-2 enemy; no normal attack |
| Crossbowman | `ranged_attack` line@2 | clear straight line; may also normal-attack |
| Berserker | `extra_maneuvers_from_stack` | pay own stack coins to keep maneuvering |
| Footman | `maneuver_each` + 2 copies | one maneuver per Footman; two may be on board |
| Pikeman | `counter_when_attacked` | adjacent attacker loses a coin (not absorbable) |
| Ensign | `grant_move` | SELECT a friendly ≤2 → it moves, ending ≤2 from the Ensign |
| Marshall | `grant_attack` | SELECT a friendly ≤2 → it makes a normal attack |
| Mercenary | `maneuver_after_recruit` | recruiting its coin grants a free maneuver |
| Scout | `deploy_adjacent_to_friendly` | deploys next to any friendly unit |
| Royal Guard | `royal_move` + `absorb_from_supply` | Royal-coin move ≤2 to a controlled loc; soaks hits from supply |
| Swordsman | `move_after_attack` | optional free move after attacking |
| Knight | `only_attackable_when_bolstered` | a stack-1 attacker cannot hit it |
| Warrior Priest | `bonus_action_after_attack_or_control` | draw a coin and use it at once |

| What changed | Detail |
|---|---|
| **Pending sub-turn machine** | Multi-step tactics resolve as masked continuation clicks via `GameState.pending`; the turn does not pass until it clears. A pending-context one-hot (`PENDING_CTX_DIM`) in the globals tells the policy which continuation it is in. 14 pending kinds. |
| **SELECT primitive** | A spatial verb (`SELECT_VERB`) whose `(r,q)` is an arbitrary *target/destination* cell (not a direction) — the piece the directional move/attack verbs can't express. Drives ranged attacks, move-to destinations, line charges, and friendly-unit grants. |
| **All 16 units** | Mechanic-named tactics (`move_then_attack`, `move_to`, `line_charge`, `ranged_attack` any/line, `grant_attack`, `grant_move`, `maneuver_each`, `royal_move`) + boolean attribute flags (`counter_when_attacked`, `move_after_attack`, `extra_maneuvers_from_stack`, `bonus_action_after_attack_or_control`, `only_attackable_when_bolstered`, `deploy_adjacent_to_friendly`, `maneuver_after_recruit`, `absorb_from_supply`, Footman `max_on_board=2`). Named by mechanic so they reuse across the roster and DLC. |
| **Shared resolution** | One `_resolve_attack` (with Pikeman counter + Royal-Guard absorb-from-supply) and `_fire_maneuver_triggers` hook feed every attack/maneuver path; a `_resolve_free_maneuver` powers granted/free/Footman maneuvers. |
| **Schema** | `N_VERBS` 31→32 (`SELECT`), `N_FACTORED_VERBS` 10→11, `PENDING_CTX_DIM`→15, `GLOBAL_DIM`→189, `ACTION_SPACE_SIZE`→1875, `OBS_VERSION` 5→8. Policy/Critic/buffer/trainer unchanged (head reads sizes from constants). |
| **Documented simplifications** | A bonus/repeat maneuver (WP drawn coin, Berserker repeat, Footman-tactic maneuver) can't use the TACTIC verb to start a *named* tactic — caps nesting, not the count of maneuvers. Granted attacks/moves (Marshall/Ensign) **do** chain the granted unit's attribute (FAQ). Royal Guard auto-absorbs from supply when able. |
| **Tests** | Test suite reorganized from phase-named files into domain files (`test_units`, `test_tactics`, `test_attributes`, `test_game_mechanics`, `test_action_space`, `test_bots`, `test_policy` + shared `_helpers`); 63 total. A 400-game random-play stress exercises all pending kinds with zero crashes/softlocks/conservation violations. |

---

## Material-at-risk + expected-opponent-hand + base-control reach observation (2026-07-03)

*Source: `docs/observation_improvement.md` (full plan/rationale) + `docs/IDEAS.md`
"base-control reach planes". Schema-breaking (`OBS_VERSION` 9→10, `BOARD_CHANNELS`
46→48, `GLOBAL_DIM` 189→211); prior saved models/pool snapshots retired. Three
actor-legibility features, each a reduction the location-blind heads compute poorly
from raw state, bundled into one bump but intended as separate A/B arms.*

| What changed | Detail |
|---|---|
| **Material-at-risk scalars** | Two globals: `own/opp_material_at_risk = Σ min(hits, stack)` over each side's on-board units, from the **raw** threat grids (pre-clip). Directly encodes the bolster/trade/extend question — how much committed material can die this turn — instead of asking the location-blind heads to reduce the enemy-threat plane × unit-occupancy spatially. |
| **`E_opp_hand`** | 17-wide expected opponent hand = `hidden_pool · opp_hand_size / hidden_total` (hypergeometric mean). Actor-side estimate of what the opponent can play *this round without redrawing*; decays to 0 as they empty their hand, unlike the static `hidden` pool. The critic already sees the true split via `PRIV_DIM`, so this closes an actor/critic asymmetry without leaking private state. The `hidden` pool's binary `≥1` gate in the threat planes was left worst-case (deliberate — max for "where can I die," mean for "how loaded are they"; see the parked "likelihood-weighted threat-plane magnitude" idea). |
| **Base-control reach planes** | 2 new board planes (`own_base_reach`, `enemy_base_reach`): 0/1 over base cells a side could move a unit onto and claim this turn — the objective-analogue of the threat planes for the win condition (control 6 bases), which the schema previously encoded only as static positions/counts. Reuses `_reachable`; new side-effect-free helpers `_maneuver_range` (1 / move_to·royal_move max_dist / Berserker stack) and `_is_claimable_base` (mirrors `Board.is_valid_claim`'s cell test), gated by coin availability exactly as the threat planes. |
| **Base-control reach scalars** | 3 globals: `bases_i_can_claim` this turn, `my_bases_under_flip_threat`, and a `win_proximity_alarm` (opponent one base from winning **and** able to take a base this turn — the most decisive state in the game, previously requiring the net to assemble it from three separate features). |
| **Tests** | New `tests/test_obs_features.py` (helper semantics: `_is_claimable_base`, `_maneuver_range`, `_base_reach_grids` incl. gating/in-place/own-base/blocked cases, material-at-risk cap, `E_opp_hand` formula). Schema pins updated in `test_action_space.py`/`test_threat_planes.py`; the `test_obs_global_vectorized.py` reference reimplements all three blocks (its exact-equality P1/P2 check relaxed to `atol=1e-6` for float32 op-order on the divide). Full suite: 95 passed. Policy+Critic forward passes verified on the new schema. |
| **Training run** | Not yet run — A/B protocol (baseline / +material / +E_opp_hand / +base-reach) is the remaining work per `docs/observation_improvement.md`. |

---

## Threat/position-aware observation + deeper trunk (2026-07-02)

*Source: `docs/IDEAS.md` "Architectural note — the agent can't see the board as
one position". Schema-breaking (`OBS_VERSION` 8→9, `BOARD_CHANNELS` 38→46);
prior saved models/pool snapshots retired.*

Three complementary fixes for the same underlying limit — a radius-2
receptive field, a location-blind global pool, and reach that isn't a fixed
radius (a single coin can chain into several hits):

| What changed | Detail |
|---|---|
| **Threat/reach planes** | 6 new board planes (`own_melee/ranged/charge`, `enemy_melee/ranged/charge`) give each cell a *graded* hit-count — how many hits a side could land there this turn — instead of asking the CNN to re-derive tactic geometry. Reuses the env's existing legal-move geometry (`_hex_distances`, `_reachable`, `_can_attack`-adjacent patterns) via new side-effect-free `_threat_*` helpers that (unlike the legal-action helpers) don't depend on `self.active_player`, since they run for both sides regardless of whose turn it is. |
| **Berserker closed form** | `extra_maneuvers_from_stack` pays 1 stack coin per *extra* maneuver (the initiating hand coin is free); solving for hits landable at hex-distance `D` from a stack-`S` Berserker gives `hits(D) = max(0, S - D + 1)` — a stack-3 Berserker threatens distance 1/2/3 for 3/2/1 hits, not just its adjacent cells. Implemented in `_threat_berserker_reach` via one `_reachable` BFS plus a per-cell neighbor lookup. |
| **Marshall grant-chaining** | A friendly unit within hex-distance 2 of a Marshall can be activated by the Marshall's coin instead of its own (`_threat_grids`'s single boolean activation gate, avoiding double-counting) — including triggering a granted Berserker's full chain, since the attribute fires on any attack regardless of which coin paid for it. Ensign's `grant_move`→Berserker-reposition combo is a documented gap (narrow reach gain, not worth the added complexity). |
| **Coordinate planes** | `row_coord`/`col_coord`, static ego-centric position — `col_coord` is the flank axis (confirmed via `Board.default_bases`), the substrate for "which side is under threat" reasoning. |
| **Split flank pooling** | `verb_head`/`facedown_head` read a two-way mean pool split along the flank axis (`_split_pool`) instead of a single location-blind global mean — doubles the pooled dim to `2*hidden_dim`. The spatial `policy_head` was untouched (never location-blind). |
| **Deeper trunk** | A 3rd `HexConv2d` layer in both `Policy` and `Critic` → receptive-field radius 3, exactly covering the Lancer's charge. |
| **Tests** | New `tests/test_threat_planes.py` (11 tests): Berserker formula (unblocked + path-blocked), Cavalry/Lancer charge geometry, Archer/Crossbowman ranged targeting, Marshall grant activation (plain unit + chained Berserker), and end-to-end `generate_observation` rotation/normalization wiring. Full suite: 86 passed. |
| **First training run** | `ppo_20260702-082214` (400 batches, same hyperparameters and same `n_batches=400` as the correct pre-change baseline `ppo_20260701-191923`). Final `wandb` summaries are nearly identical (`score_main` matches to 3 decimals; WR/Elo/critic_mae/avg_turns all close), and comparing full eval-checkpoint distributions puts `wr_vs_greedy_eval`/`elo_policy` within ~0.3 pooled standard deviations — **no measurable difference**. Full numbers and what a real test would need: `docs/experiments.md` → "Threat/position-aware observation + deeper trunk (run: `ppo_20260702-082214`)". |

---

## PPO tuning: attack-reward cut, entropy/LR anneal, sparser pool snapshots (2026-07-01)

*Source: the diagnosis of the `ppo_20260630-060400` plateau (score/WR decoupling, flat entropy, self-play treadmill) — action points P0–P3.*

| What changed | Detail |
|---|---|
| **`ATTACK_REWARD` cut 0.1 → 0.02** (P0, partial) | A raw per-attack bonus of 0.1 let a game's worth of attacks rival a full win (+1.0), the leading suspect for the measured score/WR decoupling. `CLAIM_BASE_REWARD` stayed at 0.0 (the circular-claim exploit risk) and `holding_reward` was kept, not dropped — see the annealing fix below and 2026-07-03 entry. |
| **Entropy coefficient annealed** (P1) | `entropy_coeff` linearly decays `0.025 → 0.003` over the run (`entropy_coeff_final`, `ppo.py`) instead of staying constant, so late-training exploration pressure no longer holds entropy at a high floor. |
| **LR decay** (P2, idea C8) | `lr_actor`/`lr_critic` linearly decay to `lr_*_init * lr_final_frac` (`lr_final_frac=0.0` ⇒ decay to 0) over the run, stepped once per batch in `_update_schedules`. Targets the late-run `wr_greedy` oscillation the analysis flagged. |
| **Sparser opponent-pool snapshots** (P3) | `snapshot_every` 3 → 15 (`pool_snapshot_every`), so pool opponents span a wider skill range instead of near-copies of the current policy — fewer near-zero-advantage mirror matches. |
| **`METRICS.md` entropy reference recalibrated** | Replaced the stale 14-action-era "max entropy ≈ 2.64" with the measured current-env ceiling (mean `log(n_legal)` ≈ 1.84) and added `max_entropy` / `entropy_frac` as directly-logged metrics. |

P4 (capacity) and P5 (economy shaping) from the same analysis followed on 2026-07-03, below.

---

## Reward hygiene + material PBRS + critic widening (2026-07-03)

*Source: `docs/rewards.md` (current reward table + rationale) and `docs/decision.md` (2026-07-03 entry). Non-schema-breaking (no `OBS_VERSION` bump) — existing pool snapshots remain loadable, but the reward scale changed enough that a fresh baseline run is the correct comparison.*

| What changed | Detail |
|---|---|
| **Material PBRS** | New potential `phi_material = C_MAT * (boxed_total(opp) - boxed_total(me))`, `C_MAT = 0.015`, telescoped into the reward the same way as base-diff shaping via a new `WarChestEnv.boxed_total(pid)` helper. Densifies the coin/material economy axis, which previously had zero reward gradient (a 200-game bucketed eval found the policy essentially never bolsters and never triggers a stack chain — see `docs/rewards.md`). |
| **Shaping annealing** | The holding reward and the material term are multiplied by `shaping_anneal`, linearly decayed `1.0 → 0.1` over the first half of the run then held at the floor (`SHAPING_ANNEAL_INIT/FINAL/HALF_FRAC`, `_update_schedules`). Keeps early dense guidance while the critic is weak, then hands the final policy back toward the terminal objective — the annealed-`team_spirit` pattern from OpenAI Five. Base-diff PBRS (`SHAPING_C`) is deliberately left constant. `holding_reward` was **annealed, not removed**, to preserve the base-flip-exploit tie-break it was added for. |
| **Truncation reward smoothed (C17)** | Replaced the 0 / −0.5 / −1.0 step function with a base-diff-*proportional* value (`LOSS_REWARD * (0.5 + 0.5 * deficit_frac)`), lowering critic-target variance at the truncation states the agent spends most of its time near, while preserving the old anchor values. |
| **Critic-only widening (Step 5)** | `critic_hidden_dim` 64 → 128, policy left at `hidden_dim=64`. Applied to the critic alone first — the critic's board encoder is independent of the policy's during PPO rollout — so any capacity gain is attributable to the densifier, not conflated with policy capacity. |
| **Logging** | `score_material` and `shaping_anneal` added to per-batch logs and W&B. |
| **Explicitly deferred** | `ATTACK_REWARD` (0.02) was **kept, not subsumed** into material PBRS — both currently fire on the same box-a-coin event, flagged for the next A/B (`docs/rewards.md`, `docs/IDEAS.md`). Widening the *policy* `hidden_dim` (not just the critic) is untested. The controlled A/B against the pre-change baseline is still owed (`docs/IDEAS.md` #3): a run started 2026-07-03 (`ppo_20260703-142941`) was in progress at time of writing. |

---

## Phase-4 tactic rule-correctness fixes (2026-07-12)

*Source: bugs found while play-testing the human-vs-policy game (`src/app/play.py`). Both were rule discrepancies in the tactic/attribute resolution, not learning-loop changes; no `OBS_VERSION` bump (pending kinds unchanged).*

| What was fixed | Detail |
|---|---|
| **Cavalry `move_then_attack`: attack is mandatory, not optional** | The tactic is *"move and then attack"* — both halves are required. Previously the attack step was `optional=True`, so the tactic could be started whenever the Cavalry could move and then degrade into a bare move-that-cost-a-coin. Now `_move_then_attack_moves` gates the tactic (and the move-step mask) to only those steps that land adjacent to an attackable enemy, and the follow-up attack step is `optional=False`. A Cavalry with no completable attack simply isn't offered the tactic (it still has a normal move). Earlier history row listed this as "move (mandatory) → attack (optional)" — that was the bug. |
| **Warrior Priest bonus coin can start a tactic** | `bonus_action_after_attack_or_control` draws a coin and spends it on one action. `_bonus_actions` previously filtered out `TACTIC` verbs (a deliberate simplification to avoid nesting a second pending sub-turn), so e.g. a drawn Cavalry coin could never trigger the Cavalry's tactic — the option was greyed out. Now tactic-initiation is allowed; the tactic's own pending sub-turn replaces the `bonus_action` pending, and `_perform_continuation` only clears `pending` when the bonus action did *not* install a nested one (`self.state.pending is p`). |
| **Tests** | `test_tactics.py`: attack step now asserted mandatory + new `test_cavalry_tactic_unavailable_without_a_completable_attack`. `test_attributes.py`: new `test_warrior_priest_bonus_coin_can_start_a_tactic`. Full suite: 116 passed. |
