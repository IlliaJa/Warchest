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
| **Explicitly deferred** | `ATTACK_REWARD` (0.02) was **kept, not subsumed** into material PBRS — both currently fire on the same box-a-coin event, flagged for the next A/B (`docs/rewards.md`, `docs/IDEAS.md`). Widening the *policy* `hidden_dim` (not just the critic) is untested. The controlled A/B against the pre-change baseline is still owed (`docs/IDEAS.md` #R3): a run started 2026-07-03 (`ppo_20260703-142941`) was in progress at time of writing. |

---

## Measurement + opponent infrastructure (2026-07-05 to 07-26)

*Source: the retired `docs/next_steps.md` (Steps 1, 2, 4 + its 2026-07-11 "search-augmented
policy" section), whose implemented half this entry replaces. Per-bot detail lives in
`docs/bots.md`; measured results in `docs/experiments.md` (2026-07-07, 2026-07-08). Non-
schema-breaking apart from the encoder extraction, which was byte-for-byte identical.*

**Why this block exists.** Two facts had made the project's instruments useless: WR vs
`GreedyBot` had saturated at ~100% (a myopic 1-ply bot that never bolsters, recruits, or
initiates a tactic — beating it says nothing), and "beats its own predecessors" is a *relative*
signal that a non-transitive strategy space can produce with no absolute gain. Everything below
is one push to restore a trustworthy yardstick before training longer.

| What shipped | Detail |
|---|---|
| **Versioned observation encoders** (`a931333`, 2026-07-05) | Obs encoding was extracted out of `warchest_env.py` into `environment/obs_encoders/` (registry + `v10.py`), leaving the engine obs-version-agnostic and exposing stable rules-queries (`unit_threat_footprint`, `attack_enabler_coins`, `unit_base_reach_cells`). `Policy`/`Critic` size their inputs from the paired encoder. `tests/test_obs_golden.py` guards the extraction byte-for-byte. |
| **Checkpoint metadata** (`policy/checkpoint.py`) | A `.pth` is meaningless without its matching encoder + arch + action mapping, all of which had drifted. Checkpoints now carry obs-version + arch metadata, with a legacy bare-`state_dict` fallback. This is what makes cross-era comparison possible at all. |
| **The gauntlet design contract** | The one stable boundary: *an agent receives the canonical game state and returns an action id in the absolute (unrotated) env frame* — each agent does its own ego-rotation and inverse remap internally. Only the forward pass `state → obs → net → action` is needed at eval time, and the game rules were stable across the churn, so **in-process, N instances, no serialization** works (pass the live env to each `act(env)`). Serialization + a subprocess per `git worktree` is reserved solely for resurrecting frozen old commits — deliberately **not built**: at most 1–2 checkpoints would ever justify it, and the gauntlet's value is forward, not a museum. The contract also survives a future action-space rebuild (`docs/IDEAS.md` #14). |
| **Round-robin gauntlet** (`services/gauntlet.py` + `app/gauntlet.py`) | Plays a fixed field all-pairs, K games each, with balanced colors (initiative/side is a real edge), and reports the pairwise WR matrix, a Bradley-Terry (Elo-scaled) ranking, and the **intransitive-triple fraction** — the cycle detector that tells us whether "beats predecessors" is real or rock-paper-scissors. Incompatible checkpoints are skipped rather than resurrected. `tests/test_gauntlet.py` covers the rating math + end-to-end play. |
| **Search bots as a real bar** (2026-07-08 onward) | `LookaheadBot` (alpha-beta + hand-tuned leaf heuristic + move ordering) went 25% → 93% WR vs `GreedyBot` over a tuning pass and became the first genuinely non-saturated opponent; `LookaheadCriticBot` (critic-guided beam) followed, underperformed it 20/80, and was fixed 2026-07-11 (missing critic denormalization → 68–78% vs `lookahead`). `SimGreedyBot` (forward-sim, bolster-aware) replaced the old bot as the gauntlet's default `greedy` (`808e72e`). Full catalogue, tuning log, and per-bot bugs: `docs/bots.md`. |
| **Search bots usable as *training* opponents** | The `Bot`/`GauntletAgent` split (train calls `act(obs)`, search bots need `act(env)`) was the blocker; `rollout_core.py` now routes `_SEARCH_OPP_TYPES` through the live env, and `opponent_pool.py` lazily builds `lookahead_critic`/`puct` (`_get_lookahead_bot`/`_get_puct_bot`) so pools that never sample them pay no import cost. Critic conditioning reuses the existing `pool` one-hot slot via `OPP_ONEHOT_SLOT` — widening `OPP_TYPE_IDX` would change the critic's input dim and invalidate every saved critic. |
| **Diagnostics re-pointed** (`808e72e`) | `eval_bucketed.py` now evaluates against `LookaheadBot` instead of the saturated `GreedyBot`, plus a `--opp-knight-frac` probe (a Knight can only be killed by a bolstered attacker, so forcing one into the opponent's roster stress-tests bolster usage). |
| **Human-vs-policy play mode + game records** (`6257f1a`, 2026-07-12) | `src/app/play.py` + `interactive_renderer.py`/`play_controller.py` let the user play the trained agent in a terminal/matplotlib UI with a critic eval overlay; finished games are persisted via `game_record.py` (`--save-dir data/games`) so an impression can become a metric. This was the cheapest absolute-strength signal available, and it immediately paid off — it surfaced the two rule bugs in the entry below. |
| **Policy-prior search: `PuctBot`** (`419ec07`, 2026-07-26) | The "search-augmented policy" idea: replace `LookaheadBot`'s hand-tuned move ordering with the trained policy's own priors (PUCT selection = prior × critic value over a visit-counted tree), which is what makes AlphaZero's search worth hundreds of Elo over its bare network. Wired into both the gauntlet and training as an opponent; root visit counts exposed. Also `PolicyCriticBot`/`RoundCriticBot` (`f4ba822`, policy-prior-guided and round-bounded lookahead). |
| **Expert iteration** (`419ec07`) | Closes the loop the previous row opens — "search moves become new training targets": `src/services/expert_iteration.py` + `src/app/expert_iteration.py` (gen/distill/loop CLI) self-play `PuctBot` to distill its visit distribution into the policy and game outcomes into the critic. **Outcome: it regressed** — 30 rounds made the policy monotonically weaker, root-caused to teacher ≡ student (policy/search agreement ≈ 0.95). Full diagnosis and the resulting plan: `docs/independent_opponents.md`. |
| **Dense critic targets** (`f4ba822`, 2026-07-13) | Opt-in `--dense-critic-targets`: an auxiliary MC-return regression on *opponent*-decision nodes (`rollout_core.collect_dense` → `aux_*` samples → a separate critic minibatch loop, scaled by `aux_critic_coeff=0.5`), leaving the policy path and the main GAE critic targets untouched. Off by default and **never A/B'd** — tracked as `docs/IDEAS.md` #12. |
| **What it measured** | The 1500-batch long run (`ppo_20260706-194732`) and the first 5-agent gauntlet, both in `docs/experiments.md`. Net: the long run's gains were real (the checkpoint topped the field at 1177 BT-Elo, beating both search bots), but `wr_greedy` had saturated by batch ~1200, so the back half trained against a dead signal — the concrete argument for keeping an independent opponent in the pool. |

---

## Phase-4 tactic rule-correctness fixes (2026-07-12)

*Source: bugs found while play-testing the human-vs-policy game (`src/app/play.py`). Both were rule discrepancies in the tactic/attribute resolution, not learning-loop changes; no `OBS_VERSION` bump (pending kinds unchanged).*

| What was fixed | Detail |
|---|---|
| **Cavalry `move_then_attack`: attack is mandatory, not optional** | The tactic is *"move and then attack"* — both halves are required. Previously the attack step was `optional=True`, so the tactic could be started whenever the Cavalry could move and then degrade into a bare move-that-cost-a-coin. Now `_move_then_attack_moves` gates the tactic (and the move-step mask) to only those steps that land adjacent to an attackable enemy, and the follow-up attack step is `optional=False`. A Cavalry with no completable attack simply isn't offered the tactic (it still has a normal move). Earlier history row listed this as "move (mandatory) → attack (optional)" — that was the bug. |
| **Warrior Priest bonus coin can start a tactic** | `bonus_action_after_attack_or_control` draws a coin and spends it on one action. `_bonus_actions` previously filtered out `TACTIC` verbs (a deliberate simplification to avoid nesting a second pending sub-turn), so e.g. a drawn Cavalry coin could never trigger the Cavalry's tactic — the option was greyed out. Now tactic-initiation is allowed; the tactic's own pending sub-turn replaces the `bonus_action` pending, and `_perform_continuation` only clears `pending` when the bonus action did *not* install a nested one (`self.state.pending is p`). |
| **Tests** | `test_tactics.py`: attack step now asserted mandatory + new `test_cavalry_tactic_unavailable_without_a_completable_attack`. `test_attributes.py`: new `test_warrior_priest_bonus_coin_can_start_a_tactic`. Full suite: 116 passed. |

---

## Draw-share observation features + capacity/exploration bundle (2026-07-25 to 07-26)

*Source: `docs/IDEAS.md` ideas 2, 5, 7, 8, 9 — retired from that doc once shipped; this section is their record. Schema-breaking (`OBS_VERSION` 10→11, `GLOBAL_DIM` 211→245, `BOARD_CHANNELS` unchanged at 48); prior pool snapshots retired, `obs_encoders/v10.py` kept for the gauntlet. Everything below landed inside a two-day window, alongside `PuctBot` + the expert-iteration pipeline in the same commit (`419ec07`), so **no piece got the single-variable A/B the ideas called for** — they are mutually confounded.*

| What changed | Detail |
|---|---|
| **Draw-share features (idea 2)** | `obs_encoders/v11.py` = v10 + two own-side per-type vectors over `DECK` (C=17), both `[0,1]` shares of "what fraction of my draws is type `t`": `p_soon[t]` = expected copies of `t` in the *next* hand ÷ `HAND_SIZE` (hypergeometric mean, with the one-reshuffle branch when `bag_size < HAND_SIZE`), and `p_mean[t]` = `recirc[t] / Σ recirc` over the recirculating pool (bag + hand + both discards) — the steady-state structure recruiting moves. The **gap between them is the signal** (`p_soon > p_mean` = loaded now; `<` = key coins stuck behind a reshuffle), so no third feature. Own-side only: `p_soon` needs the bag↔discard split, which is hidden for the opponent. |
| **Why a feature and not a reward** | Both plausible reward forms are traps. "Smaller bag = better" as PBRS pays out whenever coins leave the cycle — and the biggest exit is *your own units dying* (boxed coins), so it would reward losing material, contradicting material PBRS. "Higher per-unit draw rate = better" is also wrong: a fully-recruited 17-coin bag scores `3·4/17 ≈ 0.71` per unit vs the 9-coin starting bag's `0.67`, yet is the worse bag. The real target — reliably drawing *the unit you have chosen to play next* — is **policy-defined**, so any fixed potential picks the wrong target somewhere (Herfindahl pushes monotype; fielded-average fails to flag 4-of-each). Hence observation, letting the policy own the preference. |
| **Policy widened (idea 5)** | `hidden_dim` 64 → 128 (commit `86d5ccd`), one clean step, no added conv depth — the policy-side counterpart to the 2026-07-03 critic-only widening. `critic_hidden_dim` went 128 → 192 in the same window (its `critic_mae ≈ 0.5 × return std` underfit signature, `docs/rewards_improvements.md` Step 5), so policy and critic capacity moved together. |
| **Verb-marginal entropy bonus (idea 8)** | `Policy._verb_marginal_entropy` + `verb_entropy_coeff` 0.02 → 0.01 floor (commit `808e72e`): a dedicated entropy term on the top-level 11-way `P(verb)` marginal, because the flat joint entropy is dominated by the ~1875 spatial actions and barely constrains the verb head — `bolster_per_ep` had crashed 10.2 → 0.3 within ~80 batches (`ppo_20260713-144024`). Unlike the flat coefficient it anneals to a *meaningful* floor, so rare verbs stay sampled. |
| **`lam` 0.95 → 0.97 (idea 7)** | Shipped inside a bundled "ppo params update" (`cf2a9e3`, `419ec07`), never as the dedicated sweep. Propagates the terminal win/loss further back with less bias, densifying credit for delayed-payoff actions without touching the reward. |
| **Tactic-lead logging (idea 9)** | `eval_bucketed.py` records `tactic_base_leads` — base-lead at every tactic initiation — and prints the reverse-causation vs. execution-gap read for the "tactics correlate with losing" finding (11.5% usage, WR 0.696 with vs 0.915 without). Logging only; no conclusion was ever written up. |
| **Measured effect (mixed, not attributable)** | The first v11 + `hidden_dim=128` checkpoint (`ckpt_20260727-0506`) ranked **BT-Elo 923 — last of four** in a gauntlet against three prior v10/`hidden_dim=64` checkpoints (1000–1043). The drop was never diagnosed (under-training vs. self-play-pool overfitting vs. the features/width themselves), and the owed re-tests were dropped when the effort moved to `docs/independent_opponents.md`. The same checkpoint is, however, the **strongest** agent in the later 30-round ExIt field (1156.1, beating all 30 of its own distilled descendants — `independent_opponents.md` §1). |
| **What it did *not* fix** | The blind spots these ideas targeted survived: as of 2026-07-28 the policy still essentially never bolsters and doesn't use unit-specific tactics. That negative result is what produced the coverage diagnosis in `docs/independent_opponents.md` — no opponent in the repo bolsters or punishes its absence, so an exploration bonus alone has nothing to reinforce. |

---

## Critic hygiene: the opponent one-hot dropped, its offset moved to the advantage (2026-08-09)

*Source: `docs/next_iteration.md` §3.5 and §5 row 6. Two changes that only make sense as a pair;
neither is safe alone. No `OBS_VERSION` bump — this is an arch + advantage change, not an
observation change. Every prior critic checkpoint still loads (v1 ×4, v2 ×1 on disk).*

**The problem.** `Critic` took a 3-wide opponent one-hot (`random`/`greedy`/`pool`) alongside the
board, globals and privileged vector. It moved the output more than the position did: `V(start)`
spanned **0.747** across the three slots against a **0.44** std of `V` across *positions* (§3.5).
That was not a modelling error — the win rates really are 1.000 / 0.825 / 0.525 against
random / greedy / self, so a critic blind to the opponent must under-predict against weak ones and
over-predict against strong ones, and `A = G − V` then carries a per-opponent **offset** that makes
every action taken against `random` look good and every action against a snapshot look bad,
whatever the action was. The one-hot bought that back, at three costs: it is dead weight during
finetune (`p_random = p_greedy = 0`, and the search bots are mapped onto the `pool` slot anyway),
it made the raw output meaningless to every consumer outside the training loop (each search bot has
to pick a slot arbitrarily, on top of return normalisation), and it let the head satisfy part of
the loss without reading the state at all.

| What changed | Detail |
|---|---|
| **`critic_v3`** (`policy.py`, now the default) | v2's GroupNorm trunk and board-only auxiliary head, minus the one-hot: `head_in` drops by `OPP_DIM=3`. `Critic.uses_opp_onehot` says which behaviour an instance has; `_head_input` assembles the head vector accordingly, so **every existing call site keeps its signature** — v3 ignores an `opp_onehot` it is handed and accepts `None`, v1/v2 still require one and raise a named error rather than silently zero-filling. A new arch rather than a mutation of v2, because `warchest_critic_20260808-0607.pth` is a v2 checkpoint and it is the one that demonstrated the trunk fix (§3.4). |
| **Per-opponent advantage centring** (`rollout_buffer.py`) | `compute_gae(adv_norm='per_opponent')`, the new default, subtracts each opponent group's own mean advantage before applying **one shared std**. Deliberately mean-only: returns against `random` have far less spread than returns against a snapshot, so per-group *z-scoring* would amplify the near-deterministic group's noise up to the weight of the group carrying the real signal. `adv_norm='global'` reproduces every pre-2026-08-09 run and is the A/B baseline. |
| **Why the pair is safe** | The offset is constant across the siblings of a state (the opponent does not change within a decision), so it cancels in the sibling comparison *and* in `δ_t = r_t + γV(s_{t+1}) − V(s_t)` — the ranking work the critic does is untouched by removing it. What the one-hot was actually buying was absolute level, and the advantage is where that level mattered. |
| **A finer group than the critic ever had** | `rollout_core.OPP_GROUP_IDX` (5 labels + a warned fallback) is **not** `OPP_ONEHOT_SLOT`, which collapses `lookahead_critic`/`puct` onto `pool` for v1/v2 checkpoint compatibility. Finetune is 75 % `pool` / 25 % `lookahead_critic` — two opponents of genuinely different strength — so grouping them together would have left in exactly the offset being removed. The id feeds no network, so adding a label breaks no checkpoint. |
| **Small-group fallback** | A group below `MIN_GROUP_SAMPLES = 64` keeps the batch mean instead of its own: a mean estimated from a handful of correlated steps is noise, and subtracting it injects bias rather than removing it. The batch is then re-centred so the mean advantage is still exactly 0 — a non-zero mean is a uniform push on every sampled action. |
| **Visibility** | `adv_group_spread` (max − min of the removed offsets) is logged per batch and to W&B, with the per-opponent offsets printed by name. It is the quantity that justifies the change: how much of the raw advantage was opponent identity rather than action quality. Measured on a 6-episode smoke run at `random`/`greedy` 50/50: **0.37–0.54**, with `random` consistently above `greedy`, i.e. the predicted sign. Pairing `critic_v3` with `adv_norm='global'` logs a warning at startup — that combination removes the offset nowhere. |
| **Tool fix that rode along** | `eval_privileged_ablation.py` hardcoded the head's block offsets `[pooled \| global \| opp_onehot \| privileged]`. On a v3 head every block after `global` would have been off by 3 and the sensitivity report silently wrong; the layout is now built from the arch and asserted against `head_in`. |
| **Tests** | `tests/test_rollout_buffer.py` (new, 10 tests) pins the advantage half — offset removal, that groups are *not* individually rescaled, unit std, zero mean under the fallback, `global` still biased, and the group labelling. `tests/test_critic_arch.py` gains 5 v3 tests. Full suite: **181 passed**. |

**Not included, deliberately:** a shared policy/critic encoder with stop-gradient. `next_iteration.md`
§4 demotes it to a parameter-count optimisation — it measured a wash against the critic's own trunk
at matched readout, and it would cap the critic at the actor's representation.

**Owed:** the gate is *pooled R² holds ~0.20* (§5 row 6), which needs a training run plus
`eval_board_value.py fit`. Until that runs, this is an implemented change with no measured effect.

**Bundled with it, by explicit decision: `lam` 0.97 → 0.90** (`--lam`, new flag, `IDEAS.md` L2).
`V(s_{t+1})` enters the advantage at `γ(1−λ)`, so at 0.97 the critic supplied ~3 % of the
discriminative signal and a repaired critic could not show up in the gauntlet at all; 0.90 is 3.3×
that weight, an effective horizon of ~9 main-actor decisions against a ~42-decision episode. The
standing rule wants this attributable and separate; it is bundled anyway because a run is ~9.5 h
(`ppo_20260807-203528`: 1500 batches, 20:35 → 06:07) and the owner cannot run arms frequently.
So this run is **a bundle of three changes** (`critic_v3`, per-opponent advantage centring, λ) and
nothing in it is individually attributable — recorded here so it is not later read as a clean A/B.
`--lam 0.97` reproduces the prior behaviour for a baseline arm. **Trap:** λ also determines the
critic's regression target (`returns = GAE advantage + values`), so `critic_mae` is *not*
comparable across λ arms — it should fall at 0.90 simply because the target is more bootstrapped.

---

## Reward hygiene: tempo per turn, `ATTACK_REWARD` zeroed, holding rate re-derived (2026-08-09)

`docs/IDEAS.md` L8, which bundled its own item with the two `next_iteration.md` §4 had been
carrying since 2026-07-03. All three are one-line-scale changes; none of them is measured yet.

**The tempo cost was charged per maneuver, not per turn.** `MOVE_NEG_REWARD_PER_TURN = -0.002`
sat at seven `warchest_env.py` call sites, five of them tactic continuations, so a Berserker
chain, a Footman double maneuver and a Swordsman bonus move each paid it *again per maneuver*.
In a game whose currency is maneuvers-per-coin that taxed precisely the mechanics that buy extra
maneuvers — and the sharpest edge was that the Swordsman's *free* post-attack move cost strictly
more than declining it. The penalty was already known to be mis-scoped from the other direction:
`LookaheadBot` had to special-case it out of its search accumulation (`_own_action_reward`,
`docs/lookahead_bot_plan.md`) because a depth-bounded search only ever sees the cost.

| What changed | Detail |
|---|---|
| **`TURN_TEMPO_REWARD`** (renamed from `MOVE_NEG_REWARD_PER_TURN`) | added once in `_apply_action`, at the point the turn advances. That is "once per coin spent from hand" by construction, and exactly-once by construction rather than by keeping seven sites in sync — which is how the old version drifted in the first place. All seven maneuver sites now return `0.0`. Never charged on a game-ending move, an invalid action, or a mid-tactic continuation. |
| **Why it cannot distort a decision** | every option a turn offers now pays the identical amount, so the charge is a constant *within* a decision and cancels out of the comparison. It prices elapsed turns and nothing else, which is all it was ever meant to do. |
| **The Berserker's stack-paid extras stay uncharged** | deliberately. They cost *material*, which material PBRS already prices; a tempo charge on top would be the same double-pay this whole item is about. |
| **`Action.tempo_cost`** | new field carrying the charge separately (already included in `reward`). `LookaheadBot._own_action_reward` now subtracts it instead of comparing the reward against the constant — equality stopped identifying the term once it rides on top of every turn-ending reward rather than standing alone on plain moves. |
| **`ATTACK_REWARD` 0.02 → 0.0** | it fired on the same box-a-coin event as material PBRS and is non-telescoping, so the two double-paid every attack (flagged 2026-07-03, deferred then to keep that change set minimal). Kept as a named constant so the A/B back is one line. `score_attack` is now ~0 by construction — read the attack axis off `score_material`. |
| **`score_tempo`** | new decomposition bucket. Without it the tempo cost would have landed in `score_attack`, which — with `ATTACK_REWARD` at 0 — would have made a pure tempo count look like attack reward. Doubles as a clean read on episode length in turns, and against `n_decisions` on how many main-actor clicks were free continuations. |
| **`holding_reward_rate` 0.001067 → 0.004324 (4.05×)** | the divisor was `max_rounds * HAND_SIZE = 150`, the absolute worst case on main-actor turns. Converged runs settle near 78 plies (`turns=` in `logs/ppo_20260807-203528.log`), about half the main actor's, hence `TYPICAL_MAIN_TURNS = 37` — matching the ~37 in `next_iteration.md` §3. Sizing a per-turn rate on a bound no episode reaches made the term ~4× weaker than its own design intent in every game actually played, so it was not the mechanism the base-flip fix assumed. |
| **One source for the rate** | `WarChestEnv.default_holding_reward_rate()`. `ppo.py` and `LookaheadBot` had duplicated the formula, and `LookaheadBot`'s copy also hardcoded `1.0` for `WIN_REWARD`. |
| **Tests** | `tests/test_reward_hygiene.py` (new, 9 tests) pins once-per-turn *per mechanism* — plain move, Cavalry tactic, Footman double maneuver, Berserker chain, Swordsman bonus move (including that taking it costs the same as declining) — plus no charge on a winning or invalid action, `ATTACK_REWARD == 0`, and the holding-rate sizing property. Full suite: **190 passed**. |

**The trade taken on the holding rate.** The old 0.8 factor was described as a safety margin
guaranteeing accumulated holding could never exceed `WIN_REWARD`. That guarantee is gone: an
unusually long game at a sustained 5-base lead can now accumulate more than a win. It is close to
unreachable — a 5-base lead is one claim from ending the game — and `shaping_anneal` decays this
term to a 0.1 floor over the first half of a run. If it does bite, cap the accumulation rather
than restoring the 150; restoring it just re-creates a term that is 4× weaker than designed.

**Owed, and the thing to watch.** All three changes move the reward scale, so `score`, returns and
`critic_mae` are **not** comparable across this date — only win rate and the gauntlet are. The
holding rate is the one to watch: it is the sole surviving **non-PBRS** term, the only one that
can genuinely move the optimum, and it just got 4× stronger. The predicted failure mode is the
one `METRICS.md` already names — `avg_turns` rising with a flat win rate, i.e. sitting on a lead
instead of closing. The two tempo/attack changes are hygiene and should be close to neutral.

---

## Draft pairing: tried and rejected; explicit draft lists for forced-draft evals (2026-08-09)

*Source: `docs/IDEAS.md` L5. Measurement infrastructure only — no training-loop change. Recorded
as a **negative result with a working implementation left opt-in**, because the reasoning was
sound and will otherwise be re-proposed.*

**The premise, measured first.** Same deterministic bot (`greedy_fast`) on both sides, 300 paired
games with the two compositions swapped between seats: the same composition won **both** games
190/300 = **63.3 % ± 2.8 pp**. So ~**27 % of decisive games are settled by the draft alone**.
Replaying each draft once per colour should therefore cancel it — with `p` the true win rate and
`D` the draft advantage, two unpaired games give `Var = 2p(1−p)` and a mirrored pair gives
`2p(1−p) − 2·Var(D)`.

**Why it does not work.** The reduction needs the two games of a pair to be negatively
correlated. Measured on 150 pairs each: `greedy_fast` vs `greedy_sim` **r = −0.003 ± 0.082**, and
`ckpt_20260725` vs `ckpt_20260808` **r = −0.005 ± 0.082**. The 63.3 % was measured with one
deterministic bot playing *itself*, where composition is the only thing that can decide; real
entrants differ and policy agents **sample** their actions, so an identical opening diverges on
the first ply and the shared draft never propagates. Two direct variance checks landed at ratio
1.29 and 0.77 (n=120 each) — ~1.4 σ in opposite directions, i.e. noise. Neither should be cited.

| What changed | Detail |
|---|---|
| **`build_task_list(paired=...)`**, default **off** | Consecutive colour-swapped games share a seed instead of taking a fresh one. Threaded through `round_robin` and `round_robin_parallel`; exposed as `--paired-drafts`. Off by default, so every gauntlet number recorded before this date stays reproducible bit-for-bit at its seed — the change costs nothing and is available if a future field of deterministic entrants wants it. An odd `k_games` leaves one unpartnered trailing game, which still consumes its seed so the next matchup cannot silently reuse that draft. |
| **`eval_bolster.build_draft_list`** — **kept on** | A forced-draft bot cannot use the swap in any case: its composition follows the *agent*, not the seat, so it is the treatment rather than a nuisance draw. Its control is common random numbers **across arms**, so the harness now generates the full 4/4 draft up front and pins **both** sides via `force_units`, with `--draft-seed`, `--dump-drafts` and `--drafts`. A shared `--seed` nearly achieved this already, but only by relying on every arm consuming the RNG in the same order — which breaks silently when a bot's constructor or the env's reset changes. This is a robustness fix, not a variance claim. Verified by dump-then-replay: identical W/L/D. |
| **Seat balancing: measurably pointless, kept anyway** | The base layout is exactly 180°-rotation symmetric (`(1,0)→(5,6)`, `(4,1)→(2,5)`, every neutral base maps onto another) and `set_init_state` draws `initiative_owner` independently of player id — there is no first-player advantage to cancel. It costs nothing, so it stays. |
| **Tests** | `tests/test_paired_drafts.py`, 14 tests. They pin the *schedule* and the env property it depends on (*same seed ⇒ same draft*, without which `paired=True` would silently be a no-op) — deliberately **not** a variance claim. Also pinned: the default is unpaired, seeds never collide across matchups, and the odd-`k` trailing game. Full suite: **204 passed**. |

**Method note worth keeping.** This scheme was derived from a model of the game, verified on a bot
playing *itself*, and then failed on the real field — the third time in this project that a
quantity measured on a degenerate proxy did not transfer (cf. `next_iteration.md` §4's
"reliability is not validity"). **Validate a variance-reduction scheme on the estimator you
actually use, not on a simplified stand-in.**
