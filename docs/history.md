# Training history — implemented improvements

Consolidated record of all applied fixes and architectural changes. For the reasoning behind each item see the source doc noted in each section header.

---

## Phase 1 — REINFORCE era: correctness fixes

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

## Phase 2 — Architecture migration

*Source: `docs/IDEAS.md`. Large structural changes that moved the project from REINFORCE to PPO.*

**Separate actor/critic encoders (#13).** Gave `Policy` and `Critic` independent `board_encoder + unit_encoder` instances. Critic now takes a privileged `opp_onehot` input the actor never sees; separate `Adam` instances (`lr_actor=1e-4`, `lr_critic=3e-4`) honour different learning rates without interference. See `docs/rl_algorithms.md` → *Architecture: shared vs separate actor-critic encoders* for the full rationale.

**PPO migration (#14).** Replaced REINFORCE+GAE with PPO (`src/app/ppo.py`). Same actor-critic network; added rollout buffer, clipped surrogate loss, and inner epoch loop. `src/app/reinforce.py` retained as legacy reference.

**Opponent pool (#16).** `src/services/opponent_pool.py` — weighted sampler over random / greedy / past policy snapshots. Breaks the co-adaptation cycle that caused WR to *decline* over training under pure self-play.

**Hex-aware encoder (#15 / C13).** Replaced the standard 2D `Conv2d` with `HexConv2d` (custom kernel using the 6 hex-direction offsets). Removed the false-diagonal / missing-neighbour bias of the 3×3 square kernel on a hex grid. See `docs/decision.md`.

---

## Phase 3 — PPO era: stability and tuning fixes

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
| C13 | **`HexConv2d` board encoder.** Already done as part of phase 2 (see above), confirmed in `src/services/policy/policy.py`. | — |

---

## Phase 3 — 16-unit roster + per-game disjoint drafting (2026-06-01)

*Source: `docs/full_game_plan.md` Phase 3 (with Phase-5 drafting pulled forward). Schema-breaking — a new obs/action generation (`OBS_VERSION=4`); prior saved models retired.*

| What changed | Detail |
|---|---|
| **Full 16-unit vanilla roster** | `environment/roster.py` is the single source of truth (id/icon/colour/total-coins); unit classes are generated from it in `units/__init__.py`. All units share move/attack/control/deploy/bolster (tactics/attributes are Phase 4). |
| **Per-game disjoint draft** | `set_init_state` samples 8 distinct types and gives 4 to each player, disjoint, + the shared Royal coin. Per-player `build_bag`/`build_supply` replace the old global `INITIAL_BAG/SUPPLY/INITIAL_OWNED`; `GameState.owned()` is per-composition. |
| **Full-roster sizing** | Action space 796→1776 (16 deploy verbs → `N_VERBS=30`; claim/pass over 17 coins; recruit 16×17). Board planes 10→38 (6 terrain + 16 own + 16 opp, stack-valued). `GLOBAL_DIM` 28→174, `PRIV_DIM` 9→51. |
| **No policy/critic edits** | The factored head groups by verb (8), not by type, and reads sizes from env constants — Policy/Critic/RolloutBuffer adapted automatically. |
| **Renderer + bots** | Coins use per-unit colours/glyphs (2-letter codes) from the roster; GreedyBot verb-offset constants updated; RandomBot unchanged (mask-driven). |
| **Tests** | `tests/test_phase3.py` (23 total green): disjoint-draft, roster totals, encode/decode round-trips, per-type legality/planes, recruit accounting, and a coin-conservation invariant across full random games. Old `test_phase1*` retired (dead schema); `test_phase2` (factored head) still passes. |

---

## Phase 4 — Tactics, attributes & restrictions (full base game) (2026-06-28)

*Source: `docs/full_game_plan.md` Phase 4. Implemented incrementally (Cavalry → SELECT/Archer →
clusters 1–4) until all 16 units had their real abilities. Schema-breaking; prior models retired.*

| What changed | Detail |
|---|---|
| **Pending sub-turn machine** | Multi-step tactics resolve as masked continuation clicks via `GameState.pending`; the turn does not pass until it clears. A pending-context one-hot (`PENDING_CTX_DIM`) in the globals tells the policy which continuation it is in. 14 pending kinds. |
| **SELECT primitive** | A spatial verb (`SELECT_VERB`) whose `(r,q)` is an arbitrary *target/destination* cell (not a direction) — the piece the directional move/attack verbs can't express. Drives ranged attacks, move-to destinations, line charges, and friendly-unit grants. |
| **All 16 units** | Mechanic-named tactics (`move_then_attack`, `move_to`, `line_charge`, `ranged_attack` any/line, `grant_attack`, `grant_move`, `maneuver_each`, `royal_move`) + boolean attribute flags (`counter_when_attacked`, `move_after_attack`, `extra_maneuvers_from_stack`, `bonus_action_after_attack_or_control`, `only_attackable_when_bolstered`, `deploy_adjacent_to_friendly`, `maneuver_after_recruit`, `absorb_from_supply`, Footman `max_on_board=2`). Named by mechanic so they reuse across the roster and DLC. |
| **Shared resolution** | One `_resolve_attack` (with Pikeman counter + Royal-Guard absorb-from-supply) and `_fire_maneuver_triggers` hook feed every attack/maneuver path; a `_resolve_free_maneuver` powers granted/free/Footman maneuvers. |
| **Schema** | `N_VERBS` 31→32 (`SELECT`), `N_FACTORED_VERBS` 10→11, `PENDING_CTX_DIM`→15, `GLOBAL_DIM`→189, `ACTION_SPACE_SIZE`→1875, `OBS_VERSION` 5→8. Policy/Critic/buffer/trainer unchanged (head reads sizes from constants). |
| **Documented simplifications** | A bonus/repeat maneuver (WP drawn coin, Berserker repeat, Footman-tactic maneuver) can't use the TACTIC verb to start a *named* tactic — caps nesting, not the count of maneuvers. Granted attacks/moves (Marshall/Ensign) **do** chain the granted unit's attribute (FAQ). Royal Guard auto-absorbs from supply when able. |
| **Tests** | Test suite reorganized from phase-named files into domain files (`test_units`, `test_tactics`, `test_attributes`, `test_game_mechanics`, `test_action_space`, `test_bots`, `test_policy` + shared `_helpers`); 63 total. A 400-game random-play stress exercises all pending kinds with zero crashes/softlocks/conservation violations. |
