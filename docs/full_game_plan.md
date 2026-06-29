# Full-game implementation plan

Roadmap for growing the current positional-capture prototype into real two-player
War Chest: coin/bag economy, all placement + face-down + maneuver actions, the full
16-unit roster with tactics & attributes, and drafted compositions.

Decisions locked (2026-05-30):
- **Incremental sequencing** — bag economy first behind a temporary extended-flat
  head, factored head as its own later phase. Every phase stays trainable.
- **Coin stacks from the start** — board units are stacks of coins (HP); bolster adds
  a coin, attack removes one, a unit dies only when its last coin is removed.
- **End goal: all 16 cards + drafting** — random/drafted unit sets per game; the
  policy must generalize across compositions.

See also: `docs/rl_algorithms.md` → *Action head: flat spatial vs factored /
autoregressive* (the head design this plan implements) and `docs/architecture.md`.

---

## Where we are vs. where we're going

| Mechanic | Current prototype | Real War Chest (target) |
|---|---|---|
| Turn structure | 1 action, players alternate every `step` | Round = draw 3 coins, alternate **one coin per turn** until both hands empty, then redraw |
| Coin economy | none | bag / hand (3) / discard / supply per player |
| Actions | move, attack, control, deploy | + bolster, recruit, claim-initiative, pass, tactics |
| Action cost | free | each action spends a coin (face-up maneuver / placement, or face-down) |
| Units | Swordsman only, 1 HP | 16 types, coin-stack HP, tactics + passive attributes |
| Initiative | implicit (P1 starts) | tracked; claim-initiative transfers it ≤ once/round |
| Information | perfect | **imperfect** — opponent's hand & bag hidden; face-up discards visible |
| Unit set | fixed | drafted/random per game |
| Win | control 6 locations | place all 6 control markers (== control 6 locations) — already aligned |
| Policy head | flat `[14,7,7]` softmax (686) | factored/autoregressive verb → coin → cell → target |

The win condition is the one piece already correct, so it stays untouched.

---

## Guiding principles

1. **Every phase ends in a trainable, evaluable system.** No phase leaves the env or
   the policy in a half-wired state.
2. **De-risk the env state machine before the head.** The round/bag loop is the most
   bug-prone work; validate it under the known-good flat head, then swap heads.
3. **The cross-phase regression anchor is GreedyBot, not neural snapshots.** A saved
   model is bound to its obs+action schema *and* to the rules it was trained on — a
   no-coin policy cannot play the coin game, so neural baselines are reusable as
   opponents only *within one schema generation*. Re-baseline GreedyBot each phase and
   use it as the durable yardstick. Neural snapshots are archived as *records* (weights
   + env-version tag + WR), evaluated only in their own env. What carries across a schema
   change is **warm-starting**: keep the board-encoder arch stable and partial-load the
   overlapping weights into the next-phase network, randomly-initializing new channels /
   head slots.
4. **Tests first for the state machine.** The round loop, reshuffle, hand-empty and
   initiative rules are pure logic — cover them with unit tests, not just W&B curves.
5. **Observation schema is versioned.** Each phase that changes the obs bumps a
   `OBS_VERSION` constant; saved models record the version they were trained on. (The
   current capture-game env is the implicit version 0; Phase 1a introduces the constant.)
6. **Every env change ships with its visualization.** Whenever a phase adds new state
   (hands, coin-stacks, discard, initiative, supply, tactics), extend the `render` /
   `GameRenderer` / `demo.py` output to display it in the same phase. A mechanic you can't
   see in a replay is a mechanic you can't debug.
7. **Bots stay in lockstep with the env.** Every phase that changes the action space or
   legality updates both bots in the same phase: `RandomBot` must still sample only legal
   actions, and `GreedyBot`'s heuristics must be re-expressed over the new verbs (and stay
   a meaningful yardstick — a bot that ignores coins/tactics makes "WR vs greedy"
   worthless). Re-baseline GreedyBot's strength at each phase boundary.

---

## Phase 1 — Coin/bag/round state machine (the keystone)

The single largest change. Decomposed deckbuilder-style: **fix the deck first, add the
deck-building later.** Three independently-trainable sub-phases — 1a (deterministic deck),
1b (stochastic bag), 1c (stacks + economy).

### Phase 1a — Deterministic deck (no randomness)  ✅ implemented (2026-05-30)

Empty board + deploy-from-hand; any coin may go face-down (claim-initiative / pass).
Action space 741 (spatial 735 with deploy-sword/knight verbs + 6 face-down slots); board
obs 10 channels (per-type unit planes); globals 8 (hand + initiative). 25-test suite in
`tests/test_phase1a.py`; renderer shows hands + initiative; both bots updated.

**Goal.** Replace "alternate every action" with the real round structure using a *fixed*
hand, so the turn controller is deterministically testable. The hand is always
{Swordsman, Knight, Royal}; the player spends one coin per turn (alternating with the
opponent), a spent coin is unavailable until the hand refreshes, and after all three are
spent the round ends and the hand refreshes to the same three. This is the round / turn /
coin-depletion structure with **zero stochasticity** — no bag, no reshuffle, no supply.

The "can't move the same unit twice in a row" constraint falls straight out of coin
depletion: one Swordsman action + one Knight action + one Royal (face-down) action per round.

**Env / `game_state.py` changes:**
- Add a **Knight** unit type — identical actions to Swordsman, distinct id/icon/coin.
- Restructure setup: each player starts with **1 Swordsman + 1 Knight** (not 2 Swordsmen),
  so coin types map 1:1 to board units and "one unit per type" holds.
- `GameState` gains per-player `hand` (the spent/unspent state of the 3 coins) and
  `initiative_owner`. No bag/discard/supply yet — the hand is regenerated whole each round.
- **Round controller** in `warchest_env.py`: players alternate one coin per turn; a player
  whose hand is empty is skipped; both empty → new round, initiative owner first.
- **Coin-gating:** move/attack/control of a board unit is legal only if its matching coin
  is still unspent this round; performing it marks that coin spent. The Royal coin can be
  spent on `claim_initiative` or a face-down `pass`.

**Head (temporary):** 686 spatial slots **+ a couple of appended non-spatial slots**
(`claim_initiative`, `pass`). Still one flat categorical → buffer & PPO loop unchanged.
(Your idea pulls the coin-only verbs forward to here; the cost is only these 2 slots.)

**Observation changes (`OBS_VERSION = 1`):** add a **hand encoding** (which of the 3 coins
are still unspent) and **initiative ownership**. Unit planes gain a type channel for Knight.

**Other touch points:** `Critic` (wider globals), `GreedyBot` (respect coin-gating),
`RolloutBuffer` (wider global vector), `demo.py` (show hand state + initiative).

**Tests:** coin-depletion legality, can't-repeat-a-unit-in-a-round, hand-empty skip, round
boundary, initiative default & once-per-round transfer. **Exit:** stable training; turn
controller tests pass; archive `baseline_deck_v1`.

### Phase 1b — The BAG (stochastic draws; partial observability enters)  ✅ implemented (2026-05-31)

Bag = `2 S + 2 K + 1 R` per player; draw 3/round with reshuffle; maneuvers discard face-up,
face-down actions face-down. **Coins bind to the board**: deploy moves a coin hand→board,
attack sends the enemy's coin to the **box** (out of game). Hand is now a `Counter` (multiset).
`GLOBAL_DIM` 8→23 (own known counts + opponent public counts + `hidden_pool`); **privileged
critic** wired (`PRIV_DIM=9` = opp true hand/bag/face-down, plumbed through the buffer). Action
space unchanged (741); bots unchanged. Tests: `tests/test_phase1b.py` + updated
`tests/test_phase1a.py` (35 total). Note: privileged-critic *benefit* still to be A/B'd.

**Phase 1b implements the bag.** War Chest is a *bag-builder* (not a deck-builder — there is
no "deck" in the rules). 1a gave every player the same three coins in hand each round with no
randomness. 1b replaces that fixed hand with a real **bag**: a player owns a *multiset* of
coins, each round **draws 3 of them at random** into the hand, spent coins go to a **discard
pile**, and when the bag runs dry the discard is **reshuffled** back into it. Which 3 you hold
now varies round to round — and the opponent's bag/hand become hidden, so the game turns
imperfect-information here.

**Env changes:**
- `GameState` gains `bags` (per-player multiset of owned coins) and `discards` (per-player
  list of `(type, face_up)`); the old fixed-hand `DECK` constant is replaced by a per-player
  **bag composition** (e.g. `2 Swordsman + 2 Knight + 1 Royal = 5 coins`).
- Round start: draw 3 from the bag into the hand; if the bag has < 3, reshuffle the discard
  in and keep drawing (handle the 1–2-coin edge case where fewer than 3 exist total).
- Maneuvers discard the spent coin **face-up**; claim-initiative / pass discard **face-down**.
  Face-up coins are **public** (visible to the opponent until reshuffle); face-down are hidden.

**Bag → policy features (`OBS_VERSION = 2`).** The board planes are unchanged (per-type unit
planes from 1a; stacks/HP arrive in 1c), so **the bag translates entirely into added global
scalar features** — a board-independent per-type count vector appended to the globals. The
per-type counts per zone are a **sufficient statistic** for the draw probabilities, so we feed
counts, not raw history. Concretely the policy gains, per unit type (and the Royal coin):

- **Own side — known exactly** (no probability to infer): `hand[t]`, `bag[t]`, `discard[t]`,
  and total `bag_size`. Optionally the explicit next-draw expectation
  `3 · bag[t] / bag_size` as an inductive-bias shortcut so the net doesn't have to learn it.
  Note `hand[t]` is now a **count** (you can hold 2 of a type), replacing 1a's single on/off
  flag.
- **Opponent side — only the public/deducible counts** (the policy must never see the hidden
  hand): `owned[t]` (the fixed bag composition in 1b; becomes dynamic once recruit lands in
  1c), `on_board[t]`, `faceup_discard[t]`, opponent **hand size** (coins left to play this
  round — observable from how many they've played), and the key derived feature
  `hidden_pool[t] = owned[t] − on_board[t] − faceup_discard[t]` — the coins sitting in
  {hand + bag + face-down discard}, i.e. exactly what a human counts.
- **Normalization:** divide counts by a small constant (e.g. max plausible owned, or
  `bag_size`) so features stay in `[0, 1]`.
- Existing 1a globals (turn fraction, base counts, initiative ownership) stay.

**Privileged critic:** the policy sees only the public counts above; the **critic** may
additionally receive the opponent's *actual* hidden hand/bag at train time (same asymmetric
pattern as the existing `opp_onehot`, and discarded at inference — see the privileged-critic
discussion).

**Ceiling:** feedforward + counts handles everything *deducible*; the hand/bag split within
the opponent's `hidden_pool` stays a learned reactive heuristic. Stronger belief tracking
(recurrent over history, or determinized MCTS) is a later option — **not for 1b**.

**Tests:** draw/reshuffle correctness, discard accounting, face-up/face-down tracking, the
1–2-coin draw edge. **Exit:** stable training under stochastic draws; archive `baseline_bag_v2`.

### Phase 1c — Coin stacks (HP) + bolster + recruit + supply  ✅ implemented (2026-05-31)

Units carry a `stack` height (HP). **Bolster** = new spatial verb 15 (cell-targeted; type from
the unit there) moves a matching coin hand→stack. **Attack** removes one coin to the box; unit
dies at 0. **Supply** = 2 per unit type (total owned 4 S / 4 K / 1 R); **recruit** = face-down
action paying any hand coin (full agency over pay-coin × take-type, 6 slots) and taking a supply
coin into the face-up discard. Action space 741→**796** (verb 15 + 6 recruit slots);
`GLOBAL_DIM` 23→**28** (`OBS_VERSION=3`: stack-height unit planes, own/opp supply, initiative-
transferred flag; `hidden_pool` now subtracts the public supply). Renderer shows stack badges +
supply panel. Tests: `tests/test_phase1c.py` (42 total). GreedyBot left as-is (ignores
bolster/recruit — still a legal, myopic yardstick).

**Goal.** Add the HP model and the placement/economy actions (the "coin stacks now"
decision lands here).

**Env changes:**
- **Coin stacks:** a board unit carries a stack height. `bolster` (spend a matching hand
  coin face-up) adds a coin → higher HP. `attack` removes **one** coin from the target
  stack (to the box, not discard/supply); unit destroyed at zero; re-deploy of that type
  allowed afterward.
- **`recruit`** (discard a coin face-down → take one supply coin of any type into discard);
  add `supply` (count per type) to `GameState`.
- **Royal coin** restriction (face-down actions only, + future Royal Guard tactic) enforced
  in the mask.

**Head (temporary):** appended non-spatial slots grow to include recruit-per-supply-type.
Still flat → buffer/PPO unchanged. K stays small with two unit types.

**Observation (`OBS_VERSION = 3`):** per-unit stack height on the unit planes; supply counts
per type; "initiative already transferred this round" flag.

**Tests:** bolster→attack→death sequence, recruit supply accounting, royal-coin restriction.
**Exit:** trainable two-type economy with full mechanics; archive `baseline_economy_v3`.

---

## Phase 2 — Observation finalization + factored / autoregressive head  ✅ implemented (2026-06-01)

Implemented as a **verb-level factorization**: a dedicated verb head learns `P(verb)`, and
`P(a) = P(verb(a))·P(a|verb(a))` where the within-verb conditional is a masked softmax over
that verb's legal flat actions (reusing the existing spatial-conv + face-down logits). Verbs:
move / attack / control / deploy / bolster / claim_initiative / pass / recruit (`N_FACTORED_VERBS=8`,
`VERB_OF_ACTION` map lives in the env). Both stages are conditionally masked (verbs with no legal
action are dropped; within-verb masked to legal ids). The result is still a single `Categorical`
over the 796 flat ids, so **env / buffer / critic / bots / remap are unchanged** — only how the
per-action log-probs are computed differs. Chose this over the doc's "auto-pick pay-coin" first
cut to **preserve all sub-field agency** (direction / deploy-type / pay-coin / take-type stay
learnable → no regression). The verb head gets gradient every step — the structure that pays off
as the roster grows. Tests: `tests/test_phase2.py` (47 total). Finer autoregressive sub-stages
(splitting direction/pay-coin/etc. into their own conditional heads with cross-verb parameter
sharing) remain a later incremental refinement; not needed at the 2-unit scale.

**Goal.** Replace the flat+appended head with the factored head from
`docs/rl_algorithms.md`. Env logic does **not** change — this is a pure policy/plumbing
rewrite on a validated game, which is exactly why it's isolated here.

**Head tree (per the design doc):**
```
verb ∈ {move, attack, control, deploy, bolster, recruit, initiative, pass}
 ├ move/attack/control → source_cell (the existing [7,7] map, masked by "hold matching coin")
 │     ├ move:    direction
 │     ├ attack:  target cell
 │     └ control: —
 ├ deploy/bolster → hand_coin (unit-type categorical) → dest_cell ([7,7] map)
 ├ recruit → supply_stack categorical  [+ optional pay_coin]
 ├ initiative → [optional pay_coin]
 └ pass → —
```
- Joint log-prob = sum of traversed stage log-probs; entropy = sum of stage entropies.
  PPO ratio/clip/entropy operate unchanged on the sum.
- **Conditional masking at every stage** (legal sources given verb, legal targets given
  source, hand-gated coin choices). First cut: auto-pick the pay-coin for recruit/initiative
  with a heuristic; add the explicit `pay_coin` head later for coin economy.

**Plumbing changes:**
- `Policy`: emit the staged categoricals; `act` samples stage-by-stage and returns the
  composite action + summed log-prob + summed entropy; `evaluate_actions(_batch)` recomputes
  the sum from stored stage choices.
- `RolloutBuffer`: store per-stage action indices, per-stage masks, and the joint log-prob
  (not just one int + one 686-mask).
- `Critic`: unchanged structurally (still consumes board + globals + privileged info).
- `GreedyBot`: adapt to the staged action interface (or keep a flat-enumeration adapter
  that maps its chosen flat action onto the stage tuple).
- `demo.py`, `policy_viz.py`: updated for the new head.

**Exit criteria.** Matches or beats `baseline_economy_v2` WR with the factored head; the
recruit/initiative/pass verbs are now first-class (no appended-slot kludge); freeze
`baseline_factored_v3`. After this point, adding units no longer inflates a flat joint head.

---

## Phase 3 — Unit variety (vanilla: no tactics/attributes yet)  ✅ implemented (2026-06-01)

Implemented the **full 16-unit vanilla roster** (`roster.py` is the single source of truth:
id/icon/colour/total-coins; unit classes generated from it) **and pulled Phase 5's drafting
forward** at the user's request: each game `set_init_state` samples 8 distinct unit types and
gives 4 to each player **disjoint** (players never share a unit), plus the shared Royal coin.
Per-player bag/supply replace the old global constants (`build_bag`/`build_supply`); `owned`
is per-composition. The network is sized for the **whole roster** (fixed slots, masked per
game): action space 796→**1776** (16 deploy verbs → `N_VERBS=30`; claim/pass over 17 coins;
recruit take 16 × pay 17), board planes 10→**38** (6 terrain + 16 own + 16 opp stack-valued,
chosen for Phase-4 readiness), `GLOBAL_DIM` 28→**174**, `PRIV_DIM` 9→**51**, `OBS_VERSION=4`.
Policy/Critic/buffer were **shape-agnostic** (factoring is by verb, not type) and needed no
edits. Renderer + both bots updated; `tests/test_phase3.py` (coin-conservation invariant,
disjoint draft, per-type legality/planes, encode/decode, economy) — 23 tests green. Breaks all
prior saved models (expected at a schema change); re-baseline GreedyBot for this generation.
Tactics/attributes remain Phase 4; variable-composition *eval bucketing* remains Phase 5.

**Goal.** Add several plain unit types that differ **only by coin identity**, all sharing
move/attack/control/deploy/bolster. Validates that the factored head's `hand_coin` and
per-type `source_cell` masks scale, and makes supply/recruit meaningful across types.

- Add unit classes (subclasses of `BaseUnit`) with distinct ids/icons; no tactics.
- Bag/supply setup generalizes to multiple types (2 of each in bag, rest in supply).
- Observation: hand/supply/discard encodings already per-type — extend the type dimension.
- The "one unit of each type on the board at a time" rule is enforced per type.

**Exit criteria.** Stable training with 3–4 vanilla types; recruit decisions show
non-trivial type preference; freeze `baseline_multiunit_v4`.

---

## Phase 4 — Tactics & attributes (incremental, a few units at a time) ✅ complete (2026-06-28)

**Status: all 16 units now have their tactics / attributes / restrictions.** A 400-game
random-play stress (every drafted composition) exercises all 14 pending kinds with zero
crashes, softlocks, or coin-conservation violations, and training runs clean on the final
schema. Final schema: `N_VERBS=32` (`TACTIC`, `SELECT`), `N_FACTORED_VERBS=11`,
`PENDING_CTX_DIM=15` (no-pending + 14 continuation kinds), `GLOBAL_DIM=189`,
`ACTION_SPACE_SIZE=1875`, `OBS_VERSION=8`. Policy/Critic/buffer/trainer stayed shape-agnostic
throughout (the head reads sizes from constants). Tests live in domain files
(`test_tactics.py`, `test_attributes.py`, `test_units.py`, …; 63 total).

What each unit does (see `docs/UNITS.md` for the card text):

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

**Documented simplifications** (faithful to ~all real positions; avoid a pending-inside-a-pending):

- A bonus / repeat maneuver — the Warrior-Priest's drawn-coin action, a Berserker's stack-paid
  repeat, or a Footman-tactic maneuver — can do move / attack / control / deploy / etc. but
  **cannot use the TACTIC verb to start a *named* tactic.** This does **not** limit how many
  maneuvers a unit gets (a Berserker still chains one per stack coin); it only forbids nesting a
  second multi-step tactic inside one. Rarely binds (e.g. a WP drawing a Cavalry coin can't launch
  Cavalry's move-then-attack with it).
- The WP's bonus action does not re-fire the acting unit's on-attack attribute.
- The Royal Guard auto-absorbs from supply when able (a strictly-beneficial "may").

(Granted attacks/moves from Marshall/Ensign **do** chain the granted unit's own attribute — a
granted Berserker keeps maneuvering, a granted Swordsman gets its free move — per the FAQ.)

---

### Implementation history

**Scaffolding + Cavalry slice implemented (2026-06-02).** The variety of tactics is kept
*out* of the action space via a **pending sub-turn** state machine: a multi-step tactic
resolves as a sequence of masked clicks, not one atomic id. `GameState.pending` (a `Pending`
dataclass) parks the owed continuation; while it is set the turn does **not** pass,
`get_possible_actions` returns only the legal next clicks, and those clicks **reuse the
move/attack verbs** — the policy disambiguates "normal maneuver" vs "tactic follow-up" via a
**pending-context one-hot** appended to the globals. Action-space additions are minimal: one
spatial `TACTIC` verb (initiate; the on-board unit's type picks the tactic) + one non-spatial
`DECLINE` slot (end an optional continuation). Schema bump: `N_VERBS` 30→**31**, `FACEDOWN_SIZE`
306→**307**, `GLOBAL_DIM` 174→**177** (`PENDING_CTX_DIM=3`), `N_FACTORED_VERBS` 8→**10**
(`V_TACTIC`, `V_DECLINE`), `OBS_VERSION=5`. Cavalry (`tactic='cavalry'` in `roster.py`) is the
end-to-end proof: `TACTIC@unit → move-dir (mandatory) → attack-dir (optional)`, coin paid once
at initiation; attack is optional so a move with no adjacent enemy is not a softlock. Policy/
Critic/buffer/trainer are shape-agnostic — only the constants changed; GreedyBot's deploy-verb
range was capped below `TACTIC_VERB` so it stays a tactic-ignoring yardstick. Tests:
`tests/test_phase4.py` (16: schema/encode/remap, the full + declined + mandatory sub-turn,
context one-hot, P2-via-remap, coin-conservation-with-tactics, no-softlock, bots, nets).
Remaining clusters below add a `Pending` kind + a continuation branch + a roster flag per unit.

**SELECT primitive + Archer slice implemented (2026-06-28).** Added the `SELECT` verb (a
spatial verb whose `(r,q)` is an arbitrary *target* cell, not a direction) — the primitive the
directional move/attack verbs can't express, needed for ranged attacks and friendly-unit
grants. It is only ever legal as a pending continuation click. First consumer: the **Archer**'s
ranged attack (`TACTIC@unit → SELECT a distance-2 enemy`, mandatory), plus the Archer's
*restriction* `can_normal_attack=False` (rulebook p.16). Tactics are now named by **mechanic,
not unit**, so they reuse across the roster/DLC: Cavalry's tactic is `move_then_attack`; Archer's
is `ranged_attack` with `tactic_params={'distance':2,'straight_line':False}`. The same
`_ranged_targets` already supports Crossbowman's `straight_line=True` (clear-line range 2) — only
the roster entry is missing. Schema bump: `N_VERBS` 31→**32** (`SELECT_VERB`), `N_FACTORED_VERBS`
10→**11** (`V_SELECT`), `PENDING_KINDS` +`ranged_attack` → `PENDING_CTX_DIM` 3→**4**, `GLOBAL_DIM`
177→**178**, `ACTION_SPACE_SIZE` 1826→**1875**, `OBS_VERSION=6`. Policy/Critic/buffer/trainer
unchanged (head reads sizes from constants). The test suite was also reorganized from phase-named
files into domain files (`tests/test_units.py`, `test_tactics.py`, `test_game_mechanics.py`,
`test_action_space.py`, `test_bots.py`, `test_policy.py` + shared `_helpers.py`; 42 total). New
tactic coverage (SELECT remap, mechanic-naming, full ranged flow, the no-normal-attack restriction,
target-in-range gating, mandatory-select, P2-via-remap) lives in `test_tactics.py`/`test_units.py`.

**Then clusters 1–4 completed (2026-06-28)** — the remaining grant flavor of SELECT (select a
friendly unit → a nested granted maneuver), the `move_to` / `line_charge` destination tactics,
and all the passive/triggered attributes. See the table above.

**Goal.** Add the `tactic` verb branch + conditional masks and passive `attributes`,
one cluster of units per sub-step so failures stay localized.

Clusters by complexity (each implemented end-to-end with focused tests):
1. **Simple movement/attack tactics:** Cavalry (move-then-attack) ✅, Light Cavalry (move 2) ✅,
   Lancer (move 2 then attack) ✅.
2. **Ranged tactics + restrictions:** Archer (attack 2 away via tactic only) ✅,
   Crossbowman (straight clear line, may also normal-attack) ✅.
3. **Repeating / passive-on-attack:** Berserker ✅, Swordsman ✅, Pikeman ✅, Warrior Priest ✅.
4. **Force-multipliers / special:** Marshall ✅, Ensign ✅, Mercenary ✅, Scout ✅, Knight ✅,
   Royal Guard ✅, Footman ✅.

**Exit criteria.** ✅ All tactics/attributes implemented and tested; the random-play stress
confirms legality/conservation across every drafted composition. (Re-baselining `baseline_tactics_v5`
via a training run is the remaining step before Phase 5.)

---

## Phase 5 — Variable compositions + drafting (the full game)

**Goal.** Per-game random or drafted unit sets, so the agent generalizes across the ~1820
possible 4-unit matchups instead of memorizing one.

- Setup samples each player's 4 unit types (random first; drafting variant later).
- Observation must encode **which units are in play this game** (the per-type supply/hand
  planes already carry this — verify zero-padding for absent types is clean).
- Self-play and opponent pool span compositions; eval reports WR per composition bucket.
- Optional: implement the rulebook drafting procedure as a setup mode.

**Exit criteria.** Stable WR across randomized compositions vs. greedy and pool; the agent
adapts recruit/deploy choices to its drawn unit set. This is end-state two-player War Chest.

---

## Cross-cutting workstreams (run alongside the phases)

- **Imperfect information.** From Phase 1 on, the opponent's hand/bag is hidden. Keep the
  *privileged critic* pattern (critic may see opponent hand at train time, policy never)
  to stabilize value learning — this generalizes the existing `opp_onehot` privilege.
- **GreedyBot.** Re-baseline it each phase so "WR vs greedy" stays a meaningful yardstick;
  a too-weak bot makes 90% meaningless.
- **Reward shaping.** The current base-diff potential shaping stays valid. Reconsider small
  shaping for coin economy (e.g., discourage wasteful passes) only if learning stalls.
- **Eval / Elo.** GreedyBot (re-baselined per phase) is the cross-phase yardstick. Neural
  `baseline_*` snapshots are eval/pool opponents only *within their own schema generation*;
  once the obs/action schema changes they're retired (kept as records, not opponents).
- **Curriculum.** Reuse the existing random→greedy→pool weighting; add new baselines as
  pool anchors.
- **Demo / renderer.** Each phase that adds visible state (hands, stacks, discard,
  initiative, tactics) updates `demo.py` so games stay inspectable.

---

## Suggested first three concrete steps

1. **Phase 1a** — deterministic deck: add Knight, restructure to 1 Swordsman + 1 Knight,
   implement the round/turn/coin-depletion controller + initiative, with coin-only verbs on
   appended head slots. Write the turn-controller tests *first*; extend the renderer for
   hand/initiative state and update both bots. (Zero randomness — the cleanest possible
   first cut at the keystone.)
2. **Phase 1b** — swap the fixed deck for a stochastic bag + discard + reshuffle. (Adds
   partial observability; localized change on top of 1a.)
3. **Phase 1c** — coin-stack HP + bolster + recruit + supply. (Completes the two-type
   economy; sets up the factored-head swap in Phase 2.)
