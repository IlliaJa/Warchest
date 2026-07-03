# Ideas for improving training

Implemented history: `docs/history.md`.

---

## Open items

Current state: WR vs greedy ~49% (post C1–C7 fixes, run `ppo_20260527-191432`). Target: 60%.

### Game-completeness — remaining Phase 5 work

The full base game (all 16 units + per-game disjoint drafting) is implemented; drafting was pulled forward into Phase 3, so the old `full_game_plan.md` roadmap is retired (its history now lives in `docs/history.md`). These are the only pieces of the "Phase 5 / full game" spec that were **not** carried forward:

- **P5a. Variable-composition eval bucketing.** Eval currently reports a single aggregate WR across all randomly-drafted matchups. Add per-composition (or per-composition-bucket) WR reporting so we can see *which* unit sets the agent is weak/strong on instead of averaging over the ~1820 possible 4-unit matchups. Difficulty: low-moderate.
- **P5b. Rulebook drafting mode (optional).** Setup currently assigns each player 4 random disjoint types. The real game uses a snake/alternating draft from a shared pool. Implement it as an optional setup mode for parity with tabletop games; not required for training. Difficulty: moderate.
- **P5c. Freeze `baseline_tactics` snapshot.** The Phase-4 exit step ("re-baseline via a training run") is largely satisfied by run `ppo_20260630-060400` (~70% WR vs *true* greedy). Remaining: freeze that policy as a named pool/eval anchor for the current (`OBS_VERSION=8`) schema generation. Difficulty: trivial.

The plateau / high-entropy / reward-decoupling problems observed on that run are diagnosed with prioritized action points in `docs/analysis_ppo_20260630.md` — that doc, not this section, is the live to-do for improving the full-game agent's *strength*.

### Reward shaping for the coin/unit economy

Material PBRS (§9) is now **implemented** (2026-07-03, `C_MAT=0.015`, annealed with holding); the optional board-presence term (§10) is still open. See `docs/rewards.md`.

> **⚠️ REMINDER — zero `ATTACK_REWARD` in the next A/B.** Now that material PBRS is live, the raw `ATTACK_REWARD = 0.02` (`warchest_env.py`) **double-pays** the same box-a-coin event that `phi_material` already rewards — and unlike the PBRS term it is non-telescoping (farmable in principle, adds return variance). Decision: **it can be zeroed.** Not done in the 2026-07-03 pass only to keep that change set to exactly what was requested. Set `ATTACK_REWARD = 0.0` and re-run the material-PBRS A/B so attacks are paid once, through the policy-invariant term. (Full rationale: `docs/rewards_improvements.md` §2/Step 2 and §4.1.)

### Draw-probability observation features (bag dilution / draw efficiency)

**Goal.** Give the agent a legible signal for the coin-economy nuance that over-recruiting *dilutes* the units it actually wants to play. Hand size is fixed at `HAND_SIZE=3` (`_draw_hand`, `warchest_env.py:416-437`) and draws are uniform-without-replacement from the bag, so recruiting a coin you won't reliably draw grows the cycle **without** raising actions/round — it just lowers the chance of drawing the coin that matters. All the raw state (per-type bag/hand/discard) is already in the observation, but the agent must learn the division + reshuffle timing implicitly; these features hand it the answer.

**Decision: feature-only, no reward term.** A reward is the wrong tool here (see the discussion that produced this section):
- "smaller bag = better" is a **trap** — as PBRS it pays a positive pulse whenever coins leave the cycle, and the biggest exit is *your own coins getting boxed on unit death*, so it would reward losing material (directly contradicting §9).
- "higher per-unit draw rate = better" is *also* wrong — a fully-recruited "4-of-each + Royal" (17-coin) bag scores `3·4/17 ≈ 0.71` per unit, **higher** than the 9-coin starting bag's `0.67`, yet is a bad bag (tempo wasted recruiting; no ability to draw a *specific* unit in pairs to chain/bolster).
- The real target — "reliably draw *the unit I've chosen to play next rounds*" — has a **policy-defined** target (which unit is "preferred" is the agent's call, and 1-unit play is valid only for rare comps / Berserker / Warrior Priest). A fixed potential must pick the target concentration and is wrong somewhere (Herfindahl → pushes monotype; fielded-average → fails to flag 4-of-each). So it belongs in the observation, letting the policy own the preference. This also respects the `ppo_20260630` over-shaping diagnosis (don't add dense heuristic reward).

**The two features — same quantity ("what share of my draws is type `t`") at two horizons.** Own-side only (`p_soon` needs the bag↔discard split, hidden for the opponent). Emit both as per-type vectors over `DECK` (Royal included in denominators; same layout as `bag_v`/`hand_v`).

1. **`p_soon[t]` — imminent draw share (position now).** Expected copies of `t` in the *next* hand, before any reshuffle, normalized to a share:
   ```python
   B = sum(bag)                          # current bag size
   if B >= HAND_SIZE:
       E_soon = HAND_SIZE * bag[t] / B   # == bag[t]/B share when bag is healthy
   else:                                 # bag empties mid-draw → one reshuffle
       rest, D = HAND_SIZE - B, sum(disc)   # disc = discard_faceup + discard_facedown
       E_soon = bag[t] + (rest * disc[t] / D if D > 0 else 0)
   p_soon[t] = E_soon / HAND_SIZE        # in [0,1]
   ```
   (Expectation = hypergeometric mean, so exact for the next hand without the full distribution.)

2. **`p_mean[t]` — steady-state draw share (structure).** Long-run share of the recirculating pool — the dilution/concentration signal recruiting moves:
   ```python
   recirc[t] = bag[t] + hand[t] + disc_faceup[t] + disc_facedown[t]
             = owned[t] - on_board[t] - boxed[t] - supply[t]   # same identity the coin-counter uses
   p_mean[t] = recirc[t] / sum(recirc)                          # in [0,1]
   ```

**The gap is the signal (no third feature).** Both are `[0,1]` shares, directly comparable per type: `p_soon > p_mean` = *loaded now, spend it*; `p_soon < p_mean` = *key coins stuck in the discard behind a reshuffle, can't rely on `t` this round*. Flat "4-of-each" reads as no type peaking on either horizon.

**Wiring / difficulty.** Two counter-sums + a divide per type in `generate_observation` — negligible cost. Schema change → **bump `OBS_VERSION`** (invalidates the current `OBS_VERSION=8` pool snapshots; retrain). Difficulty: low-moderate. Plan to A/B against a no-feature baseline so any gain is attributable.

### Architectural note — factored / autoregressive action head

*(Parked — full design in `docs/rl_algorithms.md` → *Action head: flat spatial vs factored / autoregressive*.)*

Once the action set grows (4+ units, coin mechanics), the flat `[A, 7, 7]` spatial softmax breaks down — coin-only verbs (recruit, initiative, pass) have no board cell to point at. **Not needed at current scale.** Revisit when the env has bag/hand/recruit mechanics.

### Architectural note — the agent can't see the board as one position

**Implemented 2026-07-02 — Parts A, B, and C below all shipped in one pass; see `docs/history.md` → "Threat/position-aware observation + deeper trunk" for the final design (including the Berserker closed-form and Marshall grant-chaining derived beyond what's written here) and `tests/test_threat_planes.py` for coverage. Kept below as the design record.**

Not one limit but **three, stacked**, and together they mean the network reasons about the board only in local patches:

1. **Receptive field is radius 2.** The trunk is two `HexConv2d` layers (3×3 hex-masked, `policy.py:73-78`), so each feature cell aggregates only **hex-distance ≤ 2**. The spatial action logits come from a **1×1 conv** (`policy_head`, `policy.py:82,106`) that adds **zero** reach — the logit for an action at cell X sees only what is within 2 hexes of X.
2. **The board-wide path is location-blind.** `verb_head`/`facedown_head` read a global average pool (`feat.mean`, `policy.py:107-111`). It spans the board but is positionally flat: it knows a lancer exists *somewhere*, not on *which flank*.
3. **Distance-3 threats fall outside both.** Lancer `line_charge` (`max_dist=2`, then strikes the enemy beyond → **distance 3**, `warchest_env.py:1338-1351`) is past the receptive field, and the pool has discarded the in-line geometry a charge depends on. Archer/Crossbowman (`ranged_attack`, dist 2) and Cavalry/Light-Cavalry (reach 2) sit right *at* the field's edge, with a single ReLU of headroom. So the agent is effectively blind to charge threats as spatial relations, and thin on range-2 ones. Any longer-range DLC unit inherits the problem.

**What this costs, in the game's own terms.** Warchest is a two-base, two-flank game: you draft units to hold each side and generally avoid crossing the middle. Reading the board means knowing *which flank* each enemy unit is on, *what type* it is, and *what it threatens* — then answering in kind: an enemy archer on the left wants a mobile unit sent left to close the distance; a slow unit parked in a lancer's lane is free material; every unit has a matchup an opponent can exploit. All of that is position- **and** type-specific — exactly what a radius-2 field plus a location-blind pool cannot represent. The "stupid losses" in `ppo_20260630-060400` (walking a unit into a threat range) are this blind spot showing up on the board.

**Feature: exact threat/reach planes + a position-aware summary.** Three complementary changes, A/B'd separately so any gain is attributable. All touch `BOARD_CHANNELS` and the observation encoder → bump `OBS_VERSION` (schema change: retrain, invalidates the current `OBS_VERSION=8` pool snapshots).

**A. Next-turn threat & reach planes** *(recommended — highest leverage, directly kills the range-3 blind spot).* Warchest ranges are exact and are *already* computed for legal-move generation (`_reachable`, `_line_charge_targets`, `_hex_distances`, `_can_attack`), so this is a reuse, not new game logic. Fill extra planes in `generate_observation` (ego-centric, same rotation as the unit planes):
- `enemy_threat` — for every cell, **how many hits an enemy could land there this turn** (a graded count, not a 0/1 flag), using each unit's *real* tactic reach. This is the plane that drives the **bolster** decision: a cell showing "2" means a 1-coin unit parked there dies and you need stack ≥ 2 to survive. It hands the 1×1 head the answer as a **local** feature and sidesteps the receptive-field limit entirely.
- `own_threat` — the symmetric plane for my units: where I project force, and how many hits I can deliver (do I break their unit, or just force them to have bolstered?).
- *(optional, but this is what makes matchup counter-play legible)* split enemy threat into a few planes **by kind** — melee / ranged / charge — or per unit type. Then the "ranged-threat" plane lighting up a flank is a signal the agent can attach the "send a mobile unit" response to, rather than having to re-derive range geometry every step.

**Multi-move reach is the crux, and it's why this must be planes, not CNN depth.** Reach is not a fixed per-unit radius — a single activation can chain into several hits, so true threat radius *and hit-count* depend on an **activation budget** the env can compute but a conv stack cannot (data-dependent, variable-length search). Turns alternate **one coin at a time** (`_advance_turn`, `warchest_env.py:366-380`), which splits the threat into two horizons:

*Immediate (one coin, their very next turn) — this is the map that matters most.* Everything that chains inside a single coin's activation:
- **Berserker** (`extra_maneuvers_from_stack`, `warchest_env.py:1115,1617-1666`): the initiating hand coin strikes once at full stack, then it self-pays coins to keep striking down to stack 1 → a stack-`S` Berserker lands **up to `S` hits in one turn**, ending fragile. Magnitude is fully observable from stack height; the only hidden part is whether the opponent holds ≥1 Berserker coin to start it. **Model worst-case (assume they spend it): if the `S` hits clear the retaliators it is a *winning* trade, not a bad one** — the self-damage is the point.
- **Cavalry** `move_then_attack` (reach 2), lancer charge (3), ranged (2), Footman `maneuver_each` (all footmen off one coin) — all single-coin, all immediate.
- Availability here is just a **gate**: can they activate unit T at all? → `hidden_T ≥ 1` (see below). The hit-count comes from the board, not from the coin count.

*Whole-round (several coins, but you act in between).* Reactivating a non-chaining unit with multiple hand coins (two Cavalry coins → two move-then-attacks) costs **multiple of their turns with yours interleaved**, so it's a softer, multi-ply threat — not something that lands before you can respond.

**Available-coins bound — already computed.** Per-type coins hidden in the opponent's hand+bag are exactly `hidden_v` in `generate_observation` (`= opp_owned − on_board − faceup − supply`, `warchest_env.py:686`) — the same "2 + recruited − on-board" idea, minus boxed/faceup. The opponent plays at most `HAND_SIZE = 3` coins per round, so type T can be activated at most `min(hidden_T, 3)` times this round. Use it as the `≥1` gate for the immediate map, and as the reactivation cap for the whole-round map.

Pick an explicit budget tier and note it in the encoder:
- **T0** — one best tactic per unit, once. Cheapest; understates the Berserker chain (the current mental model).
- **T1 (recommended)** — full single-coin reach, worst-case: Berserker stack chain (`S` hits), Footman `maneuver_each`, move-then-attack, move-after-attack, gated by `hidden_T ≥ 1`. All observable, and exactly the reach that forces bolstering.
- **T2** — additionally model whole-round reactivation up to `min(hidden_T, 3)`. More complete, but multi-ply and fuzzier (you act between their coins).

**B. Position-preserving awareness** *(fixes flank allocation — the location-blind pool).*
- Add static **coordinate / region planes** (CoordConv-style): a left↔right axis, a forward↔back axis, and/or a "my half vs opp half" plane. Cheap, static, no env compute — lets the convs and the pool encode *which side* things are on, the substrate for two-flank reasoning.
- Upgrade the global summary: replace the single mean-pool with a **location-preserving** one — concatenate per-half (or per-base-neighborhood) pooled features, or a light spatial attention — so the verb head sees "left flank under ranged pressure, right quiet" instead of a positionless average.

**C. Deeper trunk.** A 3rd/4th `HexConv2d` → radius 3/4, so features genuinely propagate board-wide. This is the "let the net learn it" counterpart to A's "hand it the answer." Cheapest single change; complementary to A/B.

**Priority.** A first (exact, cheap to compute, attacks the documented losses head-on), then B (unlocks the flank/matchup reasoning that motivated this), C as a cheap orthogonal knob to stack on or compare against.

**Difficulty.** A: moderate (schema change + threat fill, but reuses existing range code). B: moderate (coordinate planes trivial; the pooling change touches the heads). C: low.

---

### C12. **`collect_episodes=16` produces noisy advantage estimates.**

~1100 transitions per buffer is low for PPO (standard is 2048–4096). With 16 episodes per batch there is also high variance in which opponents a batch sees. Doubling would give ~140 inner updates per batch (vs ~70 now) with no inner-loop change.

**Fix.** `collect_episodes=32`.

**Relevance.** Moderate. (Run `ppo_20260630-060400` already used `collect_episodes=64`, so this is likely superseded — confirm against the current `ppo.py` default before acting.)

**Difficulty.** Trivial.

---

### Tier 4 — small / cosmetic

| # | Issue | Difficulty | Effect |
|---|---|---|---|
| C17 | `trunc_reward` is a step function (0/−0.5/−1); a base-diff-proportional value would reduce critic target variance | low | small |
| C18 | `value_single` is called during `.train()` mode at rollout (line 199); harmless today (no BN/Dropout) but if either is added, Dropout would produce stochastic noisy values during GAE and BatchNorm would compute statistics from a single sample — both silent bugs. Fix: `self._critic.eval()` before the rollout loop, `self._critic.train()` before the update. | trivial | none until BN/dropout |
| C20 | Eval runs 20 episodes — std error on a 0.49 WR estimate is ~5%. Bump to 50. | low | log readability |
| C21 | `EloTracker` updates after every eval game — noisy; consider a running average | low | log readability |
| C22 | `score_deque` reports batch-mean only; adding a 100-episode rolling mean would make regressions easier to spot | low | log readability |

**C16** (`iter_minibatches` permutation): already correct — a fresh `np.random.permutation` is generated on each call, so minibatches are uncorrelated across PPO epochs. No action needed.

**C10** (shared encoders): obsolete. See `docs/rl_algorithms.md` → *Architecture: shared vs separate actor-critic encoders*.

---

### Recommended next steps

C8 (LR decay) and C9 (clean true-greedy eval) are **done** — run `ppo_20260630-060400` uses the true greedy bot (`RANDOM_ACTION_PROB=0.0`) and reports an honest ~70% WR. That run's plateau is now the live problem; its diagnosis and prioritized action points are in `docs/analysis_ppo_20260630.md`. Start there.

