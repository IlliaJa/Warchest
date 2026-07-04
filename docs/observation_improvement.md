# Observation improvement plan — material-at-risk + expected opponent hand

**Status: IMPLEMENTED 2026-07-03** (`OBS_VERSION 9 → 10`). All three features shipped in one
schema bump: material-at-risk scalars, `E_opp_hand`, **and** the base-control reach planes +
scalars (the latter tracked in `docs/IDEAS.md` "base-control reach planes"). `BOARD_CHANNELS
46 → 48`, `GLOBAL_DIM 189 → 211`. Helpers `_maneuver_range` / `_is_claimable_base` /
`_base_reach_grids` in `warchest_env.py`; tests in `tests/test_obs_features.py` +
schema-pin updates across `test_action_space.py` / `test_threat_planes.py` /
`test_obs_global_vectorized.py`. Retrain required (invalidates `OBS_VERSION=9` pool snapshots).
The A/B protocol below is the remaining work. The section below is kept as the design record.

This plan bundles **two** new observation features into one schema bump, A/B'd as
**separate arms** so any gain is attributable. Both convert quantities *already present
in the observation* into a reduction the network computes poorly on its own (a spatial
cross-plane reduction; a multiplicative feature interaction) — the same "hand the net the
answer to a hard reduction" rationale that motivates the draw-probability features
(`docs/IDEAS.md` → *Draw-probability observation features*).

Neither is a reward. Both respect the `ppo_20260630` over-shaping diagnosis: they add
*legibility*, not dense heuristic return.

The two draw-probability features (`p_soon`, `p_mean`) are a **separate** proposal in
`docs/IDEAS.md` and are not part of this plan; if implemented together they share the same
`OBS_VERSION` bump but stay distinct A/B arms.

---

## Feature 1 — material-at-risk scalars (`own_material_at_risk`, `opp_material_at_risk`)

**Goal.** Give the location-blind heads a direct read on the question that drives the
**bolster / trade / extend** decision: *how much of my material can die next turn, and how
much of theirs can I kill?* The threat planes already encode **where** hits land and how
hard, but to answer "how much of *my committed material* is exposed" the net must multiply
the enemy-threat plane by my-unit occupancy and reduce spatially — a cross-plane reduction
a conv+mean-pool approximates noisily. Hand it two scalars.

**Definition (per side).** From the **raw** threat grids (`_threat_grids`, *before* the
`np.clip(... / THREAT_NORM)` that produces the display planes):

```python
# enemy_hits[cell] = sum over THREAT_KINDS of the enemy raw hit-count grid at that cell
own_material_at_risk = sum(min(enemy_hits[u.loc], u.stack)
                           for u in my_on_board_units)
opp_material_at_risk = sum(min(own_hits[u.loc],  u.stack)
                           for u in opp_on_board_units)
```

`min(hits, stack)` = coins actually lost if that cell is fully struck (a stack-2 unit taking
3 hits loses 2 coins; the 3rd is wasted). This is the *material* interpretation, matching how
`step()` resolves damage.

**Modeling choices to lock at implementation (each flagged; all conservative/worst-case,
consistent with the planes):**
- **Sum hits across kinds per cell.** Over-estimates *simultaneous* hits (the opponent plays
  ≤3 coins/round, one at a time), but it's the same worst-case pressure the planes already
  encode. Keep it consistent with the planes rather than modeling activation budget here.
- **Use raw grids, not the normalized planes.** The planes are clipped to `[0,1]`; the scalar
  needs the true hit-count to compare against `stack`.
- **Opponent-availability gating is inherited** from `_threat_grids` (worst-case
  `opp_hidden[t] >= 1`), so this feature does *not* need `E_opp_hand` — it stays worst-case,
  which is correct for "can this die."

**Normalization.** Divide by a coin-count cap and clip to `[0,1]`. Candidate: `OWNED_TOTAL`
(21) — safe upper bound on a side's on-board material — or a tighter board-material constant
if profiling shows the value never approaches 21. Pick one, note it in the layout comment.

**Shape / cost.** 2 scalars. (Option, not recommended for arm 1: split each by `THREAT_KIND`
→ 6 scalars, so "ranged material-at-risk" is separable. Defer unless the 2-scalar version
underperforms — more dims, more A/B ambiguity.) Cost: reuses `threat_grids` already computed
in `generate_observation`; a couple of lookups and a `min`. Negligible.

**Helps.** Actor and critic both — it's a reduction neither computes for free.

---

## Feature 2 — expected opponent hand `E_opp_hand[t]` (actor-side)

**Goal.** Let the policy estimate *what the opponent can play against me this round without
redrawing* — their live counter-attack / counter-play capacity. The obs already carries the
opponent's hidden pool `hidden_v[t]` (= hand + bag + face-down discard, per type) and their
hand **size** (`opp_hand_size`), but never combines them, so the net cannot tell a type that
is *in hand now* from one *buried in the bag*.

**Why the network can't get this for free.** It's the product
`hidden_v[t] * opp_hand_size / hidden_total` — a multiplicative interaction of two features on
different scales, which an MLP does not form cheaply. Same justification as the draw features.

**Why the actor specifically needs it.** The **critic already sees the opponent's true hand**
via `get_privileged_features` / `PRIV_DIM` (`warchest_env.py:791-807`, the `[0:C]` opp-hand
block). The actor sees only the public obs. `E_opp_hand` narrows that asymmetry for the policy
without leaking private state — it is a *statistic* of public info, not the hidden truth.

**Definition (per type, over `DECK`, len `N_COIN_TYPES=17`).** Under exchangeability of the
hidden coins (exactly how the opponent drew — uniform from the bag):

```python
hidden_total = sum(hidden_v)                       # opp hand + bag + face-down discard
opp_hand_size = sum(opp_hand.values())             # observable count, already in the obs
E_opp_hand[t] = hidden_v[t] * opp_hand_size / max(hidden_total, 1)
```

Expected copies of type `t` in the opponent's hand right now.

**Properties that make it the right signal:**
- Sums to `opp_hand_size` — a proper mass-per-type.
- Decays to 0 as they empty their hand → correctly encodes "they've committed their coins,
  safe to extend," which the static `hidden_v` misses entirely.
- Covers **all** capability (attack, recruit, claim-initiative, tactics), i.e. "how loaded are
  they," not just where they can hit.

**Known approximation (document, don't fix).** `hidden_v` includes this round's *face-down
discards*, which are no longer drawable and were never in the bag when the hand was drawn, so
the pool isn't perfectly exchangeable. The face-down-discard *size* is observable, but
reconstructing draw order to correct this is overkill. The simple form is slightly
**over**-optimistic about their hand → biased toward caution, the safe direction. Note the
assumption in the layout comment.

**Normalization.** Divide by `_TOTAL_COINS_VEC` (per-type total coins) to sit on the same
scale as the sibling coin vectors (`hand_v`, `hidden_v`) the net already reads. (Alternative:
`/ HAND_SIZE` for a clean `[0,1]` mass; prefer `_TOTAL_COINS_VEC` for cross-vector
comparability.)

**Shape / cost.** `N_COIN_TYPES = 17` dims. One sum + one broadcast-divide. Negligible.

**Explicitly NOT touching the threat planes.** The planes keep the worst-case gate
(`opp_hidden[t] >= 1`, `_threat_grids` `coin_gate`, `warchest_env.py:1593-1594`): for spatial
*safety* you want the max, not the mean — one Berserker coin they happen to hold is deadly at
low expected count. `E_opp_hand` serves the *soft, non-spatial* judgment (over-extend? recruit
now?). Max for "where can I die," mean for "how loaded are they" — two different jobs, two
different features. A variant that makes threat-plane *magnitude* proportional to `E_opp_hand`
is tracked separately in `docs/IDEAS.md` (*Likelihood-weighted threat-plane magnitude*).

---

## Wiring

All edits in `src/services/environment/warchest_env.py` unless noted.

1. **Constants / layout.**
   - Extend the `GLOBAL_DIM` formula and the layout comment block (`warchest_env.py:187-219`).
     New `GLOBAL_DIM = 189 + 2 + 17 = 208`.
   - Bump `OBS_VERSION = 9 → 10` (`warchest_env.py:220`) with a note: adds material-at-risk
     scalars + `E_opp_hand`; invalidates `OBS_VERSION=9` pool snapshots → retrain.

2. **`generate_observation` (`warchest_env.py:684-776`).**
   - Material-at-risk: after `threat_grids` is built (line 716), compute the two scalars from
     the **raw** grids and the on-board unit stacks (both sides' units are already iterated
     nearby). Append to `global_feats` in a fixed slot.
   - `E_opp_hand`: `hidden_v` and `opp_hand_size` are already in scope
     (`hidden_v`, `sum(self.state.hands[opponent].values())`, lines 742-763). Compute the
     vector, normalize, append.
   - Ego-rotation: both features are **coin-space / scalar**, not board-space — no `rot90`
     needed (unlike the planes). Confirm they're placed outside the board-plane rotation.

3. **Observation space.** `get_observation_space()` reads `GLOBAL_DIM`, so the Box shape
   updates automatically once the constant changes — verify no hard-coded 189 elsewhere
   (`grep -rn "189" src/`).

4. **Docs.** Update `docs/environment_api.md` (`GLOBAL_DIM` line, the layout summary) and
   `docs/policy_network.md` (global-features section) to describe the two new blocks. Add a
   `docs/history.md` entry when landed.

## Testing

- Extend `tests/test_threat_planes.py` (or a sibling) with cases pinning:
  - `own/opp_material_at_risk` on a hand-built position where a known unit sits under a known
    number of hits (assert `min(hits, stack)` and the cross-side symmetry).
  - `E_opp_hand` sums to `opp_hand_size`; equals `hidden_v` scaled when `opp_hand_size ==
    hidden_total`; goes to 0 when the opponent's hand is empty.
  - Ego-invariance: identical logical position from P1 and P2 perspectives yields identical
    global-feature slots (these features are frame-independent).

## A/B protocol

Three arms against the current `OBS_VERSION=9` baseline (or four if bundled with draw
features):
1. baseline (no new features),
2. + material-at-risk only,
3. + `E_opp_hand` only,
4. (+ both, to check for interaction / redundancy).

Freeze a `baseline_tactics` anchor first (`docs/IDEAS.md` P5c) so WR-vs-pool comparisons are
against a fixed reference across the schema bump.

## Discarded during design (kept so they aren't re-proposed)

- **Stack-fragility summary** (count of stack-1 units / min threatened stack). Rejected:
  frames fragility as intrinsically bad. Fragile stacks are a *deliberate* resource/tempo
  choice — you bolster only the main 1–3 units to absorb damage, and low stacks are often
  fine (or are dumped, unwanted coins freeing the cycle). A "you are fragile" scalar invites
  over-bolstering — the over-shaping failure mode.
- **Neutral/opponent-base count.** Rejected: base *position* is what matters (already in the
  base-control planes); a raw count discards it. Opponent bases matter mainly for denial, a
  spatial relation the planes carry.
- **Own-tempo / hand-size feature.** Rejected as redundant: the agent moves one unit at a
  time and observes its own hand *exactly* (`hand_v`). Hand size is its planning horizon, not
  a budget to encode. The opponent's hand is the half worth estimating → Feature 2.
