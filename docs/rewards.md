# Reward Design

See [experiments.md](experiments.md) for the full training run history.

---

## Current implementation

### Reward table

| Event | Reward |
|---|---|
| Win (6 bases controlled) | +1.0 |
| Claim base | 0.0 |
| Truncation — base lead | 0.0 |
| Truncation — tied bases | −0.5 |
| Truncation — base deficit | −1.0 |
| Invalid action attempt | −0.02 |
| Any move | −0.002 |

Constants are defined at the top of `src/services/environment/warchest_env.py`. Raw environment rewards are augmented with potential-based shaping and a holding reward in `src/app/ppo.py` before being stored in the rollout buffer. Every move costs the same step penalty regardless of destination — directional pull toward bases is handled entirely by shaping and the holding reward.

### Potential-based reward shaping

Applied by the training loop, not the environment:

```python
shaped_r = r + gamma * phi(s_next) - phi(s)
phi(s)   = SHAPING_C * (my_bases - opp_bases)    # SHAPING_C = 0.05
```

Fires a positive pulse (~+0.05) when the agent claims a neutral base and (~+0.10) when stealing an enemy base. Fires a negative pulse when the opponent claims. Advantages are z-scored batch-wide; returns are kept in the original reward scale.

### Per-turn holding reward

Applied by the training loop alongside shaping:

```python
holding_reward = holding_reward_rate * (my_bases - opp_bases)
holding_reward_rate = WIN_REWARD / ((winning_base_count - 1) * (max_actions // 2)) * 0.8
                    = 1.0 / (5 * 100) * 0.8 = 0.0016
```

Fires every agent turn proportional to the current base lead. Incentivises defending claimed bases and breaks the base-flip exploit (two policies claiming the same bases in a loop), where potential shaping alone produces near-zero net reward per cycle. The 0.8 factor is a safety margin ensuring worst-case accumulated holding (`0.0016 * 5 * 100 = 0.8`) never exceeds `WIN_REWARD = 1.0`. `holding_reward_rate` is derived from env constants at script startup so it stays valid if `max_actions` or `winning_base_count` change.

Shaping and holding are complementary: shaping delivers an immediate, one-step credit signal at the claim action; holding creates persistent per-turn pressure that requires no bootstrapping.

### Implementation notes

- Every move returns `MOVE_NEG_REWARD_PER_TURN = -0.002` regardless of destination. The binary approach rewards (`MOVE_ON_BASE_REWARD`, `MOVE_NEAR_BASE_REWARD`) were removed: they created a bias toward neutral bases over enemy bases (which are strategically more valuable), and the holding reward + shaping now provide all directional pull needed.
- An exploration bonus (`MOVE_EXPLORE_REWARD_MAX_TURN = 5`, `MOVE_EXPLORE_REWARD_PER_TURN = 0.1`) is defined as constants but is not wired into `perform_move_action`. The exploration map is updated every step for use in the observation, but no reward is computed from it.
- `CLAIM_BASE_REWARD` is 0.0. A non-zero direct claim reward caused a circular-claiming exploit: the policy learned to claim bases back and forth with pool opponents, accumulating reward far in excess of `WIN_REWARD = 1.0`. Potential shaping and the holding reward handle base value correctly without this risk.

### Truncation reward

When the episode truncates (`action_count >= max_actions = 200`), a terminal reward is added to the last main-actor step based on the base lead:

```python
if my_bases > opp_bases:    trunc_reward = 0.0   # drew from a strong position
elif my_bases == opp_bases: trunc_reward = -0.5  # true draw
else:                       trunc_reward = -1.0  # drew from a losing position
```

---

## Ideas for denser rewards

### ~~1. Per-turn base-lead bonus~~ ✅ implemented as holding reward

```python
base_lead_reward = k * (my_bases - starting_bases)  # original formulation
```

Implemented as `holding_reward_rate * (my_bases - opp_bases)` with a derived coefficient — see holding reward section above.

---

### ~~2. Opponent-loss penalty (symmetric claiming)~~ *(ruled out — covered by holding reward)*

**Ruled out**: the holding reward already penalises losing a base (reduces per-turn income). An explicit penalty at the opponent's action step would create attribution problems since the defender had no direct control over that timestep.

---

### 3. Potential-based reward shaping ✅ implemented

A theoretically clean way to add dense rewards without changing the optimal policy (Ng et al. 1999). The `gamma * phi(s') - phi(s)` terms telescope and nearly cancel over a full trajectory, leaving only boundary terms.

---

### ~~4. Distance-to-objective shaping~~ *(not worth implementing — approach rewards removed)*

```python
delta_dist = prev_min_dist_to_target - new_min_dist_to_target
proximity_reward = k * delta_dist
```

**Not worth it**: requires BFS on the hex grid every agent step (non-trivial cost on CPU). More importantly, `MOVE_ON_BASE_REWARD` and `MOVE_NEAR_BASE_REWARD` already provide a coarser version of this signal, and the holding reward creates pull toward bases via future value. The oscillation exploit risk (agent bouncing near a base to farm the reward) is real. Better to improve the critic so it bootstraps the holding reward correctly.

---

### ~~5. Graduated distance reward~~ *(superseded — approach rewards removed)*

Was a proposed replacement for the binary `MOVE_ON_BASE_REWARD` / `MOVE_NEAR_BASE_REWARD`. Both the binary approach and this graduated version were removed in favour of letting the holding reward and shaping handle all directional pull. Avoids the neutral-vs-enemy base bias entirely.

---

### ~~6. Re-enable exploration reward~~ *(not worth implementing)*

```python
explore_reward = max(0, MOVE_EXPLORE_REWARD_PER_TURN * (MOVE_EXPLORE_REWARD_MAX_TURN - visit_count[end]))
```

**Not worth it**: the board is small (7×7 hex, ~37 valid cells) and the policy needs to focus on objectives, not coverage. With only 2 units per player, coverage is not a bottleneck. An exploration bonus would distract from the objective-focused rewards already in place.

---

### ~~7. Win-speed bonus~~ *(not needed)*

```python
win_reward = WIN_REWARD + speed_bonus * (max_actions - action_count) / max_actions
```

**Not needed**: γ=0.99 discounting already makes winning sooner better than accumulating future holding rewards. The step penalty (`-0.002` per move) adds further pressure. Verified: winning at turn 70 gives `1.0` vs holding 65 more agent turns then winning gives `≈0.83` net present value.

---

### ~~8. Base-capture sequence reward~~ *(not worth implementing)*

```python
MOVE_ADJACENT_TO_UNCLAIMED = +0.002
MOVE_ONTO_UNCLAIMED        = +0.008
CLAIM_UNCLAIMED            = +0.15
```

**Not worth it**: restoring a non-zero `CLAIM_BASE_REWARD` risks the circular-claiming exploit. The potential shaping and holding reward already incentivise the full capture sequence without the exploit risk.

---

### 9. Coin / material economy potential *(proposed — coin economy)*

Every reward today is base-centric. But `attack` sends an enemy coin **to the box** (`GameState.boxed`, permanent), and losing coins shrinks the opponent's bag → fewer coins drawn per round → fewer actions → a compounding tempo/material spiral. The agent is currently paid for this **only** if the attack eventually converts to a base — a very delayed, sparse signal across the ~150-turn midgame that the `ppo_20260630` run spends most of its time in.

Reward it as a **potential-based** term (Ng et al. 1999 — policy-invariant, same telescoping trick as the existing base shaping, so it densifies the signal without changing the optimal policy):

```python
# coins only ever leave play into the box, so a player's live-coin count is
# owned_total - boxed_total; the owned offset is fixed per composition and
# cancels in the telescoping, leaving the boxed differential as the live term.
phi_material = C_MAT * (boxed_total(opp) - boxed_total(me))   # C_MAT ≈ 0.01–0.02
shaped_r    += gamma * phi_material(s_next) - phi_material(s)
```

- **Fires** a positive pulse when the agent removes an enemy coin, a negative pulse when it loses one — exactly the mechanic (attack / defend) that base shaping never touches.
- **Coefficient.** Keep `C_MAT` well below `SHAPING_C = 0.05` (bases *win* the game; material is a means, not the objective). A single killed coin is worth a fraction of a base swing.
- **Measure `phi` after the opponent's move**, like the C4 fix — otherwise the telescoping breaks.
- **Exploit safety.** `boxed` is monotonic per player, but PBRS is policy-invariant for *any* state function, so there is no farming loop (unlike a raw per-kill bonus, which the circular-claiming precedent warns against). Uses only `GameState.boxed` + `owned()` — no BFS, negligible cost.
- **Caveat (important).** `docs/analysis_ppo_20260630.md` found the agent already *over-optimizes* dense shaping relative to winning. Two reasons this term is still worth trying: (a) it's proper PBRS, unlike the non-telescoping holding reward that is the likelier decoupling culprit; (b) material advantage is **strongly win-correlated**, so it points the dense signal at something that actually predicts wins, rather than adding another base-saturated term. A/B it against the plateau; don't stack it on top of the holding reward without measuring.

---

### 10. Unit / board-presence potential *(proposed — unit economy)*

A softer companion to (9): reward having units **deployed and alive on the board** (you can't claim or hold a base with an empty board, and a dead unit's coins sit uselessly in the bag until re-deployed).

```python
phi_units = C_UNIT * (units_on_board(me) - units_on_board(opp))   # C_UNIT small
shaped_r += gamma * phi_units(s_next) - phi_units(s)
```

- **Rationale.** Encourages board tempo — deploying to contest locations and denying the opponent presence — which base shaping only rewards *after* a claim lands.
- **Overlap with (9).** Killing an enemy unit raises both `phi_material` and `phi_units`; deploying spends a coin (neutral to material until it dies) but raises presence. To avoid double-paying the same kill, prefer **(9) as primary** and treat (10) as an optional low-coefficient add-on, or use *total on-board stack height* instead of unit *count* so it reads as "committed board strength" rather than "number of bodies."
- **Over-deploy risk.** A presence bonus can push wasteful deploys (dumping coins onto the board just for the count, leaving the bag thin). PBRS keeps the *optimal* policy unchanged in theory, but the coefficient must stay small and this should be A/B'd for exactly this failure mode. Consider gating on *controlled/contested* cells rather than any cell.
- **Cheaper alternative.** If (10) doesn't pay off, the deploy/recruit decisions may be better shaped by the *material* term (9) alone plus letting the critic bootstrap board value.

---

## Implementation priority

| Idea | Status | Notes |
|---|---|---|
| Per-turn base-lead bonus (1) | ✅ Implemented | As holding reward |
| Potential-based shaping (3) | ✅ Implemented | |
| Opponent-loss penalty (2) | Ruled out | Covered by holding reward |
| Graduated distance reward (5) | Superseded | Approach rewards removed entirely |
| Distance-to-objective (4) | Not worth it | BFS cost + oscillation risk, approach rewards removed |
| Re-enable exploration (6) | Not worth it | Board too small, distracts from objectives |
| Win-speed bonus (7) | Not needed | Covered by γ discounting + step penalty |
| Sequence reward (8) | Not worth it | Exploit risk, covered by shaping + holding |
| Coin/material economy potential (9) | **Proposed** | PBRS over `boxed` differential; win-correlated. A/B vs the `ppo_20260630` plateau |
| Unit/board-presence potential (10) | **Proposed** | Optional companion to (9); watch over-deploy; keep coefficient small |
