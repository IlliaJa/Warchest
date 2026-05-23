# Reward Design

## Current implementation

### Reward table

| Event | Reward |
|---|---|
| Win (6 bases controlled) | +1.0 |
| Claim unclaimed base | +0.15 |
| Truncation (draw, 200 actions) | −1.0 |
| Invalid action attempt | −0.02 |
| Move onto unclaimed base | +0.005 |
| Move adjacent to unclaimed base | +0.001 |
| Each action taken | −0.002 |

Constants are defined at the top of `environment/warchest_env.py`. Raw reward values are passed unchanged to GAE; the training loop normalizes advantages with z-scoring and normalizes returns with a running mean/std tracker (`RunningMeanStd` in `reinforce.py`).

### Implementation notes

- Base approach rewards (`MOVE_ON_BASE_REWARD`, `MOVE_NEAR_BASE_REWARD`) fire on every qualifying move throughout the episode — there is no per-base once-only flag.
- An exploration bonus (0.1 × (5 − visit_count) per cell, clamped to 0) is implemented but commented out in `perform_move_action`.
- Claiming an enemy base is treated identically to claiming an unclaimed base (+0.15). The player who loses the base receives no reward signal.

### Problem: reward sparsity

With near-random play, ~99 % of episodes hit the 200-action truncation limit. The dominant training signal is the flat −1.0 truncation penalty, not any signal about strategic progress. Consequences:

- The actor gradient is near-zero for most of training because advantages are nearly identical across all actions.
- The policy has no incentive to differentiate *how* it plays — only *whether* it avoids truncation.
- Base claiming, which is the core mechanic, is rewarded sporadically and only at the rare moment of capture.

---

## Ideas for denser rewards

### ~~1. Per-turn base-lead bonus~~ *(ruled out — superseded by idea 3)*

```python
base_lead_reward = k * (my_bases - starting_bases)  # e.g. k = 0.01
```

Added to the reward every step. Positive when holding more than the starting 2 bases, zero at parity, negative if losing ground.

**Ruled out**: idea 3 covers the same intent with better theoretical properties and without the passive-holding distortion (see idea 3 for details).

---

### ~~2. Opponent-loss penalty (symmetric claiming)~~ *(ruled out — attribution problem)*

Idea: when the opponent claims a base, give the defender an explicit negative reward.

**Ruled out**: the penalty would appear in the defender's trajectory at a timestep causally unrelated to the base loss — the defender wasn't even acting when the claim happened. GAE would attribute it to whatever action the defender happened to take next, which is wrong. For bases near the opponent's start the defender couldn't have prevented it regardless.

Idea 3 already covers this correctly: when the opponent claims a base, `phi_before` drops on the defender's next turn, the critic sees a less valuable state, and GAE propagates the disadvantage backwards to the actions that *actually* caused it (e.g., failing to contest that base earlier). Increase `SHAPING_C` if the signal feels too weak.

---

### 3. Potential-based reward shaping ✅ implemented

A theoretically clean way to add dense rewards without changing the optimal policy (Ng et al. 1999). Define a potential:

```python
phi(s) = c * (my_bases - opp_bases)  # e.g. c = 0.05
```

Shaped reward at each step:

```python
r_shaped = r + gamma * phi(s_next) - phi(s)
```

This fires a positive pulse when you gain a base and a negative pulse when you lose one, automatically scaled by the discount factor.

"Same optimal policy" means the shaping doesn't redirect the agent toward a different goal — it just makes the path to winning less sparse. The original win/loss signal still defines what good play is. The shaped rewards cannot make a losing strategy look better than a winning one in the long run, because the `gamma * phi(s') - phi(s)` terms telescope and nearly cancel out over a full trajectory, leaving only boundary terms.

---

### 4. Distance-to-objective shaping

Reward units for reducing their distance to the nearest unclaimed or enemy base:

```python
delta_dist = prev_min_dist_to_target - new_min_dist_to_target
proximity_reward = k * delta_dist  # positive when closing in
```

Guides units toward objectives from the start of the episode rather than waiting for them to stumble onto a base. Requires computing BFS distance on the hex grid — manageable per step.

**Trade-off**: can create reward-hacking if the agent oscillates near a base without claiming it; requires a small claiming bonus to remain dominant.

---

### 5. Graduated distance reward (replaces binary near/on)

Replace the current binary `MOVE_ON_BASE_REWARD` / `MOVE_NEAR_BASE_REWARD` with a smooth gradient:

```python
min_dist = min(hex_distance(unit_loc, base) for base in unclaimed_bases)
proximity_reward = k / (1 + min_dist)  # e.g. k = 0.005
```

Gives a stronger signal the closer the unit is, rather than a step function at distance 0 and 1.

---

### 6. Re-enable exploration reward

The commented-out code in `perform_move_action` gives a bonus for visiting new cells:

```python
explore_multiplier = MOVE_EXPLORE_REWARD_MAX_TURN - visit_count[end]
explore_reward = max(0, MOVE_EXPLORE_REWARD_PER_TURN * explore_multiplier)
```

This decays to zero after 5 visits, so it fires mostly in the early game and does not create permanent incentives to wander. Useful for breaking the initial frozen-position failure mode.

---

### 7. Win-speed bonus

Scale the terminal win reward by how quickly the game ends:

```python
win_reward = WIN_REWARD + speed_bonus * (max_actions - action_count) / max_actions
```

Encourages decisive play. Keeps the max reward at `WIN_REWARD + speed_bonus` for an instant win and asymptotes to `WIN_REWARD` for a slow win.

---

### 8. Base-capture sequence reward

Small incremental bonuses for the full capture sequence — adjacent → on → claim — to encourage following through:

```python
MOVE_ADJACENT_TO_UNCLAIMED = +0.002   # currently MOVE_NEAR_BASE_REWARD
MOVE_ONTO_UNCLAIMED        = +0.008   # currently MOVE_ON_BASE_REWARD
CLAIM_UNCLAIMED            = +0.15    # unchanged
```

These are additive across steps of the sequence, so the total reward for a full capture is `0.002 + 0.008 + 0.15 = 0.16` compared to the current `0.001 + 0.005 + 0.15 = 0.156` — structurally similar but more differentiating at the approach steps.

---

## Implementation priority

| Idea | Difficulty | Expected impact | Notes |
|---|---|---|---|
| ~~Per-turn base-lead bonus (1)~~ | — | — | Ruled out; superseded by idea 3 |
| Potential-based shaping (3) | Low | High | ✅ Implemented |
| ~~Opponent-loss penalty (2)~~ | — | — | Ruled out; attribution problem, covered by idea 3 |
| Re-enable exploration (6) | Low | Medium | Already implemented, just uncomment |
| Graduated distance reward (5) | Medium | Medium | Replaces current binary approach |
| Distance-to-objective (4) | Medium | Medium | Requires BFS per step |
| Win-speed bonus (7) | Low | Low | Polish after policy learns basics |
| Sequence reward (8) | Low | Low | Marginal improvement on current design |
