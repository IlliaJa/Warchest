# Reward Design

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
| Move onto unclaimed base | +0.005 |
| Move adjacent to unclaimed base | +0.001 |
| Each action taken | −0.002 |

Constants are defined at the top of `src/services/environment/warchest_env.py`. Raw environment rewards are augmented with potential-based shaping in `src/app/ppo.py` before being stored in the rollout buffer.

### Potential-based reward shaping

Applied by the training loop, not the environment:

```python
shaped_r = r + gamma * phi(s_next) - phi(s)
phi(s)   = SHAPING_C * (my_bases - opp_bases)    # SHAPING_C = 0.05
```

This fires a positive pulse when gaining a base and a negative pulse when losing one. Advantages are z-scored batch-wide; returns are kept in the original reward scale.

### Implementation notes

- Base approach rewards (`MOVE_ON_BASE_REWARD`, `MOVE_NEAR_BASE_REWARD`) fire on every qualifying move throughout the episode.
- An exploration bonus (0.1 × (5 − visit_count) per cell, clamped to 0) is implemented but commented out in `perform_move_action`.
- `CLAIM_BASE_REWARD` is 0.0. A non-zero direct claim reward caused a circular-claiming exploit: the policy learned to claim bases back and forth with pool opponents, accumulating reward (up to ~5.0) far in excess of `WIN_REWARD = 1.0`. Potential shaping already handles base value correctly, so the direct reward is not needed.

### Truncation reward

When the episode truncates (`action_count >= max_actions = 200`), a terminal reward is added to the last main-actor step based on the base lead:

```python
if my_bases > opp_bases:   trunc_reward = 0.0   # drew from a strong position
elif my_bases == opp_bases: trunc_reward = -0.5  # true draw
else:                       trunc_reward = -1.0  # drew from a losing position
```

---

## Ideas for denser rewards

### ~~1. Per-turn base-lead bonus~~ *(ruled out — superseded by idea 3)*

```python
base_lead_reward = k * (my_bases - starting_bases)  # e.g. k = 0.01
```

**Ruled out**: idea 3 covers the same intent with better theoretical properties.

---

### ~~2. Opponent-loss penalty (symmetric claiming)~~ *(ruled out — attribution problem)*

**Ruled out**: the penalty would appear in the defender's trajectory at a timestep causally unrelated to the base loss. Idea 3 already covers this correctly via the potential term.

---

### 3. Potential-based reward shaping ✅ implemented

A theoretically clean way to add dense rewards without changing the optimal policy (Ng et al. 1999). Define a potential:

```python
phi(s) = c * (my_bases - opp_bases)  # c = 0.05
```

Shaped reward at each step:

```python
r_shaped = r + gamma * phi(s_next) - phi(s)
```

The `gamma * phi(s') - phi(s)` terms telescope and nearly cancel over a full trajectory, leaving only boundary terms. The shaped rewards cannot make a losing strategy look better than a winning one in the long run.

---

### 4. Distance-to-objective shaping

Reward units for reducing their distance to the nearest unclaimed or enemy base:

```python
delta_dist = prev_min_dist_to_target - new_min_dist_to_target
proximity_reward = k * delta_dist  # positive when closing in
```

Requires BFS distance on the hex grid per step. Can create reward-hacking if the agent oscillates near a base without claiming.

---

### 5. Graduated distance reward (replaces binary near/on)

Replace the current binary `MOVE_ON_BASE_REWARD` / `MOVE_NEAR_BASE_REWARD` with a smooth gradient:

```python
min_dist = min(hex_distance(unit_loc, base) for base in unclaimed_bases)
proximity_reward = k / (1 + min_dist)  # e.g. k = 0.005
```

---

### 6. Re-enable exploration reward

The commented-out code in `perform_move_action` gives a bonus for visiting new cells:

```python
explore_multiplier = MOVE_EXPLORE_REWARD_MAX_TURN - visit_count[end]
explore_reward = max(0, MOVE_EXPLORE_REWARD_PER_TURN * explore_multiplier)
```

Decays to zero after 5 visits, so it fires mostly in the early game.

---

### 7. Win-speed bonus

Scale the terminal win reward by how quickly the game ends:

```python
win_reward = WIN_REWARD + speed_bonus * (max_actions - action_count) / max_actions
```

---

### 8. Base-capture sequence reward

Small incremental bonuses for the full capture sequence — adjacent → on → claim:

```python
MOVE_ADJACENT_TO_UNCLAIMED = +0.002
MOVE_ONTO_UNCLAIMED        = +0.008
CLAIM_UNCLAIMED            = +0.15   # if direct claim reward is restored
```

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
| Sequence reward (8) | Low | Low | Only relevant if direct claim reward is restored |
