# Warchest — Architecture Overview

Warchest is a two-player turn-based hex-grid strategy game paired with a reinforcement learning (RL) training framework. An actor-critic policy network learns to play the game via REINFORCE with Generalized Advantage Estimation (GAE).

## High-level component map

```
reinforce.py          training entry point
│
├─ WarChestEnv        Gymnasium environment (environment/warchest_env.py)
│   ├─ Board          hex grid + cell logic   (environment/board.py)
│   ├─ GameState      snapshot for replay     (environment/game_state.py)
│   ├─ Units          Swordsman, BaseUnit     (environment/units/)
│   └─ Action         action dataclass        (environment/action.py)
│
└─ Policy             actor-critic network    (policy.py)
    ├─ board CNN encoder
    ├─ unit position encoder
    ├─ actor head  → action logits (masked)
    └─ critic head → state value
```

## Data flow per environment step

```
Observation dict
  ├─ board[6,7,7]          6-channel board encoding
  ├─ exploration_map[7,7]  per-player visit counts
  ├─ units[2,2,2]          (player, unit_idx, [row,col])
  ├─ global[4]             active_player, turn, p1_bases, p2_bases
  └─ valid_action_mask[14] legal-move binary mask
       │
       ▼
Policy.act()
  → sampled action id, log_prob, value, entropy
       │
       ▼
WarChestEnv.step(action_id)
  → next obs, reward, terminated, truncated, info
```

## Training loop summary

```
for episode in range(3000):
    reset env
    randomly assign each player as policy-controlled or random
    collect full trajectory
    compute GAE advantages (backwards pass)
    actor_loss  = -mean(log_prob * advantages)
    critic_loss = MSE(values, returns)
    total_loss  = actor_loss + critic_loss - entropy_coeff * entropy
    clip gradients (max_norm=1.0)
    optimizer.step()
    log to W&B
save model → data/warchest_policy_YYYYMMDD-HH:MM.pth
```

## Key design decisions

| Decision | Rationale |
|---|---|
| Gymnasium API | Standard interface, easy to swap environments |
| GAE (λ=0.95) | Balances bias/variance vs. pure MC or pure TD |
| Entropy regularization | Prevents premature policy collapse; scheduled 0.1→0.01 |
| Valid-action masking | Guarantees the policy never picks illegal moves |
| Self-play + random mixing | 30% chance each player is random, preventing policy overfitting to one opponent |
| Gradient clipping 1.0 | Stabilises training on sparse reward signal |

## File reference

| File | Role |
|---|---|
| `environment/warchest_env.py` | Gymnasium env: reset, step, observation, rewards |
| `environment/board.py` | Hex board, adjacency, base ownership |
| `environment/game_state.py` | Immutable state snapshot used for replay |
| `environment/game_renderer.py` | Matplotlib interactive game replay |
| `environment/units/baseunit.py` | Abstract unit (location, player ownership) |
| `environment/units/swordsman.py` | Concrete unit, only type currently |
| `environment/action.py` | Action dataclass |
| `environment/cell_ids.py` | Cell type constants |
| `policy.py` | Actor-critic neural network |
| `reinforce.py` | Training script |
| `test.py` | Evaluation: win-rate vs random, replay |
| `demo.py` | Quick random-action smoke test |
| `main.py` | Minimal game-loop demo |
| `policy_viz.py` | Export network graph to TensorBoard |
| `Dockerfile` | Container for cloud training |
| `launch-agent.yaml` | W&B Agents queue config |
