# Policy Network

`policy.py` implements an actor-critic network that takes the environment observation and outputs action probabilities and a state value.

## Input encoding

### Board encoder (CNN)

The board observation `[7,7]` is first expanded into 6 binary channels by `encode_board()`:

| Channel | Content |
|---|---|
| 0 | Invalid cells |
| 1 | Empty cells |
| 2 | Uncontrolled bases |
| 3 | Player 1 bases |
| 4 | Player 2 bases |
| 5 | Exploration map (normalised visit counts) |

Two conv layers process the `[6,7,7]` input:
```
Conv2d(6→32, kernel=3, padding=1) + ReLU
Conv2d(32→64, kernel=3, padding=1) + ReLU
Flatten → Linear(64*7*7 → hidden_dim) + ReLU
```

### Unit encoder (MLP)

Unit positions `[2,2]` (flattened to 4 values):
```
Linear(4 → 16) + ReLU
Linear(16 → 32)
```

### Global features

`global[4]` is passed directly:
- `active_player − 1` (0 or 1)
- Normalised turn count
- Player 1 base count
- Player 2 base count

## Feature fusion and heads

All encoded features are concatenated: `[hidden_dim + 32 + 4]`.

**Actor head** — outputs action logits:
```
Linear(fused → hidden_dim*2) + ReLU
Linear(hidden_dim*2 → hidden_dim) + ReLU
Linear(hidden_dim → action_space)
```
Invalid actions are masked with −1e9 before softmax.

**Critic head** — outputs scalar state value:
```
Linear(fused → 1)
```

## Key methods

| Method | Returns | Notes |
|---|---|---|
| `forward(obs)` | `(probs, value)` | Full forward pass |
| `act(obs)` | `(action, log_prob, value, entropy)` | Sample from policy |
| `evaluate_actions(obs, action)` | `(log_prob, entropy, value)` | Used during loss computation |
| `encode_board(board, exploration_map)` | `[6,7,7]` tensor | Static board encoding |

## Hyperparameters (defaults in reinforce.py)

| Parameter | Default |
|---|---|
| `hidden_dim` | 64 |
| `action_space` | 14 |
| Learning rate | 5e-3 (Adam) |
| Entropy coeff (early) | 0.1 |
| Entropy coeff (late, after 75% episodes) | 0.01 |
| Gradient clip max_norm | 1.0 |
