# Policy Network

`src/services/policy/policy.py` implements two separate networks: `Policy` (actor) and `Critic` (value function). They share the same input encoding structure but have independent weights, allowing the critic to learn value-specific representations without conflicting with the actor's policy gradient.

## Input encoding

### Board encoder (CNN)

The raw board `[7,7]` is first expanded into 6 channels by `encode_board()`:

| Channel | Content |
|---|---|
| 0 | Invalid cells |
| 1 | Empty cells |
| 2 | Uncontrolled bases |
| 3 | Active player's own bases |
| 4 | Opponent's bases |
| 5 | Exploration map (normalised visit counts, from active player's perspective) |

Channels 3 and 4 are always ego-centric (own vs opponent) regardless of which player is active. Two conv layers process the `[6,7,7]` input:

```
Conv2d(6→32, kernel=3, padding=1) + ReLU
Conv2d(32→64, kernel=3, padding=1) + ReLU
Flatten → Linear(64*7*7 → hidden_dim)
```

### Unit encoder (MLP)

Unit positions are shaped `[2, 2, 2]` (2 player slots × 2 units × (row, col)). Each unit's 2D position is encoded independently:

```
Linear(2 → 16) + ReLU
Linear(16 → 32)
```

The two units per player slot are averaged, then the two player slots are concatenated, giving a 64-dimensional unit feature vector.

### Global features

`global[3]` is passed directly:
- `turn // 2` (half-turn counter)
- Active player's base count
- Opponent's base count

## Feature fusion and heads

All encoded features are concatenated: `[hidden_dim + 3 + 64]`.

**Actor head** — outputs action logits:
```
Linear(fused → hidden_dim*2) + ReLU
Linear(hidden_dim*2 → hidden_dim) + ReLU
Linear(hidden_dim → action_space)
```
Invalid actions are masked with −1e9 before softmax.

**Critic (separate network)** — same board and unit encoders, then:
```
Linear(hidden_dim + 3 + 64 → hidden_dim) + ReLU
Linear(hidden_dim → 1)
```

## Key methods

### Policy

| Method | Returns | Notes |
|---|---|---|
| `act(obs)` | `(action, log_prob, entropy)` | Sample from policy; used during rollout |
| `evaluate_actions_batch(batch)` | `(log_probs, entropies)` | Batched re-evaluation; used in PPO update |
| `encode_board(board, exploration_map, active_player)` | `[6,7,7]` array | Static encoding, single observation |
| `encode_board_batch(boards, maps, players)` | `[N,6,7,7]` array | Vectorised encoding for a full batch |

### Critic

| Method | Returns | Notes |
|---|---|---|
| `value_single(obs)` | scalar tensor | Used during rollout collection |
| `value_batch(batch)` | `[N]` tensor | Used during PPO update |

## Hyperparameters (defaults in `src/app/ppo.py`)

| Parameter | Default |
|---|---|
| `hidden_dim` | 64 |
| `action_space` | 14 |
| Actor LR | 3e-4 (Adam) |
| Critic LR | 3e-4 (Adam) |
| `ENTROPY_COEFF` | 0.001 |
| Gradient clip max_norm | 1.0 (applied separately to actor-side and critic parameters) |
