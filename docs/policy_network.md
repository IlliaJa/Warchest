# Policy Network

`src/services/policy/policy.py` implements two separate networks: `Policy` (actor) and `Critic` (value function). They share the same input encoding structure but have independent weights, allowing the critic to learn value-specific representations without conflicting with the actor's policy gradient.

## Input encoding

### Board encoder (CNN)

The board is encoded into `BOARD_CHANNELS` (48) planes by `generate_observation()` in `warchest_env.py` (not by the policy — the policy consumes the pre-encoded tensor directly):

| Planes | Content |
|---|---|
| 0 | Invalid cells |
| 1 | Empty cells |
| 2 | Uncontrolled bases |
| 3 | Active player's own bases |
| 4 | Opponent's bases |
| 5 | Exploration map (normalised visit counts, from active player's perspective) |
| 6–21 | Own unit-type stack planes (index = unit id − 1; stack height / `STACK_NORM`) |
| 22–37 | Opponent unit-type stack planes (same indexing) |
| 38–40 | Own threat: melee, ranged, charge — graded hit-count this side could land on each cell *this turn*, clipped to 1.0 |
| 41–43 | Enemy threat: melee, ranged, charge |
| 44 | `row_coord` — static ego-centric row index / 6 |
| 45 | `col_coord` — static ego-centric column index / 6 (the flank axis — see `docs/history.md` → "Threat/position-aware observation + deeper trunk") |
| 46 | `own_base_reach` — 0/1: claimable base cells (uncontrolled or opponent-held) I can move a unit onto and claim *this turn* |
| 47 | `enemy_base_reach` — 0/1: my bases (and neutrals) the opponent can reach and take this turn (objective-analogue of the threat planes; see `docs/observation_improvement.md`) |

Planes 3/4, 6–37, 38–47 are all ego-centric (own vs opponent) regardless of which player is active; the P2 view rotates the whole board 180° so "own"/"forward" always mean the same thing. Three `HexConv2d` layers (3×3 hex-masked kernel, so the two non-hex-adjacent corners are always zero) process the `[BOARD_CHANNELS,7,7]` input — receptive-field radius 3, exactly covering the Lancer's distance-3 charge:

```
HexConv2d(BOARD_CHANNELS→32) + ReLU
HexConv2d(32→hidden_dim) + ReLU
HexConv2d(hidden_dim→hidden_dim) + ReLU
```

### Global features

`global[GLOBAL_DIM]` (211) carries round/base/initiative counters and ego-centric coin-counting per type (own hand/bag/discard/supply/owned exactly; opponent's on-board/faceup/supply/owned exactly, with a bounded `hidden` pool standing in for what can't be observed), plus (OBS_VERSION 10) **2 material-at-risk scalars** (own/opp coins that can die this turn = `Σ min(hits, stack)`), a **17-wide expected-opponent-hand vector** (`hidden · opp_hand_size / hidden_total` — actor-side estimate of live counter-capacity; the critic sees the true split via `PRIV_DIM`), and **3 base-control reach scalars** (bases I can claim this turn, my bases under flip threat, and a win-proximity alarm), then the pending-tactic-continuation one-hot — see the constant block above `GLOBAL_DIM` in `warchest_env.py` and `docs/observation_improvement.md` for the exact layout and rationale.

## Feature fusion and heads

The spatial `policy_head` (1×1 conv → per-cell verb logits) reads the full `[hidden_dim,7,7]` feature map directly, so it was never location-blind. The `verb_head`/`facedown_head` previously read a single global mean pool, which *is* location-blind — it can tell a threat exists somewhere but not which flank. They now read `_split_pool(feat)`: a two-way mean pool along the flank (column) axis, columns 0–3 and 3–6 (column 3, the board's true center, deliberately shared by both halves), concatenated to `[2*hidden_dim]`.

**Actor** — `policy_head`: `Conv2d(hidden_dim + GLOBAL_DIM → N_VERBS, kernel=1)` for the spatial/within-verb logits; `facedown_head`/`verb_head`: `Linear(2*hidden_dim + GLOBAL_DIM → ...)` on the split-pooled features. Invalid actions are masked with −1e9 before softmax.

**Critic (separate network)** — same 3-layer board encoder and split pool, concatenated with global features, a 3-d opponent one-hot, and a privileged (critic-only) hidden-coin vector, then:
```
Linear(2*hidden_dim + GLOBAL_DIM + OPP_DIM + PRIV_DIM → hidden_dim) + ReLU
Linear(hidden_dim → hidden_dim) + ReLU
Linear(hidden_dim → hidden_dim // 2) + ReLU
Linear(hidden_dim // 2 → 1)
```

## Key methods

### Policy

| Method | Returns | Notes |
|---|---|---|
| `act(obs)` | `(action, log_prob, entropy)` | Sample from policy; used during rollout |
| `act_with_encoded(obs)` | `(action, log_prob, entropy, feat)` | Also returns encoded board features, so `Critic.value_from_features` can reuse them and skip a second board-encoder pass |
| `evaluate_actions_batch(batch)` | `(log_probs, entropies)` | Batched re-evaluation; used in PPO update |

Board/global encoding itself happens in `generate_observation()` (env), not in `Policy` — there is no separate `encode_board` step on the policy side.

### Critic

| Method | Returns | Notes |
|---|---|---|
| `value_single(obs, opp_onehot, privileged)` | scalar tensor | Used during rollout collection |
| `value_from_features(feat, ...)` | scalar tensor | Reuses `Policy.act_with_encoded`'s board features, skipping the critic's own board encoder |
| `value_batch(batch)` | `[N]` tensor | Used during PPO update |

## Hyperparameters (defaults in `src/app/ppo.py`)

| Parameter | Default |
|---|---|
| `hidden_dim` (Policy) | 64 |
| `critic_hidden_dim` (Critic) | 128 — widened alone first; see `docs/decision.md`, 2026-07-03 |
| `action_space` (`ACTION_SPACE_SIZE`) | 1875 |
| Actor LR | 3e-4 (Adam), linearly decayed to `lr_final_frac * init` over the run |
| Critic LR | 3e-4 (Adam), same decay schedule |
| `entropy_coeff` | 0.025, linearly annealed to `entropy_coeff_final = 0.003` over the run |
| Gradient clip max_norm | 1.0 (applied separately to actor-side and critic parameters) |
