# Policy Network

`src/services/policy/policy.py` implements two separate networks: `Policy` (actor) and `Critic` (value function). They share the same input encoding structure but have independent weights, allowing the critic to learn value-specific representations without conflicting with the actor's policy gradient.

**Both are arch-versioned**, because checkpoints of every generation exist on disk and the
round-robin gauntlet reconstructs them from the `arch` string recorded in the checkpoint envelope
(`checkpoint.py`). This page describes the **current defaults**, `policy_factored_v2` and
`critic_v5`; where an older arch differs materially it is called out inline. `Policy`/`Critic`
build any registered arch on demand, so never mutate a released one — add the next.

| arch | what it added |
|---|---|
| `policy_factored_v1` | the original factored-verb head; raw one-hot unit planes, globals broadcast to all 49 cells |
| **`policy_factored_v2`** | **unit-type embedding + FiLM trunk conditioning (`docs/IDEAS.md` A1 + A3)** |
| `critic_v1` | original un-normalised trunk — **provably dies** (`docs/next_iteration.md` §3.4) |
| `critic_v2` | GroupNorm + a board-only auxiliary head |
| `critic_v3` | v2 minus the 3-wide opponent one-hot (offset moved into `--adv-norm per_opponent`) |
| `critic_v4` | flank-average pool replaced by a base-cell + unit-occupancy gather (A2) |
| **`critic_v5`** | **v4 + the same unit-type embedding as the policy (A1); deliberately NO FiLM** |

## Input encoding

### Board encoder (CNN)

The board is encoded into `BOARD_CHANNELS` (48) planes by the versioned obs encoder `obs_encoders/v11.py` (which `generate_observation()` delegates to; not by the policy — the policy consumes the pre-encoded tensor directly):

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

Planes 3/4, 6–37, 38–47 are all ego-centric (own vs opponent) regardless of which player is active; the P2 view rotates the whole board 180° so "own"/"forward" always mean the same thing. Three `HexConv2d` layers (3×3 hex-masked kernel, so the two non-hex-adjacent corners are always zero) process the `[BOARD_CHANNELS,7,7]` input — receptive-field radius 3, exactly covering the Lancer's distance-3 charge.

**`policy_factored_v2` inserts two things into that stack** (`docs/IDEAS.md` A1 + A3, full rationale in `docs/architecture.md` → *How the network reads a unit type*):

```
planes 6–21 / 22–37  ──contract against the shared unit-type table──►  16 + 16 channels
                        (A1: 10 frozen roster.py attribute columns + 6 learned per type)
HexConv2d(48→32)        + FiLM(globals) + ReLU
HexConv2d(32→hidden)    + FiLM(globals) + ReLU
HexConv2d(hidden→hidden)+ FiLM(globals) + ReLU
```

The contraction is `embedded[d] = Σ_t count[t] · E[t,d]`, so the 32 per-type planes become 2×16 channels whose meaning does not depend on which 4-of-16 types were drafted. With 10 + 6 = 16 = `NUM_UNIT_TYPES` the channel count happens to be unchanged (48 in, 48 out) — a coincidence of the chosen dims, which is why `ObsTypeEmbedding` recomputes both widths instead of assuming them. FiLM applies `x ← x·(1 + γ) + β` per channel with `(γ, β) = MLP(globals)`, zero-initialised so the module is exactly the identity at step 0.

`policy_factored_v1` is the same three layers with no embedding and no FiLM:

```
HexConv2d(BOARD_CHANNELS→32) + ReLU
HexConv2d(32→hidden_dim) + ReLU
HexConv2d(hidden_dim→hidden_dim) + ReLU
```

### Global features

`global[GLOBAL_DIM]` (**245** at OBS_VERSION 11) carries round/base/initiative counters and ego-centric coin-counting per type (own hand/bag/discard/supply/owned exactly; opponent's on-board/faceup/supply/owned exactly, with a bounded `hidden` pool standing in for what can't be observed), plus **2 material-at-risk scalars** (own/opp coins that can die this turn = `Σ min(hits, stack)`), a **17-wide expected-opponent-hand vector** (`hidden · opp_hand_size / hidden_total` — actor-side estimate of live counter-capacity; the critic sees the true split via `PRIV_DIM`), **3 base-control reach scalars** (bases I can claim this turn, my bases under flip threat, and a win-proximity alarm), the two **draw-share** vectors `p_soon`/`p_mean` added by v11, then the pending-tactic-continuation one-hot — see the constant block above `GLOBAL_DIM` in `obs_encoders/v11.py` and `docs/observation_improvement.md` for the exact layout and rationale.

On `policy_factored_v2`/`critic_v5` the per-type blocks of this vector (10 running over the 17-coin deck, 3 over the 16 unit types) are contracted against the same unit-type table before any layer sees them; the royal coin has no embedding row and its count passes through as a raw scalar. The block offsets are published by the encoder (`deck_block_offsets`, `unit_block_offsets`, `deck_unit_positions`, `deck_royal_position`) rather than hardcoded in the network.

## Feature fusion and heads

The spatial `policy_head` (1×1 conv → per-cell verb logits) reads the full `[hidden_dim,7,7]` feature map directly, so it was never location-blind. The `verb_head`/`facedown_head` previously read a single global mean pool, which *is* location-blind — it can tell a threat exists somewhere but not which flank. They now read `_split_pool(feat)`: a two-way mean pool along the flank (column) axis, columns 0–3 and 3–6 (column 3, the board's true center, deliberately shared by both halves), concatenated to `[2*hidden_dim]`.

**Actor** — `policy_head`: `Conv2d(hidden_dim → N_VERBS, kernel=1)` for the spatial/within-verb logits; `facedown_head`/`verb_head`: `Linear(2*hidden_dim + GLOBAL_DIM → ...)` on the split-pooled features. Invalid actions are masked with −1e9 before softmax.

`policy_factored_v1`'s `policy_head` was `Conv2d(hidden_dim + GLOBAL_DIM → N_VERBS, kernel=1)`, fed the globals broadcast onto all 49 cells. That path is gone in v2, and the reason is worth stating because it is not an efficiency argument: a value identical at every cell, run through a 1×1 conv that applies the same weights at every cell, collapses to a single constant `W_g @ g` added everywhere. The globals contributed a position-independent bias and nothing else — 245 × 32 = 7840 weights computing one offset 49 times — and an additive bias cannot switch a channel off when the hand makes it irrelevant. FiLM multiplies, which can. `facedown_head`/`verb_head` keep their globals in both archs: they read a *pooled* vector, which is an ordinary MLP input rather than a per-cell broadcast.

**Critic (separate network)** — `critic_v5`: its own 3-layer GroupNorm board encoder over the same A1-contracted planes, the A2 gather readout, then the globals and a privileged (critic-only) hidden-coin vector:
```
Linear(pool_width + GLOBAL_DIM + PRIV_DIM → hidden_dim) + ReLU
Linear(hidden_dim → hidden_dim) + ReLU
Linear(hidden_dim → hidden_dim // 2) + ReLU
Linear(hidden_dim // 2 → 1)
```
`pool_width` is `16 * hidden_dim` on `critic_v4`/`v5` (10 fixed base cells + masked mean+max over own/opponent unit cells + whole-board mean+max) against `2 * hidden_dim` for the `_split_pool` of v1–v3. `OPP_DIM` is **not** in the head from v3 onward — the opponent-identity offset it carried is removed from the advantage instead (`--adv-norm per_opponent`); v1/v2 still require their 3-wide one-hot and raise if it is missing.

`critic_v2` and later also carry `board_only_head`, a `Linear(pool_width → 1)` auxiliary value read from the pooled board **alone**, added to the loss at `--aux-board-coeff` (default 0.1). Its loss is unsatisfiable without a board representation that carries signal, which is what keeps the trunk alive. **This is why the critic deliberately has no FiLM**: conditioning the trunk on globals would put them back inside that path and silently void the fix that took the positional-sibling tie rate from 93 % to 0 %.

## Key methods

### Policy

| Method | Returns | Notes |
|---|---|---|
| `act(obs)` | `(action, log_prob, entropy)` | Sample from policy; used during rollout |
| `act_with_encoded(obs)` | `(action, log_prob, entropy, feat, global_feats)` | Also returns encoded board features (and the **raw** globals — the critic applies its own contraction) so `Critic.value_from_features` could reuse them; that fast path is unsupported on `critic_v4`/`v5`, which need the raw board for occupancy |
| `evaluate_actions_batch(batch)` | `(log_probs, entropies, verb_entropies)` | Batched re-evaluation; used in PPO update. The third value is the top-level verb-marginal entropy, which gets its own bonus (`docs/IDEAS.md` #R8) |

Board/global encoding itself happens in the versioned obs encoder (`obs_encoders/`, which `generate_observation()` delegates to), not in `Policy` — there is no separate `encode_board` step on the policy side. `Policy`/`Critic` read `BOARD_CHANNELS`/`GLOBAL_DIM`/`PRIV_DIM` from the encoder they are paired with (`obs_encoder=…`), not from a hardcoded env constant.

### Critic

| Method | Returns | Notes |
|---|---|---|
| `value_single(obs, opp_onehot, privileged)` | scalar tensor | Used during rollout collection |
| `value_from_features(feat, ...)` | scalar tensor | Reuses `Policy.act_with_encoded`'s board features, skipping the critic's own board encoder. **Raises on `critic_v4`/`v5`** — their gather readout needs the raw board tensor for unit occupancy, so it refuses rather than pooling the wrong thing |
| `value_batch(batch)` | `[N]` tensor | Used during PPO update |
| `board_only_value(board)` | `[N]` tensor | `critic_v2`+ auxiliary head; value from the board alone |
| `trunk_health(board)` | `{'alive': [f1,f2,f3], 'out_std': float}` | Per-batch collapse guard. For `critic_v2`+ **`out_std` is the condition to watch** — GroupNorm makes the alive fraction alone useless (a collapsed v2 trunk reads 1.0 alive while carrying no information) |

## Hyperparameters (defaults in `src/app/ppo.py`)

| Parameter | Default |
|---|---|
| `hidden_dim` (Policy) | 128 |
| `critic_hidden_dim` (Critic) | 192 — widened alone first; see `docs/decision.md`, 2026-07-03 |
| `--policy-arch` | `policy_factored_v2` |
| `--critic-arch` | `critic_v5` |
| `action_space` (`ACTION_SPACE_SIZE`) | 1875 |
| Actor LR | 3e-4 (Adam), linearly decayed to `lr_final_frac * init` (`lr_final_frac = 0.1`) |
| Critic LR | 3e-4 (Adam), same decay schedule |
| `entropy_coeff` | 0.025, linearly annealed to `entropy_coeff_final = 0.003` over the run |
| `verb_entropy_coeff` | 0.02, annealed to 0.01 — a non-zero floor on purpose, so rare verbs stay in the repertoire |
| `gamma` / `--lam` | 0.99 / 0.90 (λ lowered from 0.97 with the critic fix, `docs/IDEAS.md` L2) |
| `ppo_eps` | 0.2 |
| `--aux-board-coeff` | 0.1 (`critic_v2`+ board-only auxiliary loss) |
| `--adv-norm` | `per_opponent` |
| Gradient clip max_norm | 1.0 (applied separately to actor-side and critic parameters) |
