# Warchest — Architecture Overview

Warchest is a two-player turn-based hex-grid strategy game paired with a reinforcement learning training framework. A PPO actor-critic policy learns to play the game.

## High-level component map

```
src/app/ppo.py  (PPOTrainer)        training entry point
│
├─ WarChestEnv                      Gymnasium env  (src/services/environment/)
│   ├─ Board                        hex grid + cell logic
│   ├─ GameState                    state snapshot for replay (bags/hands/discards/boxed)
│   ├─ roster.py                    single source of truth: all 16 unit types + Royal coin
│   ├─ obs_encoders/                versioned observation encoders; env delegates encode()
│   │                               /observation_space()/encode_privileged() to the chosen one
│   └─ Action                       action dataclass
│
├─ Policy                           actor network   (src/services/policy/policy.py)
│   ├─ board_encoder  (3x HexConv2d — no separate unit_encoder; units live in board planes)
│   ├─ policy_head    → per-cell verb logits (masked), 1x1 conv
│   └─ verb_head / facedown_head → non-spatial verb logits, from a flank-split pool
│
├─ Critic                           separate value network, wider trunk (critic_hidden_dim=128)
│   ├─ board_encoder  (3x HexConv2d, independent weights)
│   └─ head           → scalar state value, fed board features + globals + opponent
│                        one-hot + a privileged (critic-only) hidden-coin vector
│
├─ OpponentPool                     opponent sampler  (src/services/opponent_pool.py)
│   ├─ RandomBot                    uniform random over valid actions
│   ├─ GreedyBot                    priority: attack → control → move-toward-base → deploy → pass
│   └─ frozen Policy snapshots      rolling window (snapshot every 15 batches, max 20)
│
└─ RolloutBuffer                    GAE buffer  (src/utils/rollout_buffer.py)
```

## Data flow per environment step

```
Observation dict (generate_observation() → obs_encoders/v10.py, OBS_VERSION=10)
  ├─ board[48,7,7]             base/terrain + per-type unit stacks + threat + coord + base-reach planes
  ├─ global[211]                round/base/initiative + per-type coin counts + material-at-risk
  │                              + E_opp_hand + base-reach scalars + pending one-hot
  └─ valid_action_mask[1875]   legal-action binary mask (factored verb x cell + face-down)
       │
       ├──► Policy.act()       → sampled action, log_prob
       ├──► Critic.value_single() → state value V(s), given a privileged opponent-hand vector too
       │
       ▼
WarChestEnv.step(action_id)
  → next obs, reward, terminated, truncated, info
  (turn does not advance while a multi-step tactic's `state.pending` is set)
```

For `active_player == 2`, the encoder returns the board/globals pre-rotated 180°
so the network always sees "my units" the same way; `WarChestEnv.remap_action` performs the
matching inverse remap on the chosen action id. Full schema: `docs/environment_api.md`.

**Observation encoding is versioned and pluggable** (`src/services/environment/obs_encoders/`).
The engine holds only the stable game rules and rules-queries
(`unit_threat_footprint`, `attack_enabler_coins`, `unit_base_reach_cells`); an encoder
(`v10.py`, selected via the registry) owns the version-varying part — the coin-availability
model, feature aggregation, normalizers, plane layout, and ego-rotation. `WarChestEnv(obs_encoder=…)`
picks the encoder (default: latest); `Policy`/`Critic` size their input layers from the paired
encoder's dims. Because `encode(view)` is a pure function of the game state, agents built for
different obs versions can encode one shared `WarChestEnv` independently — the basis for the
cross-era round-robin gauntlet (`src/services/gauntlet.py`; design rationale in `docs/history.md`
→ *Measurement + opponent infrastructure*).
`tests/test_obs_golden.py` guards the encoding byte-for-byte.

## File reference

| File | Role |
|---|---|
| `src/app/ppo.py` | PPOTrainer class: collect, update, eval, log |
| `src/app/reinforce.py` | Legacy REINFORCE+GAE trainer (kept for reference, not the primary path) |
| `src/app/demo.py` | Evaluate saved model vs random + interactive replay |
| `src/app/eval_bucketed.py` | Per-composition eval bucketing (`docs/IDEAS.md` #R1) |
| `src/app/gauntlet.py` | Round-robin gauntlet CLI: load checkpoints + baselines, print WR matrix / Elo / transitivity |
| `src/app/main.py` | Minimal random-action smoke test |
| `src/services/environment/warchest_env.py` | Gymnasium env: reset, step, rewards, action encode/decode, rules-queries; delegates obs encoding |
| `src/services/environment/obs_encoders/` | Versioned observation encoders (`v10.py` + registry); owns plane layout, normalizers, ego-rotation, feature derivation |
| `src/services/environment/board.py` | Hex board, adjacency, base ownership |
| `src/services/environment/game_state.py` | State snapshot: bags/hands/discards/boxed/pending, used for replay |
| `src/services/environment/roster.py` | Single source of truth for all 16 unit types + Royal coin (id/icon/colour/total-coins) |
| `src/services/environment/game_renderer.py` | Matplotlib interactive game replay |
| `src/services/environment/coin_render.py` | Per-unit coin colours/glyphs for rendering |
| `src/services/environment/units/baseunit.py` | Unit class, generated per-type from `roster.py` |
| `src/services/environment/action.py` | Action dataclass |
| `src/services/environment/cell_ids.py` | Cell type constants |
| `src/services/policy/policy.py` | Policy (actor) and Critic networks, `HexConv2d`; obs dims from the paired encoder |
| `src/services/policy/checkpoint.py` | Checkpoint (de)serialization with obs-version + arch metadata (legacy bare-`state_dict` fallback) |
| `src/services/gauntlet.py` | Gauntlet agents (`act(env)` → absolute action), `round_robin`, Bradley-Terry/Elo, transitivity |
| `src/services/opponent_pool.py` | Weighted sampler: random / greedy / pool snapshots |
| `src/services/bots/base.py` | Bot ABC |
| `src/services/bots/random_bot.py` | Uniform-random valid-action bot |
| `src/services/bots/greedy_bot.py` | Priority attack → control → move → deploy → pass bot |
| `src/utils/rollout_buffer.py` | Transition storage + GAE computation |
| `src/utils/elo.py` | Elo rating tracker |
| `Dockerfile` | Container for cloud training |
| `launch-agent.yaml` | W&B Agents queue config |
