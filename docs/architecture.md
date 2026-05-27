# Warchest — Architecture Overview

Warchest is a two-player turn-based hex-grid strategy game paired with a reinforcement learning training framework. A PPO actor-critic policy learns to play the game.

## High-level component map

```
src/app/ppo.py  (PPOTrainer)        training entry point
│
├─ WarChestEnv                      Gymnasium env  (src/services/environment/)
│   ├─ Board                        hex grid + cell logic
│   ├─ GameState                    state snapshot for replay
│   ├─ Units                        Swordsman, BaseUnit
│   └─ Action                       action dataclass
│
├─ Policy                           actor network   (src/services/policy/policy.py)
│   ├─ board_encoder  (CNN)
│   ├─ unit_encoder   (MLP)
│   └─ actor_head     → action logits (masked)
│
├─ Critic                           separate value network
│   ├─ board_encoder  (CNN, independent weights)
│   ├─ unit_encoder   (MLP, independent weights)
│   └─ head           → scalar state value
│
├─ OpponentPool                     opponent sampler  (src/services/opponent_pool.py)
│   ├─ RandomBot                    uniform random over valid actions
│   ├─ GreedyBot                    BFS toward nearest unclaimed/enemy base
│   └─ frozen Policy snapshots      rolling window of past policy states
│
└─ RolloutBuffer                    GAE buffer  (src/utils/rollout_buffer.py)
```

## Data flow per environment step

```
Observation dict
  ├─ board[7,7]                raw cell-id grid
  ├─ exploration_map[7,7]      per-player visit counts
  ├─ units[2,2,2]              (player_slot, unit_idx, [row,col])
  ├─ global[3]                 turn, my_bases, opp_bases
  └─ valid_action_mask[14]     legal-move binary mask
       │
       ├──► Policy.act()       → sampled action, log_prob
       ├──► Critic.value_single() → state value V(s)
       │
       ▼
WarChestEnv.step(action_id)
  → next obs, reward, terminated, truncated, info
```

## Training loop summary (PPOTrainer)

```
for batch in range(n_batches):
    # collect
    for episode in range(collect_episodes):
        sample opponent from OpponentPool (random / greedy / frozen snapshot)
        run episode; store main-actor (obs, action, log_prob, shaped_reward, value)
    compute GAE advantages + returns across all episodes

    # update (ppo_epochs inner epochs, minibatch_size=64)
    for epoch in range(ppo_epochs):
        for minibatch in buffer:
            ratio    = exp(log_pi_new - log_pi_old)
            L_actor  = -min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)
            L_critic = MSE(V(s), return)
            L_total  = L_actor + L_critic - entropy_coeff * H
            clip gradients (max_norm=1.0)
            optimizer.step()
            early-stop epoch if approx_kl > KL_TARGET

    pool.maybe_snapshot(policy)      # save current weights to rolling window
    eval vs greedy + random          # every eval_every batches
    log to W&B
```

## Key design decisions

| Decision | Rationale |
|---|---|
| PPO | 4–10× more gradient steps per episode than REINFORCE; clip prevents destructive updates |
| Separate Policy and Critic | Independent encoders let the critic develop value-specific representations without interfering with the actor's gradients |
| Opponent pool (random + greedy + frozen snapshots) | Prevents self-play cycling; greedy bot provides consistent pressure; random bot teaches basic coverage |
| Dynamic finetune phase | When eval WR vs random ≥ 90%, training pool switches to greedy+pool only, raising the bar automatically |
| Potential-based reward shaping | Dense per-step reward `γ·φ(s') − φ(s)` where `φ = c·(my_bases − opp_bases)` guides base control without distorting the optimal policy |
| CLAIM_BASE_REWARD = 0 | Direct claim reward caused circular-claim exploitation; potential shaping already values net base position correctly |
| Valid-action masking | Guarantees the policy never picks illegal moves |
| Gradient clipping 1.0 | Stabilises training on sparse reward signal |

## File reference

| File | Role |
|---|---|
| `src/app/ppo.py` | PPOTrainer class: collect, update, eval, log |
| `src/app/reinforce.py` | Legacy REINFORCE+GAE trainer (kept for reference) |
| `src/app/demo.py` | Evaluate saved model vs random + interactive replay |
| `src/app/main.py` | Minimal random-action smoke test |
| `src/services/environment/warchest_env.py` | Gymnasium env: reset, step, observation, rewards |
| `src/services/environment/board.py` | Hex board, adjacency, base ownership |
| `src/services/environment/game_state.py` | State snapshot used for replay |
| `src/services/environment/game_renderer.py` | Matplotlib interactive game replay |
| `src/services/environment/units/baseunit.py` | Abstract unit (location, player ownership) |
| `src/services/environment/units/swordsman.py` | Concrete unit type |
| `src/services/environment/action.py` | Action dataclass |
| `src/services/environment/cell_ids.py` | Cell type constants |
| `src/services/policy/policy.py` | Policy (actor) and Critic networks |
| `src/services/opponent_pool.py` | Weighted sampler: random / greedy / pool snapshots |
| `src/services/bots/base.py` | Bot ABC |
| `src/services/bots/random_bot.py` | Uniform-random valid-action bot |
| `src/services/bots/greedy_bot.py` | BFS-greedy base-seeking bot |
| `src/utils/rollout_buffer.py` | Transition storage + GAE computation |
| `src/utils/elo.py` | Elo rating tracker |
| `Dockerfile` | Container for cloud training |
| `launch-agent.yaml` | W&B Agents queue config |
