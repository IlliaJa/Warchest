# Training Guide

## Quick start

```bash
pip install -r requirements.txt
python -m src.app.ppo
```

A W&B run is created automatically. The trained model is saved to:
```
data/warchest_ppo_YYYYMMDD-HH:MM.pth
```

The legacy REINFORCE trainer is kept at `src/app/reinforce.py` for reference but is no longer the primary training path.

---

## Algorithm: PPO with GAE

The trainer (`PPOTrainer` in `src/app/ppo.py`) uses Proximal Policy Optimization with Generalized Advantage Estimation.

### Advantage computation

After each batch of episodes the trajectories are processed backwards within each episode boundary:

```
delta_t  = r_t + gamma * V(s_{t+1}) - V(s_t)     (TD residual)
A_t      = delta_t + gamma * lambda * A_{t+1}      (GAE)
return_t = A_t + V(s_t)                            (target for critic)
```

Advantages are z-scored batch-wide before the PPO update. Returns are kept in the original reward scale so the critic can be calibrated correctly.

### PPO loss

```
ratio    = pi_new(a|s) / pi_old(a|s)
L_actor  = -mean( min(ratio * A,  clip(ratio, 1-eps, 1+eps) * A) )
L_critic = MSE(V(s), return)
L_total  = L_actor + L_critic - entropy_coeff * H
```

An epoch is stopped early when the per-minibatch approximate KL exceeds `KL_TARGET = 0.015`.

### Reward shaping

Raw environment rewards are augmented with potential-based shaping before being stored:

```python
shaped_r = r + gamma * phi(s_next) - phi(s)
phi(s)   = SHAPING_C * (my_bases - opp_bases)   # SHAPING_C = 0.05
```

This fires a positive pulse on gaining a base and a negative pulse on losing one without distorting the optimal policy.

---

## Opponent sampling

`OpponentPool` (`src/services/opponent_pool.py`) provides three opponent types with configurable weights:

| Type | Description |
|---|---|
| `random` | Uniform over valid actions |
| `greedy` | BFS toward nearest unclaimed or enemy base (30% random handicap) |
| `pool` | Frozen snapshot of the policy from a past batch (rolling window of 20) |

### Finetune phase

After each eval block, pool weights are set automatically:

- If eval WR vs random ≥ `wr_random_finetune_threshold` (0.90) → switch to finetune weights (random removed from training)
- Otherwise → restore initial weights

This raises the training bar automatically as the policy matures, without any one-way flag.

| Phase | `p_random` | `p_greedy` | `p_pool` |
|---|---|---|---|
| Initial | 0.40 | 0.20 | 0.40 |
| Finetune | 0.00 | 0.40 | 0.60 |

---

## Hyperparameters

| Parameter | Value | Effect |
|---|---|---|
| `n_batches` | 300 | Total batch updates |
| `collect_episodes` | 16 | Episodes collected per batch before update |
| `max_t` | 1000 | Hard cap on steps per episode |
| `ppo_epochs` | 1 | Inner gradient epochs per batch (KL early stop active) |
| `ppo_eps` | 0.2 | PPO clip parameter |
| `KL_TARGET` | 0.015 | Approx-KL threshold for early stopping an epoch |
| `gamma` | 0.99 | Discount factor |
| `lam` | 0.95 | GAE trace decay |
| `lr_actor` | 3e-4 | Adam LR for encoder + actor head |
| `lr_critic` | 3e-4 | Adam LR for critic |
| `hidden_dim` | 64 | Network width |
| `ENTROPY_COEFF` | 0.001 | Entropy bonus coefficient |
| `SHAPING_C` | 0.05 | Potential-shaping scale factor |
| Pool `max_size` | 20 | Rolling snapshot window length |
| `eval_every` | 10 | Evaluate every N batches |
| `eval_episodes` | 20 | Episodes per evaluation block |
| `wr_random_finetune_threshold` | 0.90 | WR vs random that triggers finetune phase |

---

## W&B metrics logged

### Per batch (logged by `_log_batch`)

| Metric | Description |
|---|---|
| `score_main` | Rolling mean episode reward (main actor) |
| `wr_vs_pool_train` | Win rate vs pool opponents (rolling 100 training episodes) |
| `wr_vs_greedy_train` | Win rate vs greedy opponent (rolling 100 training episodes) |
| `actor_loss` | Mean PPO actor loss averaged over minibatch updates |
| `critic_loss` | Mean critic MSE loss averaged over minibatch updates |
| `approx_kl` | Approximate KL divergence from old to new policy |
| `entropy` | Mean policy entropy |
| `grad_norm_actor` | Post-clip gradient norm, actor-side parameters |
| `grad_norm_critic` | Post-clip gradient norm, critic parameters |
| `clip_frac` | Fraction of timesteps where PPO ratio was clipped |
| `critic_mae` | Mean absolute error of critic predictions vs actual returns |
| `critic_mean` | Mean predicted state value across the batch |
| `critic_std` | Std of predicted state values across the batch |
| `advantage_std` | Std of raw (pre-normalised) advantages |
| `return_mean` | Mean of raw GAE returns |
| `return_std` | Std of raw GAE returns |
| `avg_turns` | Mean episode length in the batch |

### Per eval block (logged by `_maybe_eval`, every `eval_every` batches)

| Metric | Description |
|---|---|
| `wr_vs_random_eval` | Win rate vs RandomBot over `eval_episodes` games |
| `wr_vs_greedy_eval` | Win rate vs GreedyBot over `eval_episodes` games |
| `elo_policy` | Elo rating of the current policy |
| `elo_greedy` | Elo rating of GreedyBot |

---

## Evaluation

The eval block (every `eval_every` batches) always plays both opponents:

```python
for _ in range(eval_episodes):
    play vs GreedyBot    → record win/lose/draw → update Elo
    play vs RandomBot    → record win/lose/draw → update Elo
```

After eval, pool weights are updated unconditionally based on `wr_vs_random_eval`.

---

## Cloud training (Docker + W&B Agents)

Build and push the image, then enqueue a run:
```bash
docker build -t warchest .
wandb launch --queue warchest
```

The `launch-agent.yaml` configures the W&B entity (`illiaja-private`), project (`warchest`), and queue name.
