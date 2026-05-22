# Training Guide

## Quick start

```bash
pip install -r requirements.txt
python reinforce.py
```

A W&B run is created automatically. The trained model is saved to:
```
data/warchest_policy_YYYYMMDD-HH:MM.pth
```

## Algorithm: REINFORCE + GAE

The training loop in `reinforce.py` uses the REINFORCE policy gradient algorithm augmented with Generalized Advantage Estimation (GAE).

### Advantage computation

After each episode the trajectory is processed backwards:

```
delta_t   = r_t + gamma * V(s_{t+1}) - V(s_t)      (TD residual)
A_t       = delta_t + gamma * lambda * A_{t+1}       (GAE)
return_t  = A_t + V(s_t)                             (target for critic)
```

### Loss

```
actor_loss  = -mean(log_pi(a|s) * A)
critic_loss = MSE(V(s), return)
entropy     = -sum(pi * log(pi))
total_loss  = actor_loss + critic_loss - entropy_coeff * entropy
```

## Opponent sampling

Each episode, each player is independently assigned as:
- **Policy-controlled** (gradient tracked)
- **Random** (no gradient, uniform over valid actions)

This gives four combinations:
- Both random → no update
- One policy, one random → single-player loss update
- Both policy → averaged loss from both players

The 30 % random probability is hard-coded and encourages robustness.

## Hyperparameters

| Parameter | Value | Effect |
|---|---|---|
| `n_training_episodes` | 3000 | Total training games |
| `max_t` | 1000 | Hard cap on steps per episode |
| `max_actions` (env) | 500 | Env truncation limit |
| `gamma` | 0.9 | Discount factor |
| `lambda` | 0.95 | GAE trace decay |
| `lr` | 5e-3 | Adam learning rate |
| `hidden_dim` | 64 | Network width |
| Entropy coeff | 0.1 → 0.01 | Scheduled at 75 % of training |

## W&B metrics logged

| Metric | Description |
|---|---|
| `loss_bot1` | Total loss for policy player |
| `score_bot1/2` | Cumulative episode reward per player |
| `winrate_bot1` | Win rate in self-play episodes |
| `winrate_against_random` | Win rate vs random opponent |
| `entropy_bonus` | Mean policy entropy |
| `avg_log_prob_bot1` | Mean log-probability of chosen actions |
| `grad_norm` | Total gradient norm before clipping |
| `episode_time` | Wall-clock seconds per episode |
| `last_turn` | Number of turns until termination |

## Evaluation

```bash
python test.py
```

Loads the latest checkpoint from `data/`, evaluates against 10 random opponents, prints win/draw/loss counts, then replays one AI-vs-AI game interactively.

## Cloud training (Docker + W&B Agents)

Build and push the image, then enqueue a run:
```bash
docker build -t warchest .
wandb launch --queue warchest
```

The `launch-agent.yaml` configures the W&B entity (`illiaja-private`), project (`warchest`), and queue name.
