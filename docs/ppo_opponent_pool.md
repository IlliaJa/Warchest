# PPO Training with Opponent Pool

## Entry point

`ppo.py` is the new training entry point. Run it the same way as `reinforce.py`:

```bash
python ppo.py
```

It replaces the per-episode REINFORCE update with two mechanisms:

1. **PPO (Proximal Policy Optimization)** — collects a batch of N episodes before updating, then runs K gradient steps per batch using a clipped surrogate loss.
2. **Opponent pool** — the main actor trains against either a random bot or a frozen past snapshot of the policy, preventing self-play cycling and strategy forgetting.

---

## Main actor concept

`reinforce.py` tracked metrics per player id (player 1 or player 2), which made scores and win rates ambiguous when the policy randomly ended up in either slot.

`ppo.py` tracks everything relative to the **main actor** — the policy being trained. At episode start, `main_pid = random.choice([1, 2])` picks which player slot the main actor occupies. The other slot (`opp_pid = 3 - main_pid`) is the opponent. All scores, win rates, and loss metrics belong to the main actor regardless of which player position it holds.

---

## Opponent pool

`OpponentPool` (in `opponent_pool.py`) maintains a rolling window of frozen policy snapshots.

```
pool = OpponentPool(max_size=20, snapshot_every=1)
```

### How it works

- After each batch update, `pool.maybe_snapshot(policy)` copies the current policy weights into the deque.
- When the deque is full (20 entries), adding a new snapshot evicts the oldest one automatically.
- At episode start, `pool.sample(policy_constructor, device, p_random=0.4)` returns either:
  - `(None, 'random')` — 40% of the time, or when the pool is empty
  - `(frozen_policy, 'pool')` — 60% of the time, drawn uniformly from the 20 stored snapshots

### Why a rolling window

The pool always holds the **last 20 policy states**. This means:
- Early, near-random snapshots are evicted as training progresses — opponents stay competitive.
- The pool naturally spans roughly `20 × collect_episodes = 160` episodes of policy history, which is enough diversity to prevent cycling without being so stale that the opponents are trivially weak.
- No tuning of a separate "snapshot interval" parameter is needed.

### Two pools, two purposes

The opponent pool is **completely separate** from PPO's importance sampling:

| Concept | What it is | Relationship to current policy |
|---|---|---|
| Opponent pool | Who the main actor plays against | Past snapshots, can be old |
| PPO's `π_old` | Frozen reference for computing importance ratio `r_t` | Always the current policy at batch start, very close |

`π_old` is implicit in `ppo.py` — the `log_probs_old` stored during data collection serve this role. The opponent pool is a separate object entirely.

---

## PPO update

### Batch collection

Each batch collects `collect_episodes` (default 8) full game episodes before any gradient update. Main-actor transitions are stored in a `RolloutBuffer`:

```
(obs_before, action, log_prob_old, shaped_reward, value)
```

Opponent transitions are discarded — only the main actor's trajectory is trained on.

### GAE computation

After all episodes are collected, `buffer.compute_gae()` computes advantages and returns for the full batch using the same GAE formula as `reinforce.py`:

```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t     = delta_t + gamma * lambda * A_{t+1}
```

Episode boundaries are respected — GAE does not bleed across episode ends.

### Inner loop

The buffer is then iterated `ppo_epochs` (default 4) times in random order. Each step:

```
r_t      = exp(log_pi_new(a|s) - log_pi_old(a|s))   # importance ratio
L_actor  = -min(r_t * A_t,  clip(r_t, 1-eps, 1+eps) * A_t)
L_critic = MSE(V(s), return)
L_total  = L_actor + L_critic - entropy_coeff * entropy
```

Gradients accumulate across all steps before a single `optimizer.step()` per epoch (gradient accumulation pattern — avoids holding the full computation graph in memory).

### KL divergence monitoring

`ppo_kl` (approximate KL: `mean(log_pi_old - log_pi_new)`) is logged to W&B. Values above ~0.05 suggest the clip is not constraining enough; values near 0 after several batches suggest the policy has stopped changing.

---

## W&B metrics

| Metric | Description |
|---|---|
| `score_main` | Rolling average of main-actor episode reward |
| `winrate_vs_random` | Win rate when opponent was random (rolling 100) |
| `winrate_vs_pool` | Win rate when opponent was from pool (rolling 100) |
| `win_rate` | Overall win rate across all opponent types (rolling 100) |
| `lose_rate` | Overall lose rate (rolling 100); truncate = 1 - win - lose |
| `actor_loss` | Mean PPO actor loss per inner step, averaged over epochs |
| `critic_loss` | Mean critic MSE loss per inner step, averaged over epochs |
| `ppo_kl` | Approximate KL divergence from old to new policy |
| `grad_norm` | Pre-clip gradient norm (last epoch of each batch) |
| `pool_size` | Number of snapshots currently in the opponent pool |
| `avg_turns` | Mean episode length across batch episodes |

---

## Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `n_batches` | 400 | Total batch updates (= 3200 total episodes with collect_episodes=8) |
| `collect_episodes` | 8 | Episodes collected per batch before update |
| `ppo_epochs` | 4 | Inner gradient steps per batch |
| `ppo_eps` | 0.2 | PPO clip parameter |
| `gamma` | 0.99 | Discount factor |
| `lam` | 0.95 | GAE trace decay |
| `lr_actor` | 1e-4 | Adam LR for encoder + actor head |
| `lr_critic` | 5e-4 | Adam LR for critic head |
| `hidden_dim` | 64 | Network width |
| Pool `max_size` | 20 | Rolling window length |
| Pool `p_random` | 0.4 | Probability of choosing random over pool opponent |
