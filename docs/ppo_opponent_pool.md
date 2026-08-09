# PPO Training with Opponent Pool

## Entry point

```bash
python -m src.app.ppo
```

`PPOTrainer` (in `src/app/ppo.py`) replaces the per-episode REINFORCE update with two mechanisms:

1. **PPO** — collects a batch of N episodes before updating, then runs K inner epochs with a clipped surrogate loss.
2. **Opponent pool** — the main actor trains against a weighted mixture of random, greedy, and frozen past policy snapshots.

---

## Main actor concept

Everything is tracked relative to the **main actor** — the policy being trained. At episode start, `main_pid = random.choice([1, 2])` picks which player slot the main actor occupies. All scores, win rates, and transitions belong to the main actor regardless of player position.

---

## Opponent pool

`OpponentPool` (`src/services/opponent_pool.py`) samples from three opponent types according to internal weights.

```python
pool = OpponentPool(
    max_size=20,
    snapshot_every=1,
    p_random=0.40,
    p_greedy=0.20,
    p_pool=0.40,
)
```

### Three opponent types

| Type | Description |
|---|---|
| `random` | `RandomBot` — uniform over valid actions |
| `greedy` | `GreedyBot` — BFS toward nearest unclaimed/enemy base; 30% random handicap |
| `pool` | Frozen `Policy` snapshot drawn uniformly from the rolling window |

When the pool is empty (start of training), only random and greedy are used and their weights are renormalised.

### Snapshots

After each batch update, `pool.maybe_snapshot(policy)` copies the current policy weights into a `deque(maxlen=20)`. The oldest snapshot is evicted automatically when full. The rolling window always spans the most recent `max_size × collect_episodes` episodes of training.

### Finetune phase

After every eval block, `PPOTrainer` unconditionally calls `pool.set_weights()` based on the current eval WR vs greedy:

```python
if wr_greedy_eval >= wr_greedy_finetune_threshold:  # default 0.90
    pool.set_weights(p_random=0.00, p_greedy=0.40, p_pool=0.60)
else:
    pool.set_weights(p_random=0.40, p_greedy=0.20, p_pool=0.40)
```

This is recalculated every eval — if WR later drops below the threshold the initial weights are restored automatically.

### Pool vs PPO's π_old

| Concept | What it is | Staleness |
|---|---|---|
| Opponent pool | Who the main actor plays against | Can be 20 batches old |
| PPO's `log_probs_old` | Frozen reference for importance ratio `r_t` | Always from the batch just collected |

These are completely independent.

---

## PPO update

### Batch collection

Each batch collects `collect_episodes` (default 16) full game episodes. Main-actor transitions stored per step:

```
(obs_before, action, log_prob_old, shaped_reward, value)
```

Opponent transitions are discarded.

### Reward shaping

Shaped reward applied before storage:

```python
shaped_r = r + gamma * phi(s_next) - phi(s)
phi(s)   = SHAPING_C * (my_bases - opp_bases)    # SHAPING_C = 0.05
```

### GAE computation

After all episodes are collected, `buffer.compute_gae()` computes advantages and returns respecting episode boundaries:

```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t     = delta_t + gamma * lambda * A_{t+1}
```

Advantages are z-scored batch-wide. Returns stay in the original reward scale.

### Inner loop

```
for epoch in range(ppo_epochs):                   # default 1
    for minibatch in buffer (size 64):
        ratio    = exp(log_pi_new - log_pi_old)
        L_actor  = -min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)
        L_critic = MSE(V(s), return)
        L_total  = L_actor + L_critic - entropy_coeff * H
        clip gradients separately: actor-side and critic (max_norm=1.0)
        optimizer.step()
        if approx_kl > KL_TARGET: break epoch early
```

`approx_kl = mean((ratio − 1) − (log_pi_new − log_pi_old))` is the early-stop signal, not the logged metric. The logged `approx_kl` is `mean(log_pi_old − log_pi_new)`.

---

## W&B metrics

### Per batch

| Metric | Description |
|---|---|
| `score_main` | Rolling mean episode reward (main actor) |
| `wr_vs_pool_train` | Win rate vs pool opponents (rolling 100 training episodes) |
| `wr_vs_greedy_train` | Win rate vs greedy opponent (rolling 100 training episodes) |
| `actor_loss` | Mean PPO actor loss per update |
| `critic_loss` | Mean critic MSE loss per update |
| `approx_kl` | Mean `log_pi_old − log_pi_new` per update |
| `entropy` | Mean policy entropy |
| `grad_norm_actor` | Post-clip norm, actor-side params |
| `grad_norm_critic` | Post-clip norm, critic params |
| `clip_frac` | Fraction of timesteps where PPO ratio was clipped |
| `critic_mae` | Mean absolute error of critic predictions |
| `critic_mean` | Mean predicted state value |
| `critic_std` | Std of predicted state values |
| `advantage_std` | Std of raw advantages |
| `return_mean` | Mean of raw GAE returns |
| `return_std` | Std of raw GAE returns |
| `avg_turns` | Mean episode length |

### Per eval block

| Metric | Description |
|---|---|
| `wr_vs_random_eval` | Eval win rate vs RandomBot |
| `wr_vs_greedy_eval` | Eval win rate vs GreedyBot |
| `elo_policy` | Policy Elo rating |
| `elo_greedy` | GreedyBot Elo rating |

---

## Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `n_batches` | 300 | Total batch updates |
| `collect_episodes` | 16 | Episodes collected per batch |
| `ppo_epochs` | 1 | Inner epochs per batch (KL early stop active) |
| `ppo_eps` | 0.2 | PPO clip parameter |
| `KL_TARGET` | 0.015 | Approx-KL epoch early-stop threshold |
| `ENTROPY_COEFF` | 0.001 | Entropy bonus coefficient |
| `gamma` | 0.99 | Discount factor |
| `lam` | 0.95 | GAE trace decay |
| `lr_actor` | 3e-4 | Adam LR for encoder + actor head |
| `lr_critic` | 3e-4 | Adam LR for critic |
| `hidden_dim` | 64 | Network width |
| Pool `max_size` | 20 | Rolling snapshot window |
| `snapshot_every` | 1 | Snapshot after every N batch updates |
| `eval_every` | 10 | Eval every N batches |
| `eval_episodes` | 20 | Games per eval block |
| `wr_greedy_finetune_threshold` | 0.90 | WR vs greedy to trigger finetune weights |
