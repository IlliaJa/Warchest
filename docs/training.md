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

Raw environment rewards are augmented with potential-based shaping (base-diff **and**
material), a per-turn holding reward, and per-run annealing of the holding + material terms
before being stored — see [Reward Design](rewards.md) for the full, current reward table and
rationale (this section previously duplicated it and had drifted out of sync).

---

## Opponent sampling

`OpponentPool` (`src/services/opponent_pool.py`) provides three opponent types with configurable weights:

| Type | Description |
|---|---|
| `random` | Uniform over valid actions |
| `greedy` | Priority: attack → control → move toward nearest base → deploy → pass (no random handicap; `RANDOM_ACTION_PROB = 0.0`) |
| `pool` | Frozen snapshot of the policy from a past batch (rolling window of `pool_max_size=20`, snapshotted every `pool_snapshot_every=15` batches so pool opponents span a wide skill range instead of near-copies) |

### Finetune phase

After each eval block, pool weights are set automatically:

- If eval WR vs greedy ≥ `wr_greedy_finetune_threshold` (0.90) → switch to finetune weights (random removed from training; greedy kept as a small fixed anchor)
- Otherwise → restore initial weights

This raises the training bar automatically as the policy matures, without any one-way flag.

| Phase | `p_random` | `p_greedy` | `p_pool` |
|---|---|---|---|
| Initial | 0.40 | 0.20 | 0.40 |
| Finetune | 0.00 | 0.10 | 0.90 |

---

## Hyperparameters

| Parameter | Value | Effect |
|---|---|---|
| `n_batches` | 1500 | Total batch updates |
| `collect_episodes` | 64 | Episodes collected per batch before update |
| `max_t` | 1000 | Hard cap on steps per episode |
| `ppo_epochs` | 4 | Inner gradient epochs per batch (KL early stop active) |
| `ppo_eps` | 0.2 | PPO clip parameter |
| `KL_TARGET` | 0.015 | Approx-KL threshold for early stopping an epoch |
| `gamma` | 0.99 | Discount factor |
| `--lam` | 0.90 | GAE trace decay. Lowered from 0.97 together with the critic trunk fix — at 0.97 only ~3 % of the discriminative signal came from `V(s_{t+1})`, so a better critic bought PPO nothing (`docs/IDEAS.md` L2) |
| `lr_actor` / `lr_critic` | 3e-4 | Adam LR, both actor and critic; linearly decayed to `lr_final_frac * init` (`lr_final_frac=0.1`) over the run |
| `hidden_dim` (Policy) | 128 | Policy network width |
| `critic_hidden_dim` (Critic) | 192 | Critic widened alone first — the densifier of the sparse terminal reward (`docs/decision.md`, 2026-07-03) |
| `--policy-arch` | `policy_factored_v2` | Unit-type embedding + FiLM trunk conditioning (`docs/IDEAS.md` A1 + A3). `policy_factored_v1` reproduces the pre-2026-08-16 net as an A/B baseline |
| `--critic-arch` | `critic_v5` | `critic_v4` + the same unit-type embedding; no FiLM, deliberately (it would leak globals into the board-only auxiliary head). Older archs are selectable to reproduce a baseline; `critic_v1`'s trunk provably dies |
| `--aux-board-coeff` | 0.1 | Weight of the `critic_v2`+ board-only auxiliary loss — the gradient pressure that keeps the trunk carrying board signal |
| `--adv-norm` | `per_opponent` | Centre advantages within each opponent group (mean-only, one shared std). Pairs with `critic_v3`+ dropping the opponent one-hot; `global` reproduces pre-2026-08-09 runs |
| `entropy_coeff` | 0.025 → `entropy_coeff_final` 0.003 | Entropy bonus coefficient, linearly annealed over the run |
| `SHAPING_C` | 0.05 | Base-diff potential-shaping scale factor (constant, not annealed) |
| `C_MAT` | 0.015 | Material (boxed-coin) potential-shaping scale factor |
| `shaping_anneal` | 1.0 → 0.1 over first half of run | Multiplier applied to the holding reward and material shaping (`docs/rewards.md`) |
| Pool `max_size` / `snapshot_every` | 20 / 15 | Rolling snapshot window length / batches between snapshots |
| `eval_every` | 10 | Evaluate every N batches |
| `eval_episodes` | 20 | Episodes per evaluation block |
| `wr_greedy_finetune_threshold` | 0.90 | WR vs greedy that triggers finetune phase |

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
| `max_entropy` | Batch's own entropy ceiling, `mean(log(n_legal))` — see `docs/METRICS.md` |
| `entropy_frac` | `entropy / max_entropy` — the decisive-ness ratio (1.0 = random) |
| `entropy_coeff` | Current (annealed) entropy bonus coefficient |
| `lr` | Current (decayed) actor learning rate |
| `grad_norm_actor` | Post-clip gradient norm, actor-side parameters |
| `grad_norm_critic` | Post-clip gradient norm, critic parameters |
| `clip_frac` | Fraction of timesteps where PPO ratio was clipped |
| `critic_mae` | Mean absolute error of critic predictions vs actual returns |
| `advantage_std` | Std of raw (pre-normalised) advantages |
| `avg_turns` | Mean episode length in the batch |
| `score_attack` / `score_shaping` / `score_holding` / `score_material` / `score_terminal` / `score_other` | Per-episode-mean decomposition of `score_main` into its reward sources — see `docs/METRICS.md` |
| `shaping_anneal` | Current anneal multiplier applied to the holding + material shaping terms |

`critic_mean`/`critic_std` (predicted-value mean/std) and `return_mean`/`return_std` (raw GAE
return mean/std) are computed each batch and printed to the text log, but are **not** sent to
W&B.

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
