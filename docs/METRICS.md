# PPO Training Metrics Reference

Metrics logged every batch to W&B and to the run log file.
All values are averages over the minibatch updates performed in that batch.

---

## `clip_frac` — PPO clip fraction

**What it measures:** fraction of timesteps in a minibatch where the probability
ratio `r = π_new / π_old` fell outside `[1−ε, 1+ε]` (ε = 0.2 by default),
meaning the surrogate objective was clipped by the PPO clamp.

**Ideal range:** `0.05 – 0.25`

| Value | Interpretation |
|---|---|
| < 0.05 | Policy updates are too small; clip never activates. Equivalent to doing one tiny gradient step. LR may be too low or `ppo_eps` too large. |
| 0.05 – 0.25 | Healthy — meaningful learning without overstepping. |
| > 0.4 | Policy changes too aggressively per update. Expect instability. Lower LR or raise `ppo_eps`. |

**Trend:** roughly flat once training stabilises. Temporary spikes when a harder
opponent is introduced.

---

## `ppo_kl` — per-minibatch KL divergence

**What it measures:** average `E[log π_old(a|s) − log π_new(a|s)]` across a
minibatch. Measures how much the policy drifted from the version that collected
the data. A separate approximation (`(r−1) − log r`) is used for early stopping
inside the epoch loop; the logged value is the direct log-ratio mean.

**Ideal range:** `0.003 – 0.012` per minibatch average

| Value | Interpretation |
|---|---|
| Near 0 with low clip_frac | Updates are doing almost nothing. |
| 0.003 – 0.012 | Healthy drift — the policy is learning without destabilising. |
| Consistently hitting `kl_target` (0.015) | Epochs terminate early; policy wants to move faster than the KL budget allows. Reduce `ppo_epochs` or LR. |
| > 0.02 | Large distribution shift; old trajectories are stale. Likely to cause training instability. |

**Trend:** roughly stable or slightly decreasing as the policy converges.

---

## `ret_mean` — mean of raw returns

**What it measures:** average discounted cumulative reward across all timesteps
in the batch. Returns include shaped rewards (base-difference term) plus terminal
signals (win = +1, loss = −1, truncation = −0.5 or 0 depending on base lead).

**Ideal behaviour:** should **increase over time**, trending from negative toward
positive.

| Phase | Typical value | Why |
|---|---|---|
| Early (near-random) | −0.5 to −1.0 | Policy loses often and truncates; dominated by −1 / −0.5 terminals. |
| Mid training | −0.2 to 0.0 | Fewer losses, more draws/truncations with a base lead. |
| Strong policy | > 0 | Policy wins more often than it loses against its training mix. |

**Trend: monotonically increasing** (with noise). This is the clearest top-level
learning signal — a flat or falling `ret_mean` means no improvement.

---

## `ret_std` — standard deviation of raw returns

**What it measures:** spread of returns within a batch. High std = outcomes vary
widely (some wins, some losses, some truncations). Low std = outcomes are
consistent.

**Ideal behaviour:** should **decrease over time** as the policy becomes more
consistent, but with caveats.

| Phase | Expected std | Why |
|---|---|---|
| Early (near-random) | 0.5 – 1.0 | Outcomes are a coin flip. |
| Mid training | decreasing gradually | Policy wins more consistently. |
| After adding a harder opponent | temporary spike | Greedy episodes introduce reliable losses initially. |
| Converged | 0.2 – 0.5 | Still some variance; game is non-deterministic. |

**Warning:** if `ret_std` collapses toward 0 but `ret_mean` is still negative,
the policy has converged to consistently losing — stable but broken. Also watch
for near-zero `ret_std` causing NaN advantages during return normalisation.

---

## `critic_mae` — critic mean absolute error

**What it measures:** `mean(|V(s) − R|)` — how accurately the critic predicts
actual returns. Since returns are in roughly [−1, +1], this is directly
interpretable in reward units.

**Ideal behaviour:** should **decrease** from a high starting value and
**stabilise** at a low level.

| Value | Interpretation |
|---|---|
| 0.5 – 1.0 (early) | Critic is guessing. Expected. |
| Steadily decreasing | Critic is learning; advantages become a better signal for the actor. |
| < 0.15 (late) | Critic has good accuracy. |
| Stays high after many batches | Critic LR too low, or shared encoder not expressive enough. |
| Spikes after being low | Distribution shift from a new opponent or a sudden policy jump. |

**Trend:** decreases and stabilises. Spikes are a warning sign.

---

## `critic_mean` — mean predicted value

**What it measures:** average `V(s)` the critic predicts across all states in
the batch. Represents the critic's estimate of expected return from a typical
game state seen during training.

**Ideal behaviour:** should **track `ret_mean` closely** once the critic has
learned. Early in training they will diverge (critic is random); as `critic_mae`
falls, `critic_mean` should converge toward `ret_mean`.

| Condition | Interpretation |
|---|---|
| `critic_mean` ≈ `ret_mean` | Critic is calibrated. Advantages are centered correctly. |
| `critic_mean` << `ret_mean` | Critic is pessimistic — underestimates policy quality. Advantages will be positively biased, pushing the actor toward all actions equally. |
| `critic_mean` >> `ret_mean` | Critic is optimistic — advantages will be negatively biased, suppressing learning. |
| `critic_mean` drifts upward while `ret_mean` is flat | Value estimates are diverging; check for gradient issues or LR imbalance between actor and critic. |

**Trend:** starts anywhere (random init), converges toward `ret_mean`, then
rises together with `ret_mean` as the policy improves.

---

## Quick-reference table

| Metric | Early training | Healthy mid | Target late | Bad signs |
|---|---|---|---|---|
| `clip_frac` | any | 0.10 – 0.20 | 0.05 – 0.25 | < 0.02 or > 0.45 |
| `ppo_kl` | any | 0.005 – 0.010 | stable | hitting kl_target every batch |
| `ret_mean` | −0.5 to −1.0 | rising | > 0 | flat or falling |
| `ret_std` | 0.5 – 1.0 | decreasing | 0.2 – 0.5 | collapses to 0 |
| `critic_mae` | 0.5 – 1.0 | decreasing | < 0.15 | plateau or spike |
| `critic_mean` | noisy | converging to ret_mean | ≈ ret_mean | diverging from ret_mean |
