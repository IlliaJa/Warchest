# PPO Training Metrics Reference

Metrics logged every batch to W&B and to the run log file.
All update-related values are averages over the minibatch steps performed in that batch.

---

## Win-rate metrics

### `wr_vs_pool_train` / `wr_vs_greedy_train` — training win rates

Rolling win rates (last 100 training episodes) split by opponent type. These reflect in-training performance on whatever opponent mix is currently active.

**Trend:** `wr_vs_greedy_train` is the harder signal; expect it to climb slowly. `wr_vs_pool_train` can be misleading early when the pool holds near-random snapshots.

---

### `wr_vs_random_eval` / `wr_vs_greedy_eval` — evaluation win rates

Win rates measured every `eval_every` batches against fixed opponents (RandomBot and GreedyBot) over `eval_episodes` games each. Not rolling — fresh each eval block.

**Trend:** `wr_vs_random_eval` should reach > 0.9 quickly; once it does, training pool switches to finetune weights (random removed). `wr_vs_greedy_eval` is the main long-term quality signal.

---

### `score_vs_reference_eval` / `wr_vs_reference_eval` / `draw_rate_vs_reference_eval` — vs the previous generation

Result of playing the current policy against a **frozen saved checkpoint**, `eval_episodes` games every `eval_every` batches. The checkpoint is the newest `data/warchest_ppo_*.pth` by mtime unless `--reference-policy` names one, and — this is the part that makes the number mean anything — it is resolved **once at startup**, before the first batch. Resolving it per eval would let checkpoints the run itself saves become its own baseline, which pins the score near 0.5 no matter how much the policy gains. `--no-reference-eval` skips the match; so does a `data/` with no policy checkpoint in it (a first run has nothing to compare against, and the run logs that at startup).

`score_vs_reference_eval` is the one to read: `(wins + 0.5 × draws) / n`, so **0.5 is parity** with the saved generation and above it the current policy is ahead. Prefer it to `wr_vs_reference_eval` — truncations count for neither side and are common in a near-mirror match, so the bare win rate sags as games get longer even when nothing regressed. `draw_rate_vs_reference_eval` is there to tell those two cases apart.

These games are deliberately **not** fed into the Elo tracker: the reference plays nobody but the current policy, so the pair would float freely and drag `elo_policy` off the greedy/random anchor that makes it comparable across runs.

**Trend:** should sit above 0.5 and drift up. Sustained ≤ 0.5 means the run is not beating the checkpoint it started from. Note the baseline differs between runs — the W&B config records it as `reference_policy`, and two runs' numbers are only comparable when that matches.

---

## Elo metrics

### `elo_policy` / `elo_greedy` — Elo ratings

Elo ratings updated after each eval block. `elo_greedy` is a fixed reference point (it does not train, so its Elo reflects only how the policy has historically performed against it).

**Trend:** `elo_policy` should rise steadily. A widening gap `elo_policy − elo_greedy` means the policy is consistently outplaying the greedy bot.

---

## PPO update metrics

### `clip_frac` — PPO clip fraction

**What it measures:** fraction of timesteps in a minibatch where the probability
ratio `r = π_new / π_old` fell outside `[1−ε, 1+ε]` (ε = 0.2 by default),
meaning the surrogate objective was clipped.

**Ideal range:** `0.05 – 0.25`

| Value | Interpretation |
|---|---|
| < 0.05 | Updates too small; clip never activates. LR may be too low. |
| 0.05 – 0.25 | Healthy — meaningful learning without overstepping. |
| > 0.4 | Policy changes too aggressively. Lower LR or increase `ppo_eps`. |

**Trend:** roughly flat once training stabilises. Temporary spikes when training switches to finetune phase.

---

### `approx_kl` — per-minibatch KL divergence

**What it measures:** average `E[log π_old(a|s) − log π_new(a|s)]` across a
minibatch. Measures how much the policy drifted from the version that collected
the data. Epoch terminates early when this exceeds `KL_TARGET = 0.015`.

**Ideal range:** `0.003 – 0.012` per minibatch average

| Value | Interpretation |
|---|---|
| Near 0 with low `clip_frac` | Updates are doing almost nothing. |
| 0.003 – 0.012 | Healthy drift — the policy is learning without destabilising. |
| Consistently hitting `KL_TARGET` | Epochs terminate early every batch. Reduce LR. |
| > 0.02 | Large distribution shift; old trajectories are stale. |

**Trend:** roughly stable or slightly decreasing as the policy converges.

---

### `actor_loss` / `critic_loss` — component losses

`actor_loss`: mean PPO surrogate loss (negative, so improvements show as rising values).
`critic_loss`: mean MSE between predicted values and GAE returns.

**Trend:** `critic_loss` should fall over time as the critic calibrates. `actor_loss` oscillates more; its absolute magnitude matters less than whether it stays non-zero (a flat-zero actor loss means advantages are not informative).

---

### `entropy` — policy entropy

**What it measures:** mean `−Σ p log p` over the *masked* action distribution (only legal
actions). The ceiling is therefore set by how many moves are legal per turn, **not** by the
1875-wide raw action space.

**Reference scale (current env).** Measured over 8000 random self-play decision points: legal
actions per turn average **7.8** (median 6, p10 3, p90 16). The mean per-state maximum entropy
(`mean of log(n_legal)`) is therefore **≈ 1.84**, not the 2.64 quoted for the old 14-action
version. Read entropy *relative to 1.84*:

- ~1.84 → uniform-random over legal moves (no policy);
- ~1.3 → still ~70 % of max; the policy is barely committing (as seen on the
  `ppo_20260630-060400` plateau — see `docs/history.md`);
- ~0.7–0.9 (≈40–50 % of max) → a healthily decisive policy for this game.

**Trend:** should start near 1.84 and decrease as the policy becomes decisive. The entropy
coefficient is now linearly annealed (`entropy_coeff` 0.025 → 0.003 over the run,
`src/app/ppo.py`), so expect a steady downward drift; a flat entropy near 1.3 for the whole run
is a warning sign (over-weighted entropy bonus and/or a reward the policy can't sharpen
against). A collapse to near 0 *early* still means premature convergence.

**Now logged directly (no mental math):**
- `max_entropy` — the batch's own ceiling, `mean(log(n_legal))` over all main-player
  decisions. Watch for drift: if the env changes what's legal, "high entropy" means something
  different.
- `entropy_frac` = `entropy / max_entropy` — the decisive-ness ratio. 1.0 = random, target
  ~0.4–0.5 by end of run. This is the number to watch instead of raw entropy.

### `score_attack` / `score_shaping` / `score_holding` / `score_material` / `score_terminal` / `score_other` / `score_tempo`

**What it measures:** the per-episode-mean decomposition of `score` into its reward sources
(attack rewards, base-diff potential shaping `γφ′−φ`, the annealed holding reward, the annealed
material-PBRS term, terminal win/loss/truncation, the per-turn tempo cost, and everything else).
The seven sum to `score`.

**Why it matters:** catches **proxy/objective decoupling** — the failure mode on the
`ppo_20260630-060400` run (see `docs/history.md`) where `score` rose +50 % while win rate
stayed flat. If
`score_attack`/`score_shaping` climb while `score_terminal` (≈ wins) is flat, the policy is
farming dense reward instead of learning to win. `score_terminal` should be the component that
rises over training; watch `score_holding` for stalling (persistently positive + rising
`avg_turns` = sitting on a lead instead of closing).

**Two changes on 2026-08-09** (`docs/IDEAS.md` L8) that break comparison across that boundary:

- `score_attack` is **~0 by construction** now — `ATTACK_REWARD` was zeroed because material
  PBRS already pays the box-a-coin event. Read the attack axis off `score_material` instead.
  The key is kept so older runs stay plottable, not because it still measures anything.
- `score_tempo` is new: the per-turn tempo cost, peeled out of `score_attack`/`score_other`
  so it cannot masquerade as either. It is ≈ `-0.002 × (main-actor turns)`, so it doubles as a
  clean read on **episode length in turns** — and, against `n_decisions`, on how many main-actor
  clicks were free continuations rather than fresh turns (the gap is exactly the tactic and
  bonus-maneuver usage that `score_tempo` used to be charged for).

---

## Gradient metrics

### `grad_norm_actor` / `grad_norm_critic`

Post-clip gradient norms for the actor-side parameters (board encoder + unit encoder + actor head) and critic parameters separately. Both are clipped at max_norm = 1.0.

**Ideal:** comfortably below 1.0 most steps. Consistently hitting 1.0 means gradients are large and clipping is doing heavy lifting — consider lowering LR.

---

## Critic quality metrics

### `critic_mae` — mean absolute error

**What it measures:** `mean(|V(s) − R|)` — how accurately the critic predicts actual returns. Since returns are in roughly [−1, +1], this is directly interpretable in reward units.

**Trend:** falls from ~0.5–1.0 early and stabilises at a lower level. Spikes after opponent pool changes are normal.

| Value | Interpretation |
|---|---|
| 0.5 – 1.0 (early) | Critic is guessing. Expected. |
| Steadily decreasing | Critic is learning. |
| < 0.15 (late) | Good accuracy. |
| Stays high after many batches | Critic LR too low or network not expressive enough. |

---

### `critic_mean` — mean predicted value

**What it measures:** average `V(s)` across all states in the batch. Should track `return_mean` once the critic has learned.

| Condition | Interpretation |
|---|---|
| `critic_mean` ≈ `return_mean` | Critic is calibrated. |
| `critic_mean` << `return_mean` | Critic is pessimistic — advantages positively biased. |
| `critic_mean` >> `return_mean` | Critic is optimistic — advantages negatively biased. |

---

### `critic_std` — std of predicted values

Spread of the critic's predictions across states. A near-zero std means the critic predicts the same value for every state — it has not learned to differentiate positions. Should be non-trivial (> 0.1) once the critic is useful.

---

## Return and advantage metrics

### `return_mean` / `return_std` — GAE return statistics

Raw (un-normalised) GAE returns across all timesteps in the batch. Includes shaped rewards and terminal signals.

**`return_mean` trend:** should **increase over time**, trending from negative toward positive. This is the clearest top-level learning signal — a flat or falling `return_mean` means no improvement.

| Phase | Typical `return_mean` | Why |
|---|---|---|
| Early (near-random) | −0.5 to −1.0 | Dominated by losses and truncation penalties |
| Mid training | −0.2 to 0.0 | Fewer losses, more draws with base lead |
| Strong policy | > 0 | Wins more often than loses against training mix |

**`return_std` trend:** should **decrease** as play becomes more consistent. A temporary spike is expected when the finetune phase introduces harder opponents.

---

### `advantage_std` — raw advantage spread

Std of GAE advantages before normalisation. Should be non-zero (> 0.05) — near-zero means all actions look equally good to the critic, which zeros out the actor gradient regardless of normalisation.

---

## Quick-reference table

| Metric | Early training | Healthy mid | Target late | Bad signs |
|---|---|---|---|---|
| `clip_frac` | any | 0.10 – 0.20 | 0.05 – 0.25 | < 0.02 or > 0.45 |
| `approx_kl` | any | 0.005 – 0.010 | stable | hitting KL_TARGET every batch |
| `return_mean` | −0.5 to −1.0 | rising | > 0 | flat or falling |
| `return_std` | 0.5 – 1.0 | decreasing | 0.2 – 0.5 | collapses to 0 |
| `critic_mae` | 0.5 – 1.0 | decreasing | < 0.15 | plateau or spike |
| `critic_mean` | noisy | converging to `return_mean` | ≈ `return_mean` | diverging from `return_mean` |
| `critic_std` | near 0 | rising | > 0.1 | stays near 0 |
| `advantage_std` | any | > 0.05 | > 0.1 | near 0 (actor gradient dies) |
| `entropy` | ~1.8 (≈ max) | decreasing | 0.7 – 0.9 | flat near 1.3, or collapses to 0 early |
| `wr_vs_random_eval` | 0 | rising | > 0.90 | not rising by batch 50 |
| `wr_vs_greedy_eval` | 0 | rising | > 0.60 | flat after finetune phase |
| `score_vs_reference_eval` | < 0.5 | rising past 0.5 | > 0.5 | still ≤ 0.5 late in the run |
