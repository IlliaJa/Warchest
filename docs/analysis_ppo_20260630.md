# Run analysis — `ppo_20260630-060400`

Analysis of the long PPO run started 2026-06-30 06:04, stopped at **batch 710 / 1000**
(~12.5 h wall-clock). Log: `logs/ppo_20260630-060400.log`.
W&B: https://wandb.ai/illiaja-private/warchest/runs/13lpgbwu

## TL;DR

The policy learned the rules and beats greedy (~70% WR) and random (~100%), but **stopped
improving around batch 130** and then oscillated for the remaining ~580 batches (~10 h) with
no gain in real strength. Two hard signals confirm the "plays like a newbie" impression:

1. **Entropy never fell.** Final policy entropy ≈ **1.3**, which is **~70 % of the maximum
   possible** for this game (mean legal-action count is only 7.8; see below). The policy is
   still spreading its probability over ~half the legal moves instead of committing to a plan.
2. **The training reward decoupled from winning.** Between batch 40 and 600 the shaped
   `score` climbed +50 % (0.93 → 1.45) while `wr_greedy` stayed flat at ~0.7 **and games got
   longer** (127 → 150 turns). The agent is increasingly optimizing the dense shaping signal,
   not wins.

This is not a "train longer" problem — the last 10 hours bought nothing. It is a
reward-design + exploration-pressure problem.

---

## What the run actually did

Hyperparameters (from log line 2):
`collect_episodes=64, ppo_epochs=4, ppo_eps=0.2, gamma=0.99, lam=0.95,
entropy_coeff=0.025, lr_actor=lr_critic=3e-4, hidden_dim=64, minibatch=64`,
opponent schedule `random/greedy/pool = 0.4/0.2/0.4` initial → `0/0.4/0.6` finetune
(finetune triggered almost immediately since `wr_random` hit 1.0 by batch ~4).

Trajectory (eval every 10 batches, 20 games each → resolution 0.05, ±0.10 ≈ 2 games of noise):

| phase | batches | `wr_greedy` | `elo_policy` | `entropy` | `score` |
|---|---|---|---|---|---|
| ramp-up | 1–40 | 0.20 → 0.55 | 1051 → 1200 | 1.75 → 1.40 | −0.45 → 0.93 |
| plateau | 130 → 710 | 0.6–0.9 (mean ~0.72, **no trend**) | 1290–1426 (**no trend**) | ~1.30 (**barely moves**) | 1.0 → 1.5 (**still rising**) |

- `wr_random` ≈ 1.0 throughout (random is trivially beaten — near-useless as a signal after
  batch 5). Note the **greedy eval is now the true greedy bot**: `GreedyBot.RANDOM_ACTION_PROB
  = 0.0` (`src/services/bots/greedy_bot.py:37`), so idea C9 is resolved and 0.72 WR is honest.
- `elo_policy` peaks at 1426 (batches 230, 360) and never beats that — flat for the second
  half of the run.

---

## Diagnosis

### 1. Entropy is high because the action space is small — the METRICS doc is stale

`docs/METRICS.md:86` still says "for 14 actions, maximum entropy ≈ 2.64" and lists an ideal
band of 0.5–1.5. That was written for an older 14-action version. **Measured on the current
env** (40 random self-play games, 8000 decision points):

```
legal actions per turn: mean 7.8, median 6, p10 3, p90 16, max 35
average max entropy (mean of log(n_legal)) = 1.84
27% of states have ≤3 legal actions
```

So the correct yardstick today is **~1.84, not 2.64**. Against that:

- start of run: entropy 1.75 ≈ **95 % of max** → essentially uniform-random over legal moves;
- end of run: entropy 1.30 ≈ **71 % of max** → effective branching `exp(1.30) ≈ 3.7` moves
  out of ~6–8 legal.

A decisive policy in a game this tactical should be well below that. The absolute number 1.3
looks "fine" only because the doc's reference point is wrong. **The entropy barely moved
because the two forces below hold it up.**

### 2. The entropy bonus is over-weighted for this advantage scale

`entropy_coeff = 0.025` is constant for the whole run (no decay). Meanwhile advantages are
small and noisy — `adv std ≈ 0.20–0.42` across the run. When the entropy-maximizing gradient
(scale ≈ `entropy_coeff`) is a large fraction of the advantage-weighted policy gradient
(scale ≈ `|adv|`), the optimum sits at a high-entropy floor: the policy stops sharpening
because doing so costs more entropy bonus than it gains in clipped advantage. 0.025 is on the
high side for PPO (typical 0.0–0.01) and, crucially, **never anneals**, so the pressure is
identical at batch 700 as at batch 1. This is the leading suspect for the flat entropy.

### 3. The dense shaping reward is being optimized instead of winning

Reward table (`src/services/environment/warchest_env.py:201-206`, shaping in
`src/app/ppo.py:234-256`):

```
WIN=+1.0  LOSS=−1.0  CLAIM_BASE=0.0  ATTACK=+0.1  MOVE=−0.002/turn  INVALID=−0.02
potential shaping: phi = 0.05 * base_diff * winning_base_count   (policy-invariant)
holding_reward   = 0.00107 * base_diff  per main turn             (NOT potential-based)
```

Problems:

- **`ATTACK_REWARD = 0.1` is huge relative to the win signal.** Ten attacks in a game equal a
  full win (+1.0). A policy can accumulate large positive return by trading blows without ever
  converting a material edge into a base claim. This is consistent with the observed
  score↑/WR-flat/turns↑ divergence.
- **`holding_reward` breaks potential-shaping invariance.** Potential-based shaping
  (`γφ' − φ`) is provably policy-invariant, but an *extra* per-turn reward proportional to the
  current base lead is a genuine bias toward *grabbing an early lead and stalling* rather than
  closing the game — again matching the "games get longer" observation.
- **`CLAIM_BASE_REWARD = 0.0`.** The actual win condition (controlling bases) gets no direct
  dense credit — it only shows up through the shaping potential, which the attack reward can
  drown out.
- Nothing in the reward references **unit types / coin economy**, which (per your point) is
  where most of Warchest's skill lives. The agent has no gradient telling it that committing to
  2 units and playing them well beats spreading thin — hence "buys all the supply."

Net: the proxy the agent maximizes (shaped return) and the objective you care about (wins) have
measurably decoupled after ~batch 300.

### 4. Self-play treadmill masks the plateau

The pool snapshots the policy **every 3 batches** (`src/app/ppo.py:123`) and after finetune
60 % of training games are vs pool snapshots. `wr_pool` sits at ~0.45–0.60 by construction
(you're playing near-copies of yourself), so ~60 % of every batch carries almost no learning
signal (advantages average to ~0). The 40 % greedy games are the only fixed yardstick in the
gradient — and against that fixed yardstick the policy is flat at 0.7. Self-play is fine, but
right now it dilutes an already-weak signal rather than driving improvement.

### 5. Secondary contributors

- **`hidden_dim = 64` is small** for a `[6,7,7]` board + 1875-way factored action head with
  full unit rules (`docs/policy_network.md`). A tactical game may simply be capacity-limited at
  this width; worth a controlled bump once the reward issues are fixed.
- **No LR decay** (idea C8, still open — `src/app/ppo.py:661-662` use plain Adam, constant
  3e-4). Late-training oscillation of `wr_greedy` (0.6 ↔ 0.9) is exactly the symptom C8
  predicts.
- **Actor KL early-stop is mostly benign.** `KL_TARGET=0.015` triggers an early stop in ~50 %
  of batches, but almost always in the *last* epoch (198 of 357 at epoch 3) and median
  `n_actor` updates is 264 — so the actor is not being starved. The very low counts (17, 37)
  are only early batches when games were short. Not a primary issue; leave it.

---

## Action points (prioritized)

**P0 — fix the reward so "win" dominates "attack".** This is the root cause of the
score/WR decoupling.
- Cut `ATTACK_REWARD` from 0.1 to ~0.02–0.03 (or remove and rely on shaping), so a game's
  worth of attacks cannot rival a win.
- Give the win condition direct dense credit: set `CLAIM_BASE_REWARD` > 0 (small, e.g. 0.1),
  since claiming bases *is* the path to winning.
- Drop `holding_reward` (the non-potential term) entirely, or make it symmetric/decaying —
  keep only the policy-invariant `γφ' − φ` potential term. Test: does mean `turns` fall?
- Re-verify after the change that `score` and `wr_greedy` move *together* (add the correlation
  to the eval log — see diagnostics).

**P1 — let the policy commit: anneal the entropy bonus.**
- Linearly decay `entropy_coeff` 0.025 → ~0.003 over training (or step down at a WR milestone).
- Target: entropy trending toward ~0.7–0.9 (≈40–50 % of the 1.84 max) by end of run, *while*
  WR holds or rises. If WR drops as entropy falls, the ceiling is capacity/reward, not
  exploration.
- **Fix `docs/METRICS.md`**: the entropy reference is 1.84 (max) for the current env, not 2.64;
  restate the ideal band relative to that.

**P2 — add LR decay (idea C8).** `LambdaLR: 1 − step/total_batches`, stepped once per outer
batch, on both optimizers. Cheap, directly targets the late-run oscillation.

**P3 — make self-play pull its weight.**
- Snapshot less often than every 3 batches (e.g. every 10–20) so pool opponents are
  meaningfully weaker than current, giving positive advantages.
- Cap / down-weight the pool share, or bias sampling toward stronger recent snapshots, so
  fewer batch samples are ~0-advantage mirror matches.
- Consider keeping a small (~0.1) greedy share as a permanent fixed anchor.

**P4 — only after P0–P3: scale capacity.** Try `hidden_dim=128` (and possibly a third conv
layer) in a controlled A/B. Do this last so you can attribute any gain to capacity rather than
the reward/exploration fixes.

**P5 — reward-shape toward unit-type play (optional, higher effort).** If you want the agent to
value unit economy without hard-coding known openings, add a small shaping term for material
kept in play / initiative held, rather than only attacks and base lead. Keep it potential-based
to preserve invariance. This is the principled way to nudge "commit to a few units" without
scripting strategy.

## Diagnostics to add before the next long run

These would have made this diagnosis a 5-minute read instead of a log dig:
- Log **entropy as a fraction of `mean log(n_legal)`** for the batch, not just raw entropy.
- Log the **decomposition of `score`**: sum of attack rewards vs shaping vs terminal, per
  batch. The decoupling would jump out immediately.
- Log **mean legal-action count** per batch (drift here changes what entropy "means").
- Track **correlation between `score` and `wr_greedy`** over a sliding window — flag when it
  goes negative.
- Bump eval to ≥40 games (currently 20 → ±0.10 noise per 2 games) so plateau vs. progress is
  distinguishable.

## One-line verdict

Stop the run early; the ceiling here is **reward design (attack reward drowns out winning) and
a never-annealed entropy bonus**, not training time or (yet) network size. Fix P0–P2 and
re-run — expect entropy to fall and `wr_greedy` to break past ~0.72 only once `score` and wins
are re-coupled.
