# Experiment Log

Chronological record of training runs — what changed, when, and what the metrics showed.

---

## 2026-05-28 — Reward redesign + PPO fixes (run: solar-sun-127)

**Changes:** Removed `MOVE_ON_BASE_REWARD` / `MOVE_NEAR_BASE_REWARD` (created a neutral-over-enemy bias) and added a per-turn holding reward (`holding_reward_rate = 0.0016`) to break the base-flip exploit. Also applied PPO fixes: epochs 1→4, value-loss clipping (C5), normalised global features (C2), snapshot_every 1→3 (C3).

**Results:** `critic_mae` and `wr_vs_greedy_eval` remained at similar levels to the previous run (max ~33%). The main visible improvement was `avg_turns` dropping from ~80–140 to ~40–80, indicating the policy now ends games decisively rather than stalling — a sign the holding reward is working as intended.

---

## 2026-05-28 — Board mirroring for P2 perspective (run: lively-pyramid-129)

**Changes:** Implemented C6 fix — when `active_player == 2` the observation is rotated 180° so the policy always sees its own units at the top-left of the board. Unit coordinates remapped `(r,c) → (6-r, 6-c)`, action mask remapped with self-inverse offset flip `[3,4,5,0,1,2]`, and all action IDs reverse-mapped before `env.step()` in both the training loop and evaluator.

**Results:** `wr_vs_greedy_eval` peaked at **49%** vs **33%** maximum in the previous run — a clear step change driven by the policy now learning a single unified strategy instead of two mirrored ones. `wr_vs_random_eval` remained stable at ~1.0 throughout. `wr_vs_greedy_train` shows a consistent upward trend across the full 300 batches, and `score_main` is trending up vs the previous run's oscillation. The 60% WR target is not yet reached but the trajectory is positive.

---

## 2026-05-29 — Hex-correct board encoder (C13)

**Changes:** Implemented C13 fix — replaced the 2D 3×3 convolutions in `Policy.board_encoder` and `Critic.board_encoder` with `HexConv2d`, a custom module that gathers only the 6 hex neighbours plus center via `F.unfold` and projects with a 1×1 conv. The two anti-diagonal positions that are not hex-adjacent under `Board.offsets` are excluded from the kernel. See `docs/decision.md` for the full rationale.

**Results:** `wr_vs_greedy_eval` climbed steadily through training, crossing **60% by ~step 300** and peaking at **80% around step 360 and again around step 580** over a ~650-batch run. From step ~300 onward the curve oscillates in the 0.4–0.8 range — variance is high (`eval_episodes=20` → std error ≈ 11pp on a 0.6 estimate) but the mean is clearly above the 60% target. This is the first run where the agent meaningfully exceeds greedy on the current rule set; the goal stated at the top of `improvement_ideas.md` is met.

**Implication.** The prototype rule set (2 units, move-only, claim bases) is now solved well enough that further encoder / hyperparameter tuning would be optimising against a saturated target. Next focus shifts to expanding the rule set toward the original game — see `docs/decision.md` § *2026-05-29 — Focus shift: expand rule set toward original Warchest*.

---

## 2026-05-30 — Attack + deploy actions, spatial conv head (run: ~step 270)

**Changes:** Added two new action types (`attack`: instant-kill adjacent enemy unit; `deploy`: place a new unit on a controlled empty base, capped at `MAX_DEPLOYS=4` lifetime uses). Simultaneously migrated the policy to a spatial cell-keyed action head: the board encoder now outputs a `[Cf, 7, 7]` feature map (no flatten), unit positions moved into board planes (channels 6–7), the separate `unit_encoder` MLP was removed (planes-only), and the actor head became a `Conv2d → [14, 7, 7]` logit map (`action_dim = 14×49 = 686`). P2 rotation now applies uniformly to the full spatial grid (no per-action-type remapping tables needed). Global features expanded from 3 → 5 dims to expose remaining deploy budgets to both policy and critic. GreedyBot rewritten with priority attack → control → move. See `docs/decision.md` § *2026-05-30* for the full rationale.

**Results:** `wr_vs_greedy_train` climbed from near 0 to **~90% by step ~270**, well above the 80% peak of the previous run on the move-only game. The curve shows a consistent upward trend with no regression — the richer action set (attack pressure, deploy recovery) and the spatially-coherent head appear to make the learning problem structurally easier for the agent, not harder. Attack and deploy actions are both used by the trained policy.

![wr_vs_greedy_train](../src/app/../.claude/image-cache/d6caddcd-b06a-442e-a19c-41da97c16ea1/2.png)

---

## 2026-05-31 — Full coin economy (Phase 1c), ~650 batches

**Changes:** First training run on the complete Phase 1 coin/bag economy (Phase 1a→1c). The env is now faithful two-unit-type War Chest: a per-player **bag** (2 Swordsman + 2 Knight + 1 Royal) with a random 3-coin draw each round, discard + reshuffle; **round/turn controller** with initiative (randomised at setup, claim-initiative transfers it ≤ once/round); coins **bind to the board** (deploy moves a coin hand→board; **coin-stack HP** via bolster; attack removes one coin to the box); **recruit** from a per-type supply; **pass**. Action space is the temporary flat head, now `796` (spatial `16×49` incl. deploy-S/K + bolster, plus 12 face-down slots for claim/pass/recruit). Observation `OBS_VERSION=3`, `GLOBAL_DIM=28`: ego-centric coin-counting features (own hand/bag/discard/supply known; opponent on-board/face-up-discard/supply + derived `hidden_pool`), unit planes carry stack height. **Privileged critic** wired (`PRIV_DIM=9` — opponent's true hidden hand/bag/face-down split, never seen by the policy).

**Results:** Strong, healthy convergence over ~650 batches.
- `wr_vs_greedy_eval` rises to **~0.9** and holds there from ~step 220 onward (peaks near 1.0); `wr_vs_greedy_train` tracks it at **~0.9**.
- `wr_vs_random_eval` saturates at **1.0** within ~20 steps and stays flat.
- `wr_vs_pool_train` (self-play vs the opponent-pool snapshots) sits at **~0.5**, oscillating 0.4–0.6 — the expected equilibrium for self-play and a sign the pool is keeping pace.
- `score_main` plateaus around **0.9–1.0**; `grad_norm_critic` stays low (~2–5) with only occasional spikes (≤27) — stable value learning, no divergence.

The full coin economy is learnable end-to-end: the agent masters the much larger action/observation space and beats greedy at the same ~90% it reached on the far simpler move+attack+deploy prototype, while self-play stays balanced.

**Caveats.** (1) `GreedyBot` ignores bolster and recruit, so "WR vs greedy" is a softer bar than full-economy play — the agent is beating an economy-blind, myopic bot. (2) The privileged critic is in use but **not yet A/B'd** against a public-only critic, so its contribution to this result is unmeasured. Both are open items before reading too much into the 90%.

![Phase 1c training curves](assets/2026-05-31-phase1c.png)

---

## 2026-06-30 — Full base game, long run (run: `ppo_20260630-060400`, ~710/1000 batches)

**Changes:** First long training run on the **complete base game** — all 16 units with their
tactics/attributes/restrictions and per-game disjoint drafting (`OBS_VERSION=8`,
`ACTION_SPACE_SIZE=1875`). This is the culmination of Phases 3–4. Hyperparameters:
`collect_episodes=64, ppo_epochs=4, ppo_eps=0.2, gamma=0.99, lam=0.95, entropy_coeff=0.025,
lr_actor=lr_critic=3e-4, hidden_dim=64, minibatch=64`; opponent schedule `random/greedy/pool`
`0.4/0.2/0.4` → `0/0.4/0.6` (finetune triggered almost immediately once `wr_random` hit 1.0).
`GreedyBot.RANDOM_ACTION_PROB=0.0`, so greedy eval is now the **true** bot (idea C9 resolved) —
this WR is honest, not softened.

**Results:** the agent learns the full ruleset and beats greedy, but **plateaus early and then
oscillates for ~10 h with no real gain**. Reading the panels (see image):

- **`elo_policy`** ramps ~1050 → ~1400 by batch ~130–250, then oscillates 1350–1420 flat for
  the entire second half (peaks 1426 at batches 230/360, never beaten). `wr_greedy` (not in this
  panel set) mirrors it: ~0.72 mean, no trend; `wr_random` ≈ 1.0 throughout.
- **`entropy`** falls only 1.75 → ~1.25 and stalls — that's ~70% of the *measured* max entropy
  (mean legal-action count is just 7.8, so max ≈ 1.84). The policy never commits; it keeps
  spreading mass over ~half the legal moves.
- **`score_main`** keeps climbing (~0.9 → ~1.4–1.5) across the whole run **while WR/Elo stay
  flat** — the agent is optimizing the dense shaping signal, not wins. `avg_turns` sits flat
  ~125–140 (the analysis log notes a mild 127→150 drift), i.e. games aren't getting more
  decisive as score rises. This score-vs-win decoupling is the headline problem.
- **Training-health signals are all clean:** `critic_loss` drops ~0.5 → ~0.12 and stays low;
  `critic_mae` flat ~0.05–0.07; `approx_kl` ~0.005; `clip_frac` steady ~0.06–0.08;
  `advantage_std` stable ~0.3; `actor_loss` ~0; grad norms low with only occasional spikes
  (actor ≤8, critic ≤3). Nothing is diverging — this is **not** an optimization-instability or
  train-longer problem.

**Takeaway.** The full game is learnable end-to-end and the agent reaches ~70% vs true greedy,
but the last ~580 batches (~10 h) bought nothing. Root cause is **reward design + exploration
pressure**, not compute or stability. This diagnosis drove the entropy/LR annealing, sparser
pool snapshots, `ATTACK_REWARD` cut, critic widening, and material-PBRS fixes recorded in
`docs/history.md`. (`docs/METRICS.md`'s entropy band has since been recalibrated to the
current max ≈ 1.84.)

![Full base game — long run W&B panels (ppo_20260630-060400)](assets/2026-07-01-full-game-implemented.png)

---

## 2026-07-02 — Threat/position-aware observation + deeper trunk (run: `ppo_20260702-082214`)

*Design/implementation: `docs/history.md` → "Threat/position-aware observation + deeper trunk".*

**Changes:** Implemented the `docs/IDEAS.md` "the agent can't see the board as one position" note in full: 6 graded threat/reach planes (`own`/`enemy` × `melee`/`ranged`/`charge`, with an exact Berserker closed-form and Marshall grant-chaining), 2 static ego-centric coordinate planes, a flank-split pool feeding `verb_head`/`facedown_head` in place of the old location-blind global mean, and a 3rd `HexConv2d` trunk layer (receptive field radius 2→3). `BOARD_CHANNELS` 38→46, `OBS_VERSION` 8→9 — schema-breaking, fresh run. Same hyperparameters and **same `n_batches=400`** as the correct pre-change baseline, run `ppo_20260701-191923` (`collect_episodes=64, ppo_epochs=4, gamma=0.99, lam=0.95, entropy_coeff=0.025→0.003, lr=3e-4, hidden_dim=64`) — a genuinely equal-length comparison.

**Note on an earlier version of this entry:** the first two drafts compared against `ppo_20260630-060400` (`n_batches=1000`, manually interrupted at batch 711) — the wrong run, with unequal length, that happened to be the most recently-documented baseline in this file. The correct comparison is against `ppo_20260701-191923` (wandb `f3oyn0sb`), the run actually shown alongside this one in the panel image below, which used the identical `n_batches=400` schedule. All numbers below use the correct run.

**Results — headline: no measurable difference.** The W&B panel image overlays both runs: **orange = this new run (`ppo_20260702-082214`)**, **magenta = the correct baseline (`ppo_20260701-191923`)**. The two final `wandb` summaries:

| Metric | Baseline (`f3oyn0sb`) | This run (`u62wwlnc`) |
|---|---|---|
| `wr_vs_greedy_train` | 0.880 | 0.890 |
| `wr_vs_greedy_eval` | 0.700 | 0.800 |
| `elo_policy` | 1433 | 1457 |
| `entropy` (raw) | 0.632 | 0.508 |
| `entropy_frac` | 0.256 | 0.195 |
| `critic_mae` | 0.341 | 0.312 |
| `avg_turns` | 72.8 | 60.7 |
| `score_main` | 0.708 | 0.708 |

`score_main` is identical to three decimal places; `critic_mae` and `avg_turns` are in the same range either way. Comparing the full **distribution** of the two metrics with the widest apparent gap, over each run's eval checkpoints from batch 200 on (not just the two final numbers):

| Metric | New run: mean ± std (n=21) | Baseline: mean ± std (n=18) | Gap in pooled std |
|---|---|---|---|
| `wr_vs_greedy_eval` | 0.783 ± 0.143 | 0.747 ± 0.092 | **0.31σ** — noise |
| `elo_policy` | 1410 ± 62 | 1396 ± 37 | **0.29σ** — noise |

A ~0.3 pooled-std gap, from a single run per architecture, is indistinguishable from noise. **These two runs play about the same.** This matches what the panel image shows directly — the two curves track each other closely across every panel, not just on average. Retracting both earlier drafts' framings in full: neither "every number favors the new run" nor even the more hedged "weak, and n=1" reading survive comparison against the correct baseline — the previous drafts' entire premise (that there was a gap worth explaining away) was built on the wrong reference run.

**Attribution caveat.** Moot given no measurable effect either way, but worth remembering if a future run does show one: three sub-changes (threat/reach planes, coordinate planes + split pooling, deeper trunk) shipped together in one pass rather than being A/B'd separately as the originating note suggested, so a future confirmed gain still couldn't be attributed to a specific piece without an ablation.

**Implication.** This run provides no evidence the architecture change helped or hurt. The entropy/critic numbers hint the new network isn't obviously worse, but nothing here clears the noise bar. To actually test the hypothesis: 2–3 seeds per architecture, same `n_batches`, comparing distributions (as above) rather than single runs or endpoints.

![Threat/position-aware observation vs. correct baseline — W&B comparison panels, orange=ppo_20260702-082214 (new) vs magenta=ppo_20260701-191923 (baseline, same n_batches=400)](assets/2026-07-02-spacial-awareness.png)

---

## 2026-07-04 — New obs features + annealed material PBRS + PPO tuning, parallelized (~600 batches)

**Changes.** This run bundles everything landed since the 2026-07-02 entry — it is *not* an
isolated A/B:
- **Observation** (`docs/history.md` 2026-07-03): material-at-risk, expected-opponent-hand
  (`E_opp_hand`), and base-control reach planes added (commit `70c948b`, `OBS_VERSION` bump).
- **Reward + critic** (`docs/history.md` 2026-07-03): annealed material PBRS reward; critic
  widened to `critic_hidden_dim=192` (commit `33e33d2`).
- **PPO dynamics** (commit `dcaabdd`): `lr_final_frac` 0.0 → **0.1** (LR now decays to
  `3e-4 → 3e-5`, not to 0, targeting late-run WR oscillation); `KL_TARGET` raised to **0.03**
  with per-minibatch KL-skip instead of an early-stop of the whole epoch.
- **Speed:** episode-level multiprocessing (`n_workers=6`, `docs/parallel_rollouts.md`).
  Speed-only in expectation — it changes the concrete episode sample (parallel RNG streams)
  but not the learning dynamics.

Config: `n_batches=600, collect_episodes=64, ppo_epochs=4, ppo_eps=0.2, gamma=0.99, lam=0.95,
entropy_coeff=0.025→0.003, lr=3e-4→3e-5, hidden_dim=64, critic_hidden_dim=192, minibatch=64,
pool_snapshot_every=15`, shaping anneal `1.0→0.1` over the first 50% then floored. Greedy eval
is the true bot (`RANDOM_ACTION_PROB=0`), so WR-vs-greedy is honest.

**Results — new best on every headline metric, and the score-vs-win decoupling is gone.**
- **`elo_policy`** climbs ~1000 → **~1500** (peak ~1520), essentially monotone across the whole
  run — a clear step above the prior highs (1457 on 07-02, 1433 on 06-30).
- **`wr_vs_greedy_eval`** rises 0 → **~0.9**, oscillating 0.8–1.0 through the second half
  (vs 0.80 / 0.70 in the two prior logged runs); `wr_vs_greedy_train` tracks it at ~0.9;
  `wr_vs_random_eval` saturates at 1.0.
- **Elo/WR keep climbing together with `score_main`** (~0 → ~0.6) instead of the score-up /
  WR-flat split that defined the 2026-06-30 run. This is the headline: the agent is now
  converting dense shaping into actual wins, and WR is still trending up at the end rather
  than plateauing.
- **Shaping annealed cleanly:** `shaping_anneal` 1.0 → 0.1 by ~batch 300; `score_material` and
  `score_holding` both decay to ~0 as intended, so the late-run gains are terminal-driven.
- **Exploration:** `entropy` falls 1.75 → ~0.7 while `max_entropy` *rises* to ~2.5 (mean legal
  count grows as games open up); `entropy_frac` ends ~0.25 — the policy still spreads mass over
  ~a quarter of legal moves, same regime as before, not a hard collapse.
- **Training health is clean:** `clip_frac` 0.15 → ~0.05 (consistent with LR/entropy anneal),
  `critic_loss` ~0.2 flat, `actor_loss` ~-0.02 → 0, `advantage_std` ~0.4 stable, grad norms low
  with only occasional late spikes (actor ≤15, critic ≤5). `critic_mae` drifts *up* 0.05 → ~0.22
  over the run (return scale grows as the policy improves) but ends below the 0.34 of the 07-02
  baseline — worth watching, not alarming.

**Caveats.**
1. **Bundled changes, n=1.** Three substantive changes (obs features, reward+critic, PPO
   tuning) shipped together, so the ~1500 Elo / ~0.9 WR gain can't be attributed to any one
   piece without an ablation. WR oscillates with `eval_episodes=20` (~11pp std error), so the
   peak-vs-plateau shape from a single run is not distinguishable from noise.
2. **`wr_vs_pool_train` ≈ 0.7, not the ~0.5 self-play equilibrium** — it sits above 0.5,
   oscillating ~0.6–0.8, i.e. the policy is outrunning its own snapshots. The pool is lagging
   (improvement outpacing `pool_snapshot_every=15`), so the greedy/Elo gains are real but
   self-play here is not a tight equilibrium. Denser snapshots may keep the pool honest.
3. Parallel rollouts were correctness-verified but the head-to-head speed benchmark
   (`n_workers=1` vs `6`) is still pending (`docs/parallel_rollouts.md`).

![New obs + annealed material PBRS + PPO tuning — W&B training panels (~600 batches)](assets/2026-07-04.png)

---

## 2026-07-07 — Long run, 1500 batches (run: `ppo_20260706-194732` → `warchest_ppo_20260707-0026.pth`)

*Backfilled 2026-08-01 from the retired `docs/next_steps.md` (Step 4), which was the only
record of this run outside git history.*

**What it was.** The 1000–1500-batch long run the roadmap had been deferring until a
non-saturated yardstick existed. It ran *before* `LookaheadBot` and the round-robin gauntlet
were fully in place, so — like every prior run — it was read against greedy while it trained.

**Result.** `wr_greedy` saturates to **1.0 by batch ~1200**, the same shape as the
2026-06-30 plateau run: the back half of the run was optimizing a dead signal. Despite that,
the checkpoint was **validated after the fact** by the first gauntlet (below): it is the
strongest agent in that field, beats the previous checkpoint (`ckpt_20260704-1243`) 70/30, and
beats both search bots. So this run's gains were real, not a saturated-signal artifact — this
time.

**Lesson carried forward.** A long run should train against a curriculum that still moves at
batch 1200 (an independent opponent in the pool) and be read against the gauntlet from the
start rather than after the fact.

---

## 2026-07-08 — First round-robin gauntlet (5 agents, `app/gauntlet.py`)

*Backfilled 2026-08-01 from the retired `docs/next_steps.md`.* First measurement against a
fixed, non-self-referential field after the versioned-encoder + gauntlet work
(`docs/history.md` → *Measurement + opponent infrastructure*).

```
Win-rate matrix (row vs column):
                         ckpt_2  ckpt_2  greedy  lookah  lookah
ckpt_20260704-1243[v10]     -      0.30    0.90    0.55    0.57
ckpt_20260707-0026[v10]    0.70     -      1.00    0.65    0.75
                 greedy    0.10    0.00     -      0.15    0.47
              lookahead    0.45    0.35    0.85     -      0.80
       lookahead_critic    0.42    0.25    0.53    0.20     -

Bradley-Terry ranking (Elo-scaled, field mean = 1000):
  ckpt_20260707-0026[v10]   1177.1
                lookahead   1070.6
  ckpt_20260704-1243[v10]   1051.4
         lookahead_critic    909.2
                   greedy    791.7
```

**Reading it:**

- **The 1500-batch checkpoint is the strongest agent in the field**, beating everything
  including both lookahead bots. First time "beats predecessors" was checked against a
  genuinely non-saturated yardstick and held up.
- **Plain `lookahead` beat `lookahead_critic` 80/20**, which was backwards from its design
  intent (critic-guided beam search was supposed to search *better*). Diagnosed 2026-07-11 as
  a **missing critic denormalization**; post-fix it wins 68–78% vs `lookahead`
  (`docs/bots.md`). **This ranking is therefore stale** — treat it as a historical first
  measurement, not a current standing.
- **No cycles found by hand-checking triples**, but 5 agents is too small a field to trust
  that qualitatively; the gauntlet's own intransitive-triple metric is the check as the field
  grows (it later measured 0.11 on a 31-agent ExIt field — `docs/independent_opponents.md` §1).