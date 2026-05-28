# Improvement ideas — PPO vs GreedyBot (run `ppo_20260527-191432`)

Goal: WR vs GreedyBot >= 60% by end of training. Current observed plateau: 5–15% (peak 0.15 at batch 80/130/150).

This document analyses the **current PPO setup** (`src/app/ppo.py`, `src/services/policy/policy.py`, `src/utils/rollout_buffer.py`) against the log `logs/ppo_20260527-191432.log` (300-batch run, interrupted at batch 194). It looks for fundamental issues that would explain the symptoms the user reported:

- gradient norms not descending (frequent spikes to 5–25, several into 50+);
- entropy decreasing fast (2.18 → 0.78 by batch 188);
- critic MAE INCREASING over training (0.13 at batch 7 → 0.30–0.45 from batch 80 onward);
- training score peaks at batch 130 (`0.61`) and then **regresses** to `0.19` by batch 194.

Yes, both hypothesised pathologies are happening:

- **the actor IS hunting a ghost.** Critic predictions are ~60–80% as wrong as the spread of the returns themselves (`critic_mae ≈ 0.35`, `ret_std ≈ 0.55`), so normalised advantages are dominated by critic noise rather than real outcome signal.
- **critic returns are partly meaningless.** Returns are non-stationary because (a) opponent distribution shifts as the pool refreshes; (b) `wr_random=0.90` is never reliably reached so the "fine-tune" weights never activate but the underlying policy strength does drift; (c) global feature `action_count // 2` is unnormalised and slides from 0 to ~100 across an episode, so the critic input distribution drifts within every game.

The fixes below are ordered by **impact-per-effort**, not by safety. The top six are the ones I would change before touching anything else.

---

## Symptoms → root causes (cross-reference)

| Symptom | Most likely cause(s) |
|---|---|
| Critic MAE grows over training | C1 (PPO epochs=1 starves critic), C2 (unnormalised `action_count`), C3 (non-stationary opponent distribution), C5 (no value-loss clipping) |
| Entropy collapses fast | C1 (only 1 epoch but full minibatch sweep), C7 (entropy coeff too low for the amount of clipping happening), C6 (board not mirror-encoded → policy collapses to one perspective faster) |
| Gradient norms not descending / large spikes | C4 (advantages noisy → loss noisy), C5 (no value clipping), C8 (no LR decay) |
| WR vs random oscillates 0.7–0.95 (never reliably >= 0.9) | C6 (no board mirroring → each perspective half-trained), C3 (pool overwrites old snapshots, fine-tune never triggers) |
| WR vs greedy plateaus at 0.05–0.15 | C9 (no mid-game shaping signal for stealing enemy bases), C1 (PPO underused) |
| Two `critic_std=nan` events (batches 133, 143) | C5 (no value clipping + occasional outlier return) |
| Score regresses after batch 130 | C3 + C2 + C1 stacked: small noisy gradient + drifting opponent pool + drifting `action_count` input |

---

## Tier 1 — fundamental fixes (do these first)

### C1. ✅ **`PPO_EPOCHS = 1` is silently turning PPO into A2C.**

**Evidence.** `src/app/ppo.py:523` sets `'ppo_epochs': 1`. With one epoch, the inner `for epoch in range(self._ppo_epochs)` loop runs once, the ratio `lp_new / lp_old` is ~1 everywhere (clipping never bites), and the early-KL-stop almost never triggers (it does fire on ~5% of batches: 65, 73, 74, 96, 111, 141, 168, 193 — those are likely outliers where the SINGLE epoch already overshot). The clip mechanism that justifies PPO over A2C is unused.

**Fix.** Set `ppo_epochs = 4` (canonical PPO default). Keep `KL_TARGET = 0.015` as the safety brake. Verify by logging `clip_frac` — you want it in the 0.05–0.20 range, not the current 0.00–0.05.

**Implementation difficulty.** Trivial — one line.
**Expected effect.** Biggest single fix. Roughly 4× more gradient updates per same data → 2–4× faster convergence on the same wall-clock. Typical PPO benchmarks show this is the single biggest knob for sample efficiency. Expect WR vs greedy to roughly double over the same run length if nothing else is wrong.

---

### C2. ✅ **`global` feature `action_count // 2` is unnormalised and drifts to ~100.**

**Evidence.** `src/services/environment/warchest_env.py:230`:
```python
global_feats = np.array([self.action_count // 2, my_bases, opp_bases], dtype=np.float32)
```
This is concatenated unchanged with the conv features (range ~0–1) and unit features. It grows from 0 to ~100 across an episode. The critic and actor both have to absorb this drifting input. Bases are bounded 0–6, action_count grows unbounded. The first MLP layer sees one feature 100× larger than the others — its weights for that feature are forced to be near-zero, and the gradient noise on those weights destabilises the rest.

This is also the most likely single explanation for the **critic-MAE-grows-over-training** symptom: as policies get better, episodes get shorter, so the distribution of `action_count` at the typical decision-point shifts. The critic was implicitly memorising "value vs turn count" — change the turn-count distribution and the memorisation breaks.

**Fix.** Normalise. Either divide by `max_actions` (so it's in `[0, 1]`) or drop it entirely (the network can infer time pressure from the board state). Also normalise `my_bases / 6.0` and `opp_bases / 6.0` while you're there, so all globals live in `[0, 1]`.

**Implementation difficulty.** Trivial — one line, and one matching `low/high` update in `observation_space`.
**Expected effect.** Should make critic MAE trend downward instead of upward. Expect critic MAE to settle near `0.15` instead of `0.35`, which directly tightens the advantage signal and makes actor updates ~2× less noisy.

---

### C3. ✅ **Pool snapshots too frequent — pool loses diversity by batch 50.**

**Evidence.** `snapshot_every=1`, `max_size=20`: pool always contains the most recent 20 snapshots. By batch 50, the pool no longer has any "weak self" — only the last 20 self-copies. This makes the opponent distribution drift continuously, which is the worst case for the critic.

**Fix.** `snapshot_every = 3` to keep older, weaker copies in the pool alongside recent ones.

**Implementation difficulty.** Trivial.
**Expected effect.** Stabilises the critic target distribution (returns become more stationary).

---

### C4. ✅ **Reward shaping is not formally potential-based, and per-step `MOVE_ON_BASE_REWARD` competes with the actual claim reward.**

**Evidence.** Two separate issues, both in the shaping pipeline:

1. **Non-PBR shaping.** `src/app/ppo.py:164–180`:
   ```python
   phi_before = SHAPING_C * (my_bases - opp_bases)  # state s_t, my turn
   state, reward, ... = self._env.step(action)
   phi_after = SHAPING_C * (my_bases - opp_bases)   # state s_t' — AFTER my move, BEFORE opp
   shaped_reward = reward + gamma * phi_after - phi_before
   ```
   By Ng et al. (1999), PBR requires `F = γ·Φ(s_{t+1}) - Φ(s_t)` where `s_{t+1}` is the *next state in the MDP I'm modelling*. In a 2-player turn-based game, the next state at my decision-point is *after the opponent has moved*, not after my own move. As written, the shaping captures only my contribution to base diff and silently ignores what the opponent did, so the telescoping that makes PBR policy-invariant breaks. In practice it still works as a dense bonus, but it isn't doing what the docstring expects, and the discrepancy creates extra variance.
2. **Reward conflict.** `MOVE_ON_BASE_REWARD = 0.005` fires *on the move that steps onto an unclaimed base*, but `CLAIM_BASE_REWARD = 0.0` and base-claim shaping is only `SHAPING_C * 1 = 0.05`. So moving-onto-an-unclaimed-base (0.005) and then claiming it next turn (0.05 shaping) gives 0.055 total — fine. But moving onto an *enemy*-controlled base gives **`MOVE_NEG_REWARD_PER_TURN = -0.002`** (enemy bases aren't in `unclaimed_bases`). Stealing an enemy base is therefore *immediately punished* until the claim-shaping arrives on the next step. This is plausibly why GreedyBot wins: the policy never learns to steal.

**Fix.**
1. Compute `phi_after` at the NEXT main decision point, not right after your own move. Simplest implementation: cache `phi_before` at step k, and use the **next** step's pre-action `phi_before` as `phi_after_k`. For the terminal step, `phi_after_terminal = 0`. This makes the shaping a true PBR.
2. Replace `MOVE_ON_BASE_REWARD` / `MOVE_NEAR_BASE_REWARD` with a uniform "approach unclaimed-OR-enemy base" signal so stealing is incentivised at the same scale as exploring. E.g. shape on `-min_BFS_distance_to_target_base / 6` where `target` includes enemy bases.

**Implementation difficulty.** Medium for #1 (refactor `_collect_episode` to defer the phi calc), low for #2 (one boolean change in `perform_move_action`).
**Expected effect.** Should unblock the "policy never steals" failure mode and make critic targets cleaner (lower variance). Expect WR vs greedy to gain another 10–20 points if C3 doesn't already address it.

---

### C5. ✅ **No value-loss clipping, plus the critic NaN events.**

**Evidence.** `_update_critic` uses plain `F.mse_loss(val, ret)`. Standard PPO uses *clipped* value loss to bound how far the critic can move per minibatch. Without it, occasional outlier returns (e.g. a long episode with shaping accumulated against you) produce huge MSE gradients. Two events in the log (batches 133 and 143) show `critic std=nan` with `grad_c=46.108` and `grad_c=24.196` — the critic outputs blew up on those minibatches. The gradient clip at 1.0 prevented a total NaN-cascade, but the critic state is now temporarily corrupted and the next few batches show critic_mae spiking (batch 133: 0.33; batch 143: 0.31 in noisy regime).

**Fix.** Add the PPO-style clipped value loss:
```python
v_clipped = batch['values_old'] + (val - batch['values_old']).clamp(-clip_range, clip_range)
loss_v_unclipped = (val - ret) ** 2
loss_v_clipped = (v_clipped - ret) ** 2
critic_loss = 0.5 * torch.max(loss_v_unclipped, loss_v_clipped).mean()
```
This requires storing the rollout-time `value` in the buffer (which you already do — `self._values`), and adding it to `iter_minibatches`. Use `clip_range = 0.2` to start.

**Implementation difficulty.** Low — ~15 lines.
**Expected effect.** Removes the rare NaN events and stabilises critic in late training. Won't change peak WR by much, but will eliminate the regression-from-batch-130 pattern.

---

### C6. ✅ **Board CNN sees the board in absolute orientation; the agent must learn two mirror policies.**

**Evidence.** Encoding swaps channels 3/4 for active perspective (good) but the spatial layout is unchanged. Player 1 always starts at `(1,0)+(4,1)`, player 2 always at `(2,5)+(5,6)`. When the active player is 2, the agent sees its own units in the bottom-right corner with channel 3 marking them; when active is 1, they're in the top-left with channel 3 marking them. The CNN must learn that the same strategy applies in both — which roughly doubles the data requirement.

This is the same class of bug previously fixed for the units / global features (see `docs/IDEAS.md` #1), but the *spatial* dimension was never canonicalised. The hex board has a 180° rotational symmetry that maps player-1 starting bases onto player-2 starting bases.

**Fix.** Rotate the board 180° (i.e. `np.rot90(board, 2)` and the same for `exploration_map`) and remap unit coordinates when `active_player == 2`. Now the network always sees "my side at top-left" regardless of which player is acting. This is a true equivariance fix.

**Implementation difficulty.** Medium — needs careful coordinate remapping (the action IDs reference offsets from *current* unit positions, which is already perspective-correct in `get_move_info`, so the action side does not need changes — but the observation rotation must be matched in `encode_board` and any unit-feature path).
**Expected effect.** Large. Effectively doubles the data efficiency for any state observed from both perspectives. This is the same magnitude of effect as the earlier IDEAS #1 fix. Expect 30–50% faster convergence (so WR vs greedy 30%+ by batch 150 instead of 300).

---

## Tier 2 — important tuning

### C7. ✅ **Entropy coefficient too low for how aggressively the masked softmax pushes entropy down.**

**Evidence.** Entropy goes 2.18 → ~0.78 by batch 188. Effective action space (masked) is typically 5–8, so max-entropy bound is `ln(8) ≈ 2.08`. Final entropy of 0.78 means the policy has roughly 2 effective actions per state — that's premature commitment given WR vs greedy is only 0.10. With `ENTROPY_COEFF=0.005` and `ent ≈ 1.0`, the entropy bonus is 0.005 in loss-units, while actor loss magnitude is `1e-3` to `5e-2`. The bonus is meaningful at the start but quickly outscaled by the actor loss; if you increase `ppo_epochs` (C1), the entropy will collapse even faster.

**Fix.** Raise `ENTROPY_COEFF` to `0.02` and add a linear decay schedule: `0.02` at batch 1, `0.005` at batch 300. Or simpler: keep `0.01` constant.

**Implementation difficulty.** Trivial.
**Expected effect.** Maintains exploration deep into training. Expect WR-vs-greedy ceiling 5–10 points higher because the policy keeps probing alternatives instead of locking in on the first "good enough" move.

---

### C8. **Linear LR decay missing.**

**Evidence.** `lr_actor=1e-4` and `lr_critic=3e-4` are constant. Standard PPO uses linear decay to zero across the training horizon. Without decay, late-training noise has the same step size as early-training large corrections.

**Fix.** Add a `LambdaLR` scheduler: `lambda step: 1.0 - step / total_batches`. Step it once per outer batch.
**Implementation difficulty.** Trivial.
**Expected effect.** Reduces late-training gradient-norm spikes and the score regression after batch 130. Probably worth 5–10 points of WR ceiling.

---

### C9. **Greedy bot is handicapped with 30% randomness — and the agent never sees clean greedy.**

**Evidence.** `src/services/bots/greedy_bot.py:20`: `RANDOM_ACTION_PROB = 0.30`. The training distribution and eval distribution both face this softened greedy. The agent learns to beat "70% greedy + 30% random", which is an easier target than what the user means by "greedy bot". And because the handicap is stochastic per-action, the gradient signal vs greedy episodes is noisier than necessary.

**Fix.** Train with `RANDOM_ACTION_PROB = 0.30` (curriculum) but **eval with `0.0` against true greedy**. Better yet: anneal the training handicap from 0.30 → 0.05 over the run, and add a *separate* eval call against `RANDOM_ACTION_PROB = 0.0`.

**Implementation difficulty.** Low — pass `random_prob` to `GreedyBot.__init__`, anneal it from the trainer, and add a second `_eval_episode` loop.
**Expected effect.** Mainly clarifies what your WR number means. The current 0.10 WR is vs softened greedy; the policy may be even weaker vs clean greedy. Hitting 60% vs *softened* greedy is roughly as hard as ~45% vs clean greedy.

---

### C10. ~~**Critic and actor encoders are not shared.**~~ **— OBSOLETE**

**Status (2026-05-28): superseded.** This recommendation predates the current design and would actively interact badly with it. See `docs/rl_algorithms.md` → *Architecture: shared vs separate actor-critic encoders* for the full analysis. Summary of why sharing is no longer the right call:

- the critic now takes a privileged `opp_onehot` input the actor must not see — a shared encoder leaks opponent-conditional features into the actor;
- `lr_actor=1e-4` vs `lr_critic=3e-4` cannot both be honoured on a shared trunk;
- the actor has an independent KL early-stop (`KL_TARGET=0.015`) which the critic does not — a combined loss cannot stop one without the other;
- two disjoint `Adam` instances over disjoint param sets make the partition typecheck-visible; the previous shared-trunk version with a single optimiser had several failure modes around which params actually get stepped.

The ~30% sample-efficiency claim below assumed a setup this codebase no longer has. Original analysis preserved for context:

> **Evidence.** `Policy` has its own `board_encoder` + `unit_encoder`; `Critic` has independent copies. The docstring justifies it ("Independent encoders let the critic develop value-optimized representations"), but in practice:
> - the actor encoder *never* sees value gradients (only policy gradients, which are small) → its features are slow to converge;
> - the critic encoder *never* sees policy gradients → it can't focus on what matters for action selection;
> - each encoder has ~200K params (Linear(3136, 64)); training two of them on a small dataset doubles the data needed.
>
> Standard PPO shares a trunk and forks at the last layer. This is also what MOST AlphaZero-style work does (a single trunk feeding two heads).
>
> **Fix.** Share `board_encoder + unit_encoder` between Policy and Critic. Add a `value_head` to Policy (or have Critic accept already-encoded features). Re-tune `lr` slightly because the trunk now sees a sum of two gradient sources.
>
> **Implementation difficulty.** Medium — refactor Policy/Critic class boundary, but maybe 50 lines of code. Will need to verify training stability.
> **Expected effect.** Faster convergence (~30% fewer batches to reach the same WR), better critic accuracy. This is the single architectural change most likely to lift the WR ceiling vs greedy.

---

### C11. ✅ **Hidden dim 64 is small for a 7×7 board with 14 actions.**

**Evidence.** `hidden_dim=64`. Board encoder projects `64*7*7 = 3136 → 64`, a 49× compression. That's a hard bottleneck. The actor head then expands back to 128 before predicting 14 logits.

**Fix.** Try `hidden_dim=128` (matches the actor head expansion). Watch out for overfitting if data volume stays low — combine with `ppo_epochs=4` and you should be fine.

**Implementation difficulty.** Trivial (already parameterised).
**Expected effect.** Modest — maybe 5–10% faster convergence. Most useful if combined with the shared-trunk fix (C10) so the larger trunk is actually used.

---

### C12. **Collect more episodes per batch.**

**Evidence.** `collect_episodes=16`, typical episode ~70 steps → ~1100 transitions per buffer → ~17 minibatches of 64. PPO usually wants 2048–4096 transitions per buffer for stable advantage estimates. With 16 episodes per batch you also get high variance in *which opponents* the batch saw — e.g. batch 87 shows 12 of 16 episodes vs greedy by chance, batch 95 shows mostly random/pool.

**Fix.** `collect_episodes=32` (with the same `minibatch_size=64`, you'll get ~35 minibatches per epoch, and with C1's `ppo_epochs=4` → ~140 updates per batch). Halves the number of batches but each batch is much more informative.

**Implementation difficulty.** Trivial.
**Expected effect.** Lower per-batch variance, smoother loss curves, helps the critic. Combined with C1, the actor sees ~8× more updates per *episode* than today.

---

## Tier 3 — bigger restructuring (optional, high-payoff)

### C13. **Switch to a hex-aware encoder (or just a flat MLP).**

**Evidence.** `Policy.board_encoder` uses 3×3 2D convolutions on a hex grid stored in a square array. The 3×3 kernel sees diagonals that aren't hex neighbours and misses one hex neighbour. For 7×7 = 49 cells × 6 channels = 294 inputs, even a flat MLP with one hidden layer of 256 has ~75K params — quite trainable, and topology-correct by definition.

**Fix (cheap).** Replace `board_encoder` with `Linear(6 * 7 * 7, 256) → ReLU → Linear(256, 128)`.
**Fix (clean).** Implement hex convolution by manually unrolling the 6-offset kernel.

**Implementation difficulty.** Low (cheap) / High (clean).
**Expected effect.** Removes a small but persistent representation bias. Modest gain on its own; meaningful when combined with C10/C11.

### C14. **Try DQN / Double DQN with prioritised replay.**

**Evidence.** Action space is small (14), board is small (7×7), episodes finite, deterministic transitions given action. DQN is *very* well-suited to this regime and is much more sample-efficient than on-policy methods when the replay buffer can store transitions across many opponents. See `docs/rl_algorithms.md` — DQN was already considered.

**Fix.** New trainer: standard DQN with masked-argmax target, ε-greedy exploration, replay of ~50K transitions, target-net every 1K steps.
**Implementation difficulty.** High — ~300 lines new trainer.
**Expected effect.** Plausibly the fastest path to "WR vs greedy > 60%". Off-policy methods exploit the replay buffer, which sidesteps the non-stationary opponent issue (C3). Worth doing as a parallel benchmark.

### C15. **MCTS + policy/value (AlphaZero-style).**

**Evidence.** Tiny branching factor (avg ~6 valid actions, max 14), deterministic transitions, perfect information. This is the ideal setting for MCTS.

**Fix.** Wrap the current Policy/Critic as the network in an AlphaZero loop: do `n_simulations=50` MCTS rollouts per move using policy as prior and critic as leaf value, then train policy toward MCTS visit distribution and critic toward MCTS outcome.

**Implementation difficulty.** High — ~500 lines, plus reference implementations exist.
**Expected effect.** Largest plausible win. AlphaZero-style methods routinely hit >90% WR vs hand-coded heuristics on similar-size games. The cost is implementation complexity and slower wall-clock per-game.

---

## Tier 4 — small / cosmetic / nice-to-have

| # | Issue | Difficulty | Effect |
|---|---|---|---|
| C16 | `iter_minibatches` re-permutes per call but reuses the same buffer across epochs (now relevant with C1) | trivial | minor — keeps minibatches uncorrelated across epochs |
| C17 | Truncation outcome handling: `trunc_reward` jumps 0/-0.5/-1 in a step function; a smoother base-diff-proportional reward would be lower variance | low | small — cleaner critic target |
| C18 | `value_single` calls `.train()` mode during rollout, then `eval()` at eval — irrelevant today (no BN/dropout) but worth `.eval()` before rollout to be future-safe | trivial | none until BN/dropout added |
| C19 | Pool stores full `state_dict` deep-copied — fine, but consider sharing the policy_constructor closure with a state-dict-only loader (already done) | trivial | none |
| C20 | Eval is only 20 episodes, every 10 batches — std error on a 0.1 WR estimate is ~6.7%. Bump to 50 episodes to make the wr-vs-greedy curve readable | low | makes logs more interpretable |
| C21 | `EloTracker` updates after every eval game, double-updating WR signal — fine but elo numbers in the log are noisy; consider a running-average | low | log readability only |
| C22 | `score_deque` length is `print_every * collect_episodes = 160` — fine, but reporting both batch-mean and 100-episode rolling mean would make regressions easier to spot | low | log readability |

---

## Recommended order (concrete plan)

If you only do four things, do these (in this order):

1. **C1 — set `ppo_epochs = 4`** *(15 minutes)*. Biggest single fix. Validates that the rest of the loop works at all when PPO actually does PPO.
2. **C2 — normalise `action_count` in globals** *(15 minutes)*. Should be visible in `critic_mae` within ~30 batches. If C1 + C2 doesn't visibly help the critic, the problem is somewhere unexpected and the remaining items become a debugging tool.
3. **C3 — fix opponent pool diversity + fine-tune phase trigger** *(2 hours)*. Stabilises the return distribution so the critic can actually converge, and forces more greedy exposure in training.
4. **C6 — board mirror equivariance** *(2 hours)*. Doubles data efficiency for every state.

After those: C5, C7, C11. Then re-measure and decide whether you need C13–C15. (C10 is superseded — see header.)

If after C1+C2+C3+C6 you are still under 30% WR vs greedy at the *end* of a 300-batch run, the bottleneck is most likely architectural — go to C11 (hidden_dim=128) and C13 (hex-aware encoder). Do **not** revisit shared encoders — see `docs/rl_algorithms.md` → *Architecture: shared vs separate actor-critic encoders*.

If you want a step-change rather than incremental improvements, skip to C14 (DQN) or C15 (MCTS) — but only after the foundational fixes, because both reuse the encoder.

---

## Verification checklist (after applying fixes)

You should expect to see, in this rough order:

- **critic_mae trends DOWN** over the run (currently trends up). Target: < 0.20 by batch 200.
- **entropy decays smoothly to ~0.6, not 0.4**.
- **grad_a stays under 5** with rare spikes, instead of frequently touching 10–25.
- **clip_frac sits in 0.05–0.20** during PPO updates (currently 0.00–0.05 because epochs=1 means ratio≈1).
- **wr_random eval crosses 0.95 and stays there** by batch ~60, triggering the fine-tune phase.
- **wr_greedy eval climbs past 0.30 by batch 150**, past 0.50 by batch 250, hitting target 0.60+ by batch 300.

If `critic_mae` is still climbing after C1+C2, the root cause is something this analysis missed — most likely a deeper bug in `compute_gae` or in how `done` is handled at terminal vs truncation boundaries. Re-audit those before adding more shaping.
