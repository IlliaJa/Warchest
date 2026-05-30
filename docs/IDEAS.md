# Ideas for improving training

Implemented history: `docs/history.md`.

---

## Open items

Current state: WR vs greedy ~49% (post C1–C7 fixes, run `ppo_20260527-191432`). Target: 60%.

### Architectural note — factored / autoregressive action head

*(Parked — full design in `docs/rl_algorithms.md` → *Action head: flat spatial vs factored / autoregressive*.)*

Once the action set grows (4+ units, coin mechanics), the flat `[A, 7, 7]` spatial softmax breaks down — coin-only verbs (recruit, initiative, pass) have no board cell to point at. **Not needed at current scale.** Revisit when the env has bag/hand/recruit mechanics.

---

### C8. **Linear LR decay missing.**

`lr_actor=1e-4` and `lr_critic=3e-4` are constant throughout training. Standard PPO decays both linearly to zero so late-training noise has a smaller step size than early large corrections.

**Fix.** Add a `LambdaLR` scheduler: `lambda step: 1.0 - step / total_batches`. Step once per outer batch.

**Relevance.** Still open. At 49% WR the policy is in late-middle training — late-training oscillation is the most likely explanation if the WR curve flattens or regresses. Low effort, worth doing before the next long run.

**Difficulty.** Trivial.

---

### C9. **GreedyBot eval uses 30% randomness — the WR number is misleading.**

`src/services/bots/greedy_bot.py:29`: `RANDOM_ACTION_PROB = 0.30` is hardcoded. Both training and eval face the softened bot. The current 49% WR is vs "70% greedy + 30% random", not vs true greedy. Hitting 60% vs softened is roughly equivalent to ~45% vs the real thing.

**Fix.** Add a `random_prob` parameter to `GreedyBot.__init__`. At eval time pass `0.0` (a separate eval call). Keep the training handicap as-is or anneal 0.30 → 0.05 over the run as a curriculum.

**Relevance.** High — the WR number you're optimising for is off by ~15 points vs the implied target. Worth adding the clean eval before claiming 60%.

**Difficulty.** Low.

---

### C12. **`collect_episodes=16` produces noisy advantage estimates.**

~1100 transitions per buffer is low for PPO (standard is 2048–4096). With 16 episodes per batch there is also high variance in which opponents a batch sees. Doubling would give ~140 inner updates per batch (vs ~70 now) with no inner-loop change.

**Fix.** `collect_episodes=32`.

**Relevance.** Moderate. Probably worth the wall-clock cost once LR decay (C8) is in.

**Difficulty.** Trivial.

---

### Tier 4 — small / cosmetic

| # | Issue | Difficulty | Effect |
|---|---|---|---|
| C17 | `trunc_reward` is a step function (0/−0.5/−1); a base-diff-proportional value would reduce critic target variance | low | small |
| C18 | `value_single` is called during `.train()` mode at rollout (line 199); harmless today (no BN/Dropout) but if either is added, Dropout would produce stochastic noisy values during GAE and BatchNorm would compute statistics from a single sample — both silent bugs. Fix: `self._critic.eval()` before the rollout loop, `self._critic.train()` before the update. | trivial | none until BN/dropout |
| C20 | Eval runs 20 episodes — std error on a 0.49 WR estimate is ~5%. Bump to 50. | low | log readability |
| C21 | `EloTracker` updates after every eval game — noisy; consider a running average | low | log readability |
| C22 | `score_deque` reports batch-mean only; adding a 100-episode rolling mean would make regressions easier to spot | low | log readability |

**C16** (`iter_minibatches` permutation): already correct — a fresh `np.random.permutation` is generated on each call, so minibatches are uncorrelated across PPO epochs. No action needed.

**C10** (shared encoders): obsolete. See `docs/rl_algorithms.md` → *Architecture: shared vs separate actor-critic encoders*.

---

### Recommended next steps

1. **C9 — clean greedy eval** *(1 hour)*. Calibrates whether 49% is actually close to the 60% target.
2. **C8 — LR decay** *(15 min)*. Before the next long run.
3. **C12 — `collect_episodes=32`** *(5 min)*. After C8.
4. Re-measure. If still under 50% vs *true* greedy after a 300-batch run, escalate to C15 (MCTS) or C14 (DQN benchmark).

