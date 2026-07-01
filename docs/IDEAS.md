# Ideas for improving training

Implemented history: `docs/history.md`.

---

## Open items

Current state: WR vs greedy ~49% (post C1–C7 fixes, run `ppo_20260527-191432`). Target: 60%.

### Game-completeness — remaining Phase 5 work

The full base game (all 16 units + per-game disjoint drafting) is implemented; drafting was pulled forward into Phase 3, so the old `full_game_plan.md` roadmap is retired (its history now lives in `docs/history.md`). These are the only pieces of the "Phase 5 / full game" spec that were **not** carried forward:

- **P5a. Variable-composition eval bucketing.** Eval currently reports a single aggregate WR across all randomly-drafted matchups. Add per-composition (or per-composition-bucket) WR reporting so we can see *which* unit sets the agent is weak/strong on instead of averaging over the ~1820 possible 4-unit matchups. Difficulty: low-moderate.
- **P5b. Rulebook drafting mode (optional).** Setup currently assigns each player 4 random disjoint types. The real game uses a snake/alternating draft from a shared pool. Implement it as an optional setup mode for parity with tabletop games; not required for training. Difficulty: moderate.
- **P5c. Freeze `baseline_tactics` snapshot.** The Phase-4 exit step ("re-baseline via a training run") is largely satisfied by run `ppo_20260630-060400` (~70% WR vs *true* greedy). Remaining: freeze that policy as a named pool/eval anchor for the current (`OBS_VERSION=8`) schema generation. Difficulty: trivial.

The plateau / high-entropy / reward-decoupling problems observed on that run are diagnosed with prioritized action points in `docs/analysis_ppo_20260630.md` — that doc, not this section, is the live to-do for improving the full-game agent's *strength*.

### Architectural note — factored / autoregressive action head

*(Parked — full design in `docs/rl_algorithms.md` → *Action head: flat spatial vs factored / autoregressive*.)*

Once the action set grows (4+ units, coin mechanics), the flat `[A, 7, 7]` spatial softmax breaks down — coin-only verbs (recruit, initiative, pass) have no board cell to point at. **Not needed at current scale.** Revisit when the env has bag/hand/recruit mechanics.

---

### C12. **`collect_episodes=16` produces noisy advantage estimates.**

~1100 transitions per buffer is low for PPO (standard is 2048–4096). With 16 episodes per batch there is also high variance in which opponents a batch sees. Doubling would give ~140 inner updates per batch (vs ~70 now) with no inner-loop change.

**Fix.** `collect_episodes=32`.

**Relevance.** Moderate. (Run `ppo_20260630-060400` already used `collect_episodes=64`, so this is likely superseded — confirm against the current `ppo.py` default before acting.)

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

C8 (LR decay) and C9 (clean true-greedy eval) are **done** — run `ppo_20260630-060400` uses the true greedy bot (`RANDOM_ACTION_PROB=0.0`) and reports an honest ~70% WR. That run's plateau is now the live problem; its diagnosis and prioritized action points are in `docs/analysis_ppo_20260630.md`. Start there.

