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
