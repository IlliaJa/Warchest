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
