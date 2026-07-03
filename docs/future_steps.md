# Future steps — toward a stronger, legible strategy

Sequential action plan, to be executed one step at a time with a measured A/B
after each change — not applied all at once. Context: the goal is a policy
that (a) wins reliably and (b) has a legible strategy (uses tactics, manages
the coin economy), not just a confident-but-arbitrary policy. See
`docs/analysis_ppo_20260630.md` for the original plateau diagnosis and
`docs/experiments.md` (2026-07-02 entry) for why single-run, wrong-baseline,
or endpoint-only comparisons have already produced false conclusions once —
the standing rule at the bottom of this doc exists to not repeat that.

**On the "100% vs greedy" framing:** this is very likely the wrong target,
not just an ambitious one. Two structural reasons:
1. Irreducible randomness (random disjoint 8-of-16 draft split 4/4, random
   coin draws from the bag, random initiative) means even an optimal policy
   has WR < 100%. The real ceiling is unknown — that's exactly what Step 0
   below measures.
2. `GreedyBot` (`src/services/bots/greedy_bot.py`) never initiates a tactic,
   never bolsters, never recruits, and attacks with the first legal option
   rather than the best one. Optimizing directly against it risks exploiting
   its specific blindness rather than building a general strategy. Greedy
   should stay a fixed ruler, not the training objective — self-play WR /
   Elo trend is the real strategy signal.

---

## Step 0 — Measure before changing anything else

**Status: done (2026-07-02).** Tool: `src/app/eval_bucketed.py`
(`python -m src.app.eval_bucketed --games 200`). Raw records:
`logs/eval_bucketed_20260702-1442.csv`.

We don't know whether the ~75% eval WR ceiling is bag/draft variance or a
real, fixable skill gap. Every step below is a guess without this answer.

- Bucketed eval by drafted composition (`docs/IDEAS.md` P5a): run 200+ eval
  games on the saved checkpoint (`data/warchest_ppo_20260702-1442.pth`),
  report win rate per composition (or composition class), not one aggregate.
- Loss autopsy: for lost games, log final base score, both compositions, and
  whether the losing side ever used a tactic verb.
- This answers: is the ceiling structural (bad drafts) or fixable (skill)?
  Which specific compositions are weak?

Difficulty: low-moderate.

### Findings (200 games, checkpoint `warchest_ppo_20260702-1442.pth`, vs GreedyBot)

- **WR = 0.890 ± 0.022** (95% CI ≈ ±0.043) — a much tighter estimate than the
  training-loop eval's 20-game, ±0.10-noise number.
- **No dominant bad matchup.** Per-unit-type WR swings (own composition and
  opponent composition both checked) top out around ±0.10–0.15, mostly within
  ~2 standard errors of each other at n≈40–60 per bucket. Losses are not
  concentrated on a few toxic compositions — they look roughly evenly spread.
  **Conclusion: the ~11% loss rate does not look like a draft-variance
  ceiling; it looks like a real, roughly uniform skill gap.**
- **Tactic usage is rare and, when it happens, correlates with losing more:**
  used a tactic in only 23/200 games (11.5%); WR with a tactic used = 0.696
  vs. 0.915 without. Two readings, not yet distinguished: tactics get reached
  for mostly in already-losing games (reverse causation), or the policy
  executes them poorly. Either way this is a direct, measured confirmation
  that the policy has **not** learned to use the tactical layer that
  distinguishes Warchest from a vanilla move/attack/control game — this is
  likely the concrete shape of "no legible strategy."
- Losses skew toward longer games (many 60–100+ turns vs. an overall
  `avg_turns` around 60–90) — losses look more like grinding attrition than
  fast blowouts.
- **The policy essentially never bolsters — 1 bolster action across 200 full
  games (0.5% of games).** Follow-up (`used_tactic` alone is blind to this:
  `extra_maneuvers_from_stack` triggers via the `'extra_maneuver'` pending
  state, not `TACTIC_VERB`, since it's a triggered attribute, not a named
  tactic). Checked specifically: in all 50/200 games where Berserker was
  drafted, its stack never once reached 2 — the chain was never even offered.
  This generalizes the tactic-underuse finding: it isn't just the tactic
  layer, the policy has also dropped a whole *normal* action (`BOLSTER_VERB`)
  from its repertoire, most likely because its payoff is purely delayed
  survival with no direct reward term, and entropy annealed to a near-zero
  floor (`entropy_coeff_final=0.003`) before random exploration ever surfaced
  a case where bolstering paid off.
- **Caveat on whether §9 (material PBRS) actually fixes this:** not
  guaranteed for every unit. For an *offensive* stack-chain unit (Berserker)
  the link is direct — higher stack → more hits per the closed-form derived
  for the threat planes (`hits(D) = max(0, S-D+1)`) → more `boxed(opp)`. For a
  purely defensive use (e.g. Knight's bolstered-attacker requirement).
  bolstering then dying anyway sends the *same total* coins to `boxed(me)`
  either way (one per hit taken) — it only pays off if survival meaningfully
  reduces the number of hits actually landed (attacker gives up / game ends
  first), not mechanically. So §9 is a plausible partial fix, not a
  guaranteed one — worth checking bolster-usage rate specifically as part of
  the Step 2 A/B, not just aggregate WR.
- **Revises Step 2's priority:** material PBRS (§9) still targets a real gap
  (nothing rewards the coin economy) and now has a second, more concrete
  target (bolster/stack-chain avoidance) beyond the original "tactic
  underuse" framing. Still open: whether tactic underuse specifically is
  reverse-causation (tactics reached for mostly when already behind) or poor
  execution — not yet distinguished.

Tool coverage note: `src/app/eval_bucketed.py` now also reports
`bolster_count` and, for stack-chain units specifically, whether their chain
was ever offered/accepted (`has_stack_chain_unit`, `chain_offered`,
`chain_used` per game, all present in `--out-csv` output).

---

## Step 1 — Reframe the target

**Status: done (2026-07-02).** Confirmed directly: `wr_vs_greedy` is tracked
for the user's own sense of agent strength, not as the training objective —
self-play pool WR / Elo trend remains the real strategy signal. No further
action needed; Step 0's bucketed/per-behavior numbers are exactly the
richer, non-single-number picture this step called for.

---

## Step 2 — Strategic reward signal: material PBRS

**Status: implemented 2026-07-03 (A/B still owed).** `phi_material = C_MAT *
(boxed(opp) - boxed(me))` with `C_MAT = 0.015`, wired into `ppo.py` via the new
`WarChestEnv.boxed_total(pid)` helper, measured at the base-shaping telescoping
points, and **annealed** together with the holding reward (see Step 3). Logged
as `score_material`. `ATTACK_REWARD` was kept, not subsumed — flagged for the
A/B (see `rewards.md` §9 note). Still to do: run the controlled A/B (same
`n_batches`, settled-phase distributions) and check **bolster/chain rate**
specifically, not just aggregate WR.

Current reward table (`warchest_env.py`) is entirely base-centric — nothing
rewards the coin/unit economy, which is where most of Warchest's skill
actually lives. This is the most direct lever toward a *legible* strategy,
as opposed to Step 0/1 which are about measurement and framing.

- Implement `docs/rewards.md` §9: `phi_material = C_MAT * (boxed(opp) -
  boxed(me))`, potential-based (policy-invariant, same telescoping trick as
  the existing base shaping). Keep `C_MAT` well below `SHAPING_C = 0.05`.
- A/B against the current baseline — same `n_batches`, correct baseline run,
  compare settled-phase distributions (see standing rule below). Do not
  stack this blind on top of other changes in the same run.

Difficulty: low to implement, but the measurement discipline matters more
than the code here.

---

## Step 3 — Reward hygiene

**Status: partially implemented 2026-07-03.**

- `holding_reward` — **addressed by annealing rather than removal**: it (and the
  material term) are now linearly decayed 1.0 → 0.1 over the first half of the
  run, then held at 0.1 (`ppo.py`, `_update_schedules`, `shaping_anneal`). This
  keeps the flip-loop tie-break it was added for while the critic is weak, and
  removes the non-PBRS distortion late in training where the policy crystallizes
  — the reversible middle path over outright removal (rationale: `decision.md`
  2026-07-03, and the `holding_reward` discussion in `rewards_improvements.md`
  Step 1). If the A/B shows the flip-loop returning late, keep a higher floor
  (e.g. 0.2) instead of decaying to 0.1.
- **Still not started:** re-test whether a small `CLAIM_BASE_REWARD` is safe now
  that pool snapshotting is much rarer than when it was zeroed out (the original
  circular-claim exploit relied on frequent near-identical pool opponents).
- Also part of the C17 hygiene bucket and **done**: the truncation reward is now
  base-diff-proportional (see Step 5).

Can be tested alongside Step 2 in the same A/B round if practical, but keep
each change individually attributable.

---

## Step 4/5 — Capacity: strengthen the densifier (critic first)

**Status: critic-only widening implemented 2026-07-03 (A/B still owed);
full-net width still untested.**

`docs/analysis_ppo_20260630.md` flagged `hidden_dim=64` as a possible
capacity bottleneck for a `[46,7,7]` board input with a 1875-way factored
action head. The threat-planes change (`docs/history.md` /
`docs/experiments.md` 2026-07-02) changed the *input*, not the width, and
showed no measurable effect either way.

Because the **critic is the real densifier** of the sparse terminal reward
(`decision.md` 2026-07-03; `rewards_improvements.md` Step 5), the width increase
was applied to the **critic alone** first: new hp `critic_hidden_dim = 128`
(policy left at `hidden_dim = 64`). This is safe — the critic's board encoder is
independent of the policy's during PPO rollout (`value_from_features` is unused
there) — and keeps the capacity A/B attributable to the critic rather than
conflating it with a wider policy.

- **Done:** `critic_hidden_dim` 64→128 (critic only).
- **Still open:** controlled A/B of the critic widening; and separately, whether
  widening the *policy* (`hidden_dim` 64→128) helps — untested, do last.

---

## Backlog — bundle with the next observation-schema change, not standalone

- **Draw-probability observation features** (bag dilution / draw efficiency,
  `p_soon`/`p_mean` per coin type — `docs/IDEAS.md`). Feature-only, not a
  reward: no fixed potential can pick the agent's own preferred-unit target
  without risking a penalty on a valid strategy. Deprioritized because: (a)
  self-assessed as a weak effect, and weak effects are exactly what the
  current ~0.10 eval-noise band cannot detect; (b) `GreedyBot` never
  recruits, so it doesn't punish bag dilution — this barely touches the
  "beat greedy" axis; (c) it's the same class of change as the threat planes
  ("hand the network a value it could in principle already compute from raw
  state"), and that class just produced a measured **no-effect** result. Add
  it opportunistically when another `OBS_VERSION` bump is already planned,
  so the retrain cost is shared rather than spent on this alone.

---

## Standing rule for every step above

No conclusion from a single run. Every A/B must:
- use the same `n_batches` for both sides (not "run until it looks done");
- use the *correct* baseline log — see `docs/experiments.md`'s 2026-07-02
  entry for how easy it is to grab the wrong one (an `n_batches=1000` run
  interrupted early looks superficially similar to a completed
  `n_batches=400` run, but is not a fair comparison);
- compare **distributions** over each run's settled phase (e.g. eval
  checkpoints from batch 200 on), not endpoints or peaks. The threat-planes
  result taught us directly: a ~0.3–1.15 pooled-std gap from a single run
  per side is noise, not signal, no matter how good the story sounds.
