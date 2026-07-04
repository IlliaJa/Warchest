# Ideas for improving training

Implemented history: `docs/history.md`.

---

## Open items

**Standing rule for every A/B below** (carried over from the retired `future_steps.md`): no
conclusion from a single run. Every A/B must use the same `n_batches` on both sides (not "run
until it looks done"), use the correct baseline log (an interrupted long run is not a fair
comparison against a completed short one — see `docs/experiments.md`'s 2026-07-02 entry for a
concrete case this bit us), and compare **distributions** over each run's settled phase (e.g.
eval checkpoints from batch 200 on), not endpoints or peaks — a ~0.3 pooled-std gap from a
single run per side is noise, not signal.

Current state: WR vs greedy ~49% (post C1–C7 fixes, run `ppo_20260527-191432`); vs *true*
greedy on the full base game, ~70-90% depending on run (`docs/experiments.md`). Target: keep
improving self-play Elo/WR, not a fixed number vs. greedy — `GreedyBot` never bolsters,
recruits, or initiates a tactic, so optimizing directly against it risks exploiting its
specific blindness rather than building a general strategy (`docs/rewards.md`).

### 1. Game-completeness — remaining Phase 5 work

The full base game (all 16 units + per-game disjoint drafting) is implemented; drafting was carried forward into Phase 3, so the old `full_game_plan.md` roadmap is retired (its history now lives in `docs/history.md`). These are the only pieces of the "Phase 5 / full game" spec that were **not** carried forward:

- **P5a. Variable-composition eval bucketing.** Eval currently reports a single aggregate WR across all randomly-drafted matchups. Add per-composition (or per-composition-bucket) WR reporting so we can see *which* unit sets the agent is weak/strong on instead of averaging over the ~1820 possible 4-unit matchups. **Done** — `src/app/eval_bucketed.py` (`docs/experiments.md`'s `ppo_20260702-1442` eval run). Difficulty: low-moderate.
- **P5b. Rulebook drafting mode (optional).** Setup currently assigns each player 4 random disjoint types. The real game uses a snake/alternating draft from a shared pool. Implement it as an optional setup mode for parity with tabletop games; not required for training. Difficulty: moderate.
- **P5c. Freeze `baseline_tactics` snapshot.** Freeze a strong policy as a named pool/eval anchor for the current (`OBS_VERSION=9`) schema generation, so future changes have a fixed comparison point instead of always comparing against whatever the last run happened to be. Difficulty: trivial.

### 2. Draw-probability observation features (bag dilution / draw efficiency)

**Goal.** Give the agent a legible signal for the coin-economy nuance that over-recruiting *dilutes* the units it actually wants to play. Hand size is fixed at `HAND_SIZE=3` (`_draw_hand`, `warchest_env.py:416-437`) and draws are uniform-without-replacement from the bag, so recruiting a coin you won't reliably draw grows the cycle **without** raising actions/round — it just lowers the chance of drawing the coin that matters. All the raw state (per-type bag/hand/discard) is already in the observation, but the agent must learn the division + reshuffle timing implicitly; these features hand it the answer.

**Decision: feature-only, no reward term.** A reward is the wrong tool here (see the discussion that produced this section):
- "smaller bag = better" is a **trap** — as PBRS it pays a positive pulse whenever coins leave the cycle, and the biggest exit is *your own coins getting boxed on unit death*, so it would reward losing material (directly contradicting the material PBRS term).
- "higher per-unit draw rate = better" is *also* wrong — a fully-recruited "4-of-each + Royal" (17-coin) bag scores `3·4/17 ≈ 0.71` per unit, **higher** than the 9-coin starting bag's `0.67`, yet is a bad bag (tempo wasted recruiting; no ability to draw a *specific* unit in pairs to chain/bolster).
- The real target — "reliably draw *the unit I've chosen to play next rounds*" — has a **policy-defined** target (which unit is "preferred" is the agent's call, and 1-unit play is valid only for rare comps / Berserker / Warrior Priest). A fixed potential must pick the target concentration and is wrong somewhere (Herfindahl → pushes monotype; fielded-average → fails to flag 4-of-each). So it belongs in the observation, letting the policy own the preference. This also respects the `ppo_20260630` over-shaping diagnosis (don't add dense heuristic reward).

**The two features — same quantity ("what share of my draws is type `t`") at two horizons.** Own-side only (`p_soon` needs the bag↔discard split, hidden for the opponent). Emit both as per-type vectors over `DECK` (Royal included in denominators; same layout as `bag_v`/`hand_v`).

1. **`p_soon[t]` — imminent draw share (position now).** Expected copies of `t` in the *next* hand, before any reshuffle, normalized to a share:
   ```python
   B = sum(bag)                          # current bag size
   if B >= HAND_SIZE:
       E_soon = HAND_SIZE * bag[t] / B   # == bag[t]/B share when bag is healthy
   else:                                 # bag empties mid-draw → one reshuffle
       rest, D = HAND_SIZE - B, sum(disc)   # disc = discard_faceup + discard_facedown
       E_soon = bag[t] + (rest * disc[t] / D if D > 0 else 0)
   p_soon[t] = E_soon / HAND_SIZE        # in [0,1]
   ```
   (Expectation = hypergeometric mean, so exact for the next hand without the full distribution.)

2. **`p_mean[t]` — steady-state draw share (structure).** Long-run share of the recirculating pool — the dilution/concentration signal recruiting moves:
   ```python
   recirc[t] = bag[t] + hand[t] + disc_faceup[t] + disc_facedown[t]
             = owned[t] - on_board[t] - boxed[t] - supply[t]   # same identity the coin-counter uses
   p_mean[t] = recirc[t] / sum(recirc)                          # in [0,1]
   ```

**The gap is the signal (no third feature).** Both are `[0,1]` shares, directly comparable per type: `p_soon > p_mean` = *loaded now, spend it*; `p_soon < p_mean` = *key coins stuck in the discard behind a reshuffle, can't rely on `t` this round*. Flat "4-of-each" reads as no type peaking on either horizon.

**Wiring / difficulty.** Two counter-sums + a divide per type in `generate_observation` — negligible cost. Schema change → **bump `OBS_VERSION`** (invalidates the current `OBS_VERSION=9` pool snapshots; retrain). Difficulty: low-moderate. Plan to A/B against a no-feature baseline so any gain is attributable.

**Priority: backlog, bundle with the next observation-schema change rather than standalone.** Deprioritized because: (a) self-assessed as a weak effect, and weak effects are exactly what the current ~0.10 eval-noise band cannot detect; (b) `GreedyBot` never recruits, so it doesn't punish bag dilution — this barely touches the "beat greedy" axis; (c) it's the same class of change as the threat planes ("hand the network a value it could in principle already compute from raw state"), and that class already produced a measured **no-effect** result (`docs/history.md` → "Threat/position-aware observation"). Add it opportunistically when another `OBS_VERSION` bump is already planned, so the retrain cost is shared rather than spent on this alone.

### Likelihood-weighted threat-plane magnitude

*(A variant of the shipped `E_opp_hand` feature — `docs/history.md` 2026-07-03 — not a duplicate.)*

The enemy threat planes gate opponent availability worst-case: a unit type contributes its full hit-count to a cell if the opponent holds **≥1** hidden coin of that type (`_threat_grids` `coin_gate`, `warchest_env.py:1593-1594`). This is correct for spatial *safety* — one Berserker coin they happen to hold is lethal even at low expected count, and you don't want the plane to average that tail away.

**The idea:** scale each unit's threat contribution by its **likelihood of being playable this round** — the shipped `E_opp_hand[t]` feature (`docs/observation_improvement.md`) instead of the binary `≥1` gate. The plane would then read *"how likely am I to actually be hit here,"* not *"could I possibly be hit here."*

**Why it's parked, not planned:** it understates exactly the tails that lose material. Worst-case planes + an expected-hand **global** scalar (the split shipped in `observation_improvement.md`) already give both signals — max for "where can I die," mean for "how loaded are they" — without diluting the spatial safety read. Only revisit likelihood-weighting the *planes* if the worst-case version makes the agent measurably too timid (over-bolstering, refusing good trades). If pursued, A/B against the worst-case planes directly; keep the two mutually exclusive within the threat-plane block.

### 3. A/B the 2026-07-03 reward + capacity bundle

Material PBRS, annealed holding + material shaping, the base-diff-proportional truncation reward, and the critic-only widening all shipped together on 2026-07-03 (`docs/history.md`). The controlled A/B against the pre-change baseline is still owed — a run was started same-day (`ppo_20260703-142941`) but its outcome wasn't known at time of writing. Since all four changes landed in one pass, a clean A/B can only attribute the *bundle*, not the individual pieces; if the result is ambiguous, consider ablating one term at a time (start with material PBRS alone, since it's the one targeting the specific measured gap below).

**What to check beyond aggregate WR:** bolster/stack-chain rate specifically (`eval_bucketed.py` already emits `bolster_count`, `chain_offered`, `chain_used`). The original motivation was a measured gap — the policy essentially never bolsters (1/200 games) and never triggers a Berserker stack chain (never reached in 50/200 games it was drafted) — so success means that rate moving, not just WR.

**Caveat.** Advantages are z-scored but returns are kept in original scale and fed through a return normalizer (`ppo.py`); adding `phi_material` and the anneal schedule shifts the return distribution. Re-verify the normalizer's running stats settle and the critic loss scale stays sane after the change, or a "no-effect" A/B result might actually be a normalization artifact rather than a real null result.

### 4. Re-test a small `CLAIM_BASE_REWARD`

`CLAIM_BASE_REWARD` was zeroed early on because a non-zero value caused a circular-claiming exploit under frequent near-identical pool opponents (`docs/rewards.md` §2). Pool snapshotting is now much rarer (`pool_snapshot_every=15` vs. the `snapshot_every=1`-era exploit), so the exploit's precondition may no longer hold. Worth a cheap re-test; keep the value small and watch training-loop score for the exploit's signature (reward far exceeding what a single win should pay) if it's reintroduced.

### 5. Widen the policy network too

The critic was widened alone first (`critic_hidden_dim=128`, `hidden_dim=64` unchanged) so any capacity gain would be attributable to the densifier specifically (`docs/decision.md`, 2026-07-03). Widening the *policy* the same way is untested. Do this **after** idea 3's A/B lands, so a policy-capacity gain isn't conflated with the reward-bundle change.

### 6. Unit / board-presence PBRS (coin-economy §10)

A softer companion to material PBRS: reward having units **deployed and alive on the board** (you can't claim or hold a base with an empty board). Full proposal, rationale, and failure modes (over-deploy risk, overlap with material shaping): `docs/rewards.md` § *Unrealized ideas*. Only worth pursuing if ideas 3–5 above still leave a measurable tempo gap — treat material PBRS as primary and this as a low-coefficient add-on, never a replacement.

### 7. GAE-λ sweep (reward-neutral densification)

Reward sparsity is fundamentally a *credit-assignment* problem, not only a reward-design one — tune the propagation, not just the terms. `lam=0.95` has never been swept. λ closer to 1 propagates the terminal reward further back with less bias (the TD-Gammon eligibility-trace lesson), which densifies the effective per-step signal **without** touching the reward or risking any of the distortion/exploit issues shaping changes carry. Cheap A/B; do independently of ideas 3–6 since it doesn't interact with the reward terms.

### 8. Tactic/bolster underuse may be an exploration problem, not (only) a reward one

Entropy is annealed to a near-zero floor (`entropy_coeff_final=0.003`) fairly early in training — plausibly *before* random exploration ever surfaced a case where bolstering or a named tactic paid off, meaning the behavior could have been dropped from the repertoire before any reward (including material PBRS) had a chance to reinforce it. Material PBRS only helps if the behavior is *sometimes sampled*. Two options, either decaying and kept tiny (the board is small — this is about *action* coverage, not spatial coverage):
- a higher, or again-annealed (rather than monotonic), entropy floor; or
- a small decaying **intrinsic/count-based bonus on rarely-used verbs** (`BOLSTER`, `TACTIC`) — the approach JP-DouZero used to break an analogous collaboration-behavior gap.

Do idea 9 first — if tactic usage turns out to be reverse-causation, this isn't the right fix.

### 9. Disambiguate tactic reverse-causation before acting on idea 8

"Tactics correlate with losing" (11.5% usage, WR 0.696 with vs 0.915 without, from the same 200-game eval that motivated material PBRS) has two readings that imply *opposite* responses: reached-for-only-when-already-behind (reverse causation — no fix needed) vs. executed poorly when reached for (an execution/exploration gap — ideas 6/8 are on target). Cheap disambiguator: log tactic usage **conditioned on base-lead at the time of use**. If usage clusters in already-behind states, it's reverse causation; if it's spread across lead states, it's execution. Difficulty: low (logging only, no training change).

### 10. Factor direction out of the move/attack spatial head

The verb head already groups all 6 move directions under one `V_MOVE` and all 6 attack directions under one `V_ATTACK` (`warchest_env.py:113-115`, `N_FACTORED_VERBS=11`) — `P(verb)` doesn't distinguish move-north from move-south. But the *within-verb* spatial logits still come from `policy_head`'s single `Conv2d(hidden_dim + GLOBAL_DIM → N_VERBS=32, kernel=1)` (`policy.py:106`), which gives each of the 12 move/attack directions its own independently-learned output channel — direction isn't a shared factor, so the network can't transfer "which way is good here" between moving and attacking.

**The idea:** add one more factorization level (verb → cell → direction) — collapse move/attack down to 2 spatial channels (or fold them into per-cell logits without a direction dimension) plus a small shared 6-way direction head reused by both verbs, analogous to how `verb_head` already shares parameters across directions at the top level.

**Why parked, not planned:** the channel-count saving is modest (32 → ~22), and the real cost is reworking `policy_head` from a monolithic conv into cell-logits + a conditional direction head, then re-deriving `_joint_log_probs`/masking for a third factorization level (legal directions differ between move and attack per cell — masking must stay consistent with the new hierarchy). Worth it only if move/attack direction choices turn out to share enough structure that sample efficiency is actually gated on it; no evidence either way yet. Difficulty: moderate.

### 11. Parallel rollout collection — remaining phases 4 & 5

Full design in `docs/parallel_rollouts.md`. **All phases 1-5 are now implemented** (P11a dynamic balancing via a shared atomic episode counter; P11b `overlap_collection` hides the rollout wall behind the GPU update). Remaining open work is validation + tuning, not implementation:

- **P11c. Real-config speed + learning-quality A/B.** Overlap adds 1-step off-policy staleness (behavior weights are one update behind), which interacts with the KL-skip (larger old→new gap → more skipped minibatches). Compare `overlap=True` vs `False` on **elo/wr trajectory**, not just wall-clock, before trusting it. Also confirm RAM headroom — overlap holds a second in-flight buffer (the box was at 87% RAM when this landed).
- **P11d. IPC via shared memory (if it ever bottlenecks).** Currently worker→main transfer is pickle-through-Queue (~55 MB/batch). Steady-state IPC is small (→0 when hidden by overlap); move board arrays to `multiprocessing.shared_memory` only if profiling shows it on the critical path.

Standing-rule reminder: measure over ≥10 batches per config, not a single batch — spawn/import startup (batch 1) and pool-phase (`p_pool→0.9`) opponent cost both skew early/late batches.

---

### Tier 4 — small / cosmetic

| # | Issue | Difficulty | Effect |
|---|---|---|---|
| C18 | `value_single` is called during `.train()` mode at rollout (`_collect_batch`, `ppo.py`); harmless today (no BN/Dropout) but if either is added, Dropout would produce stochastic noisy values during GAE and BatchNorm would compute statistics from a single sample — both silent bugs. Fix: `self._critic.eval()` before the rollout loop, `self._critic.train()` before the update. | trivial | none until BN/dropout |
| C20 | Eval runs 20 episodes — std error on a 0.49 WR estimate is ~5%. Bump to 50. | low | log readability |
| C21 | `EloTracker` updates after every eval game — noisy; consider a running average | low | log readability |

**C16** (`iter_minibatches` permutation): already correct — a fresh `np.random.permutation` is generated on each call, so minibatches are uncorrelated across PPO epochs. No action needed.

**C10** (shared encoders): obsolete. See `docs/rl_algorithms.md` → *Architecture: shared vs separate actor-critic encoders*.

**C17** (truncation reward step function): **done** — replaced with a base-diff-proportional value 2026-07-03. See `docs/history.md`.

**C22** (`score_deque` rolling mean): already satisfied — `_score_deque` has `maxlen = print_every * collect_episodes` (currently 640), so `score_main` already averages across the last ~10 batches, not just the current one. No action needed.

---

### Recommended next steps

C8 (LR decay) and C9 (clean true-greedy eval) are **done** — run `ppo_20260630-060400` uses the true greedy bot (`RANDOM_ACTION_PROB=0.0`) and reports an honest ~70% WR. That run's plateau (score/WR decoupling, flat entropy) drove the fixes now recorded in `docs/history.md`. Start at idea 3 above (the owed A/B for the 2026-07-03 bundle) for the current live plan.
