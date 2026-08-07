# Independent opponents — breaking the self-play collapse

**Written 2026-07-28.** Goal set by the user: build the *strongest possible* training
signal for the main policy, specifically to fix two standing blind spots — the policy
**never bolsters** and **doesn't use unit-specific tactics** — after a multi-round
expert-iteration (ExIt) run came back *weaker every round*.

This doc (a) roots-causes that collapse from the run logs, (b) argues the fix is not "a
stronger single bot" but a set of **independent (non-policy-derived) opponents**, and
(c) lays out a measurement-first plan. It sits alongside [IDEAS.md](IDEAS.md) (the open-item
list — this doc's §7 is the live sequencing it defers to) and [bots.md](bots.md) (the bot
catalogue + prior ExIt diagnosis).

---

## TL;DR

- **The ExIt loop is a closed loop with no external reference.** The teacher (`PuctBot`)
  is the student (policy priors + a self-distilled critic) plus ~100–300 simulations of
  search. Measured **policy/search agreement ≈ 0.94–0.95** every round: the search
  reproduces the policy's own top move ~19 times in 20, so distillation re-fits the
  policy to a near-copy of itself. Errors and idiosyncrasies compound; the field goes
  **non-transitive** (cycles). This is exactly the user's intuition — *the model closed
  in on itself.*
- **The Dirichlet-noise fix (bots.md) was necessary but not sufficient.** Visit entropy
  is no longer inflating (~0.66 nats, matching the policy's own), yet the loop still
  degenerates — via a *different* mechanism (no independent signal), not the old one.
- **"Never bolsters / no tactics" is a coverage problem, not only an algorithm problem.**
  *No* bot in the repo bolsters — not `GreedyBot`, not `SimGreedyBot`, not `LookaheadBot`
  (bolstering is a tempo loss a shallow search won't pick; the `rich_eval` terms that
  would reward it are documented net-harmful). Self-play can't teach a behaviour neither
  side ever plays, and neither can a bot that never punishes its absence.
- **The highest-leverage work is therefore a *portfolio* of independent opponents that
  (a) exhibit the neglected behaviours by construction and (b) punish their absence** —
  wired into **both** the PPO pool *and* ExIt data-gen — plus keeping `PuctBot` as the
  strong teacher. Making `PuctBot` stronger in isolation does not fix the bottleneck,
  which is *signal independence*, not teacher strength.

---

## 1. Evidence: the collapse, quantified

Latest run `logs/exit_20260727-201848.log` — `loop --rounds 30 --games 200`, v11
checkpoints, `dirichlet_frac=0.03` (i.e. the bots.md fix already applied), ~11 h of
compute. Final post-round gauntlet (base policy + all 30 distilled policies, raw, no
search):

```
Bradley-Terry ranking (Elo-scaled, field mean = 1000):
  ckpt_20260727-0506[v11]   1156.1   <- BASE (pre-ExIt), strongest in the field
       round0_policy[v11]   1089.0
       round1_policy[v11]   1047.4
       round2_policy[v11]   1018.0
       ...
      round16_policy[v11]    956.2   <- late rounds cluster at the bottom
      round24_policy[v11]    961.7
      round20_policy[v11]    962.8
Intransitive-triple fraction: 0.110  (cycles present)
```

Three facts, each independently damning:

1. **The base policy beats every one of its 30 descendants.** ExIt made the policy
   monotonically weaker for the first several rounds, then plateaued *below* the start
   with the late rounds forming a bottom cluster. This is a regression, not a plateau.
2. **policy/search agreement ≈ 0.94–0.95 in every outcome-mode round** (per-round `gen`
   line). The search's most-visited move equals the raw policy's argmax ~19 times in 20.
   The distillation target is, to first order, *the policy itself*. Round 0 (still in
   `shaped` mode, seeded from the good PPO critic) was the exception — agreement 0.77
   before distill — and round 0 is the **best** ExIt round (1089). The moment the loop
   switched to the self-distilled z-critic (`outcome` mode, rounds ≥1), agreement jumped
   to ~0.94 and strength fell away.
3. **Cycles are present (0.11).** Pure self-play is drifting along a rock-paper-scissors
   manifold, not climbing a strength gradient — precisely the non-transitivity risk
   [IDEAS.md](IDEAS.md)'s guiding principle flagged.

Supporting signals:

- **The z-critic doesn't accumulate knowledge across rounds.** Each round the *previous*
  round's critic scores held-out MSE ≈ 1.2–1.55 on the new data (near-max error for a
  target in `[-1,1]`), then re-fits down to ≈ 0.03–0.05 within the round. It memorises
  each round's narrow self-play distribution and generalises to the next round's barely
  at all — churn, not learning.
- **Visit entropy is *not* the culprit this time** (~0.63–0.71 nats, flat, matching the
  pre-distill policy). The bots.md flattening mechanism is fixed; this collapse is a
  separate failure.

---

## 2. Diagnosis: why pure self-play ExIt collapses here

Five reinforcing mechanisms, all variations on *no external reference*:

1. **Teacher ≡ student.** `PuctBot` = policy priors + critic value over a tree. In
   `outcome` mode the leaf is `(1-f)·critic + f·heur` with the *distilled* z-critic, and
   the priors are the policy being distilled. At ~100–300 sims/move on a 1875-wide action
   space with `max_branching=8`, the visit distribution is dominated by the prior — hence
   agreement ≈ 0.95. The "expert" is barely more than the network it seeds. There is no
   gradient pulling the policy toward *correct* play, only toward *self-consistent* play.

2. **The search is too shallow to out-vote its own prior.** AlphaZero affords a noisy
   prior and a weak critic because ~800 sims/move let accumulated `Q` dominate the PUCT
   bonus. At this budget it can't; the prior stays the dominant term for the whole search
   (the same arithmetic that made 25%-Dirichlet catastrophic in bots.md — but now it bites
   even with the prior *un*-perturbed). Raising `--time-budget` 5× earlier "did not fix it"
   for the same reason.

3. **The z-critic is retrained from a narrow, shifting distribution each round.** ~200
   self-play games ≈ 16k samples, all drawn from the current policy's own play. A critic
   fit to that overfits (MSE 0.04) and doesn't transfer (next-round MSE 1.4). A bad value
   → a bad search → bad targets → a worse policy → an even narrower distribution. Warm-
   starting each round from the previous (already-degraded) nets compounds it.

4. **State-distribution collapse.** With only self-play, the visited states narrow to what
   the current policy tends to reach. Situations that *require* bolstering (a hanging stack
   about to be boxed) or a unit tactic (Archer/Lancer's only attack path) rarely arise,
   because neither side ever creates them. The policy is never *shown* a target that says
   "bolster here" — so it can't learn to. This is why the specific blind spots persist.

5. **No win/loss anchor.** Progress is *measured* by the post-round gauntlet, but nothing
   in the *training signal* is anchored to a fixed opponent. Combined with (4) and the
   non-transitivity in §1, the loop has no force preventing it from wandering.

### The same disease in the PPO finetune pool

This isn't only an ExIt problem. `ppo.py`'s **finetune** opponent weights are:

```
p_pool = 0.75   p_lookahead_critic = 0.25   p_greedy = 0.00   p_random = 0.00   p_puct = 0.00
```

So in finetune the policy plays **~100% policy-derived opponents**: 75% frozen snapshots
of *itself*, 25% `lookahead_critic` (a beam guided by the *trained critic*). The only
genuinely independent opponents (greedy, random) are weighted to **zero**. Whatever the
policy already ignores, nothing in finetune ever punishes. The user's "it only plays
itself" is literally true of the finetune schedule, not just ExIt.

---

## 3. Reframing the ask: what "strongest bot" should mean

The literal request — "make the strongest possible bot" — needs one correction to be the
right lever:

> The bottleneck is not the strength of the *teacher*. It is the **independence and
> behavioural coverage** of the *opponents the policy is exposed to*.

A single very strong bot that plays like the policy (which `PuctBot` in `outcome` mode
effectively is) adds nothing the policy doesn't already contain. Two roles must be filled
by *different* bots:

| Role | Requirement | Best current candidate | Gap |
|---|---|---|---|
| **Teacher** (generates strong targets) | Search that meaningfully diverges from the prior | `PuctBot` (`shaped`, good critic) | In `outcome` mode it barely diverges (agreement 0.95) |
| **Sparring / exploiter** (creates diverse, punishing states) | *Independent* of the policy; exhibits + punishes the neglected mechanics | `LookaheadBot` (strongest independent), but **it doesn't bolster either** | No bot injects bolster/tactic behaviour or states |

So "the strongest bot" resolves into **two** builds: keep/So-improve the teacher, and —
the higher-EV, cheaper move first — build a **panel of independent opponents** that do
what no existing bot does.

### What "independent" must mean here

1. **Not policy-derived.** No shared weights, no shared critic. A hand-coded strategy or a
   separately-trained net. (This is what rules `pool`, `lookahead_critic`, `puct` out as
   "independent" — they all read the policy or its critic.)
2. **Behaviourally diverse & legible.** Each opponent should *reliably* play a distinct
   archetype the policy under-explores: builds tall bolstered stacks; opens with an
   initiative rush; leans on a specific unit's tactic (Archer ranged, Lancer charge,
   Cavalry `move_to`, Berserker chain).
3. **Punishing, not merely present.** The point of an opponent that bolsters is that it
   *wins the stacks it bolsters*, so the policy pays for not contesting them. A behaviour
   the opponent shows but never benefits from teaches nothing.

Note these need **not** be objectively top-tier. In league/PSRO terms they are
*exploiters*: their value is coverage and pressure, not raw Elo. (AlphaStar's "main
exploiters" are deliberately narrow.)

---

## 4. The plan (measurement-first, phased)

### Phase 0 — confirm the mechanism *(largely done by §1)*

The 30-round log already gives the two diagnostic reads this phase existed to get:
- policy/search **agreement ≈ 0.95** ⇒ teacher ≈ student (mechanism 1–2).
- **cycles present** + base-on-top ⇒ non-transitive drift, not climbing (mechanism 5).

One cheap addition worth logging before building: **per-behaviour rates in self-play**
(bolster / recruit / tactic / claim_initiative fractions per game, already computed by
`SimGreedyBot.usage` and `eval_bucketed.py`). Expect ~0 bolster — this becomes the
baseline the whole effort is judged against.

### Phase 1 — a scripted archetype/exploiter panel *(cheapest, highest EV)*

Build a small `bots/exploiters/` family (sketched in [IDEAS.md](IDEAS.md) § *Method*),
each a hand-coded `Bot` (≈30–80 lines). Priority order by the blind spots:

1. **`BolsterBrawler`** — deploys a couple of strong units, then *bolsters them to tall
   stacks* and fights only when ahead on stack height. Directly manufactures the "face a
   bolstered stack you must contest" states self-play never produces. This is the single
   most important entrant.
2. **`TacticBot(unit)`** — one per key tactic unit (Archer, Lancer, Cavalry, Berserker):
   deploy that unit and use its tactic as the primary win condition, so the policy must
   learn to respect ranged/charge/chain threats.
3. **`InitiativeRush`** — claims initiative every round and races bases, punishing a
   passive policy on tempo.
4. **`RecruitEconomy`** — recruits aggressively to out-scale late, testing the long-game
   the policy currently coasts through.

Design notes:
- Prefer the **`act(obs)`** interface (like `GreedyBot`) where a strategy needs no
  forward-sim; fall back to **`act(env)`** only when it must simulate. Keep them *fast* —
  they run in the rollout hot path.
- They don't need to *win* the field; they need to be *distinct and punishing*. Validate
  each in the gauntlet: it should beat `random` ~100%, lose to the policy overall, but
  **win a non-trivial minority via its archetype** (e.g. `BolsterBrawler` wins the games
  that come down to a stack fight).

### Phase 2 — wire the panel into training *(the payoff step)*

Two independent integration points; **both** matter (a bot the policy never faces in the
loop that actually trains it is decoration):

- **PPO opponent pool** (`opponent_pool.py` + `ppo.py`):
  - Add a builder + weight per exploiter, mirroring `_get_puct_bot`/`_get_lookahead_bot`.
  - **Fix the finetune schedule** so it is *not* ~100% policy-derived: give the exploiter
    panel a real slice (e.g. 20–30% total) even in finetune. This is the direct antidote
    to §2's "plays only itself."
  - **Critic conditioning:** the one-hot is only 3-way (`{random, greedy, pool}`) and
    expanding it changes the critic's input dim → breaks every existing critic checkpoint.
    So route scripted heuristic exploiters onto the **`greedy`** one-hot slot (they *are*
    heuristic bots) via `OPP_ONEHOT_SLOT`, exactly as `lookahead_critic`/`puct` reuse the
    `pool` slot. No schema change, no retrain-from-scratch.
  - If an exploiter needs the live env (`act(env)`), add its label to `_SEARCH_OPP_TYPES`
    in `rollout_core.py` so `_opponent_env_action` routes it correctly.
- **ExIt data-gen** (`expert_iteration.py` + `selfplay_collector.py`) — **break the closed
  loop**:
  - Generalise `play_selfplay_game` into a *mixed-opponent* variant: `PuctBot` (teacher,
    on the learner's side) vs. an exploiter on the other side, recording **only the
    teacher's decision nodes** (its visit distribution → policy target) while the exploiter
    plays its side via its own `act`. `z` labelling stays per-mover but only learner
    samples are appended.
  - This gives the distillation targets over the *exploiter-driven* state distribution —
    states the closed self-play loop never reaches (mechanism 4) — with an external,
    non-policy-derived source of structure (mechanisms 1, 5). Mix it with regular self-play
    (e.g. 50/50) so the loop keeps its strong-vs-strong data too.

### Phase 3 — a genuinely strong *independent* search bot *(the "strongest bot" ask, higher effort)*

The scripted panel injects behaviours but caps out in raw strength. To also raise the
ceiling with an independent bot that *correctly values* bolstering and tactics:

- The lever is **not** re-enabling `rich_eval` (measured net-harmful — a shallow leaf
  can't cash in long-horizon assets). It's **depth / quiescence**: `LookaheadBot` with a
  quiescence extension that resolves stack fights to the end, so a bolster that *saves a
  stack two plies later* is actually seen and valued. That converts "bolster = tempo loss"
  into "bolster = a stack that survives," which is when a search will pick it.
- Alternatively, **behaviour-clone the exploiters + `LookaheadBot` into one fast net**
  (the distillation dry-run [IDEAS.md](IDEAS.md) #8 already wants) — a cheap, strong,
  policy-independent pool opponent that runs at network speed instead of search speed.

This phase is optional relative to Phases 1–2 and should be sequenced *after* them: if the
scripted panel + fixed schedules already move the bolster/tactic metrics and reverse the
ExIt regression, the extra strength here is polish, not the fix.

### Phase 4 — anchor & measure (keep it from silently re-collapsing)

- **Standing exploiter panel in the gauntlet.** Every exploiter becomes a permanent
  gauntlet entrant. WR-vs-each-exploiter is a per-weakness regression test that never
  saturates the way WR-vs-greedy did.
- **Per-behaviour metrics as first-class training signals**, not just win rate: bolster
  rate, tactic rate, recruit rate, chain rate — logged every eval (mostly already emitted
  by `eval_bucketed.py`). "Did ExIt/PPO make the policy stronger" must be read as *both*
  gauntlet Elo *and* these rates moving off ~0.
- **Guard for the ExIt entropy/agreement inversion every round** (the warning in
  `expert_iteration.py` already fires on entropy; add the symmetric **agreement > 0.9 ⇒
  teacher isn't teaching** warning). A round where agreement ≥ 0.9 is a round that will
  regress — stop or inject more exploiter data.
- **Exploitability probe** ([IDEAS.md](IDEAS.md) #4) reuses all of the above: each scripted
  exploiter is a free best-response sample.

---

## 5. Concrete integration map (files & seams)

| Change | Where | Notes |
|---|---|---|
| New exploiter bots | `src/services/bots/exploiters/*.py` + export in `bots/__init__.py` | Prefer `act(obs)`; keep fast |
| Gauntlet registration | `services/gauntlet.py` `build_agent` + `app/gauntlet.py --bots` | So they're permanent yardsticks |
| Pool builders + weights | `opponent_pool.py` (`sample`, `set_weights`, `__init__`) | Mirror `_get_puct_bot` lazy build |
| Finetune schedule fix | `ppo.py` `p_*_finetune` block (~L935) | Give exploiters ≥20–30% even in finetune |
| Critic one-hot slot | `rollout_core.py` `OPP_ONEHOT_SLOT` | Reuse `greedy` slot — **do not** widen `OPP_TYPE_IDX` (breaks checkpoints) |
| Env-routing (if `act(env)`) | `rollout_core.py` `_SEARCH_OPP_TYPES` | Only for forward-sim exploiters |
| Mixed-opponent ExIt gen | `expert_iteration.py` `play_selfplay_game`; `selfplay_collector.py` | Record only the teacher's nodes; mix with self-play |
| Agreement guard | `expert_iteration.py` (next to the entropy warning) | Warn/abort when agreement ≥ 0.9 |

---

## 6. Risks & what *not* to do

- **Don't widen the critic's opponent one-hot** to give exploiters distinct identities — it
  changes the critic input dim and invalidates every saved critic. Reuse the `greedy`
  slot (they're heuristic bots); losing the critic's ability to tell `BolsterBrawler` from
  `greedy` is acceptable and matches how the search bots already share `pool`.
- **Don't drop the strong teacher.** Flooding the pool/ExIt with weak exploiters would just
  teach the policy to beat weak bots (re-saturating the yardstick from the other side).
  Keep `PuctBot`(`shaped`, good critic) and pool snapshots in the mix; exploiters are an
  *addition* (~20–40% weight), not a replacement.
- **Don't re-enable `rich_eval`** to make search bots bolster — it's a documented net loss.
  Get bolstering from *scripted* bots (behaviour by construction) and from *quiescence*
  (Phase 3), not from richer shallow-leaf terms.
- **Exploiters must actually punish.** A `BolsterBrawler` that bolsters but loses the stack
  fights teaches the policy the wrong lesson (that bolstering is bad). Validate each
  archetype *wins its intended game type* before pooling it.
- **Keep ExIt's z-critic honest.** Even with mixed data, the round-to-round MSE-1.4 churn
  (§1) says the z-critic is under-fit to a shifting distribution. Consider accumulating a
  *replay buffer across rounds* rather than training each round only on its own 200 games,
  and/or not warm-starting the critic from the previous (possibly-degraded) round.

---

## 7. Recommended sequencing

1. **Phase 1** — `BolsterBrawler` + one `TacticBot` first (smallest slice that targets the
   two named blind spots). Register in the gauntlet; confirm they punish.
2. **Phase 2** — pool wiring + **finetune-schedule fix** (PPO), then mixed-opponent ExIt
   gen. Re-run a short PPO finetune *and* a short ExIt loop; read bolster/tactic rates and
   gauntlet Elo.
3. **Phase 4 guards** — agreement warning + per-behaviour metrics, so the next long run
   self-reports collapse instead of burning 11 h to discover it.
4. **Phase 3** — quiescence `LookaheadBot` / distilled fast bot, only if more ceiling is
   wanted after 1–2 land.

The success test is concrete: **the ExIt regression in §1 reverses (a round beats base),
and the policy's bolster/tactic rates move off ~0** — both, not either alone.

---

## Phase-1 result: `BolsterBot` (measured 2026-07-30)

First concrete exploiter built: `BolsterBot` (`src/services/bots/bolster_bot.py`), a
Berserker + Warrior-Priest archetype, evaluated with a *forced* draft via
`src/app/eval_bolster.py` (registered in the gauntlet as `bolster`). Goal set by the
user: beat `lookahead` ≥70% at a training-viable budget (≤0.5s/move, ideally 0.3).

**Three implementations tried, each measured vs `lookahead` (its default 0.1s):**

| Approach | Result | Read |
|---|---|---|
| Durability **leaf term** (reward alive+bolstered key stacks) | 30% (w=0.20) → 25% (w=0.35), n=40 | Reproduces the `rich_eval` failure (docs/bots.md): a depth-bounded search trades tempo for stack height and loses. **Stronger weight = worse.** |
| **Scripted controller** driving the key units (build Berserker, then kill/claim) | build_target=1 (rush) 43%, build_target=2 51%, n=60 @0.3 | Naive scripting of the key units ≈ neutral-to-harmful vs letting the search move them; a scripted advance walks the Berserker into danger. |
| **Pure `LookaheadBot` + forced draft** (archetype off), 3× budget | **43%, n=60 @0.3** | The composition itself is a slight *disadvantage*: strong general play at 3× the opponent's budget still loses. |

**Two control measurements that settle the cause:**

- `lookahead`@0.3 vs `lookahead`@0.1 (random drafts): **42% (10/24)** — **more search budget
  does not help in this game.** Alpha-beta saturates by ~depth 5 at 0.1s; extra depth
  adds nothing (echoes bots.md's "quality is not monotonic in depth"). So budget is *not*
  a lever, and the noisy n=40 readings that looked like 60% were sampling noise.
- `BolsterBot`@0.3 vs the trained **policy** (`warchest_ppo_20260727-0506`): **42.5%** — it
  *loses* to the current policy (consistent with the gauntlet's policy > `lookahead`).

**Conclusion — a genuine negative result.** The Berserker/Priest **bolster archetype is
not competitive with `lookahead` at a training-viable budget** (~43-51%, not 70%), and
no injection method changes that. Root cause, confirmed rather than assumed: this env is
a **fast 6-base race** (games end ~round 11), and **bolstering is a tempo loss the race
punishes** — the exact reason §2/§3 give for why *nothing* bolsters. Building a Berserker
costs 3-4 tempo-turns while `lookahead` rushes bases; traces show the built Berserker
sitting on its home base as the opponent reaches 6 bases. The composition is a mild
handicap, not an edge, against a base-rusher — and search depth can't buy the win back.

**What this means for the plan.** The `BolsterBot` value is **behavioural, not raw
strength** (as §3 warned: exploiters are for coverage/pressure, not Elo). It *does*
bolster and use the Berserker chain / Priest bonus action, so it still injects the states
self-play never produces — but it is ~`lookahead`-strength and currently loses to the
policy, so its worth as a *teacher* rests on that coverage, and on facing the policy
earlier in training when the policy is weaker. **Open question for a strong independent
opponent:** beating `lookahead` ≥70% needs a better *search/heuristic* (depth and this
archetype both proven not to be levers) or a fundamentally different exploit — not this
composition. Whether a Priest-tempo/initiative engine or a different archetype fares
better is untested.
