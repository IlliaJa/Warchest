# Next steps — strategic plan

Written 2026-07-04, after the run logged in `docs/experiments.md` (2026-07-04) reached
**WR vs greedy ≈ 100%** and self-play showed the agent beating its own predecessors.

This document is the *live* strategic plan (the "why" and the ordering). Concrete numbered
implementation items live in `docs/IDEAS.md`; per-run results in `docs/experiments.md`.

---

## The core problem: the measurement instruments are saturated

Two facts define where we are:

1. **WR vs `GreedyBot` = 100%.** `GreedyBot` is a myopic 1-ply priority list
   (`attack → control → move → deploy → pass`, `greedy_bot.py`). It **never bolsters, recruits,
   initiates a tactic, values stack HP, or looks ahead**. Beating it 100% now says essentially
   nothing about absolute strength — this yardstick is exhausted. `eval_bucketed.py`, being also
   vs greedy, is saturating for the same reason.
2. **"Beats predecessors" is a *relative* signal.** Self-play / opponent-pool improvement is
   necessary but not sufficient for real strength: strategy spaces can be **non-transitive**
   (rock-paper-scissors), where every generation beats the last without any of them being
   objectively strong. `wr_vs_pool_train ≈ 0.7` (not the ~0.5 self-play equilibrium) already
   hints the pool is lagging behind the policy.

Compounding this: no atomic A/B tests were run for the last ~2 days of changes (a reasonable
time trade — each change was logically justified and smoke-tested on ~100 episodes). But that
means we have **neither per-change attribution nor a discriminating aggregate signal**. Fixing
measurement is therefore doubly important.

### Guiding principle

> **Restore a trustworthy yardstick before training longer or shipping big features.**
> Do not optimize, or ship online, against saturated instruments.

Every option below is ranked against this principle.

---

## Prioritized roadmap

| # | Step | Why now / why not | Effort |
|---|---|---|---|
| **1** | **Round-robin gauntlet + transitivity check + richer `eval_bucketed`** | Restores a discriminating, absolute-ish signal and tells us whether "beats predecessors" is real or cyclic. Highest info per hour. | med |
| **2** | **Raise the bar: stronger opponent** (economy-aware greedy → shallow lookahead / MCTS) | Re-establishes an objective WR *with headroom*, and doubles as a training opponent that forces neglected mechanics. | med–high |
| **3** | **Exploitability metric (+ optional PSRO)** — the Nash direction | The only measure of *unexploitability*; the principled answer to non-transitivity. Reuses the round-robin machinery. | med |
| **4** | **Long run (1000–1500 batches)** | Low value *until* there's a harder opponent/curriculum — otherwise it optimizes a saturated signal (the 2026-06-30 run already showed a ~10 h plateau). | low |
| **5** | **Online play vs humans** | The ultimate absolute test, but a big feature with real risks — do it last, once we trust the agent has no glaring blind spots. | high |

The rest of this document details each, plus the design decisions worked out for the
round-robin and the Nash direction.

---

## Status update — 2026-07-08

Steps 1, 2 (mostly), and 4 are done. First real round-robin result, 5 agents, `app/gauntlet.py`:

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

- **`ckpt_20260707-0026` (the 1500-batch long run, Step 4) is the strongest agent in the
  field, beating everything including both lookahead bots.** This is the first time "beats
  predecessors" has been checked against a genuinely non-saturated, non-self-referential
  yardstick and it held up — the long run's Elo/WR gain vs greedy was real progress, not an
  artifact of optimizing a saturated signal. That risk was live (`wr_greedy` in the run's own
  log saturates to 1.0 by batch ~1200, the same shape as the 2026-06-30 plateau run) but this
  result says it wasn't fatal here.
- **`lookahead` (plain alpha-beta search) clearly beats `lookahead_critic` (critic-guided beam
  search), 80/20**, and sits ahead of the older checkpoint. `lookahead_critic` is in fact the
  *weakest* non-greedy agent in the field (909 Elo, barely above `greedy`'s 792) — surprising
  given it was built specifically to search deeper by replacing hand-tuned heuristics with a
  trained value function. Plausible causes, not yet diagnosed: the fixed `beam_width=5` cutting
  too much of the tree, the critic's value estimates degrading on off-distribution states a
  beam search reaches but on-policy rollout collection never visits, or simply needing the same
  kind of tuning pass `lookahead_bot_plan.md` documents for `LookaheadBot`. Worth resolving
  before leaning on this bot for anything further (training opponent, exploitability probe).
- **No cycles found by hand-checking triples in this 5-agent field** — the ranking looks
  transitive. Re-check with the gauntlet's own intransitive-triple metric as the field grows;
  5 agents is too small to trust this qualitatively for long.
- **Not yet done:** this run (`ppo_20260706-194732` → `warchest_ppo_20260707-0026.pth`) and this
  gauntlet result are not recorded in `docs/experiments.md` yet. `eval_bucketed.py` is still
  hardwired to `GreedyBot` — pointing it at `lookahead` (the stronger of the two search bots)
  would make its per-composition/bolster/tactic diagnostics discriminating again, per the
  still-open half of Step 1 below.

**Effect on priority order:** `LookaheadBot`, not `LookaheadCriticBot`, is the one worth wiring
into `opponent_pool.py` as a training opponent first (Step 2, below) — it's the stronger of the
two and close enough to the trained policy in Elo (1070 vs 1177) to be a useful sparring
partner rather than either a pushover or an undiagnosed regression. Revisit
`LookaheadCriticBot` once its underperformance is understood.

---

## Status update — 2026-07-11 — analysis + new directions

Written after `LookaheadCriticBot`'s underperformance was diagnosed and fixed (missing critic
denormalization; see `docs/bots.md`), which was the blocker the 2026-07-08 note above said to
wait on before sequencing the Nash direction. This section supersedes that gate and adds ideas
for what comes after the currently-planned Step 2 remainder.

### The 2026-07-08 gauntlet ranking is now stale

`lookahead_critic` at 909 Elo (weakest non-greedy agent) was measured *before* the
denormalization fix. Post-fix, `docs/bots.md` reports 68-78% WR vs `lookahead` (up from 30-35%)
— the field has likely reordered, possibly putting the critic bot ahead of `lookahead` and
closer to the policy checkpoint. **Re-run the round-robin gauntlet before making any further
strategic reads off the 2026-07-08 numbers** — this includes the Step 1/3 status notes above
that deferred sequencing until the critic bot was "understood": it now is.

### Nash direction — scope it as measurement, not as the engine

The instinct that Nash "seems good but may not get the model to a new level" is correct, and
the reasoning is worth making explicit:

- **Exploitability measurement is cheap now and highly informative.** Two probes, cheapest
  first:
  1. *Search best-response proxy* (hours): `LookaheadCriticBot` with `see_opponent_hand=True`
     vs. the frozen best checkpoint — already sketched in the 2026-07-08 note above, now
     unblocked by the denormalization fix.
  2. *RL best response* (a day or two): train a fresh PPO run whose opponent pool contains
     **only** the frozen checkpoint. The WR trajectory over that run *is* the exploitability
     curve. No new infra — it's an `opponent_pool.py` config, reusing the existing training
     stack end to end.
- **The probe result is a decision procedure, not a score.** If the best response reaches
  ~80-90% WR, the agent has a real exploitable hole → PSRO-style pool weighting (meta-Nash over
  the Bradley-Terry matrix, nearly free on top of the gauntlet) is justified, and the
  best-response policy itself becomes a new pool opponent — that's PSRO iteration 1. If the
  best response caps at ~55-60%, the policy is already hard to exploit *within this policy
  class*, and further Nash investment (R-NaD, deeper PSRO) buys robustness that can't currently
  be measured as improvement, not raw strength — stop there.
- **Why Nash alone is unlikely to be "the new level":** equilibrium-seeking optimizes the worst
  case against the current strategy population. At this compute scale the binding constraint is
  more likely elsewhere — a `hidden_dim=64` policy with no search at inference (IDEAS.md #5).
  Nash refinement polishes what exists; it doesn't grow model capacity or add search. Treat
  exploitability as the thermometer, not the heater.

### New idea — search-augmented policy (the AlphaZero direction)

Not previously in this plan. The pieces already exist and aren't connected:

1. **Policy-prior search at inference (cheap, do soon).** `LookaheadCriticBot` currently orders
   and prunes moves with `LookaheadBot`'s hand-tuned `_ordering_key`. The trained *policy* is the
   strongest agent in the current field and is exactly a learned move-orderer — replace the
   ordering key with policy priors (PUCT-style: policy prior × critic value for selection), keep
   the existing 0.7/0.3 critic/heuristic leaf blend. This should produce an agent strictly
   stronger than the raw policy **with zero retraining** — the same effect that makes AlphaZero's
   search worth ~400+ Elo over its bare network. Becomes the new gauntlet ceiling, the natural
   best-response opponent for the exploitability probe above, and the candidate to field online
   (see Step 5 below) instead of the raw policy.
2. **Expert iteration (medium-term, the more likely route to a genuine step up).** Once
   policy+critic+search beats the raw policy, close the loop: search's chosen moves become new
   policy training targets, game outcomes become value targets, retrain, repeat (ExIt/
   AlphaZero-style). This is a more proven route past a PPO self-play plateau in a board game
   than further reward/observation tuning. Caveat: Warchest has hidden information (bag/hand
   composition) and AlphaZero assumes perfect information; both existing search bots already
   handle this via single-determinization sampling, and determinized search's practical strength
   vs. its theoretical impurity (DeepNash's core objection) is exactly what the exploitability
   probe above will measure empirically.
3. **Distill `LookaheadBot` into a fast network.** The planned `Bot`/`GauntletAgent` merge makes
   `LookaheadBot` usable as a training opponent, but at ~0.1s/move it will dominate rollout
   wall-clock once wired into `opponent_pool.py`. Behavior-clone it into a small net (supervised,
   from a few thousand self-play games) and pool the fast clone instead. Doubles as a dry run of
   the distillation machinery expert iteration (above) needs regardless.

### Online play — reframe, don't just defer

Two refinements to the existing "defer to last" call, not a reversal of it:

- **"Plays like a newbie" has a known, specific shape, not a vague one.** The 200-game eval
  already on record: the policy essentially never bolsters, rarely recruits meaningfully, never
  triggers a stack chain. IDEAS.md #9 (log tactic usage conditioned on base-lead at time of use —
  logging-only, no training change) is nearly free and should run before investing in the
  exploration fixes (#8) that assume execution-gap rather than reverse-causation.
- **The agent that goes online should be the search-augmented one, not the raw policy** — it
  directly papers over the tactical blunders that read as "newbie" to a human opponent, and
  costs nothing extra to prepare given the search-augmented-policy idea above.
- **Cheapest absolute-strength signal available right now: play it yourself, locally, before any
  Playwright/web-agent work.** A terminal human-vs-agent mode (the renderer + interactive replay
  already exist in `demo.py`) gets the human-eval signal at a fraction of the online-play cost,
  with none of the ToS/rules-parity risk. A handful of self-played games will surface blind spots
  faster than most metrics.

### From human games to training changes — closing the loop

"Play 10 games yourself" is only step zero. Its value is *discovery*, and discovery is worthless
without a pipeline that turns each qualitative observation into (a) an automated, re-runnable
metric and (b) the right training lever. The pipeline:

**0. Record every game.** `demo.py`'s `play_game` already keeps full history for rendering —
persist it (states + actions + who won) instead of discarding after render. An unrecorded
impression can't be quantified, and the logged positions feed everything below. (The
human-input mode itself is small: an agent whose `act(env)` renders the board and prompts for
one of the legal actions.)

**1. Convert each observation into an automated probe.** Four probe types, by observation shape:

- *Counter metric* — "it never bolsters" → bolster/recruit/chain rates, already emitted by
  `eval_bucketed.py`. Nothing to build.
- *Bucketed metric* — "it plays badly against Cavalry" → per-composition WR (`eval_bucketed`,
  P5a) conditioned on the opponent's draft containing the suspect unit.
- *Scripted exploiter bot* — "it's fragile to an initiative rush" → encode the strategy *you*
  used to beat it as a ~30-line scripted bot; its WR vs. the frozen policy is the metric.
  Human-found strategies are usually trivial to script once discovered — discovery was the hard
  part, and that's exactly what the human games buy. This is the poor-man's exploitability
  probe: every human win is a free best-response sample.
- *Puzzle/scenario suite* — extract the specific blunder positions from the logged games into a
  frozen set of (state, known-correct-response) scenarios ("must bolster here or lose the stack
  next turn", "must block the charge lane"). Run the policy over the suite at every eval:
  regression tests for gameplay, and the strongest form of the "opponent-strength-independent
  quality metrics" Step 1 already calls for.

**2. Classify the cause — five buckets, each with a different lever.** "The agent doesn't do X"
has at least five distinct causes, and applying the wrong lever wastes a training run:

| Cause | Diagnostic | Lever |
|---|---|---|
| **Can't see it** (obs gap) | Does the *critic* misjudge the puzzle positions too? If both nets are blind, the state isn't legible | New obs feature (bundle with next `OBS_VERSION` bump) |
| **Never tries it** (exploration gap) | Training action-frequency logs: was the verb *ever* sampled early, or dropped before reward could reinforce it? | Entropy floor / count-based verb bonus (IDEAS #8) |
| **Tries it, unrewarded** (credit gap) | Verb sampled early at normal rates, then decays | PBRS term, GAE-λ sweep (IDEAS #3/#7) |
| **Never faced it** (opponent gap) | Pool opponents never use/punish the mechanic (greedy never bolsters, rushes, or initiates tactics) | Pool composition: exploiter bots, `LookaheadBot`, meta-Nash weighting |
| **Can't fit it** (capacity gap) | Everything above ruled out; loss plateaus | Widen policy (IDEAS #5) — last resort, test after the others |

Cheap disambiguators before any retraining: run the critic over the puzzle positions (knows-but-
doesn't-play ⇒ policy-side problem; blind ⇒ obs/capacity), and IDEAS #9's usage-conditioned-on-
base-lead logging.

**3. Fix, then re-measure the probe *and* the full gauntlet.** The probe confirms the targeted
weakness moved; the gauntlet + exploiter-WR panel guards against whack-a-mole (patching the
rush weakness while dropping general Elo). Scripted exploiters accumulate into a standing
panel — every discovered weakness stays measured forever.

**Worked examples** (the hypothetical 10-0 human result: "never bolsters, fragile to an
initiative rush, weak against Cavalry"):

- *Never bolsters* — already measured (1/200 games). Run the #9 disambiguator; opponent gap is
  also implicated (greedy never punishes unbolstered stacks — `LookaheadBot` in the pool, already
  planned, attacks this directly). If bolster-rate stays ~0 after the material-PBRS A/B (#3) and
  the harder pool, escalate to #8's count-based verb bonus.
- *Fragile to initiative rush* — almost certainly an opponent gap: nothing in the pool rushes.
  Script the rush bot, verify it reproduces the human result vs. the frozen policy, add it to the
  pool at meaningful weight, retrain, watch WR-vs-rushbot climb. This is a mini PSRO iteration
  with a hand-coded best response standing in for a trained one.
- *Weak vs Cavalry* — confirm via composition-bucketed WR; check the charge threat planes
  actually fire on the blunder positions (if the critic sees the threat, it's not an obs gap);
  then oversample Cavalry-containing drafts in training (prioritized composition curriculum,
  below) and add the charge positions to the puzzle suite.

### Further ideas (2026-07-11, second pass)

- **Blunder finder — automated post-mortem of logged games.** For every position in a logged
  game (human, gauntlet, or self-play), compare the policy's move against the search agent's
  choice and plot the critic's value trajectory; large policy/search disagreement combined with
  a value drop flags a candidate blunder automatically. Turns any pile of games into a ranked
  list of concrete positions for the puzzle suite — no human required after the first pass.
- **Prioritized composition curriculum.** `set_init_state` drafts uniformly; `eval_bucketed`
  already knows which compositions the agent is weak on. Sample training drafts from a
  distribution tilted toward weak buckets (re-estimated at each eval), the draft-level analogue
  of prioritized fictitious self-play. Cheap: a sampling-weights hook in `set_init_state` + a
  weights file the eval refreshes.
- **Scripted-exploiter panel as standing infrastructure.** Formalize the exploiter bots from the
  loop above: a `bots/exploiters/` family, each a small scripted strategy, all registered as
  gauntlet entrants and pool candidates. The panel is the cheap, cumulative version of the
  league's "main exploiters" (AlphaStar) and feeds the meta-Nash pool weighting when that lands.
- **Belief auxiliary head (actor-side hand inference).** The actor gets `E_opp_hand` (analytic
  hypergeometric mean); the critic gets the true hand. Add a small auxiliary head on the *actor*
  trunk trained to predict the opponent's actual hand (available as a supervised target at
  training time, never at inference) — privileged-information distillation. Unlike the analytic
  mean, a learned head can condition on *behavior* (what the opponent chose to do reveals what
  they hold). Cost: one head + one loss term, no schema change, no inference-time leak. A/B-able.
- **Warm-start vs. from-scratch on curriculum changes.** Every run so far starts fresh. When the
  pool gains `LookaheadBot`/exploiters, also try *continuing* `ckpt_20260707-0026` on the new
  curriculum as a second arm — if fine-tuning holds general strength while patching the gaps, it
  halves the cost of every future curriculum iteration.
- **Checked and rejected — mirror-symmetry data augmentation.** The base layout is 180°-rotation
  symmetric (already exploited: C6 ego-rotation) but **not** mirror-symmetric — e.g. reflection
  `(r,q)→(q,r)` maps P1 base (1,0) onto *neutral* base (0,1) (`board.py` `default_bases`), and
  hex reflections flip direction handedness. Mirrored positions are not legal Warchest positions;
  augmentation would train on states outside the game. Recorded so it isn't re-proposed.

### Revised sequencing

1. Re-run the round-robin gauntlet with the fixed `LookaheadCriticBot`; record it plus the
   `ppo_20260706-194732` run in `docs/experiments.md` (both still owed per the 2026-07-08 note).
2. Finish the already-planned Step 2 remainder (`Bot`/`GauntletAgent` merge, a lookahead bot or
   its distilled clone into `opponent_pool.py`, `eval_bucketed.py` re-pointed), then the
   properly-sequenced long run — still the single biggest owed *training* improvement.
3. Exploitability probe (search proxy, then RL best-response). Its result decides how much
   further Nash investment (PSRO-lite pool weighting) is justified.
3.5. Local human-play mode **with game logging** (decided 2026-07-11: right after step 2). The
   mode itself is dependency-free and can be built anytime; playing is scheduled here so the
   games are against the post-step-2 checkpoint (the agent we intend to keep) and the findings
   feed the human-eval loop above (exploiter panel / puzzle suite) as inputs to steps 3-4
   rather than afterthoughts.
4. Build the policy-prior search agent — new gauntlet ceiling, new best-response opponent, new
   online-play candidate.
5. Online play with the search-augmented agent — after 2-4, not before.
6. Expert iteration as the next big training investment if strength gains are still wanted after
   that.

---

## Step 1 — Round-robin gauntlet + diagnostics

> **Status (implemented):** the obs-encoder was extracted from `warchest_env.py`
> into versioned modules (`environment/obs_encoders/`, registry + `v10.py`); the
> engine now delegates encoding and exposes stable rules-queries
> (`unit_threat_footprint`, `attack_enabler_coins`, `unit_base_reach_cells`) so the
> *availability model + feature layout* (the version-varying part) lives in the
> encoder. `Policy`/`Critic` take their obs dims from the paired encoder;
> checkpoints now carry obs-version + arch metadata (`policy/checkpoint.py`, with a
> legacy bare-`state_dict` fallback). The in-process round-robin
> (`services/gauntlet.py` + `app/gauntlet.py`) plays a field all-pairs with balanced
> colors and reports the WR matrix, a Bradley-Terry (Elo-scaled) ranking, and the
> intransitive-triple fraction. A golden-output test
> (`tests/test_obs_golden.py`) guards the extraction byte-for-byte.
> *Still open:* the opponent-independent quality metrics in `eval_bucketed`, and
> pointing diagnostics at a stronger opponent (Step 2). Resurrecting pre-v10 /
> pre-`conv` checkpoints remains the subprocess/worktree path (not built — the
> gauntlet skips incompatible checkpoints).

### What it is (and what it is *not*)

A **fixed** set of agents (checkpoints from different eras + `GreedyBot` variants) played
all-pairs, K games per pair, to produce (a) a stable Elo/Bradley-Terry ranking anchored to
fixed reference points instead of a moving pool, and (b) a **transitivity metric** to detect
cycles.

### The compatibility blocker (and the key simplification)

A `.pth` checkpoint is meaningless without its matching **obs-encoder + network class +
action-space mapping** — all three drifted over the last 2 days. `wr_vs_pool` works only
*within* a run because everything is identical there.

The simplification that makes cross-era comparison tractable:

- **At eval time we need only the forward pass** `state → obs → net → action`. No old rewards,
  training, or rollout code.
- The **game rules (Board, legal moves, win conditions) were stable** across the recent churn —
  only ML-facing code (obs, arch, reward) changed.
- Therefore the one stable contract to build around is:

  > **An agent receives the canonical game state and returns an action id in the absolute
  > (unrotated) env frame.**

  The absolute frame matters because eras handled P2 ego-rotation differently (early: per-verb
  remap tables; later: full spatial rotation). Each agent does its own rotation + inverse-remap
  *internally* and hands back an absolute env action. This contract also survives a future
  **action-space rebuild** (the user's own idea; see below) — the env-facing boundary never
  changes.

### Two implementation paths

| | Serialization | Coexistence requirement | Verdict |
|---|---|---|---|
| **In-process, N instances** | **none** — pass the live env object to each `act(env)` | all agents' code must coexist in one interpreter | **preferred for forward** comparisons |
| **Subprocess per git-worktree** | needed, as transport across the process boundary | none — full code isolation | fallback for resurrecting old commits |

**Clarification on serialization** (this caused confusion): serializing the state does **not**
mean storing all possible states — there is exactly one live state per turn, and serialization
is just the *per-turn wire payload* to ship it to a subprocess. If agents run in-process, no
serialization is needed at all — you pass the live Python object directly. Serialization is
purely the transport for the subprocess design.

### Recommended shape

The blocker to the clean in-process path is that **the obs-encoder is baked into
`warchest_env.py`**, so two generations of that file can't coexist. Fix it with one refactor
that pays off regardless:

1. **Extract obs encoding out of `warchest_env.py` into a versioned module**
   (`obs_encoder_v9.py`, `_v10.py`, …). Decouples "game rules" from "observation encoding".
2. **Keep versioned `Policy` classes** (don't delete old archs).
3. **Store architecture metadata alongside each checkpoint** (which encoder + layer sizes).

Then the current code can construct *any* registered architecture, instantiate N policies
in-process, and play them through one authoritative `WarChestEnv` — exactly the "policy class on
disk, several instances, call them" design. Reuse the existing `Bot` ABC: each gauntlet entrant
is a `Bot` wrapper. Subprocess + serialization is reserved solely for resurrecting **frozen old
commits** (via `git worktree add <commit>` + saved `.pth`), which is worth doing for **at most
1–2** checkpoints — do not build a museum. The smoke-test-per-change discipline already gives
reasonable confidence nothing broke; the gauntlet's lasting value is **forward**.

### The round-robin itself (trivial once agents are callable)

- All pairs, K games each, **alternating colors** (half the games with the agent as P1, half as
  P2 — initiative/side is a real edge).
- Build the pairwise WR matrix; fit **Elo or Bradley–Terry** for a global ranking.
- **Transitivity metric:** fraction of triples `(i,j,k)` with `i>j>k>i` (cycles), or the
  disagreement between the WR matrix and the Elo-implied order. Many cycles ⇒ "beats
  predecessors" is partly illusory ⇒ league/population play is warranted (Step 3).

### Diagnostics (`eval_bucketed.py`)

The script is already rich (per-composition WR, unit-presence swings, bolster/tactic/Berserker
usage, loss autopsy) but evals vs greedy → saturating. Two upgrades:

- Point it at the **stronger opponent** from Step 2 so buckets discriminate again.
- Add **opponent-strength-independent quality metrics**: does it ever `recruit`? does it leave
  the Royal exposed? material efficiency (coins-to-box ratio), tempo, win-by-base-control vs
  win-by-elimination. These tell us *what to fix* and de-risk Step 5.

---

## Step 2 — Raise the bar

`GreedyBot` ignores whole mechanics. Two escalation levels:

- **Economy-aware greedy + shallow lookahead** (1-ply value or 2-ply minimax): cheap; restores
  an objective WR with headroom, and as a **pool opponent** forces the agent to learn to
  use/punish bolster, recruit, tactics, and stack HP — mechanics it currently gets away with
  ignoring. Heuristics have a ceiling, so it will eventually saturate too.
- **MCTS / deeper lookahead opponent**: the "correct" high-ceiling bar. More effort, but usable
  both as an eval bar and a strong training opponent.

> **Status (mostly done, 2026-07-06 to 07-08):** `LookaheadBot` (alpha-beta + hand-tuned leaf
> heuristic + move ordering, `docs/lookahead_bot_plan.md`) went through an extensive tuning
> pass — 25% → 93% WR vs `GreedyBot` — and, more importantly, is a genuine, non-saturated
> opponent for the trained policy (see the 2026-07-08 gauntlet result above: 1070 Elo vs the
> current best checkpoint's 1177). `LookaheadCriticBot` (critic-guided beam search) was added
> next but currently *underperforms* plain `LookaheadBot` (loses 80/20 in the round-robin) —
> not yet diagnosed, see the status update above. **Still open:** neither bot is wired into
> `opponent_pool.py` — both only implement `GauntletAgent.act(env)` (needs the live env to
> forward-simulate), while training's opponent sampling calls `Bot.act(obs)`. The
> `Bot`/`GauntletAgent` interface merge (noted as M4 in `lookahead_bot_plan.md`, "not blocking")
> is the actual prerequisite for using either bot as a *training* opponent rather than just an
> eval/gauntlet one. Do this for `LookaheadBot` first, per the priority note above.

---

## Step 3 — The Nash direction (exploitability, PSRO, R-NaD)

Warchest is a **two-player, zero-sum, imperfect-information** game (hidden hand/bag — hence the
privileged critic). That is exactly the Stratego / DeepNash regime, so pursuing an approximate
Nash equilibrium — a policy no opponent can beat >50% — is a legitimate north star for
**robustness (unexploitability)**.

### Round-robin vs Nash — different layers, not substitutes

- **Round-robin = measurement of *relative* strength** (and transitivity).
- **Nash = a training objective / solution concept** (*what* you train toward).
- **Exploitability = measurement of Nash-ness**: freeze the agent, train a best-response
  against it, see how badly it loses. Low exploitability ≈ near Nash. This is a *different*
  measurement from round-robin, and the one that actually tracks Nash progress.

### Is round-robin required to pursue Nash?

**Not strictly** — it depends on the method:

- **R-NaD** (DeepNash's engine): model-free, single-network, no population, no payoff matrix →
  needs no round-robin at all. Large, separate project; **overkill for now**.
- **PSRO** (Policy-Space Response Oracles): the payoff matrix over the population **is** the
  round-robin. Maintain a population → compute the matrix → solve for a meta-Nash mixture →
  train a best-response to it → add to population → repeat. The current opponent pool is a crude
  PSRO **without** the meta-Nash solve (it samples with fixed weights instead of the equilibrium
  mixture).

Practically: you don't *need* round-robin to dig toward Nash, but you *do* need some yardstick,
and right now there is none. Round-robin is the cheapest to build, gives transitivity for free,
and **becomes the PSRO engine** if you go that way — so it is the natural first step regardless.

DeepNash reached an **approximate** Nash (exact is intractable at that scale) — calibrate
expectations accordingly.

> **Note (2026-07-08):** the 2026-07-08 gauntlet result (status update above) makes this step
> better-motivated than when this section was written — the field's ranking looks transitive
> by hand-inspection and the round-robin machinery + `LookaheadCriticBot`'s forward-simulation
> harness (built for Step 2) are most of what an exploitability probe needs: freeze
> `ckpt_20260707-0026`, use the search harness as a cheap best-response proxy (already noted as
> a "stress-test" reading of `see_opponent_hand=True` in `lookahead_bot_plan.md`), and measure
> how badly it loses. Sequence after `LookaheadCriticBot`'s underperformance is understood,
> since an exploitability probe built on a bot that isn't behaving as intended would be
> measuring the bug, not the policy.

### Recommended Nash-ward increments

1. **Exploitability metric** (best-response vs the frozen agent) — the first honest measure of
   unexploitability. Reuses the same best-response machinery.
2. **PSRO-style pool weighting**: solve the meta-Nash over the round-robin matrix and sample
   opponents by it instead of fixed weights. Nearly free on top of Step 1 + the exploitability
   engine.
3. **Full R-NaD** — only if (1) shows the agent is meaningfully exploitable *and* the pool fails
   to catch it. Not now.

---

## Step 4 — Long run (1000–1500 batches)

Low value **right now**: the 2026-06-30 run already spent ~10 h on a plateau against a saturated
eval (score kept rising while WR/Elo stayed flat — the score-vs-win decoupling). Training longer
against opponents already beaten 100% just optimizes a saturated signal. This becomes worthwhile
**after** Step 2 provides a harder opponent / curriculum and Step 1 provides a signal that can
still move. Run it *then*, and read the plateau against the gauntlet, not against greedy.

> **Status (done, 2026-07-06/07, run `ppo_20260706-194732` → `warchest_ppo_20260707-0026.pth`):**
> ran before `LookaheadBot`/the round-robin gauntlet were fully in place, so — same as every
> prior run — it was read against greedy at the time, and `wr_greedy` in this run's own log
> saturates to 1.0 by batch ~1200, the same shape this section warns against. However, the
> 2026-07-08 gauntlet result (status update above) validated it *after the fact*: this
> checkpoint beats every other agent in the field including both lookahead bots, and beats the
> previous checkpoint (`ckpt_20260704-1243`) 70/30 — so the run's gains were real, not a
> saturated-signal artifact, this time. **Not yet done:** this run isn't recorded in
> `docs/experiments.md`. The next long run should be against a curriculum that includes
> `LookaheadBot` (once wired into `opponent_pool.py`, Step 2) so `wr_greedy` saturating mid-run
> doesn't leave training coasting on a dead signal for the back half, the way it did here.

---

## Step 5 — Online play vs humans

The ultimate absolute-strength test — real, non-cyclic, vs humans. But it is the **wrong next
step** and the **right eventual one**. Reasons to defer:

- **Big feature.** Playwright driver, action mapping to the site's UI, state parsing — see
  `docs/web_agent.md` / `config/web_agent.sample.toml`.
- **Rules-parity risk.** The site must match our env *exactly*; any divergence makes the numbers
  meaningless. Verify rule-by-rule before trusting a single game.
- **Low statistical power.** Human games are slow → few samples → wide confidence intervals.
- **ToS / anti-bot risk.** Check terms and rate-limit conservatively.

Do it once we trust — via Step 1 diagnostics — that the agent has no glaring blind spot to
embarrass itself with, and after a careful rules-parity audit.

---

## Side idea — rebuild the policy's action space

Noted for later. This is the schema break that the round-robin contract is explicitly designed
to absorb: as long as each agent maps its own action head → an **absolute env action id**, an
action-space rebuild does not break the gauntlet. Related to `docs/IDEAS.md` #10 (*factor
direction out of the move/attack spatial head*). Worth pursuing on its own merits (sample
efficiency), but it is an *improvement*, not a *measurement fix* — so it sits behind Step 1 in
priority. If done, land it as a new versioned arch so it drops straight into the gauntlet.

---

## Summary ordering

1. **Refactor obs-encoder out of the env → versioned encoders**, then build the **in-process
   round-robin gauntlet** (Elo + transitivity) and extend `eval_bucketed` with
   opponent-independent quality metrics. *(Restore the signal.)* — **done**, `eval_bucketed`
   quality metrics not yet re-pointed at `LookaheadBot`.
2. **Strengthen the opponent** (economy-aware greedy + shallow lookahead, optionally MCTS).
   *(Raise the bar.)* — **done for eval** (`LookaheadBot`, `LookaheadCriticBot`); **not yet
   done for training** (opponent-pool wiring, blocked on the `Bot`/`GauntletAgent` merge).
3. **Add an exploitability metric**; optionally upgrade the pool toward **PSRO**. *(Nash
   direction, measured.)* — not started; better-motivated now (see the 2026-07-08 note above),
   sequence after `LookaheadCriticBot` is understood.
4. **Only then** run 1000–1500 batches against the harder opponent and read the plateau against
   the gauntlet. — a 1500-batch run happened (`ppo_20260706-194732`) but *before* the harder
   opponent was wired into training, so it was read against greedy (which saturated mid-run) and
   only validated against the gauntlet after the fact. The properly-sequenced version of this
   step — training against `LookaheadBot` as an opponent — is still owed.
5. **Last:** build online play as the final human validation, after a rules-parity audit. —
   correctly still deferred; nothing above changes that.

**Immediate next steps (2026-07-08), in order:** — superseded by the 2026-07-11 status update
above; step 1 below is done (see `docs/bots.md`), which unblocks the Nash/search-augmented-policy
ideas in that section.
1. ~~Diagnose why `LookaheadCriticBot` underperforms plain `LookaheadBot`~~ — **done, 2026-07-11**:
   missing critic denormalization, fixed via `_calibrate_value_scale()` / exact
   `return_mean`/`return_std` recovery for new checkpoints (`docs/bots.md`). Now beats
   `lookahead` 68-78% (up from 30-35%).
2. Merge `Bot`/`GauntletAgent` interfaces enough to let `LookaheadBot` run inside
   `opponent_pool.py`, and add it to the training opponent schedule.
3. Point `eval_bucketed.py` at `LookaheadBot` instead of `GreedyBot` so its per-composition/
   bolster/tactic diagnostics are discriminating again.
4. Record the `ppo_20260706-194732` run and the 2026-07-08 gauntlet result in
   `docs/experiments.md` — both are currently only in this file and the git history.
5. Then: a fresh long run against the `LookaheadBot`-inclusive curriculum, read against the
   gauntlet from the start rather than after the fact.

**Immediate next steps (2026-07-11), in order:** see "Revised sequencing" in the 2026-07-11
status update above for the full list and rationale; short form:
1. Re-run the gauntlet (critic bot fix invalidates the 2026-07-08 ranking) + backfill
   `docs/experiments.md` (steps 3-4 above).
2. Finish Step 2's remaining wiring (steps 1-2 above) + the properly-sequenced long run.
3. Local human-play mode + game logging, played against the post-step-2 checkpoint; findings
   feed the exploiter panel / puzzle suite (decided 2026-07-11).
4. Exploitability probe (search best-response, then RL best-response) — decides PSRO investment.
5. Policy-prior search agent (new idea, AlphaZero-style) — new ceiling + online-play candidate.
6. Online play with the search-augmented agent.
7. Expert iteration, if still wanted after that.
