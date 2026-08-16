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

**Guiding principle — measurement first** (from the retired `next_steps.md`, still the rule):

> **Restore a trustworthy yardstick before training longer or shipping big features.**
> Do not optimize, or ship online, against saturated instruments.

Two failure modes it guards against, both of which have already happened here: an *absolute*
yardstick that saturates (WR vs `GreedyBot` reached ~100% — a myopic 1-ply bot that never
bolsters, recruits, or initiates a tactic, so beating it says nothing), and a *relative* one
that can rise without any real gain (self-play "beats its own predecessors" is compatible with
a non-transitive strategy space — the 30-round ExIt field measured a 0.11 intransitive-triple
fraction). The gauntlet's Bradley-Terry ranking + cycle metric (`app/gauntlet.py`) is the
measurement of record; per-behaviour rates (bolster/tactic/recruit/chain) are the
opponent-independent complement.

Current state: the live plan is **not** in this file — it is
[next_iteration.md](next_iteration.md) §5, which supersedes both the sequencing in
[independent_opponents.md](independent_opponents.md) §7 and this file's *Recommended next steps*.
(The intermediate step was the self-play-collapse diagnosis in `independent_opponents.md`: the
binding constraint is opponent *independence* and behavioural coverage, not reward terms. That
still stands; what moved on is the ordering.) A new block of proposals sits below the numbered
items — [New directions (2026-08-07)](#new-directions-2026-08-07--opponent-pool-architecture-learning-process),
groups **B** / **A** / **L** for opponent pool, architecture and learning process, with its own
suggested order in §N.4. Items **1–9 are retired** — the shipped ones are in `docs/history.md`, the rest are
dropped; the map is in [Retired items](#retired-items-19) below. Numbering of what's left is
deliberately unchanged so existing references in code and docs still resolve. Items **12–22**
came from `next_steps.md` when that doc was dissolved (2026-08-01); its implemented half is in
`docs/history.md` → *Measurement + opponent infrastructure*, its measurements in
`docs/experiments.md` (2026-07-07, 2026-07-08).

### Likelihood-weighted threat-plane magnitude

*(A variant of the shipped `E_opp_hand` feature — `docs/history.md` 2026-07-03 — not a duplicate.)*

The enemy threat planes gate opponent availability worst-case: a unit type contributes its full hit-count to a cell if the opponent holds **≥1** hidden coin of that type (`_threat_grids` `coin_gate`, `warchest_env.py:1593-1594`). This is correct for spatial *safety* — one Berserker coin they happen to hold is lethal even at low expected count, and you don't want the plane to average that tail away.

**The idea:** scale each unit's threat contribution by its **likelihood of being playable this round** — the shipped `E_opp_hand[t]` feature (`docs/observation_improvement.md`) instead of the binary `≥1` gate. The plane would then read *"how likely am I to actually be hit here,"* not *"could I possibly be hit here."*

**Why it's parked, not planned:** it understates exactly the tails that lose material. Worst-case planes + an expected-hand **global** scalar (the split shipped in `observation_improvement.md`) already give both signals — max for "where can I die," mean for "how loaded are they" — without diluting the spatial safety read. Only revisit likelihood-weighting the *planes* if the worst-case version makes the agent measurably too timid (over-bolstering, refusing good trades). If pursued, A/B against the worst-case planes directly; keep the two mutually exclusive within the threat-plane block.

### 10. Factor direction out of the move/attack spatial head

The verb head already groups all 6 move directions under one `V_MOVE` and all 6 attack directions under one `V_ATTACK` (`warchest_env.py:113-115`, `N_FACTORED_VERBS=11`) — `P(verb)` doesn't distinguish move-north from move-south. But the *within-verb* spatial logits still come from `policy_head`'s single `Conv2d(hidden_dim + GLOBAL_DIM → N_VERBS=32, kernel=1)` (`policy.py:106`), which gives each of the 12 move/attack directions its own independently-learned output channel — direction isn't a shared factor, so the network can't transfer "which way is good here" between moving and attacking.

**The idea:** add one more factorization level (verb → cell → direction) — collapse move/attack down to 2 spatial channels (or fold them into per-cell logits without a direction dimension) plus a small shared 6-way direction head reused by both verbs, analogous to how `verb_head` already shares parameters across directions at the top level.

**Why parked, not planned:** the channel-count saving is modest (32 → ~22), and the real cost is reworking `policy_head` from a monolithic conv into cell-logits + a conditional direction head, then re-deriving `_joint_log_probs`/masking for a third factorization level (legal directions differ between move and attack per cell — masking must stay consistent with the new hierarchy). Worth it only if move/attack direction choices turn out to share enough structure that sample efficiency is actually gated on it; no evidence either way yet. Difficulty: moderate.

### 11. Parallel rollout collection — remaining phases 4 & 5

Full design in `docs/parallel_rollouts.md`. **All phases 1-5 are now implemented** (P11a dynamic balancing via a shared atomic episode counter; P11b `overlap_collection` hides the rollout wall behind the GPU update). Remaining open work is validation + tuning, not implementation:

- **P11c. Real-config speed + learning-quality A/B.** Overlap adds 1-step off-policy staleness (behavior weights are one update behind), which interacts with the KL-skip (larger old→new gap → more skipped minibatches). Compare `overlap=True` vs `False` on **elo/wr trajectory**, not just wall-clock, before trusting it. Also confirm RAM headroom — overlap holds a second in-flight buffer (the box was at 87% RAM when this landed).
- **P11d. IPC via shared memory (if it ever bottlenecks).** Currently worker→main transfer is pickle-through-Queue (~55 MB/batch). Steady-state IPC is small (→0 when hidden by overlap); move board arrays to `multiprocessing.shared_memory` only if profiling shows it on the critical path.

Standing-rule reminder: measure over ≥10 batches per config, not a single batch — spawn/import startup (batch 1) and pool-phase (`p_pool→0.9`) opponent cost both skew early/late batches.

### 12. Exploitability probe → PSRO-lite pool weighting

The Nash direction, scoped as **measurement, not as the engine**. Warchest is two-player,
zero-sum, imperfect-information (hence the privileged critic) — the Stratego/DeepNash regime —
so "a policy no opponent can beat >50%" is a legitimate north star for *unexploitability*. But
round-robin measures *relative* strength; exploitability is the measure of Nash-ness: freeze the
agent, train a best response (BR) against it, see how badly it loses.

Two probes, cheapest first, both on existing infra:

1. **Search BR proxy** (hours): `LookaheadCriticBot`/`PuctBot` with `see_opponent_hand=True` vs
   the frozen best checkpoint. The hand-seeing "stress-test" mode already exists
   (`docs/bots.md`, `docs/lookahead_bot_plan.md`).
2. **RL BR** (a day or two): a fresh PPO run whose opponent pool contains **only** the frozen
   checkpoint. The WR trajectory over that run *is* the exploitability curve. No new infra — an
   `opponent_pool.py` weight config, reusing the whole training stack.

**The result is a decision procedure, not a score.** BR reaches ~80–90% ⇒ a real exploitable
hole ⇒ PSRO-style pool weighting is justified: solve the meta-Nash over the Bradley-Terry matrix
the gauntlet already produces and sample opponents by that mixture instead of fixed weights,
and add the BR policy itself as a pool opponent (that is PSRO iteration 1 — the current pool is
already a crude PSRO *without* the meta-Nash solve). BR caps at ~55–60% ⇒ the policy is hard to
exploit within this policy class ⇒ stop; further Nash investment buys robustness that can't be
measured as improvement.

**Why Nash is the thermometer, not the heater:** equilibrium-seeking optimizes the worst case
against the *current* population; at this compute scale the binding constraint is elsewhere
(opponent coverage, no search at inference, capacity). Full R-NaD theory and why it is not the
next step: `docs/rl_algorithms.md` § *R-NaD*. Every scripted exploiter
(`docs/independent_opponents.md` Phase 1) is a free BR sample, so this probe gets cheaper as
that panel grows.

### 13. Opponent-independent quality metrics in `eval_bucketed`

Half-done. `eval_bucketed.py` already emits per-composition WR, bolster/tactic/chain counts, the
tactic-initiation base-lead breakdown, the initiative "wasted move" split, and the Knight
bolster probe. Still missing from the original list, all opponent-strength-*independent* so they
cannot saturate the way WR-vs-greedy did: **does it ever `recruit`**, **does it leave the Royal
exposed**, **material efficiency** (coins-to-box ratio), **tempo**, and **win-by-base-control vs
win-by-elimination**. Cheap — counters inside `play_one_game`. Same family as the per-behaviour
metric panel in `docs/independent_opponents.md` Phase 4. Difficulty: low.

### 14. Puzzle/scenario suite + automated blunder finder

Two halves of one pipeline over logged games (`game_record.py`, `data/games` — already written
by `src/app/play.py`):

- **Blunder finder.** For every position in a logged game (human, gauntlet, or self-play),
  compare the policy's move against a search agent's choice and track the critic's value
  trajectory; large policy/search disagreement *plus* a value drop flags a candidate blunder.
  Turns any pile of games into a ranked list of concrete positions with no human in the loop
  after the first pass. Half-built already: ExIt computes policy/search agreement per round
  (`docs/independent_opponents.md` §1).
- **Puzzle suite.** Freeze those positions into a set of (state, known-correct-response)
  scenarios — "must bolster here or lose the stack next turn", "must block the charge lane" —
  and run the policy over the suite at every eval: gameplay regression tests, and the strongest
  form of the opponent-independent metric #13 asks for. Bonus: running the **critic** over the
  same positions is the cheap obs-gap-vs-policy-gap disambiguator (knows-but-doesn't-play ⇒
  policy side; blind ⇒ observation/capacity).

### 15. Wire the independent `LookaheadBot` into the training pool

`opponent_pool.py` samples `pool` / `lookahead_critic` / `puct` — *all* policy-derived (frozen
snapshots of itself, or a beam/tree guided by its own critic) — with greedy and random at
**zero** weight in the finetune schedule. Plain `LookaheadBot` is the strongest genuinely
*independent* agent in the repo and is still not wired in. The old blocker (the `Bot` vs
`GauntletAgent` interface split) is gone: `_SEARCH_OPP_TYPES` routing and `OPP_ONEHOT_SLOT`
already exist, so this is a lazy builder + a weight. Caveat: ~0.1 s/move will dominate rollout
wall-clock, which is what motivates #16. This is the same fix `docs/independent_opponents.md`
§2 argues for from the coverage side. Difficulty: low.

### 16. Distill the search bots into a fast network

Behaviour-clone `LookaheadBot` (and the scripted exploiters) into a small net from a few
thousand games, then pool the *clone* — network speed instead of search speed, keeping an
independent opponent affordable in the rollout hot path. Doubles as a dry run of the
distillation machinery. `docs/independent_opponents.md` Phase 3 lists this as one of the two
routes to a strong independent opponent; the other is a **quiescence extension** for
`LookaheadBot` (resolve stack fights to the end so a bolster that saves a stack two plies later
is actually valued — the diagnosed reason no search bot ever bolsters). Difficulty: moderate.

### 17. Prioritized composition curriculum

`set_init_state` drafts uniformly, while `eval_bucketed` already knows which compositions the
agent is weak on. Sample training drafts from a distribution tilted toward the weak buckets,
re-estimated at each eval — the draft-level analogue of prioritized fictitious self-play. Cheap:
a sampling-weights hook in `set_init_state` plus a weights file the eval refreshes. Difficulty:
low-moderate.

### 18. Belief auxiliary head (actor-side hand inference)

The actor sees `E_opp_hand` (an analytic hypergeometric mean); the critic sees the true hand.
Add a small auxiliary head on the **actor** trunk trained to predict the opponent's actual hand
— a supervised target available at training time and never at inference (privileged-information
distillation). Unlike the analytic mean, a learned head can condition on *behaviour*: what the
opponent chose to do reveals what they hold. Cost: one head + one loss term; no schema change,
no inference-time leak; directly A/B-able. Difficulty: moderate.

### 19. Warm-start vs. from-scratch on curriculum changes

Every run so far starts fresh. When the pool gains exploiters / `LookaheadBot` (#15), also run
*continuing* the best existing checkpoint on the new curriculum as a second arm. If fine-tuning
holds general strength while patching the gaps, it halves the cost of every future curriculum
iteration. Difficulty: trivial (a config arm, not new code).

### 20. A/B the dense-critic-targets opt-in

`--dense-critic-targets` shipped 2026-07-13 and is **off by default, never measured**
(`docs/history.md` → *Measurement + opponent infrastructure*): an auxiliary MC-return regression
on *opponent*-decision nodes (`rollout_core.collect_dense` → `aux_*` samples → a separate critic
minibatch loop at `aux_critic_coeff=0.5`), leaving the policy path and the main GAE targets
untouched. The rationale is that those nodes are collected anyway and carry a return signal the
main path discards. Cheap to test — it is a flag. Read `critic_mae` against the return std plus
the elo/WR trajectory, per the standing rule. Difficulty: trivial.

### 21. Online play vs humans (deferred, deliberately last)

The ultimate absolute-strength test — real, non-cyclic, vs humans. Design exists, code does not:
`docs/web_agent.md` + `config/web_agent.sample.toml` (Playwright driver, action mapping to the
site's UI, state parsing). Reasons it stays last:

- **Rules-parity risk.** The site must match our env *exactly*; any divergence makes every
  number meaningless. Audit rule-by-rule before trusting a single game.
- **Low statistical power.** Human games are slow → few samples → wide confidence intervals.
- **ToS / anti-bot risk.** Check terms; rate-limit conservatively.
- **Big feature** relative to its information yield while cheaper probes are still open.

Two standing decisions: field the **search-augmented** agent (`PuctBot` over the best
checkpoint), not the raw policy — search directly papers over the tactical blunders that read as
"newbie" to a human; and prefer the local substitute first, which already exists —
`src/app/play.py` (human vs policy, with game logging that feeds #14).

### 22. Rebuild the policy's action space

Noted for later; the deeper version of #10. The gauntlet contract absorbs it by design: as long
as each agent maps its own head → an **absolute env action id**, an action-space rebuild does not
break cross-era comparison (`docs/history.md` → *the gauntlet design contract*). Worth pursuing
on sample-efficiency merits, but it is an *improvement*, not a measurement fix, so it sits
behind everything above. If done, land it as a new versioned arch so it drops straight into the
gauntlet.

---

## New directions (2026-08-07) — opponent pool, architecture, learning process

Written after the critic dead-trunk diagnosis (`docs/next_iteration.md` §3.4) landed and while
the fix is being run, as a deliberate step *outside* the within-state-ranking investigation that
has occupied the last week. Three groups, numbered **B** (bots), **A** (architecture), **L**
(learning) so nothing collides with #10–#22 or C20/C21.

Everything below is grounded in one of three things: a number measured *today* (§N.0), a rule in
`docs/UNITS.md` / `roster.py`, or a measurement already in the docs (cited). Where an idea looks
like something already tried and rejected, the difference is stated explicitly — several of these
sit next to a recorded negative result and the reason they are not the same thing is the load-
bearing part of the argument.

Nothing here is sequenced ahead of `docs/next_iteration.md` §5 rows 2b/3 (run the shipped critic
fix). §N.4 is the suggested order once those report.

### N.0 Three measurements this section rests on

Taken 2026-08-07 on the 16-core box, `torch.set_num_threads(1)` (how the rollout workers run —
these numbers only compose with the per-worker budget at that setting). Tables A and B are
reproducible with **`python src/app/probe_costs.py`** — landed under `src/app/` for this section
rather than left in a scratchpad; Table C is arithmetic on an existing training log. None of the
three appears in any other doc.

**Table A — per-decision cost.** This is the table that decides what "fast enough for the rollout
hot path" means.

| component | cost | note |
|---|---|---|
| `env.step` + `get_possible_actions` | 0.35 ms | the floor for anything |
| `get_possible_actions` alone | 27 µs | |
| `_clone_state` | 10 µs | what search pays per candidate |
| `Policy.act`, `hidden_dim=128` | **0.86 ms** | the reference "network speed" |
| `Policy.act`, `hidden_dim=64` | 0.57 ms | |
| `Critic.value_single`, 192 | 0.85 ms | |
| `Critic.value_batch(64)`, 192 | 0.37 ms/state | batching buys only ~2× on CPU |
| `RandomBot.act(obs)` | 0.008 ms | |
| `GreedyBot.act(obs)` | ~~**0.83 ms**~~ → **0.04 ms** | *was as expensive as a policy forward* — **fixed 2026-08-09**, see below |
| `ThreatAwareGreedyBot.act(obs)` | **0.11 ms** | added 2026-08-09 (B5) |
| `SimGreedyBot.act(env)` | 18.0 ms | ~21× a policy forward |
| `LookaheadBot.act(env)` @0.02 s | 20.6 ms | |
| `LookaheadBot.act(env)` @0.1 s | 104 ms | ~121× a policy forward; ~10–12 k nodes/s |

Two readings that are not obvious from the rows:

- **The obs-only "cheap" bot is not cheap** — *was not; fixed 2026-08-09.* `GreedyBot` cost the
  same as a 128-wide policy forward, because `_best_move_toward_base` ran a fresh whole-board BFS
  for *every candidate move × every target base* — a product of two quantities that both grow as
  the game opens up. The prescription written here ("**one multi-source BFS**, indexed, linear in
  candidates, comfortably under 0.3 ms") has now been applied **to `GreedyBot` itself**:
  `0.90 ms → 0.04 ms`, **~23×**, on a metric proven identical cell-by-cell and an action choice
  proven identical over 500+ real decision states (`tests/test_greedy_bot_speed.py`, which keeps
  the original as the reference implementation — the bot is a training-pool opponent and the
  yardstick every historical number is quoted against, so *no* behaviour change was admissible).
  The BFS now lives in `bots/board_geometry.py`; B3 should import it rather than re-derive it.
  Consequence for Table C: the `greedy` share of rollout opponent cost is now negligible, and a
  policy forward is **~20×** an obs-only opponent rather than ~1×.
- The env-taking rows are backed out of complete games (4–5 games each, the bot moving on ~half
  the plies) as `2·(ms_per_ply − env) − opponent`, not read off one probe state: a single state's
  `legal_at_root` swings a search bot's node count and reachable depth by an order of magnitude.
  Only the nodes/s figure is state-independent.

**Table B — how much of the observation is zero.** 2757 real decision states.

| | dead per decision | of which structural |
|---|---|---|
| board planes | **29.8 / 48 (62 %)** | 24 — only 4 of 16 unit types are drafted per side |
| global dims | **201.2 / 245 (82 %)** | ~156 — the ten length-17 coin vectors carry 5 live entries |
| whole input surface (`48·49 + 245`) | **64 % exactly zero** | |

The structural share never fills in: the draft is 4/4 disjoint out of 16, so 12 own + 12 opponent
unit planes are identically zero *for the whole game*, and **which 24 changes every game**. The
remainder is transient (a drafted type not yet deployed, an empty supply). See A1.

**Table C — where the rollout wall-clock actually goes.** From `logs/ppo_20260726-203902.log`
batch 1495–1500 (finetune phase, `p_lookahead_critic = 0.25`), 64 episodes, `turns ≈ 85` plies:

```
rollout=20s wall (6 workers ⇒ ~120 core-s) | env=9.7s | model_play=97s | actor_grad=7s critic_grad=5s
```

So **model inference is ~89 % of rollout core-time and env simulation is ~9 %**. Arithmetic on
the 25 % search-opponent slice: `0.25 · 64 episodes · ~42 opponent plies · 0.104 s ≈ 70 s` — i.e.
roughly **two thirds of all rollout compute is spent by an opponent that plays a quarter of the
episodes**. (Estimate, not an instrumented split: the remaining ~27 s is the policy's own
forwards, the pool snapshots', and the search bot overrunning its budget on critic calls.) The
direct consequence is B4: this is the single largest throughput lever in the repo, worth ~2.5–3×.

---

### B — a pool of fast opponents

The standing diagnosis (`docs/independent_opponents.md`) is right and is *still* not acted on:
the finetune schedule is ~100 % policy-derived (`p_random = p_greedy = 0`). What follows is a
different route to fixing it than "write four hand-coded archetypes", which was tried once
(`BolsterBot`) and produced a documented negative result.

#### B1. A randomised-coefficient evaluator family — a *continuum* of independent opponents

`HeuristicEvaluator` already exposes eight coefficients (`POS_COEFF`, `RISK_COEFF`, `DUR_COEFF`,
`ECON_COEFF`, `INIT_COEFF`, `PROG_COEFF`, plus the `SHAPING_C`/`C_MAT` weights it imports).
Sample a coefficient vector **θ per episode** and you get hundreds of distinct, fast,
policy-independent playstyles for the cost of a constructor argument: base-rusher, material
grinder, bolster brawler, recruit-economy, tempo/initiative bot — the exact archetype list
`independent_opponents.md` Phase 1 wants to hand-write, generated instead.

**Why the `rich_eval` negative result does not kill this.** That measurement
(`docs/bots.md`, `rich_eval=True` vs `False` = 20 %) is about the *bot's strength*: a
depth-bounded leaf cannot cash a long-horizon asset, so a bot that chases one loses. But an
opponent-pool entrant is not judged on Elo — `independent_opponents.md` §3 says so explicitly
("their value is coverage and pressure, not raw Elo"). A θ that makes the bot spam `recruit` is
not a broken bot, it is *the state distribution self-play never produces*, which is mechanism 4
of the collapse. The negative result rules θ out as a way to make a **strong** bot; it says
nothing about θ as a way to make a **varied** one.

Cost: SimGreedy speed (~18 ms/move, Table A), so this needs either a small pool weight,
`reply_branching=2`, or B4. Test: gauntlet ~8 sampled θ and read the per-behaviour rates — the
family is working if bolster/recruit/tactic/initiative rates *span a wide range* across θ, and
failing if every θ collapses onto the same profile. Difficulty: low.

> **IMPLEMENTED AND MEASURED, 2026-08-09.** `evaluation.py` takes `theta`, `RandomEvalBot` is
> the sampled-θ `SimGreedyBot`, `src/app/eval_theta_family.py` is the harness, and
> `OpponentPool(p_random_eval=)` / `ppo.py --p-random-eval-finetune` are the wiring (default
> **0.0** — the coverage is measured, the training benefit is not). Full write-up:
> `docs/bots.md` § *`RandomEvalBot` — the θ family*. Four things the measurement changed
> about the proposal above:
>
> 1. **The gate passes, with a control.** "A wide range across θ" is not evidence by itself —
>    any 16-game sample produces a range. Against the *same* bot re-seeded, the spread ratio
>    (mean pairwise total-variation distance between verb profiles) is **4.0x** vs `greedy_sim`
>    and **5.1x** vs `lookahead_critic`. Bolster and recruit span 25–40x the noise floor.
> 2. **Only two of the six named archetypes exist.** Per-dial sweeps: `economy` buys recruit
>    0.02 → 0.19 nearly free; `pos` buys the racer. **`tempo` and `progress` are inert** —
>    claim_initiative moved 0.126 → 0.133 across a 0…20 tempo sweep, because that verb's rate
>    is set by the rules, not the evaluation. There is no "tempo/initiative bot" in this class.
> 3. **`durability` is a trap, and it rewrites the `rich_eval` post-mortem.** It does not make
>    a bolster brawler: bolster saturates at 0.087 by weight 0.5 while `pass` climbs 0.11 →
>    0.65 and the win rate goes 0.75 → 0.19 → 0.00. The paragraph above blames `economy` for
>    the `rich_eval` collapse (following `docs/bots.md`); per-dial, economy is the *cheap* one
>    and durability is what kills the bot. `THETA_RANGES` caps durability at 1.0 as a result —
>    which cut unhealthy arms 4/8 → 1/8 on fresh θ seeds, at the cost of 6.7x → 4.0x spread,
>    almost all of it the `pass` column.
> 4. **"Independent" is half-delivered.** A 9-agent gauntlet (k=12) is **fully transitive**:
>    the arms straddle `greedy_sim` over ~375 Elo but none counters another. They are
>    independent of the *policy* — nothing learned is in the loop — and that is the state
>    coverage `independent_opponents.md` §3 asks for, but they are not a diverse threat model.

#### B2. θ-search as an automated best-response oracle — a cheaper, *interpretable* #12

Given B1, run CMA-ES or plain random search over the 8-dim θ to maximise win rate against the
frozen policy. One evaluation is ~40 paired games ≈ 30 s; 200 evaluations ≈ 2 CPU-hours.

That is a **measured best response with no training run**, and unlike #12's RL-BR arm the answer
is *nameable*: the winning θ says which term the policy fails to respect. Same decision procedure
as #12 — best θ ≥ ~65 % ⇒ a real exploitable hole and PSRO-style weighting is justified; ≤ ~55 %
⇒ hard to exploit within this class, stop. Every θ found this way is also a free permanent
gauntlet entrant and a free curriculum signal (L7). Difficulty: low-moderate. **This supersedes
the expensive half of #12** — keep #12's RL-BR arm only if the θ family turns out to be too
narrow a policy class to find anything.

> **DEPRIORITISED by B1's measurement, 2026-08-09.** It is too narrow a policy class. In the
> B1 gauntlet every one of six sampled θ scored **0.00–0.08** against `ckpt_20260808-0607`,
> and *unmodified* `greedy_sim` scored 0.17. Six draws are not CMA-ES, but they bound where a
> search would start: the binding constraint is the `SimGreedyBot` class, not the coefficients
> inside it, and no reweighting of a 2-ply leaf gets from 0.08 to the 65 % decision threshold.
> Re-scope this onto a stronger base bot (`LookaheadBot`, or B4's distilled net) before
> spending the 2 CPU-hours, or fall back to #12's RL-BR arm.

#### B3. `RaceBot` — the archetype the game's own rules say is strongest, and nobody plays it

Four facts, all already established:

- Win at **6 of 10** bases; games end at **round ~11** (`next_iteration.md` §3.3).
- `Board.is_valid_claim` requires *your own unit standing on the cell*, and you cannot move onto
  an occupied cell ⇒ **a parked unit is an absolute lock on a base**.
- 30 games produced **53 steals from occupied bases and 0 from empty ones** (§3.3).
- Deploy is restricted to bases *you control* (Scout excepted), so bases are also your
  deployment infrastructure — losing one costs geometry, not just a point.

So Warchest, as implemented, is a **claim-and-park race** in which combat happens only where two
locks collide. Nothing in the repo plans that. `GreedyBot._best_move_toward_base` walks whichever
unit gets nearest to *any* target base — no assignment (two units happily chase the same base),
no parking rule (it will walk a unit *off* a base it holds), and its BFS ignores unit occupancy.
`LookaheadBot` has `_nearest_dist` as a **tie-break term**, not a plan. `SimGreedyBot` is 2 ply.

**Design.** Multi-source BFS from each of my units and each deployable base to each capturable
base (37 cells, a handful of BFS ≈ tens of µs) → solve a tiny assignment (≤5 units × ≤8 targets;
greedy or Hungarian) → a "tempo to 6 bases" estimate → play the action that reduces it most,
subject to: never vacate a lock that is currently contested, and fight only when a lock blocks the
cheapest assignment. Target ~0.2–0.3 ms — *below* `GreedyBot`'s measured 0.83 ms, because one
multi-source BFS replaces its per-move × per-target BFS, so **cheap enough for the hot path at
any weight**.

**Falsifiable prediction, which is the point of building it:** it beats `lookahead`@0.1 s. The
evidence for that is already on record — `lookahead`@0.3 vs @0.1 measured **42 %** (depth is not a
lever here), and the bolster archetype lost to `lookahead` precisely *because* `lookahead`
out-raced it to bases (`independent_opponents.md` Phase-1 result). If a bot that races
*deliberately* does not beat one that races by tie-break, the race framing is wrong and that is
worth knowing. Difficulty: moderate. **Highest-value single bot in this list.**

#### B4. Distil one small net that serves four roles at once

Behaviour-clone into a **32-wide, 2-layer HexConv net** — not the full 128-wide policy arch, the
opponent does not need it. Scaling from Table A's measured 0.57 ms for the 64-wide 3-layer policy,
that lands near **0.25 ms**: roughly **70×** cheaper than `SimGreedyBot` and **400×** cheaper than
`LookaheadBot`@0.1. (Only the 0.25 ms is an estimate; every other figure here is measured.)

**A cleaner target than move-cloning:** regress `HeuristicEvaluator.evaluate(s′)` for every legal
action, from the *current* observation — one forward pass yields a 1875-vector of
simulated-consequence scores whose argmax *is* `SimGreedyBot`'s 1-ply choice. No game outcomes
are needed, so labels can be generated from **random and adversarially-seeded positions**, and
coverage is not limited by any policy's state distribution — which is the exact failure
(mechanism 4) that sank ExIt.

One artifact, four uses:

1. a fast, genuinely independent **pool opponent** at real weight in *finetune*;
2. a **PPO warm start** (L6);
3. an **independent PUCT prior**, which breaks ExIt's teacher ≡ student (agreement 0.94–0.95);
4. a policy-independent **reference for the agreement metric**, so "is the teacher teaching" stops
   being measured against the student.

Plus Table C: removing the 0.1 s search opponent from the finetune pool is worth ~2.5–3× rollout
throughput on its own. This is IDEAS #16 with a concrete target, a concrete arch, and a
concrete budget. Difficulty: moderate.

#### B5. `ThreatAwareGreedy` — a sub-millisecond prophylaxis bot the encoder already feeds for free

**Built 2026-08-09 (`src/services/bots/threat_greedy_bot.py`, gauntlet kind `threat_greedy`). The
cost half replicated; the strength half did not — see § Measured below before citing this item.**

v11 board planes **38–43** are graded own/enemy threat hit-counts per cell, and the globals carry
`own_at_risk` / `opp_at_risk`. **No bot reads any of them** — every existing obs-only bot predates
those planes.

Ladder, pure obs, no simulation: (1) take a capture that is free (own threat ≥ target stack, and
the landing cell is not under enough enemy threat to lose the attacker); (2) un-hang any unit
whose incoming hits ≥ its stack; (3) claim / park; (4) march; (5) deploy. That is the user's
domain claim — *you attack so that your unit is not attacked* — instantiated as an opponent, at a
cost that permits any pool weight. It also punishes the specific thing the policy is suspected of
doing (hanging material) without needing anyone to prove that first. Difficulty: low.

##### B5 § Measured (2026-08-09)

**Cost: the technique replicated; the *advantage* did not survive the same day.**
`ThreatAwareGreedyBot.act(obs)` costs **0.11 ms**, under the 0.3 ms target, via the mechanism this
section named — one multi-source BFS over 49 cells instead of a whole-board BFS per candidate move
× per target. But the obvious follow-up was to apply that same BFS to `GreedyBot`, which took it
from 0.90 ms to **0.04 ms** (~23×, behaviour bit-identical — see Table A's amended note). So the
new bot is now **~2.7× more expensive** than the baseline it was meant to undercut. What B5
actually bought was the *technique*, and the technique's biggest payer was the old bot.

**Strength: no effect.** 1600 games vs `greedy_fast`, colours balanced: **0.515 ± 0.024** — the
interval contains 0.5. In the four-bot gauntlet (k=40/pair) it ranks top of the non-search field
(BT 897 vs `greedy_fast` 876, `greedy_sim` 871) and is still crushed by `lookahead_critic`
(0.03, BT 1357). Reading the threat planes is worth, in playing strength, approximately nothing.

**The ladder as written in this item loses 0.26 to `greedy_fast`.** Three separate corrections,
each measured, each pointing the same way — *the threat model is worst-case over the opponent's
entire hidden pool, so "covered" is nearly everywhere, and any rule that lets "covered" outrank
tempo hands over the race*:

| symptom | what the literal reading does | fix | worth |
|---|---|---|---|
| rung 2 above rung 3 | the unit standing on a claimable base is exactly the unit the opponent covers, so it retreats instead of claiming — `claim_base` falls 12.7 % → 9.3 % of decisions | claim before un-hang | +0.15 |
| safety-first march | every cell near a base reads as covered, so the march walks away from the win condition | park, then distance, then safety | ~0 once the above lands |
| safety filter on lethal blows | the planes count the hits of the unit the blow removes, so a stack-1 unit refuses every even trade | a lethal blow is always taken | +0.037 |

**Ablations vs `greedy_fast`** (120–400 games each, shared seeds): dropping the **un-hang rung
entirely changes the win rate by 0.000**; dropping the Pikeman-counter guard is worth **−0.017**
(i.e. nominally *better* without it); never attacking costs **−0.167**. So of the whole
prophylaxis apparatus, the part that measurably matters is the part that attacks. The
`own_at_risk` / `opp_at_risk` globals are still unread by anything.

**What this does and does not unblock — nothing, on the current evidence.** It does not satisfy
B8's requirement: an opponent that ties `greedy_fast` is not strong *relative to this policy*, and
after per-opponent advantage centring its episodes would contribute near-noise. Nor is it a
throughput win any more, now that `GreedyBot` carries the same BFS. There is currently **no reason
to give it pool weight** — it is kept as a gauntlet entrant and as the worked example that the
threat planes, consumed this way, are worth ~0. The unblock is still **B2** or **B3**.

For B3: its cost target ("*below* `GreedyBot`'s measured 0.83 ms, because one multi-source BFS
replaces its per-move × per-target BFS") is now demonstrated rather than predicted, but the bar it
must clear has moved with it — `board_geometry.distance_to` is the piece to import, and 0.04 ms is
the number to beat.

#### B6. Tempo-multiplier archetypes, chosen by what the roster actually encodes

Read `roster.py` as a whole and one pattern dominates: **ten of sixteen units buy more than one
maneuver — or more than one hex of maneuver — per coin.** Swordsman (`move_after_attack`), Cavalry
and Lancer (move *and* attack on one coin), Light Cavalry (two spaces on one maneuver, so distance
rather than an extra action), Berserker (`extra_maneuvers_from_stack`), Footman
(`maneuver_each`, plus `max_on_board=2`), Mercenary (`maneuver_after_recruit`), Ensign and
Marshall (grant a maneuver to *another* unit), Warrior Priest
(`bonus_action_after_attack_or_control`).

A game is ~11 rounds × 3 coins ≈ **33 coins**, so a single free maneuver is ~3 % of a player's
entire budget. **Maneuvers-per-coin is the game's real currency, and no component of this system —
not the reward, not the evaluator, not the obs, not any metric — represents it.**

The archetype that follows is the one `independent_opponents.md` closes on as untested: a
**`PriestTempoBot`**, because the Priest's bonus action triggers on *control* — the thing a racer
does anyway — making it the only unit whose tempo bonus is free rather than conditional on
winning a fight. Force the draft the way `eval_bolster.py` already does. Difficulty: low-moderate.

#### B7. Search-bot speed engineering, only if search must stay in the loop

Table A gives `LookaheadBot` ~10–12 k nodes/s, i.e. ~85–100 µs/node against a 10 µs
`_clone_state` — so most of a node is *not* the clone. Make/unmake instead of
clone-per-candidate, plus a Zobrist-keyed transposition table and the existing ordering key as a
killer-move heuristic, is a standard 3–5×: `lookahead` at 20–30 ms/move. Worth doing eventually,
but **worth less than B4**, which is two orders of magnitude and whose artifact is reused four
ways. Difficulty: moderate.

#### B8. Pool hygiene — **half of this was wrong, corrected 2026-08-09**

*As written this item said "reserve ≥20–30 % of the finetune schedule for non-policy-derived
opponents" and called it trivial, as though **weight** were the constraint. It is not, and the
error is worth keeping visible: there is nothing in the repo worth giving that weight to.*

**The independence half is blocked, not trivial.** Every independent bot here loses to the
current policy — `greedy_fast` ~100 %, `greedy_sim` mid-tier (BT 1124 vs `lookahead` 1318),
and `BolsterBot` was measured at **42.5 % against this exact checkpoint**
(`independent_opponents.md`, Phase-1 result). Giving 20 % of finetune to an opponent you beat
buys little, and it is worse than it looks now that advantages are centred per opponent (§5
row 6): a group whose returns have little spread contributes near-noise *after* centring, so
those episodes cost full rollout time and move the gradient barely at all. The unblock is an
opponent that is strong **relative to this policy** — i.e. **B2** (θ best-response search,
which manufactures an exploiter rather than hoping a generic bot is strong) or **B3**
(`RaceBot`). Do not re-propose "add weight to the existing bots"; that is this item's mistake.

**The half that survives has nothing to do with independence.** The `lookahead_critic` slice is
a bad trade on its own terms: it is board-blind (`lookahead_critic_v4.pth`, `alive3 = 0.0`,
`out_std = 0.0`), the policy beats it **95 %** by end of run (`wr_lookahead` 0.04 → 0.95 over
`ppo_20260807-203528`), and it costs **~4.2 s/episode against ~36 ms for a pool snapshot** —
roughly two-thirds of all rollout compute for a quarter of the episodes. Cutting
`p_lookahead_critic_finetune` 0.25 → 0.10 (into `p_pool`) takes opponent cost from ~69 s to
~29 s per batch, i.e. ~1500 batches in ~6 h instead of ~9.5 h, with no new bot and no new
tests. The measured trajectory argues for annealing it (hard opponent early — `wr` 0.04 at
batch 10 — free wins late) rather than a flat cut. Difficulty: trivial, and it is a *throughput*
change, so it neither helps nor hurts the coverage problem above.

If a bot is ever added: route heuristic entrants onto the existing `greedy` one-hot slot via
`OPP_ONEHOT_SLOT` (do not widen `OPP_TYPE_IDX`, it breaks every v1/v2 critic checkpoint), give
it its own `OPP_GROUP_IDX` entry or its advantages land in the warned fallback bucket, add it to
`_SEARCH_OPP_TYPES` if it is `act(env)` — and fix the hardcoded `opp_type` if/elif chain in
`ppo.py::_log_batch`, which silently drops the win rate of any opponent it does not know about.

---

### A — architecture

#### A1. Unit-type embeddings instead of 32 one-hot planes

Table B: **62 % of board planes and 82 % of global dims are exactly zero on any given forward
pass**, and the structural floor — 24 planes, ~156 globals — never fills in, because only 4 of 16
unit types are drafted per side. Worse, *which* 24 changes every game. So the network maintains
sixteen disjoint parameter sets for what are really four roles, and each set is trained on ~1/4 of
the data. There are 1820 × 495 possible drafts; most compositions are seen a handful of times in a
96 k-episode run.

**Fix:** one learned embedding per unit type (dim ~8), initialised from — or concatenated with —
the rules attributes `roster.py` already stores: `can_normal_attack`, the `tactic` mechanic and
its params, `counter_when_attacked`, `only_attackable_when_bolstered`, `deploy_adjacent_to_friendly`,
`max_on_board`, `total_coins`. The 32 unit planes become `Σ_units stack · E[type]` → ~16 planes;
the ten length-17 coin vectors become `Σ count · E[type]`.

Two payoffs, and the second is the real one: parameter sharing across types, and **generalisation
to compositions never seen** — a rare unit is legible through its attributes rather than through
an under-trained plane index. Cost: an obs schema change ⇒ new `OBS_VERSION`, fresh run; the
gauntlet contract absorbs it. Pair with A2 so one version bump buys both. Difficulty: moderate.

**Shipped 2026-08-16 as `policy_factored_v2` / `critic_v5`** (`src/services/policy/unit_embedding.py`,
both now the default arch). **The `OBS_VERSION` bump above was wrong and did not happen** — the
contraction has to run inside the net (the learned half of the table cannot exist in the numpy
encoder), and the 32 unit planes already *are* the per-type count tensor it consumes, so the
observation is byte-identical and v11 is untouched. What v11 gained is descriptive metadata only
(`deck_block_offsets` / `unit_block_offsets` / `deck_unit_positions` / `deck_royal_position`), so
the net can read the flat 245-vector as per-type blocks. All existing checkpoints still load.

Two design points settled during implementation and worth carrying:

* **Table shape 16 = 10 frozen + 6 learned, and no royal row.** The frozen columns are a fixed
  function of `roster.py`, a non-persistent buffer that never takes gradient — freezing is
  load-bearing, since learnable per-type columns would be a rotation of the one-hot being
  replaced. Every frozen column is shared by ≥2 units *by rule*: a singleton column (there were
  six in the first draft — `counter_when_attacked`, `move_after_attack`, …) shares nothing and
  is a one-hot slot with a nicer name, so those collapsed into `gives_extra_tempo` (5 units) and
  `has_defensive_trait` (3). The tactic block is decomposed by behaviour rather than one-hot by
  mechanic name, and its `ranged`/`charge` split deliberately mirrors v11's `THREAT_KINDS`, which
  already spends six planes separating Archer/Crossbowman from Cavalry/Lancer. The royal coin gets
  no row (bag-only, no board unit, no unit behaviour to describe); its count rides through as a
  raw scalar.
* **The frozen block is deliberately not injective.** Swordsman/Berserker/Mercenary, Knight/Pikeman
  and Ensign/Marshall share a frozen row and are separated by the learned block. Identity is the
  learned half's job; starting genuinely-similar types together *is* the prior being bought.
  `tests/test_unit_embedding.py` pins exactly which three collide, so a new one — two unrelated
  types merged — fails loudly.

Note what this is *not*: with 10 + 6 = 16 = `NUM_UNIT_TYPES`, `board_channels` and `global_dim`
both come out unchanged (48 / 245) and the effective per-type weight `W_eff[o,t] = Σ_d W[o,d]·E[t,d]`
is still full rank. Nothing is compressed. The change is that 10 of each type's 16 degrees of
freedom are tied to shared, rules-derived columns that take gradient from *every* game instead of
only from the ~1/4 in which that type was drafted. The gate is a run: the win is supposed to show
up on rare compositions, so `eval_bucketed.py` (per-composition buckets) is the measurement, not
the pooled win rate.

#### A2. Read the board where the game happens — a base-cell + unit-cell gather readout

`_split_pool` compresses 49 cells into **two numbers per channel**. Against a win condition that
is a function of **10 fixed base cells**, with **≤5 units per side** (`max_on_board=1`, Footman 2),
that is close to throwing the board away — and it is the readout feeding the critic head that
§3.4 measures as tying 89–93 % of purely positional sibling pairs.

**Fix:** gather trunk features at the **10 static base-cell indices** — free, a constant index
tensor, the base layout never moves — plus at each of my/their unit cells (≤10, padded, or
attention-pooled as a set), plus a global **max**-pool alongside the mean (a mean destroys "my
Berserker is hanging on the far flank"; §3.2 records that exact failure for the hand features).

Note carefully how this differs from a tested-and-unresolved idea: `board_xy` is a
*location-preserving flatten* of all 49 cells, and `board` ≥ `board_xy` on every bucket with
nothing resolving (§3.1a). A **task-relevant gather at the 10 cells that define the win condition**
is a different hypothesis, not a re-run of that one. Cheapest large change here — readout only,
no trunk change, no obs version bump. Difficulty: low-moderate.

**Shipped 2026-08-09 as `critic_v4`** (now `CURRENT_CRITIC_ARCH`, `src/services/policy/policy.py`),
built on v3's trunk and aux head. Implements the mean+max form of "attention-pooled as a set" for
units rather than padded per-unit tokens: `_gathered_pool` concatenates the 10 fixed base-cell
features (from `Board.default_bases`, symmetric under the P2 ego-rotation so one constant index
set covers both players), masked mean+max over own/opponent unit-occupied cells (occupancy read
off the *input* board's unit-stack planes via new `ObsEncoder.own_unit_channels`/
`opp_unit_channels`, since the trunk output no longer carries per-cell occupancy), and a
whole-board mean+max — `[B, 16·hidden]` vs. `_split_pool`'s `[B, 2·hidden]`. Policy's
`facedown_head`/`verb_head` still use `_split_pool` (out of scope here). `value_from_features`
cannot support it (no raw board) and now raises rather than silently pooling wrong. 42 tests in
`tests/test_critic_arch.py`; the actual §3.4 gate — does the tie rate drop further than v3's
93 %→0 % — still needs a run through `eval_board_value.py`/`eval_privileged_ablation.py`.

#### A3. FiLM the globals into the trunk instead of broadcasting 245 constant planes into the head

Today the trunk never sees the globals at all: `policy_head` is `Conv2d(hidden + 245 → 32, k=1)`,
so 245 constant planes are pasted onto every cell and the head spends 245×32 weights turning them
into a per-cell bias. But **what a board means depends on the hand** — a Berserker threat cell is
irrelevant if no Berserker coin is playable this round.

**Fix:** `MLP(globals) → (γ, β)` per channel, applied after each conv block (FiLM). A few thousand
parameters, removes the dead head weights, and gives the *trunk* hand-conditioning it currently
cannot have. It also dissolves §3.5's `opp_onehot` problem structurally — the one-hot becomes one
conditioning input among many rather than a raw head input with a 0.747 output spread. Difficulty:
low-moderate.

**Shipped 2026-08-16 as `policy_factored_v2`** (`FiLM` in `src/services/policy/policy.py`), paired
with A1 in the same arch. `x ← x·(1 + γ) + β` after each of the three conv blocks, before the
activation; `policy_head` drops its `global_dim` input block and is now `Conv2d(hidden, N_VERBS,
k=1)`. That block was exactly the 245×32 = 7840 dead weights predicted above — confirmed by the
parameter delta. `facedown_head`/`verb_head` keep their globals: they read a *pooled* vector, which
is an ordinary MLP input, not the per-cell broadcast this item is about.

The conditioning MLP's output layer is zero-initialised, so at step 0 γ = β = 0 and the module is
exactly the identity — a fresh v2 net starts from v1's trunk behaviour rather than a random
perturbation of it. The usual zero-init consequence applies: on the first step only that output
layer has a gradient, and the layers below it start moving from the second.

**Deliberately not applied to the critic.** `critic_v5` gets A1 but *not* FiLM. `board_only_head`
reads `pool(trunk(board))` and exists precisely so its loss cannot be satisfied from the globals
(`critic_v2`, next_iteration.md §3.4 — the fix that took the positional-sibling tie rate 93 % → 0 %).
Conditioning the trunk on globals would put them straight back inside that path and silently void
it. Pinned by `test_v5_board_only_value_ignores_globals`. If FiLM is ever wanted in the critic, the
auxiliary head has to bypass it first — that is a separate item, not a tweak.

#### A4. Global context and residuals — the trunk cannot see across the board

Three HexConv layers ⇒ receptive radius 3 on a 7-wide board, so a unit on one flank is invisible
to the other; whole-board context reaches the net only through the pooled path this section is
otherwise trying to fix. But the long-range interactions are real: Lancer charge 3, Ensign/Marshall
grant range 2, ranged attacks at 2 — and a base race is global by definition.

**Fix:** a squeeze-and-excite / broadcast-mean block after conv 2 (every cell gets a whole-board
summary at O(C) cost) plus residual connections, on top of the GroupNorm `critic_v2` already
ships. This is the standard AlphaZero-family block; it is cheap in parameters and ~free in FLOPs
at 7×7. Difficulty: low-moderate.

#### A5. An entity-token transformer as the serious alternative to HexConv

The entire game state is **≤10 units + 10 bases + 2 hands ≈ 22 tokens**. A 2-layer, 4-head
transformer over 22 tokens is *smaller and faster* than a 3-layer 128-wide conv over 49 cells, and
every relation the conv must approximate becomes one attention edge: attack, grant-at-range-2,
line charge, "which of my units is threatened by which of theirs". Pointer-style action heads fall
out naturally — an attack is attention from my unit token to an enemy token — which is also the
clean answer to #10 and #22 rather than another patch on the 32-channel spatial head.

Honest cost: this is the largest item in the section, and it should be landed as a new versioned
arch so the gauntlet compares eras. Do it **after** A2/A3 have reported, because if a better
readout and hand-conditioning close most of the gap, the rebuild is polish. Difficulty: high.

#### A6. Two value heads on one trunk — `V_shaped` for GAE, `V_win` for search

Two findings currently point in opposite directions. §3.3b: a shaped-return target ranks siblings
~2× better than `z`. §3.5: the critic's raw output is a z-score of a shaped return and is
therefore "meaningless to every search bot", which is why `LookaheadCriticBot` needs a
moment-matching calibration hack.

Both heads on one trunk resolves this instead of choosing: PPO keeps the target that ranks, search
gets a **calibrated win probability**, and the calibration hack is deleted. Pair with a
**categorical / HL-Gauss value loss** in place of MSE — one head and one loss change, and a
well-documented improvement in value regression. Difficulty: low-moderate. Run *after* §5 row 2b
has settled which target is better, since 2b is the experiment this builds on.

#### A7. Auxiliary heads = one-to-two-ply lookahead installed in the representation

`next_iteration.md` §1's thesis is that value lives one to two moves ahead and nothing computes
one to two moves ahead. Search fixes that at inference cost. An auxiliary **prediction** head fixes
it at zero inference cost, and the labels are already sitting in the trajectories being collected:

- **survival head** — for each of my units (or each cell), will this stack lose a coin within the
  next 2 plies? This is prophylaxis, supervised.
- **threat-forecast head** — which cells will be enemy-occupied next ply.
- **opponent-hand head** — #18, on the *actor* trunk; unlike the analytic `E_opp_hand` mean, a
  learned head can condition on what the opponent chose to do.

The precedent that this mechanism works here is `critic_v2`'s board-only auxiliary head, shipped
2026-08-07 for exactly the same reason (create gradient pressure the main head does not supply).
Difficulty: moderate. Each head is independently A/B-able and none touches inference.

---

### L — learning process

#### L1. Record both sides of self-play — a free 2× on data

`rollout_core.play_episode` appends transitions only in the `acting_pid == main_pid` branch. In
finetune, **75 % of opponents are frozen snapshots of the same network**, and every one of their
decisions is discarded (`collect_dense` recovers them as *critic* targets only, never as policy
gradient).

Add a `p_self` opponent — the **current** policy, not a snapshot — and record both sides. Twice
the samples per env-second, and because the game is zero-sum the two halves are perfectly
antithetic, which cuts advantage variance as well. Two things to get right: PPO ratio bookkeeping
for the second stream, and the opponent one-hot has no "self" slot (route to `pool`, or drop the
input per `next_iteration.md` §5 row 6). Difficulty: moderate.
**Best data-efficiency per line of code in this section.**

#### L2. Drop λ *with* the critic fix, as one A/B

`next_iteration.md` §3.6 is explicit and its consequence has not been turned into a work item: at λ = 0.97, `V(s_t)`
cancels in the action comparison and `V(s_{t+1})` enters at γ(1−λ) ≈ 0.03, so **~97 % of the
discriminative signal is the realised return** — "λ = 0.97 is accidentally the right setting for a
critic that cannot rank, and improving the critic buys PPO nothing unless λ drops with it."

The trunk fix has shipped (and, 2026-08-09, so has §5 row 6). The paired change has not.
This is the step that *converts* a repaired critic into policy improvement; without it the repair
is invisible in the gauntlet and will read as a negative result. Difficulty: trivial (a
hyperparameter arm). **Should ride along with §5 row 8's first training run.**

**Recommended value: λ = 0.90**, swept as `{0.97 (baseline), 0.90, 0.80}`. **Shipped as the
`--lam` default 2026-08-09**, bundled into the row-6 run rather than swept, because a run is
~9.5 h and arms are expensive — see `docs/history.md` for what that costs in attribution.
The arithmetic:

| λ | γ(1−λ) — the weight on `V(s_{t+1})`, i.e. on sibling ranking | vs today | effective horizon `1/(1−γλ)`, in main-actor decisions |
|---|---|---|---|
| 0.97 (today) | 0.0297 | 1× | 25.2 |
| 0.95 | 0.0495 | 1.7× | 16.8 |
| **0.90** | **0.0990** | **3.3×** | **9.2** |
| 0.80 | 0.1980 | 6.7× | 4.8 |
| 0.70 | 0.2970 | 10× | 3.3 |

Why 0.90 and not lower, given that §1's thesis is "value lives one to two moves ahead":

- **An episode is only ~42 main-actor decisions** (`turns ≈ 85` plies in the logs). At λ = 0.97 the
  25-decision horizon covers more than half the game, so the advantage is close to plain Monte
  Carlo and the critic is nearly irrelevant by construction. At 0.90 the horizon is ~9 decisions
  ≈ 2–3 rounds — enough to span a base-race exchange, short enough that the critic actually
  arbitrates.
- **The critic is newly repaired, not yet accurate.** §3.4's row-3 result is a *ranking* win —
  same-verb pairwise 46.0 % → 55.8 %, tie rate 93 % → 0 % — against a ~61 % best-observed ceiling,
  and it explicitly notes pooled Pearson `corr` *fell* while the rank metrics rose. Lowering λ
  trades return variance for critic bias; at ~56 % pairwise accuracy on the hardest bucket, 6.7×
  (λ = 0.80) is a large bet on a quantity with one measurement behind it. 0.90 is the biggest step
  that is still clearly resolvable (3.3×) without that bet.
- **Skip 0.95.** 1.7× is inside the noise this project keeps getting burned by — the standing rule
  is that a ~0.3 pooled-std difference from one run per side is not signal, and 0.0297 → 0.0495 is
  not going to clear it. Three arms, well separated, is the better use of the same compute.

Two things that make this cheaper than it looks. Dropping `opp_onehot` (§5 row 6) **does not fight
it**: the per-opponent offset is constant across a state's siblings and across `V(s_t)`/`V(s_{t+1})`
within an episode, so it cancels in `δ_t` — removing the input costs the critic nothing where a
lower λ leans on it. And the run needs no new instrumentation: read `critic_mae` against the return
std (if it is still ~0.5× the std, the critic is too weak to carry more weight and 0.80 will lose),
plus the actor gradient norm, plus the usual gauntlet + step-5 conditional metrics.

Gate, in the standing form: all three arms at the same `n_batches`, compared over each run's
settled phase — and if 0.90 does not beat 0.97 on gauntlet Elo, that is evidence about the critic's
*absolute* accuracy, not about λ, and the next move is the critic-target A/B (§5 row 2b), not a
smaller λ.

#### L3. Action-conditional baseline — use the critic where it can actually act

The env is a perfect forward model at 10 µs per clone (Table A). For a sampled subset of
decisions, evaluate `V(s′)` over K legal successors and use `A(s,a) = Q(s,a) − Σ_a π(a)Q(s,a)`
instead of the GAE scalar. The critic's *ranking* ability then enters the gradient at full weight
rather than at 0.03 — which makes the sibling-ranking quantity the entire `next_iteration.md`
investigation measures into the thing the gradient actually consumes.

Cost control: K = 4 sampled successors on 10–20 % of decisions ≈ +2 ms/decision against a 0.8 ms
baseline of 0.86 ms; batch the critic call (Table A: 0.37 ms/state at batch 64). This is the cheap version of
"search at training time". Gate it behind L2 — if λ stays at 0.97 the baseline change is
pointless. Difficulty: moderate.

#### L4. Two PBRS terms the rules justify — and why the `rich_eval` result does not transfer

**State the distinction first, because it is the whole argument.** `rich_eval` failed as a
*depth-bounded search-leaf term*: a search cannot cash a long-horizon asset inside its horizon, so
a bot that chases one trades away tempo and loses (measured three times — `docs/bots.md`,
`independent_opponents.md`). A **potential-based** shaping term in RL is a different object: it
telescopes, it provably leaves the optimal policy unchanged (Ng et al.), and it only redistributes
credit across time. Carrying the leaf-term negative result onto PBRS would be a category error,
and this repo has already paid for one of those.

Two potentials the rules argue for:

- **Lock potential.** `Φ = SHAPING_C · Σ_bases w(b)` with `w = 1.0` for a base I control *and*
  occupy, `0.6` for controlled-and-empty. Justification is B3's: a parked unit is an absolute
  lock, and 0 of 53 observed steals came from an empty base. The current potential
  (`SHAPING_C · base_diff`) prices a lock and a walk-in-able base identically, which is not how
  the game works.
- **Hanging-material potential.** `Φ_risk = −c · own_at_risk`, using the scalar the encoder
  already computes and `HeuristicEvaluator` already has. This is the prophylaxis thesis expressed
  in the one shaping form that cannot distort the optimum.

Both are one-line potentials in `play_episode`, both A/B-able, both policy-invariant. Difficulty:
low. Note this reopens the reward axis that #6 retired — deliberately, and on a different basis
(a rule, not a tuning intuition).

#### L5. Antithetic game pairs — **tried, measured, and it does not work. 2026-08-09**

*Kept in full because the reasoning was good and the result was still negative; this is the
kind of item that gets re-proposed every six months otherwise.*

**The premise is true.** Every eval and gauntlet number is confounded by draft luck — 1820 × 495
asymmetric matchups over ~11-round games. Measured: same deterministic bot (`greedy_fast`) on
both sides, 300 paired games with the two compositions swapped between seats, **the same
composition won both games 190/300 = 63.3 % ± 2.8 pp**. Chance is 50 %, so ~**27 % of decisive
games are settled by the draft alone**.

**The algebra is also right.** Writing `p` for the true win rate and `D` for the draft advantage
(mean 0, variance σ²_d): two unpaired games give `Var = 2p(1−p)`, because the draft variance is
absorbed into the Bernoulli marginal. A pair sharing a draft has the *opposite* sign of `D` for a
given agent, so `Var = 2p(1−p) − 2σ²_d` — strictly lower, by twice the draft variance per pair.
And the swap is free: `play_game` seeds the global RNG before `env.reset()`, `set_init_state`
draws the whole 4/4 disjoint draft from it, so two games at the same seed open identically and a
composition is bound to a *seat*.

**It still fails, and the reason is worth keeping.** The reduction requires the two games of a
pair to be **negatively correlated**. They are not:

| field | within-pair outcome correlation |
|---|---|
| `greedy_fast` vs `greedy_sim` (heterogeneous) | **r = −0.003 ± 0.082** (150 pairs) |
| `ckpt_20260725` vs `ckpt_20260808` (homogeneous) | **r = −0.005 ± 0.082** (150 pairs) |

`|r| < 0.16` at 95 % in both. The 63.3 % was measured with one **deterministic bot playing
itself**, where the composition is the only thing that can decide. Real gauntlet entrants
*differ*, and policy agents **sample** their actions rather than taking an argmax — so an
identical opening diverges on the first ply and the shared draft stops propagating. The
mechanism never engages.

Two direct variance checks confirm nothing either way, which is itself instructive: ratio
**1.29** (`greedy_fast` vs `greedy_sim`, n=120) and **0.77** (two checkpoints, n=120) — ~1.4 σ
each, in *opposite* directions. That is what two draws of a noisy statistic centred on 1.0 look
like. **Do not cite either number.**

**What shipped:** `build_task_list(paired=...)` / `--paired-drafts`, **off by default**, so every
previously recorded gauntlet number stays reproducible bit-for-bit at its seed. Kept as an opt-in
only for a field of deterministic entrants, where the argument may still hold — untested.

**What *did* survive, on a different argument.** A forced-draft archetype bot could never have
used the swap: its composition follows the **agent**, not the seat, so it is the *treatment*, not
a nuisance draw, and averaging over it would destroy what is being measured. Its control is
**common random numbers across arms**, and that shipped and is on by default:
`eval_bolster.build_draft_list` generates the full 4/4 draft up front and pins *both* sides, with
`--draft-seed` / `--dump-drafts` / `--drafts`. A shared `--seed` nearly achieved this already, but
only by relying on every arm consuming the RNG in exactly the same order — which breaks silently
the moment a bot's constructor or the env's reset changes. This is a robustness fix, not a
variance claim, so it needs no measurement to justify.

**Two facts recorded so they are not re-derived.** The seats are exactly symmetric — the base
layout is 180°-rotation symmetric (`(1,0)→(5,6)`, `(4,1)→(2,5)`, every neutral base maps onto
another) and `set_init_state` draws `initiative_owner` independently of player id — so the colour
balancing the gauntlet and `eval_bolster.py` already do buys essentially nothing. And *variance
reduction schemes must be validated on the estimator you actually use*: this one was derived on a
model of the game, verified on a bot playing itself, and failed on the real field. Pinned by
`tests/test_paired_drafts.py` (14 tests).

#### L6. Supervised warm start

PPO currently spends its first couple hundred batches learning what a 200-line bot already knows.
Behaviour-clone B4's net (or the θ family, for diversity) for ~1 CPU-hour, initialise the actor
from it, then run PPO — AlphaStar's ordering, and cheap here because label generation is
embarrassingly parallel across 16 cores.

Second benefit, which matters more given the standing A/B rule: runs stop differing by "how fast
did it escape the random phase", so early-batch comparisons become meaningful instead of noise.
Difficulty: low, once B4 exists.

#### L7. A curriculum over drafts, prioritised by the exploiter rather than by win rate

#17 proposes tilting `set_init_state`'s draft sampling toward buckets `eval_bucketed` says are
weak. Sharper version, once B2 exists: tilt toward the **(composition, θ) pairs where the
best-response search found the largest edge**. That closes measurement → curriculum automatically,
and it targets *exploitable* weakness rather than merely *low-win-rate* weakness — which are not
the same thing, and only the first one costs games against a real opponent. Difficulty:
low-moderate.

#### L8. Reward hygiene the tempo reading exposes — **IMPLEMENTED 2026-08-09**

`MOVE_NEG_REWARD_PER_TURN = -0.002` was charged at seven call sites in `warchest_env.py`, five of
them tactic continuations. So a Berserker chain, a Footman double maneuver and a Swordsman bonus
move each paid the penalty **again per maneuver** — in a game whose central currency is
maneuvers-per-coin (B6), the per-step cost was charged against exactly the mechanics the policy
never uses. Charge it per **coin spent** instead, or zero it.

Joins the two hygiene items `next_iteration.md` §4 already lists: zero `ATTACK_REWARD = 0.02`
(double-pays what material PBRS covers), and re-derive `holding_reward_rate` from ~37 real
main-actor turns rather than the assumed 150. Difficulty: trivial.

**What shipped** (`docs/rewards.md` §1 carries the full write-up):

1. **Tempo cost per turn, not per maneuver.** `MOVE_NEG_REWARD_PER_TURN` → `TURN_TEMPO_REWARD`,
   added once in `_apply_action` at the point the turn advances. That is "per coin spent from
   hand" by construction, and it is exactly-once by construction rather than by keeping seven
   sites in sync — the property that made the old version drift. All seven maneuver sites now
   return 0.0. Two consequences worth naming: the charge is a **constant across every option a
   turn offers**, so it cannot distort the choice *within* a turn (it prices only elapsed turns);
   and the Swordsman's free post-attack move no longer costs strictly more than declining it.
   The Berserker's stack-paid extras stay uncharged deliberately — they cost *material*, which
   material PBRS already prices, so a tempo charge on top would be the same double-pay this item
   is about. `Action.tempo_cost` carries the term separately so `LookaheadBot` and the score
   decomposition can subtract it without pattern-matching on the reward value.
2. **`ATTACK_REWARD` 0.02 → 0.0.** `score_attack` is now ~0 by construction; read the attack
   axis off `score_material`. New `score_tempo` bucket keeps the decomposition honest (it would
   otherwise have landed in `score_attack`, which would have looked like attack reward).
3. **`holding_reward_rate` 0.001067 → 0.004324 (4.05×).** Divisor is now
   `TYPICAL_MAIN_TURNS = 37`, sourced from converged runs (~78 plies in
   `logs/ppo_20260807-203528.log`, about half the main actor's) rather than the `max_rounds *
   HAND_SIZE = 150` worst-case bound no episode reaches. `WarChestEnv.default_holding_reward_rate()`
   is now the single source for `ppo.py` and `LookaheadBot`, which previously duplicated the
   formula. The trade: the old "accumulated holding can never exceed a win" bound is gone (see
   `rewards.md` for why that is close to unreachable, and what to do if it bites).

`tests/test_reward_hygiene.py` pins once-per-turn per mechanism — plain move, Cavalry tactic,
Footman double maneuver, Berserker chain, Swordsman bonus move — plus no charge on a winning or
invalid action. The property is invisible in gameplay, so nothing else would catch a regression.

**Not yet measured.** All three are reward-scale changes: `score`, returns and `critic_mae` are
**not** comparable across this boundary, only win rate and the gauntlet are. Change 3 in
particular strengthens a **non-PBRS** term 4×, which is the one term that genuinely can move the
optimum — it is the plausible regression here, and the one to look at first if the next run's
`avg_turns` rises with a flat win rate (sitting on a lead, `METRICS.md`).

#### L9. A tempo + lock metric panel

Extends #13 and `next_iteration.md` §5 row 5 with the quantities B3/B6 imply, all
opponent-independent so none can saturate:

- **maneuvers per coin spent**, and free-maneuver yield per unit type;
- **locked-base fraction** (controlled *and* occupied) and base retention;
- **coins-to-claim** — economy spent per base gained;
- `P(bolster | facing a Knight ∧ none of my units bolstered)`. The Knight rule
  (`only_attackable_when_bolstered`) makes bolstering **mandatory** to attack it, so this is the
  one conditional where bolster is provably not a tempo loss. §3.7 measures bolster as
  unconditionally suppressed at `P̄ = 0.029`; this conditional separates "collapsed mode" from
  "correctly priced" in a way the marginal cannot.

Difficulty: low — counters inside `play_one_game`.

---

### N.4 Suggested order

Not sequenced ahead of `next_iteration.md` §5 rows 2b/3. After those report:

| order | item | why first | cost |
|---|---|---|---|
| 1 | ~~**L2**~~ (λ with the critic fix) | rides along with the first training run; without it the shipped critic fix cannot show up. **Shipped 2026-08-09** — `--lam` default is 0.90 | trivial |
| 2 | **B8** + ~~**L5**~~ + ~~**L8**~~ | two lines, a seed swap, a constant — all three make every later number cleaner. **L8 shipped 2026-08-09**; it changes the reward scale, so it must land on the *same* side of any A/B boundary as B8/L5. **L5 shipped (and rejected) 2026-08-09** — `--paired-drafts`/`--draft-seed` landed, but the variance-reduction premise measured false on real fields; only B8's throughput half is still open | trivial |
| 3 | **B5**, ~~**B1**~~ | fast independent opponents at hot-path cost, no new machinery. **B1 shipped 2026-08-09** (measured: 4–5x behaviour spread over the re-seed noise floor, fully transitive field, 2 of 6 promised archetypes real). What is left of it is one A/B: `--p-random-eval-finetune 0.15` against a baseline run, on the *training* benefit the coverage is supposed to buy | low |
| 4 | ~~**A2**~~ | largest expected gain per line; readout only, no obs bump, aimed straight at §3.4. **Shipped 2026-08-09** as `critic_v4` — the §3.4 gate (does the tie rate drop further than v3's) still needs a run | low-mod |
| 5 | **B3** | the falsifiable read on whether the race framing is right | moderate |
| 6 | ~~**B2**~~ | ~~interpretable exploitability; subsumes the expensive half of #12~~ **Deprioritised 2026-08-09** — B1 measured the class ceiling at 0.00–0.08 vs the current policy (`greedy_sim` itself: 0.17), so a θ search cannot reach the 65 % decision threshold. Re-scope onto a stronger base bot first | low-mod |
| 7 | **L1**, **B4** | 2× data and ~3× rollout throughput; B4 also unlocks L6, L7 and independent ExIt | moderate |
| 8 | ~~**A1 + A3**~~ together | ~~one `OBS_VERSION` bump buys both~~ **Shipped 2026-08-16** as `policy_factored_v2` + `critic_v5`, and the premise was wrong: neither needs an obs bump, because the contraction runs inside the net and v11 is byte-identical. Both gates still need a run — A1's on `eval_bucketed.py` (rare compositions, not pooled WR), A3's on the gauntlet | moderate |
| 9 | **A6**, **A7**, **L3**, **L4** | each depends on something above having reported | moderate |
| 10 | **A5** | only if A2/A3 leave a gap worth a rebuild | high |

Standing caution for all of it: the header's A/B rule applies unchanged — same `n_batches` both
sides, distributions over the settled phase, no conclusion from a single run.

---

## Method — turning an observed weakness into a training change

Carried over from `next_steps.md`, where it was written for the human-play loop; it generalizes
to any observation ("it never bolsters", "it's fragile to a rush", "it's weak vs Cavalry").
Playing 10 games yourself is only step zero — the value is *discovery*, and discovery is wasted
without a pipeline that turns each observation into (a) an automated, re-runnable metric and
(b) the right training lever.

**0. Record every game.** Already shipped: `src/app/play.py` persists finished games via
`game_record.py`. An unrecorded impression can't be quantified, and the logged positions feed
everything below.

**1. Convert the observation into an automated probe** — four types, by observation shape:

| Observation shape | Probe | Status |
|---|---|---|
| "it never does X" | Counter metric (bolster/recruit/chain/tactic rates) | Mostly emitted by `eval_bucketed.py`; gaps in #13 |
| "it plays badly against unit/comp Y" | Bucketed metric — per-composition WR conditioned on Y being drafted | Exists (`eval_bucketed.py`) |
| "it's fragile to strategy Z" | **Scripted exploiter bot** — encode the strategy *you* used to beat it (~30–80 lines); its WR vs the frozen policy is the metric | The panel plan is `docs/independent_opponents.md` Phase 1; first entrant `BolsterBot` measured there |
| "it blundered *here*" | **Puzzle/scenario suite** from the logged position | #14 |

Human-found strategies are usually trivial to script once discovered — discovery was the hard
part. Each human win is a free best-response sample, i.e. the poor-man's version of #12.

**2. Classify the cause before picking a lever.** "The agent doesn't do X" has at least five
distinct causes, and the wrong lever wastes a training run:

| Cause | Diagnostic | Lever |
|---|---|---|
| **Can't see it** (obs gap) | Does the *critic* misjudge the same positions? If both nets are blind, the state isn't legible | New obs feature, bundled with the next `OBS_VERSION` bump |
| **Never tries it** (exploration gap) | Action-frequency logs: was the verb ever sampled early, or dropped before reward could reinforce it? | Entropy floor / count-based verb bonus — **tried and insufficient alone**, `docs/history.md` 2026-07-25 |
| **Tries it, unrewarded** (credit gap) | Verb sampled early at normal rates, then decays | PBRS term / GAE-λ (retired items 3, 6, 7 — reopen only with a probe backing it) |
| **Never faced it** (opponent gap) | Do pool opponents ever use or punish the mechanic? (nothing in the repo bolsters) | Pool composition: exploiter panel, #15, meta-Nash weighting (#12) — **the current leading hypothesis**, `docs/independent_opponents.md` |
| **Can't fit it** (capacity gap) | Everything above ruled out; loss plateaus | Widen the policy — **tried, inconclusive/negative**, retired item 5 |

**3. Fix, then re-measure the probe *and* the full gauntlet.** The probe confirms the targeted
weakness moved; the gauntlet plus the standing exploiter panel guards against whack-a-mole
(patching a rush weakness while dropping general Elo). Probes accumulate — every discovered
weakness stays measured forever.

**Worked example — "it's fragile to an initiative rush."** Cause is almost certainly the
opponent gap (nothing in the pool rushes), so: script the rush bot, verify it reproduces the
human result against the frozen policy, add it to the pool at meaningful weight, retrain, watch
WR-vs-rushbot climb while the gauntlet Elo holds. That is a mini PSRO iteration with a
hand-coded best response standing in for a trained one — and the rush bot stays a permanent
gauntlet entrant afterwards.

## Checked and rejected

- **Mirror-symmetry data augmentation.** The base layout is 180°-rotation symmetric (already
  exploited — C6 ego-rotation) but **not** mirror-symmetric: reflection `(r,q)→(q,r)` maps P1's
  base (1,0) onto a *neutral* base (0,1) (`board.py` `default_bases`), and hex reflections flip
  direction handedness. Mirrored positions are not legal Warchest positions, so augmenting with
  them trains on states outside the game. Recorded so it isn't re-proposed.
- **More search budget as a strength lever.** `lookahead`@0.3 s/move vs `lookahead`@0.1 measured
  **42%** — alpha-beta saturates by ~depth 5 here and extra time buys nothing
  (`docs/independent_opponents.md` Phase-1 result, `docs/bots.md`).
- **`rich_eval` / durability leaf terms to make a search bot bolster.** Measured net-harmful
  twice; a depth-bounded leaf can't cash in a long-horizon asset (`docs/bots.md`,
  `docs/independent_opponents.md`).

---

## Retired items (1–9)

Closed out 2026-08-01. Shipped pieces live in `docs/history.md`; the rest are dropped rather
than parked — they are all reward/observation/hyperparameter tuning read against a saturated
`GreedyBot` yardstick, and `docs/independent_opponents.md` §2–3 locates the actual bottleneck
elsewhere (every training opponent is policy-derived, and *no* bot in the repo bolsters, so
nothing in the loop ever punishes the blind spots these items were aimed at). Kept as a map
because code comments and other docs still cite these numbers.

| # | Item | Outcome |
|---|---|---|
| 1 | Game-completeness — remaining Phase 5 work | **P5a shipped** (`src/app/eval_bucketed.py`, per-composition WR). P5b (rulebook snake/alternating draft) dropped — tabletop parity with no training value. P5c (freeze a `baseline_tactics` anchor) dropped — the gauntlet's fixed agent field (`services/gauntlet.py`) now provides the fixed comparison point it wanted, and it was written for the long-gone `OBS_VERSION=9` generation. |
| 2 | Draw-probability observation features (`p_soon`/`p_mean`) | **Shipped** — `obs_encoders/v11.py`, `OBS_VERSION` 10→11. Full record incl. the "why a feature, not a reward" analysis: `docs/history.md` → *Draw-share observation features + capacity/exploration bundle (2026-07-25 to 07-26)*. The owed standalone A/B was never run and is not planned. |
| 3 | A/B the 2026-07-03 reward + capacity bundle | **Resolved** 2026-07-04 (`docs/experiments.md`): `elo_policy` ~1000→~1500, `wr_vs_greedy_eval` 0→~0.9 — an unambiguous bundle-level win. Per-piece attribution across the three sub-bundles abandoned: the win was clear, and both the reward table and the yardstick have moved on since. |
| 4 | Re-test a small `CLAIM_BASE_REWARD` | Dropped, never run. Reward micro-tuning measured against a saturated eval; the circular-claim exploit it was hedging (`docs/rewards.md` §2) is also cheaper to leave alone than to re-litigate. |
| 5 | Widen the policy network | **Shipped** — `hidden_dim` 64→128 (`86d5ccd`), with `critic_hidden_dim` 128→192 in the same window; see `docs/history.md`. The result was confounded with idea 2's `OBS_VERSION=11` bump and came back *worse* on the gauntlet (BT-Elo 923, last of four vs three v10/64-wide checkpoints); the owed pinned-encoder re-test is dropped with the rest of the list. |
| 6 | Unit / board-presence PBRS | Dropped, never started. It was explicitly gated on ideas 3–5 leaving a measurable tempo gap; the proposal text survives in `docs/rewards.md` § *Unrealized ideas* if the reward axis is ever reopened. |
| 7 | GAE-λ sweep | **Value changed, never swept** — `lam` 0.95→0.97 shipped bundled (`cf2a9e3`, `419ec07`), see `docs/history.md`. The dedicated single-variable sweep is dropped. |
| 8 | Tactic/bolster underuse as an exploration problem | **Shipped** — verb-marginal entropy bonus annealed to a non-zero floor (`Policy._verb_marginal_entropy`, `808e72e`), see `docs/history.md`. It did **not** move the blind spots: as of 2026-07-28 the policy still essentially never bolsters and doesn't use unit-specific tactics. That negative result is what reframed the problem as *coverage* (`docs/independent_opponents.md` §2 mechanism 4, §3) — exploration has nothing to reinforce when no opponent ever bolsters or punishes its absence. |
| 9 | Disambiguate tactic reverse-causation | **Shipped (logging only)** — `tactic_base_leads` in `eval_bucketed.py`, see `docs/history.md`. No conclusion was ever written up, and the question is superseded by the per-behaviour metric panel in `docs/independent_opponents.md` Phase 4 (bolster/tactic/recruit/chain rates as first-class eval signals). |

---

### Tier 4 — small / cosmetic

| # | Issue | Difficulty | Effect |
|---|---|---|---|
| C20 | Eval runs 20 episodes — std error on a 0.49 WR estimate is ~5%. Bump to 50. | low | log readability |
| C21 | `EloTracker` updates after every eval game — noisy; consider a running average | low | log readability |

**C18** (critic train/eval mode): already fixed — `self._critic.eval()`/`.train()` are toggled around the rollout loop and the update in `ppo.py` (e.g. lines 337/344/391/629/655). No action needed.

**C16** (`iter_minibatches` permutation): already correct — a fresh `np.random.permutation` is generated on each call, so minibatches are uncorrelated across PPO epochs. No action needed.

**C10** (shared encoders): obsolete. See `docs/rl_algorithms.md` → *Architecture: shared vs separate actor-critic encoders*.

**C17** (truncation reward step function): **done** — replaced with a base-diff-proportional value 2026-07-03. See `docs/history.md`.

**C22** (`score_deque` rolling mean): already satisfied — `_score_deque` has `maxlen = print_every * collect_episodes` (currently 640), so `score_main` already averages across the last ~10 batches, not just the current one. No action needed.

---

### Recommended next steps

> **Superseded twice.** The live sequencing is now `docs/next_iteration.md` §5 (run the shipped
> critic fix: rows 2b and 3). The **B/A/L** block above holds the newer proposals and its own
> order in §N.4. What follows is kept as the record of what this file recommended on 2026-08-01;
> several of its premises — notably "the policy essentially never bolsters/recruits" — were
> corrected by `next_iteration.md` §3.3 and §3.7.

**The live sequencing is `docs/independent_opponents.md` §7**, not this file: build the scripted
exploiter panel, wire it into *both* the PPO pool (fixing the ~100% policy-derived finetune
schedule) and ExIt data-gen, then add the agreement / per-behaviour guards. Read alongside it:
**#15** (the independent `LookaheadBot` into the pool) is the cheapest piece of that same fix,
and **#13** feeds its success criterion (bolster/tactic rates moving off ~0, not just Elo).

After that, in rough order of information per hour: **#12** (exploitability probe — decides
whether any PSRO investment is justified), **#14** (blunder finder → puzzle suite, which makes
every future weakness a standing regression test), then the training levers **#16–#19** and the
cheap owed A/Bs **#20** / **#11 P11c**. **#21** (online play) stays last by design, and **#10** /
**#22** (action-space work), the parked threat-plane variant, and **C20/C21** are opportunistic —
pick them up when adjacent work already pays their setup cost (an `OBS_VERSION` bump, an
action-space rebuild, a parallel-rollout tuning pass).

C8 (LR decay) and C9 (clean true-greedy eval) are **done**; ideas 1–9 are retired (above).
