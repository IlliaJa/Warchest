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

> **Read [State review (2026-08-16)](#state-review-2026-08-16--the-strength-is-already-in-the-box-nothing-is-taking-it-out)
> at the bottom of this file first.** It audits the whole list against four measurements and
> concludes that no item here is worth a step change, while a 0.74-vs-current agent already
> exists (the current checkpoint + 1 s of PUCT). It re-sequences §N.4. Then read **R.8** (what
> search actually adds, and the A1+A3 verdict), **R.9** (pre-launch list for a PPO run) and
> **R.10** (a symbol-level read of ExIt: R.8's budget fix is validated on two independent axes,
> and what is left is that the teacher reads the opponent's hand plus four missing AlphaZero
> mechanisms — R.10.1 and R.10.2 are retractions of R.10's own first draft). R.10.8 is the
> current order.

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

> **DONE — shipped 2026-08-16 (`policy_factored_v2` + `critic_v5`), measured 2026-08-18.**
> Implemented and in the default path; the *mechanism* is in place and partially trained, and
> the *strength* gate came back negative (pooled +4.0 % [−3.9 %, +11.9 %]). Nothing here is
> pending work — see the two blocks at the end of this item before reopening it.

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

**Measured 2026-08-18 (`src/app/eval_a1_a3.py`), and the strength gate is NOT met.** First v2 run
(`warchest_ppo_20260817-2102`, hidden 128) against the v1 baseline (`warchest_ppo_20260810-0802`):

* **The table did partially learn.** Learned row norms 0.73 → 0.92, mean pairwise distance
  1.04 → 1.33. Of the three planted frozen-row collisions, Knight/Pikeman separated to 2.24x the
  init scale and Ensign/Marshall to 1.47x, but **Swordsman/Berserker/Mercenary stayed at 1.15x**,
  i.e. still indistinguishable. Gradient share across the frozen columns is uneven:
  `has_defensive_trait` 22.6 %, `coin_count` 16.6 %, down to `tactic_deals_damage` 4.4 %.
* **No detectable effect on play.** `comps` mirrors each forced archetype across both nets, so a
  composition's own strength cancels: pooled difference **+4.0 % [−3.9 %, +11.9 %]** over 300
  decided games per arm. `tempo` came up **+24 % [+4.5 %, +41.1 %]** and did not survive
  Bonferroni across the six archetypes; a confirmatory single-archetype run at a fresh seed and 6x
  the games returned **+3.3 % [−4.6 %, +11.2 %]**. Textbook winner's curse — the lead is dropped.

So A1 is not harmful and is cheap, but nothing yet shows it buying strength. Two traps this run
exposed, both worth knowing before reading any future number here: forcing the archetype onto one
arm only confounds "this net plays the deck worse" with "the deck is weak" (unmirrored, `support`
read 0 % for v2 when the mirror shows v2 *ahead*), and `Policy.act` output must go through
`WarChestEnv.remap_action` for P2 or every P2 ply silently degrades to a random legal move and P1
wins nearly every game.

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

> **DONE — shipped 2026-08-16 (`policy_factored_v2`), measured 2026-08-18.** The one item in
> this section whose mechanism is *confirmed against a provable control*: the hand now re-ranks
> the board (46.4 % of verbs change their top-1 cell) where v1 was arithmetically incapable of
> it (0.00 %). Strength effect: none detected. Policy only — `critic_v5` deliberately has no
> FiLM, see the last paragraph of this item.

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

**Measured 2026-08-18 (`eval_a1_a3.py hand`) — A3's mechanism is confirmed, its payoff is not.**
The test holds a board *and its legal-action mask* fixed, substitutes other states' `global`
vectors, and asks how much the within-verb preference over the 49 cells moves. It comes with a
control that cannot be argued with: on v1 the answer is **provably exactly zero**, because the
broadcast globals shift all 49 logits of a verb by the same constant and cancel in the softmax
across cells. Result on the first v2 run: **0.419 mean total-variation distance and 46.4 % of
verbs changing their top-1 cell**, against v1's 0.00000 / 0.00 %. FiLM is also demonstrably not
stuck at its zero init — mean |γ| 1.83 / 1.77 / 1.13 across the three blocks, and, the number that
matters, per-channel γ spread *across observations* 0.74 / 0.60 / 0.38 (a large but constant γ
would be a learned per-channel gain, not conditioning).

So the hand now re-ranks the board where it structurally could not before. That is the capability
A3 was for, and it is installed. What it has **not** yet produced is strength: see A1's measurement
block above — the pooled head-to-head difference is +4.0 % [−3.9 %, +11.9 %], consistent with zero.

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

**DEFERRED 2026-08-16 — measured, and the cheap fix wins. Full record in `docs/decision.md`.**
Both prerequisites were run. §5 row 2b (`src/app/eval_critic_target_ab.py`): at matched arch,
`hidden_dim` and data the shaped-return advantage is **~1.3×, not ~2×**, and the board-blind
control gains almost as much as the board arm — so most of it is "a less noisy target for
everyone". A6 itself (`src/app/eval_value_calibration.py`): against `as_is` / `platt` /
`isotonic` — three monotone maps of the same scalar, sharing one AUC by construction, with
isotonic the ceiling on *any* recalibration — a frozen-trunk `z`-head buys **+0.008 AUC** and
is **worse calibrated** than plain rescaling (ECE 0.036 vs 0.018). Two Platt floats capture
**78 %** of the total achievable Brier gain (ECE 0.118 → 0.031). And the search half of the
argument does not hold either: Q-spread / typical-PUCT-U = **1.27**, so the shaped scale is
already commensurate with `c_puct`. **Do instead:** save `a`/`b` next to
`return_mean`/`return_std` in `save_critic_checkpoint` and delete
`LookaheadCriticBot._calibrate_value_scale`. Blind spot that could reopen this: the gate's head
sits on a **frozen** trunk, so it cannot see what joint training would buy — but proving that
now costs a full PPO run. If revived, build it KataGo-style (shared trunk; separate winrate /
margin / ownership-like heads) **together with A7**, not as a standalone arch.

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
| 8 | ~~**A1 + A3**~~ together | ~~one `OBS_VERSION` bump buys both~~ **Shipped 2026-08-16** as `policy_factored_v2` + `critic_v5`; the bump premise was wrong (the contraction runs inside the net, v11 is byte-identical). **Measured 2026-08-18 via `eval_a1_a3.py`: A3's mechanism confirmed (46.4 % of verbs re-rank their top-1 cell when only the hand changes, against a provable 0.00 % for v1), A1's table partially learned (2 of 3 planted collisions separated), and NEITHER moved strength — pooled +4.0 % [−3.9 %, +11.9 %].** Not harmful, not yet paying. Next lever is elsewhere: row 7 (L1, B4) is still open and is data/throughput rather than architecture | moderate |
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

---

## State review (2026-08-16) — the strength is already in the box; nothing is taking it out

Written in answer to a direct question: *looking at what is left on this list, is there anything
that would put a new model at ~70 % against the previous one?* Read on its own terms — it is not
a new idea, it is an audit of where the remaining strength is, with four measurements taken for
it. Two of them are **§5 row 9 of `next_iteration.md`, which had never been run** in ~6 weeks of
it sitting at the bottom of the sequencing table.

**The short answer.** No single item on the B/A/L list is worth 70 %. But a **0.74 agent against
the current checkpoint already exists today** — it is the current checkpoint with one second of
PUCT in front of it. The whole gap is that nothing puts that strength back into the weights. The
list below has spent five weeks improving the *evaluator* (critic_v2 → v3 → v4 → v5, each worth
a few pp on a sibling-ranking metric) and zero weeks on the thing that consumes an evaluator.

### R.0 The four numbers this review rests on

None appears in any other doc. R.0.1/R.0.2 are new runs; R.0.3/R.0.4 are arithmetic on an
existing training log. Commands in R.7; the whole set is reproducible in ~15 min.

**R.0.1 — Search beats the raw policy, and it scales.** `next_iteration.md` §5 row 9, first run.
`ckpt_20260810-0802` against `PuctBot` built on *that same policy* (priors) plus
`lookahead_critic_v5.pth` (`critic_v2`, alive trunk). 100 games per pair, colours balanced,
`se(WR) ≈ 4.7 pp`:

| PUCT budget | node expansions / move (measured) | WR of **PUCT** vs the raw policy | Elo gap |
|---|---|---|---|
| 0.1 s | **~14** (12–19 under 12-way contention; ~30 idle; 2 on a loaded box) | **0.66** | +114 |
| 1.0 s | **~97** | **0.74** | +180 |

Both resolve — 0.66 and 0.74 are 3.4 σ and 5.1 σ from 0.5. (The gauntlet prints
"intransitive-triple fraction 0.000", but a two-agent field has no triples; ignore it.)

**Amended by R.8 (2026-08-18).** Repeated on the same checkpoint with a *different* tool and
the newer `critic_v4` leaf, the 0.1 s row measures **0.562 ± 0.055** rather than 0.66 — 1.4 σ
apart, so read the thin-search row as *"somewhere around 0.5–0.6, i.e. little to nothing"*,
which is what R.0.2 predicts and what R.8 confirms on a second checkpoint (0.487). The 1.0 s
row reproduced almost exactly (0.725 on the new checkpoint). **The load-bearing claim is the
slope, and the slope got steeper, not flatter.**

Two readings, and the second is the one that matters:

- **Search works in this game.** Every previous statement to the contrary
  (`independent_opponents.md`'s ExIt collapse, "more search budget is not a strength lever" in
  *Checked and rejected*) was measured with a **board-blind critic** —
  `next_iteration.md` §3.4 says so explicitly ("every search result on record has a
  board-blind leaf", `lookahead_critic_v4.pth` md5-identical to the dead
  `warchest_critic_20260727-0506.pth`). This is the first search measurement on a live trunk,
  and the sign flipped.
- **The scaling is ordinary and therefore extrapolable.** 14 → 97 expansions is 2.8 doublings
  for +66 Elo, i.e. **~23 Elo per doubling of search**. That is a normal MCTS curve. Three more
  doublings (~800 expansions, AlphaZero's actual setting) projects to ~0.81 and is affordable
  *offline*, which is exactly where a distillation teacher lives.

**R.0.2 — The game tree is tiny, and the search budget is smaller.** Over 12 random-play games:
**mean 10.5 legal actions per decision, median 9, p10 3, p90 19, max 37.** Against that,
`PuctBot`'s shipped configuration (`max_branching=8`, `time_budget=0.1`, and the *same* 0.1 s in
`ppo.py`'s `puct_time_budget` **and** in `expert_iteration.py`'s `--time-budget` default) buys
~14–30 expansions. The root alone consumes 8 of them.

**So the ExIt teacher's visit distribution is built from ~3 visits per root child.** PUCT's first
`b` selections visit each child exactly once (the `U` term dominates at `N=0`), so at 14
expansions the visit counts are *the prior, plus about one bit of information per child*. The
recorded ExIt symptom — teacher/student agreement **0.94–0.95**, every round weaker than base —
is not a tuning failure or a self-play-collapse subtlety. **It is arithmetic.** Compounding it,
`--dirichlet-frac` defaults to **0.03** where AlphaZero uses 0.25, so root exploration is 8×
weaker than the reference too.

Cost breakdown for why 14: an expansion is one policy forward (0.86 ms) + one critic forward
(0.85 ms) + clone/apply, unbatched, single-threaded CPU — ~3.3 ms measured end to end. Table A
already had every input to this number; nobody divided.

*Correction to a figure in circulation:* `independent_opponents.md` §2 mechanism 1 says
"~100–300 sims/move on a 1875-wide action space with `max_branching=8`". Both halves are off.
The expansion count is **14–30** at the shipped budget (measured from `PuctBot`'s own
`nodes_visited`, printed per worker by the gauntlet), and the *effective* branching factor is
**10.5 legal actions**, not 1875 — `max_branching=8` is therefore only mildly truncating
(p90 = 19), which is worth knowing because it means widening it is **not** the fix. The fix is
the number of evaluations, and what is done with them.

**R.0.3 — The dense reward is 41 : 1 base-differential over material, and out-earns winning.**
From `logs/ppo_20260809-195643.log`, the last completed run, per-episode score decomposition at
batch 1497–1500 (anneal at its 0.1 floor):

```
shaping=0.205   terminal=0.125   material=0.005   holding=0.001   tempo=-0.060   attack=0.000
```

`r_shaping` is the base-diff PBRS **only** (`rollout_core.py:261`), and it is the one term
deliberately **not** annealed (`ppo.py`, "Base-diff PBRS (SHAPING_C) is intentionally left
constant"). So, late in every run:

| what the agent is paid for | per unit | per game |
|---|---|---|
| one base of differential | `SHAPING_C` = **0.050** | up to 0.30 |
| one boxed enemy coin | `C_MAT · anneal` = 0.015 · 0.1 = **0.0015** | 0.006–0.015 |
| one elapsed turn | `TURN_TEMPO_REWARD` = −0.002 | −0.060 |
| winning | ±1.0 | mean **+0.125** |

**One base = 33 boxed coins.** The user's own domain note (`next_iteration.md` §6) is that real
games box 4–6 of 16–20 coins and hard games reach ~10 — so *the entire material axis of a whole
game is worth less than a third of one base* in this reward. And `γ^T·Φ(s_T) = 0.205` against a
mean terminal of 0.125: **the shaping potential pays more per episode than the win does.** PBRS
is optimal-policy-invariant in the limit, but a potential whose range (0 … 0.30) is a third of
the terminal's (±1) and whose *realised* per-episode payout exceeds it is not a hint toward the
objective, it is a competing objective under any finite-horizon, entropy-regularised,
function-approximated optimiser — which is what PPO is.

**R.0.4 — The policy is converged, collapsed, and not improving.** Same log, final batches:
`ent = 0.498` against `ent_max = 2.515` (**ent_frac 0.20**), `verb_ent = 0.298` against
`ln(11) = 2.40` (**12 %**), `bolster_per_ep` 3.5 → **0.19**. And the eval that matters:
**`vs warchest_ppo_20260809-1734: score=0.450 (9W/11L/0D)`** — 1500 batches, ~9 h, and the
result is a *loss* to its own predecessor. `critic_mae 0.226` against `ret std 0.617` is
~0.37, i.e. the critic is no longer the underfit it was. **The plateau is not the critic.**

### R.1 "It rushes bases and has no strategy" — the technical translation

Four distinct mechanisms. They are not competing hypotheses; all four are measured, and they
compose. Ordered by how much of the complaint each explains.

**R.1.1 — There is no lookahead at inference, and that alone is 180 Elo (R.0.1).** "Plays like a
beginner" is the standard, textbook signature of a raw policy network with no search: locally
plausible moves, no line, no threat conversion, no prophylaxis. `next_iteration.md` §1 named this
in the first sentence it ever wrote and then spent six weeks on the critic. The measurement now
exists: the *same* network, given 1 second of tree, is a 0.74 agent against itself.

**R.1.2 — The credit-assignment horizon is shorter than the payoff horizon of every non-race
strategy.** `λ = 0.90` (shipped L2) gives an effective horizon `1/(1−γλ) = 9.2` **main-actor
decisions**, against ~42 per game. A base claim pays 0.05 *now*. The kill → steal → hold chain
that §3.3 measured as the **only** way a base ever changes hands (53 steals from occupied bases,
0 from empty) pays 0.0015 for the kill, at t, and 0.05 for the steal, at t+1..t+4, and the hold
is what actually decides the game 20 decisions later. Within a 9-decision window, racing to an
uncontested base strictly dominates fighting for a contested one — *in the reward*, whatever is
true of the game. This is the credit gap in the §*Method* table, and the diagnostic it asks for
(is the verb sampled early then decayed?) reads **yes**: `bolster_per_ep` 3.5 → 0.19.

Note the interaction with L2, which is worth stating because it looks like an argument against a
shipped change: dropping λ was correct *for making a repaired critic count*, and it simultaneously
shortened the window in which a slow strategy can be seen. Both are true. The fix is not to raise
λ back — it is R.1.3.

**R.1.3 — Material has no terminal consequence in the env, so the reward is not wrong so much as
the MDP is shallow.** `docs/game_mechanics.md`: the win table has exactly **two** rows —
control 6 bases, or truncate at round 50. There is no elimination condition. Coins in the box are
permanently gone and the game never registers it. So a boxed coin is worth only what it opens up
positionally, and the shaping (R.0.3) prices that at 1/33 of a base.

~~**Consequence, and it is the uncomfortable one: "rush bases" may be correct play in the game as
implemented, and the strategy the user is looking for may not be in the MDP.**~~

> **ANSWERED 2026-08-18, and the escape hatch is gone (see R.8).** The user's ruling: the only
> win condition really is 6 bases, but reaching 6 against a competent opponent goes through
> positional control and removing enemy units — **the game cannot be won by racing alone.** So
> the env is fine and R.0.3 is a confirmed *defect*, not a possible feature: the reward prices
> the goal at 33× its only means. `next_iteration.md` §6 parity question 1 is **closed**; do not
> re-open it. `L4`'s hanging-material potential is re-opened, on a rule.

For the record of what the evidence looked like before that ruling, and why it was not enough:
B1's per-dial sweep found `pos` (the racer dial) is what buys wins while `durability` takes the
win rate 0.75 → 0.00, and B3's design rests on the rules making a parked unit an absolute lock.
Both are measurements *inside* a pool where nothing punishes a racer, so neither could have
settled it — which is why it took a domain answer rather than another run.

**R.1.4 — The yardstick cannot see a 70 % improvement even if one happened.** `wr_vs_greedy_eval`
is pinned at 0.95–1.00 (saturated, and the standing rule says do not optimise against it).
`wr_vs_reference_eval` compares against *the previous checkpoint*, which drifts every run — a
moving target in a space with a measured 0.11 intransitive-triple fraction, so "beats its
predecessor" is compatible with going in circles. And it is 20 games (`C20`), `se ≈ 11 pp`. The
target "70 % vs the previous model" is currently being measured with an instrument whose noise
floor is a third of the effect. **A fixed absolute anchor is missing**, and R.0.1 supplies the
obvious one: `PuctBot @ 1.0 s` over a frozen critic is stronger than the policy, does not drift,
and is transitive in every field it has been placed in.

### R.2 Why nothing currently on the list is worth a step change

Not a criticism of the items — a statement about what class they are in.

| class | items | ceiling |
|---|---|---|
| evaluator quality | A1, A2 ✓, A3, A4, A5, A6 ✗, critic_v2..v5 | each measured at +3 to +10 pp on a *sibling-ranking* metric. A1+A3 (running now) will land here too — the honest prior is ±0, exactly as predicted |
| opponent coverage | B1 ✓, B3, B5 ✗, B8, #15 | B8 measured the blocker: every independent bot in the repo **loses** to the policy, and after per-opponent advantage centring a group with no spread contributes near-noise. Coverage cannot help until an opponent is *strong* |
| data / throughput | L1, B4, B7, #11 | multipliers on a loop that is not producing improvement. 2× of ~0 is ~0 |
| reward | L4, L8 ✓ | R.1.3 says the ceiling here is set by the win condition, not the weights |
| measurement | #13, #14, L9, row 5 | necessary, never sufficient |

The pattern is visible in the sequencing table itself: §N.4 has ten rows and **nine of them
improve an input to a loop whose output has been flat for three runs.** The one that does not is
B4, and its stated purpose is throughput.

### R.3 The step change that is available: put the search in the loop, properly

The premise ExIt needs — *a teacher measurably stronger than the student* — was **false every
time ExIt was run** and is **true today**:

| ExIt's recorded fault | status |
|---|---|
| "teacher ≡ student", agreement 0.94–0.95 (`independent_opponents.md`) | **explained and stale.** ~14 expansions over 8 root children cannot disagree with the prior (R.0.2), and `dirichlet_frac=0.03` is 8× below AlphaZero's. Not a property of self-play |
| the critic objective is `MSE(critic, z)` (§3.3b) | **confirmed, and `independent_opponents.md` §1 already contains the controlled experiment nobody read as one.** Round 0 ran in `shaped` mode off the good PPO critic: **agreement 0.77, and it is the best round of the 30 (BT 1089)**. Rounds ≥1 switched to the self-distilled z-critic: agreement jumped to ~0.94 and strength fell away monotonically. The critic's *target* is the variable that moved, and row 2b independently measured its sign |
| *(unrecorded, and the largest)* every ExIt run used the **dead-trunk** critic | §3.4 pins the byte-identity. The trunk was repaired 2026-08-07 and ExIt has never been re-run |
| *(operational)* the search bots load `data/lookahead_critic/lookahead_critic_v{N}.pth`, not `data/warchest_critic_*.pth` | the newest there is **v5 = `critic_v2` (08-08)**. `critic_v4` (08-10, the shipped A2 default) was never copied in, so **every gauntlet and ExIt search since 08-10 has used a one-generation-stale critic** — including R.0.1, which is therefore a *lower* bound. Fix: copy the file, or make `_latest_critic_path` fall back to the newest `warchest_critic_*.pth` |

So the work is a single coherent change with four parts, and each part has a named precedent.

**R.3a — Make the teacher a real search (throughput).** 14 expansions is a budget problem, and
the budget is 3.3 ms of unbatched CPU inference per node. Four multipliers, all standard, roughly
independent:

1. **Batched leaf evaluation with virtual loss** (Leela/KataGo). Table A: `value_batch(64)` is
   0.37 ms/state vs 0.85 single. ~2.3× on the critic side alone; more once the policy prior is
   batched with it.
2. **Tree reuse across moves.** `_act_once` builds a fresh `_Node` root every call and throws the
   subtree away. Reusing the chosen child's subtree is a free ~2–3× effective sims in a game
   where the opponent's reply is one of ~10.
3. **A small distilled net for the search leaf** — this is exactly **B4**, and R.0.1 is the first
   argument for it that is about *strength* rather than throughput. 32-wide ≈ 0.25 ms est.
4. **Offline is not the hot path.** ExIt data generation is not the rollout loop. Even with none
   of the above, `--time-budget 2.0` is affordable for a distillation teacher and buys ~500
   expansions today.

**R.3b — Use a search target that is valid at low simulation counts: Gumbel AlphaZero.**
*Policy improvement by planning with Gumbel*, Danihelka et al., ICLR 2022. Gumbel top-k sampling
at the root + sequential halving over the sampled actions + a policy target built from
**completed Q-values** instead of raw visit counts. Its entire point is that it gives a
**guaranteed** policy improvement at **n = 16–32 simulations**, where vanilla AlphaZero's
visit-count target is provably not an improvement operator. Warchest's numbers could not fit the
paper's premise better: branching **10.5** (its worked examples are 9×9 Go and smaller), and a
budget that currently delivers **14**. This is the single highest-leverage borrowed idea in this
document — it converts the existing 14-expansion search from "noise around the prior" into a
valid teacher *without buying any throughput at all*, and the throughput work in R.3a then
compounds on top.

**R.3c — Train the ExIt critic on n-step bootstrapped returns, not `z`.** Row 2b's re-scoped
conclusion, never applied. This is MuZero's value target and it is the correct one the moment
intermediate rewards exist (`docs/decision.md` 2026-08-16 already writes this down).

**R.3d — Re-derive the exploration constants at the new budget — do *not* just restore
AlphaZero's.** An earlier draft of this item said "`--dirichlet-frac 0.03 → 0.25`". That is
wrong as an instruction: **0.03 is a measured value, not an oversight.** The CLI's own help
records why — at 0.25 the mean visit entropy came out 0.87 nats against a pre-distill policy
entropy of ~0.6, so distillation *flattened* the policy every round, and 0.03 measured 0.586.
The correct reading is that **noise fraction and search budget are not independent**: 0.25 is
affordable exactly when Q has enough visits to outcompete a noisy prior, and at 14 expansions it
does not. So raise the budget first, then re-test 0.25 and read the entropy pair
(`preflight` gate 4 prints it before a full generation). Same treatment for
`--temperature`/`--temp-moves` against the ~42-decision game length, and for `c_puct` now that
`eval_value_calibration.py puct` has measured Q-spread/U = 1.27.

**The gate, and it is cheap.** Before generating a single game of ExIt data, measure the teacher
against the student in the gauntlet — the R.0.1 command. **A teacher below ~0.65 is not worth
distilling**; today, at 1.0 s, it is 0.74. Then, and only then, one distillation round with the
agreement metric read against something that is not the student (B4's net, or the pre-round
policy frozen).

### R.4 What to borrow from outside, and what each is evidence for

Ordered by expected value here, not by fame. Every row names why it applies to *this* project's
measured numbers rather than in general.

| # | Idea | Source | Why it fits Warchest specifically | Cost |
|---|---|---|---|---|
| 1 | **Gumbel AlphaZero** — Gumbel top-k root sampling + sequential halving + completed-Q policy target | Danihelka et al., ICLR 2022 | valid policy improvement at n=16–32 sims; branching is 10.5 and the current budget is 14 (R.0.2). Directly repairs the mechanism that made ExIt's teacher ≡ its student | mod |
| 2 | **Ownership + score auxiliary heads** | KataGo (Wu 2019) | KataGo's headline sample-efficiency result. The Warchest analogue is *free*: **per-base final-control** (10 bases, a 10-way label straight off the trajectory) and **final base differential**. This is `A7` with a concrete high-signal target, and it installs "which bases will I hold" — the exact quantity §1's thesis says the net cannot compute — at zero inference cost | mod |
| 3 | **Playout-cap randomisation** | KataGo | most self-play moves get a cheap search, a small fraction get a full one, and **only full-search moves produce a policy target**. Exactly the right trade when a full search costs 500× a policy forward (Table A) and games are 85 plies | low |
| 4 | **Tree reuse + virtual loss + batched leaf eval** | Leela Chess Zero, KataGo | R.3a. ~5× on the same hardware, no new ideas | mod |
| 5 | **Forced playouts + policy-target pruning** | KataGo | makes root exploration noise usable at low visit counts instead of poisoning the target — the failure mode `dirichlet_frac=0.03` was presumably set to avoid | low |
| 6 | **n-step bootstrapped value target** | MuZero | R.3c; row 2b already measured the direction | trivial |
| 7 | **Supervised warm start from the search** | AlphaStar | `L6`, but the teacher is now the PUCT bot rather than a 200-line heuristic. Also removes "how fast did it escape the random phase" as a confound in every A/B (the standing-rule problem this file opens with) | low |
| 8 | **League / main-exploiter play** | AlphaStar; PSRO | the principled fix for B8's blocker (no independent opponent is strong enough to be worth pool weight). Expensive, and it is a *robustness* lever — park it behind R.3 | high |
| 9 | **R-NaD** | DeepNash | unexploitability, not strength. `docs/decision.md` 2026-07-03 already argues it is orthogonal. Not the constraint |  — |
| 10 | **Transposition table + make/unmake + killer moves** | classical chess engines | `B7`. At branching 10.5 a real engine reaches depth 6–8, which would make `LookaheadBot` a genuinely strong independent opponent and unblock B8 from the other side | mod |

Row 2 deserves one extra sentence, because it is the cheapest thing in this document that is not
a measurement: **the ownership head is a supervised, dense, per-cell label available for free in
every trajectory already being collected**, it cannot leak at inference, it is independently
A/B-able, and the precedent (`critic_v2`'s board-only aux head, which took the positional tie
rate 93 % → 0 %) is *in this repo*.

### R.5 Direct answer — should a PUCT model be trained?

**The measurement was worth an hour and it changed the picture; the training run is worth doing
next, but not with today's ExIt defaults.**

- The *measurement* (R.0.1) is done and is the most informative hour spent on this project in
  weeks. It is `next_iteration.md` §5 row 9 and it should have been run before the last three
  critic architectures.
- A *training* run of ExIt as currently configured would reproduce the recorded collapse, because
  R.0.2 shows the teacher is the student by arithmetic at `--time-budget 0.1`. Do not spend the
  compute.
- A training run with **R.3b (Gumbel) + R.3d (noise) + `--time-budget 1.0` + R.3c (value target)**
  is the highest-expected-value run available, and it is the only route on this whole list with a
  plausible path to 70 %: the teacher is *measured* at 0.74 today, before any of the four fixes.
- Cheapest possible first step, ~2 h, no new code: copy `warchest_critic_20260810-0802.pth` to
  `data/lookahead_critic/lookahead_critic_v6.pth`, re-run R.0.1 (does the newer critic search
  better?), then one ExIt round at `--time-budget 1.0 --dirichlet-frac 0.25` and read the
  agreement metric. If agreement drops off 0.94 and the distilled policy beats base, the whole
  R.3 programme is justified by a two-hour experiment.

### R.6 Suggested order

Supersedes §N.4 for anything not already in flight. The A1+A3 run should finish first — not
because it will move the needle, but because interleaving arches with a search change would make
both unattributable.

| order | work | why | cost |
|---|---|---|---|
| 0 | ~~**Env parity audit** (R.1.3)~~ | **done 2026-08-18** — 6 bases is the only win condition, but it is reached through control and removing units; racing alone cannot win. So the reward *is* the defect and every reward item below is unblocked (R.8) | ~0 |
| 1 | **Fix `_latest_critic_path`** + re-run R.0.1 on `critic_v4` | one-line staleness bug silently affecting every search measurement since 08-10 | ~0 |
| 2 | **Absolute anchor** — add frozen `PuctBot @ 1.0 s` as a permanent eval opponent, and raise eval to 50 games (`C20`) | R.1.4: no current instrument can resolve the improvement being aimed for | low |
| 3 | **R.3b Gumbel root** + **R.3d exploration constants** + `--time-budget 1.0` | the teacher repair. No throughput work needed to test it | mod |
| 4 | **One ExIt round**, gated on teacher-vs-student ≥ 0.65 and agreement measured off-student | the two-hour experiment that justifies or kills the programme | low |
| 5 | **R.4 row 2 — ownership + base-diff auxiliary heads** (= A7, concretised) | cheapest non-measurement item with a strong outside precedent; independent of 3–4, so it can run in parallel | mod |
| 6 | **R.0.3 reward re-balance** — ~~anneal `SHAPING_C` like the others~~ **done 2026-08-18**, `SHAPING_C` now rides `shaping_anneal` (`base_shaping_anneal` in `rollout_core.play_episode`, `--no-anneal-base-shaping` for the old arm). Holds base : material flat at 3.3 : 1 instead of drifting to 33 : 1, and puts the late-run dense payout ~6× under the terminal. **Not yet A/B'd.** Still open: add **L4's `Φ_risk`** | no longer conditional (row 0 answered). Stop paying more for base differential than for winning, and install the prophylaxis the teacher provably does *not* carry (R.8.1) | trivial–low |
| 7 | **R.3a throughput** (batched leaf + tree reuse + B4) | only worth it once 3–4 say search is the lever; then it is a 5× on the lever | mod |
| 8 | **B7 / #14** — real engine for `LookaheadBot`; blunder finder | unblocks B8 from the strength side; turns "plays like a beginner" into a per-move number | mod |

### R.7 Reproducing R.0

```bash
# R.0.1 — next_iteration.md §5 row 9, the search-vs-policy head to head
python src/app/gauntlet.py --bots policy puct \
    --checkpoints data/warchest_ppo_20260810-0802.pth \
    --k-games 100 --puct-time-budget 0.1 --n-workers 12
python src/app/gauntlet.py --bots policy puct \
    --checkpoints data/warchest_ppo_20260810-0802.pth \
    --k-games 100 --puct-time-budget 1.0 --n-workers 12
# read `nodes_visited avg` in the per-worker puct log lines — that is R.0.2's expansion count
```

R.0.2's branching factor is 12 random games under `env.get_possible_actions()`; R.0.3 and R.0.4
are `grep score_parts` / the final `[eval]` line of `logs/ppo_20260809-195643.log`.

**Standing caveats.** R.0.1 is 100 games per budget, one policy checkpoint, one critic — it
establishes *that* search helps and roughly *how much per doubling*, not a precise curve, and it
has not been repeated across checkpoints. The 0.81-at-800-expansions projection is a linear
extrapolation in log-sims from two points and should be treated as a hypothesis, not a number.
R.1.3 is the one item here that could invalidate several others, and it is unresolved.

---

## R.8 Pre-flight for the ExIt run (2026-08-18)

R.1.3 is **answered by the user, and the answer removes the escape hatch**: the only win
condition really is 6 bases, but reaching 6 against a competent opponent goes through positional
control and removing enemy units — *the game cannot be won by racing alone.* So the env is not
the problem, and every consequence flips:

* **R.0.3 is no longer a suspicion, it is a confirmed defect.** A reward that pays 0.05 for a
  base and 0.0015 for a boxed coin prices the goal at 33× the only means of reaching it. The
  agent's base-rushing is not correct play that merely looks bad — it is the local optimum this
  reward defines, and it survives only because no opponent in the pool punishes it. The user
  confirms a human punishes it immediately.
* **`L4`'s hanging-material potential moves up**, and the reward axis is properly re-opened —
  on a rule this time, per L4's own argument.
* The parity branch of `next_iteration.md` §6 is **closed**. Do not re-open it.

Everything below was built and measured to make the next ExIt run count.

### R.8.1 What the search actually adds — `src/app/eval_search_delta.py` (new)

R.0.1 said search is worth ~180 Elo; it did not say what of. That matters, because you only
distil what the teacher has. New tool: plays `PuctBot` against the raw policy that supplies its
priors, colours balanced on shared seeds, and records per side the verb mix, `own_at_risk` /
`opp_at_risk` read off the mover's own ego observation (globals 208/209 — which, per B5, nothing
in the repo had ever read), enemy coins boxed, and bases held at the end.

80 games, `ckpt_20260817-2102` + `critic_20260817-2102` (`policy_factored_v2` / `critic_v5` — the
A1+A3 run), 1.0 s budget:

| | puct | policy | Δ |
|---|---|---|---|
| WR | **0.725** | 0.275 | ±0.050 |
| enemy coins boxed / game | **6.09** | 4.78 | **+1.31 (+27 %)** |
| bases held at end | **5.16** | 4.01 | **+1.15** |
| `own_at_risk` while to move | 0.0220 | 0.0220 | **±0.0000** |
| attack share | 0.1055 | 0.0928 | +1.3 pp |
| control share | 0.0854 | 0.0743 | +1.1 pp |
| tactic / select share | 0.0664 / 0.0471 | 0.0575 / 0.0356 | +0.9 / +1.2 pp |
| deploy / move share | 0.1512 / 0.2719 | 0.1719 / 0.2916 | −2.1 / −2.0 pp |

**The answer is half of what was hoped, and the half that is missing is the half the user's
domain claim is about.** Search wins by *cashing* material and converting it — 27 % more enemy
coins boxed, more attacks, more tactics, more control, less aimless deploying and walking. It
does **not** win by hanging less: `own_at_risk` is identical to four decimal places. So the
teacher is a better *attacker and converter*, not a better *defender*.

Two consequences, and they are the operative ones:

1. **ExIt is still the right lever, and its target is legible.** "Take the kill, then convert" is
   exactly the mechanism §3.3 measured as the only way a base ever changes hands, and it is what
   the raw policy does not do. Distilling it teaches a real, named skill.
2. **Search will not supply prophylaxis, so something else has to.** "You attack so your unit is
   not attacked" is not in the teacher. That is a direct argument for the two items that install
   it *supervised* rather than by search: **L4's `Φ_risk = −c · own_at_risk`** potential (which
   cannot distort the optimum) and **A7's survival head** (will this stack lose a coin within 2
   plies). These are now complements to ExIt, not alternatives to it.

### R.8.2 The A1+A3 run did not come back worse — the in-run eval is too noisy to tell

The run's own final line reads `vs warchest_ppo_20260810-0802: score=0.350 (7W/13L/0D)`, which
looks like a regression. It is 20 games, `se ≈ 11 pp`. Re-measured in the gauntlet at **200
games**:

| | WR vs the other | BT Elo |
|---|---|---|
| `ckpt_20260817-2102` (A1+A3, `policy_factored_v2`) | **0.55 ± 0.035** | 1015.6 |
| `ckpt_20260810-0802` (baseline) | 0.46 | 984.4 |

So ~+31 Elo, 1.4 σ — a small positive, not the loss the log implied. **This is R.1.4 biting in
practice: the shipped instrument reported a 0.55 as a 0.35.** Raising eval to 50 games (`C20`) is
no longer cosmetic; it is the difference between reading a run correctly and abandoning a change
that worked.

And A1's *own* gate — behaviour on the mechanics the embedding was supposed to make legible —
moved a long way, which the pooled win rate hides entirely:

| | old ckpt | A1+A3 ckpt |
|---|---|---|
| `bolster_per_ep` (training log) | 0.20 | **2.52** |
| bolster share of decisions (gauntlet) | 0.0071 | **0.0746** — ~10× |
| plies / game | 73.5 | 83.9 |
| enemy coins boxed / game | 4.38 | 4.78 |

The unit-type embedding did what A1 argued it would: unit-specific mechanics came back into the
repertoire. `next_iteration.md` §3.7's "bolster is a collapsed mode at `P̄ = 0.029`" is **out of
date as of this checkpoint**.

### R.8.3 What shipped to make the run work

**`src/app/expert_iteration.py preflight` (new)** — one command that checks the four gates that
have each cost a run at least once, before paying for another. It exits with a verdict block:

| gate | what it measures | threshold | why it exists |
|---|---|---|---|
| 1 staleness | is the critic in force the newest one? | — | `--critic` resolves `data/lookahead_critic/`, which PPO does not write; the two drifted a full generation for a week (R.3, last row) |
| 2 search depth | real expansions/move at the configured budget | ≥ 50 | R.0.2. Run it on an **idle** box: measured 30/move idle, 14 under 12 busy workers, **2** on a loaded one, all at 0.1 s |
| 3 teacher strength | `PuctBot` vs the raw policy, head to head | ≥ 0.60 | `next_iteration.md` §5 row 9 — never checked before the 30-round run that got monotonically weaker |
| 4 target sharpness | `visit_entropy` vs `policy_entropy`, and agreement | visit ≤ policy, agree < 0.90 | the recorded `--dirichlet-frac 0.25` flattening failure, now readable on 6 games instead of a full generation |

Verified against the current checkpoint at the old 0.1 s budget, and it correctly refuses:
`search depth 2/move FAIL`, `teacher strength 0.500 FAIL`, `teacher divergence agreement 0.903
FAIL`, `target sharpness ok`. That is the run the project was about to launch.

**`--freeze-critic`** (`src/app/expert_iteration.py` + `distill(train_critic=False)`) — distils
the policy only, keeps the critic bit-identical and keeps the search in `value_mode='shaped'` for
every round. This is the configuration the record argues for and nobody ever ran: ExIt's only
round that made the policy stronger was **round 0**, the one round still on the PPO shaped-return
critic (agreement 0.77, BT 1089, best of 30), and every round on the self-distilled `z`-critic got
monotonically weaker. Freezing also removes the one place a scale bug can enter — the
`return_mean=0 / return_std=1` overwrite, which is correct for a `z`-critic and wrong for a
shaped one. With it, the loop has exactly **one** moving part, so a negative result is
attributable.

**`--time-budget` default 0.1 → 1.0**, with the reasoning in the help. Generation is offline; a
teacher that is the student by arithmetic costs 100 % of a run and returns nothing.

**Agreement guard in `_run_distill`** — warns loudly when pre-distill agreement ≥ 0.90, which
`independent_opponents.md` §7 asked for in July and which was never added.

312 tests pass.

### R.8.4 The run to launch, in order

```bash
# 0. the critic the search will use must be the newest one (R.3, last row)
cp data/warchest_critic_20260817-2102.pth data/lookahead_critic/lookahead_critic_v6.pth

# 1. gates, on an idle box. Tune --time-budget until gate 2 clears ~50 expansions/move
#    and gate 3 clears 0.60; expect somewhere around 0.5-1.0 s.
python src/app/expert_iteration.py preflight --time-budget 1.0 --preflight-games 60

# 2. once (and only once) all four gates pass — policy-only distillation, shaped critic,
#    small rounds so a regression is visible early
python src/app/expert_iteration.py loop --rounds 3 --games 300 \
    --time-budget 1.0 --freeze-critic --gauntlet-k-games 50

# 3. read, per round: post-round gauntlet BT (the only real answer), agreement before->after,
#    and re-run the behaviour delta on the distilled policy to check WHAT moved
python src/app/eval_search_delta.py --policy data/exit/round0_policy.pth \
    --critic data/warchest_critic_20260817-2102.pth --games 80 --puct-time-budget 1.0
```

The success condition is not CE/MSE falling. It is: **the distilled policy beats the base policy
in the post-round gauntlet at ≥ 50 games/pair, and `eval_search_delta` shows its boxed-coins and
control shares have moved toward the teacher's** — i.e. it absorbed the conversion skill rather
than just the prior. If round 1 is flat, stop and do not run rounds 2–3; the earlier 30-round run
is the cautionary case.

Two things to do **in parallel**, because they are independent of the loop and target the half of
the gap the teacher does not carry (R.8.1): **L4's `Φ_risk` potential** and **A7's survival head**.
And one thing to do **before the next PPO run** rather than this ExIt one: R.0.3's shaping
re-balance, now that R.1.3 has confirmed it is a defect.

---

## R.9 Pre-launch checklist for the reward-rebalance run (2026-08-18)

Context: R.0.3's re-balance is being implemented (base-diff PBRS joins the holding/material
anneal, `--no-anneal-base-shaping` reproduces the old arm). That fix is correct and correctly
plumbed — `annealed_base = base_shaping_anneal * (γΦ′ − Φ)` with one multiplier per episode, so
the telescope is intact; the knob reaches the parallel workers through
`rollout_collector.py`; `tests/test_shaping_anneal.py` pins it. What follows is what *else* the
run needs, ordered by what it costs to skip. **The constraint is that there is no budget for
small runs**, so everything here is either free, or buys the one run more batches.

### R.9.1 The blast radius of a reward change — one blocker, now fixed

`HeuristicEvaluator` imported `SHAPING_C` and `C_MAT` straight from `rollout_core`, and **five of
its eight coefficients derived from them** (`_c_base`, `_c_mat`, `RISK_COEFF`, `DUR_COEFF`,
`ECON_COEFF`). Everything downstream inherits that: `greedy_sim`, `lookahead`,
`lookahead_critic`, `policy_theta`, `random_eval`, and `PuctBot`'s `_leaf_potential` — which is
to say **the gauntlet's entire fixed agent field, the measurement of record**, plus the ExIt
teacher's leaf.

So re-balancing the training reward would have silently re-tuned every baseline it is measured
against. The yardstick would move with the treatment, in the one place the project has decided
it must not: every historical gauntlet number is quoted against these bots.

**Fixed 2026-08-18:** `evaluation.py` now defines its own `EVAL_BASE_C = 0.05` /
`EVAL_MAT_C = 0.015` — the values in force from the project's start through today, so every bot
is bit-identical to what produced every recorded result — and no longer imports from
`rollout_core`. `tests/test_evaluation.py::test_evaluator_scales_are_decoupled_from_the_training_reward`
pins it both ways (the module must not re-acquire the constants, and the values must not drift),
so this cannot come back as a side effect of touching the reward.

This is the same reasoning the concurrent
`test_shaping_anneal_scales_only_material_not_base` already applies to `shaping_anneal`: the
evaluator scores *leaf positions* for search bots and is deliberately not a mirror of the
training reward. R.9.1 extends that to the two anchor scales. If a *reward-matched* bot is
wanted later, add it as a θ member (`theta['base']` / `theta['material']` scale exactly these
two) so the frozen yardstick and the matched variant can both stand in the field.

### R.9.2 The one change that makes the run bigger: cut the finetune search opponents

Finetune is `p_pool 0.5 / p_lookahead_critic 0.3 / p_puct 0.2`. Arithmetic on the last run
(`logs/ppo_20260817-063713.log`, `rollout=34.1s`, `model_play=185.4s` aggregate over 6 workers,
`env=10.6s`), at 64 episodes × ~42 opponent plies × 0.1 s/move:

| slice | weight | core-s / batch |
|---|---|---|
| `lookahead_critic` | 0.30 | ~81 |
| `puct` | 0.20 | ~54 |
| `pool` snapshots | 0.50 | ~1 |
| main policy + env | — | ~13 |

**Two search opponents consume ~68 % of rollout core time for half the episodes** — and both are
now *measured to be weaker than the policy they are teaching*: the run's own final line reads
`wr_lookahead=0.710`, and R.0.1/R.8.1 put `PuctBot` at the same 0.1 s budget at **0.49–0.56**
against the raw policy. They also both load `data/lookahead_critic/lookahead_critic_v5.pth` —
`critic_v2`, 2026-08-08, two critic generations stale (R.3, last row).

So this is not B8's throughput argument any more, it is a strength argument: **the compute is
being spent on opponents that are no stronger than a pool snapshot costing 1/100th as much.**
Cutting `p_lookahead_critic_finetune 0.3 → 0.10` and `p_puct_finetune 0.2 → 0.0` into `p_pool`
takes rollout core time from ~196 s to ~55 s per batch, i.e. **roughly 3× the batches in the same
wall-clock** — 1500 batches in ~3 h instead of ~9 h, or ~4500 batches in the same 9 h.

Keep the *initial*-phase slice as it is (`p_lookahead_critic_initial = 0.30`): at batch 10 the
policy scored 0.04 against it, so early on it is a genuine teacher. It is only the finetune
slice, where the policy wins 71 %, that is pure waste. B8 predicted exactly this shape
("anneal it rather than flat-cut it") and now has the strength half of the argument too.

If some of the freed budget should buy *independence* rather than batches, the only
policy-independent option that is cheap enough is B1's `--p-random-eval-finetune 0.10` (~18 ms/move
against the search bots' ~100). Note B8's caveat honestly: `RandomEvalBot` is also weaker than the
policy, so it buys **state coverage**, not pressure — the θ family's measured contribution is a
4–5× behaviour spread, not Elo.

### R.9.3 Without this, the run cannot be read

R.8.2 is the direct evidence: the in-run eval reported **0.35** for a checkpoint that measures
**0.55 ± 0.035** over 200 games. At 20 games `se ≈ 11 pp`, which is a third of the effect being
looked for. Two fixes, both trivial, and they are the difference between reading an expensive run
and guessing at it:

1. **`eval_episodes` 20 → 50** (`C20`, on the list since forever, still cosmetic-tier in the
   table — it is not cosmetic for a single-run project).
2. **Pin the reference opponent.** `reference_policy_path` defaults to
   `latest_policy_checkpoint()`, so the yardstick drifts every run and "beats its predecessor"
   is measured against a moving target in a field with a recorded 0.11 intransitive-triple
   fraction. Pass `--reference-policy data/warchest_ppo_20260810-0802.pth` explicitly and keep
   that same path for every arm on this side of the reward boundary.

Also worth having, and nearly free at this point: a **frozen `PuctBot @ 1.0 s`** as a second eval
opponent (R.1.4). It does not drift, it is measurably stronger than the policy (0.725), and it is
the only absolute anchor available.

### R.9.4 What belongs in *this* reward edit, because it costs no extra run

**L4's two potentials.** R.8.1 is the argument, and it is a measurement rather than an intuition:
the search — the strongest thing in the repo — wins by boxing **27 % more** enemy coins while its
`own_at_risk` is *identical* to the raw policy's. So search does **not** carry prophylaxis, and
distilling it never will. The reward is the only place "you attack so your unit is not attacked"
can enter, and the reward is already open on the bench:

- **`Φ_risk = −c · own_at_risk`** — the encoder already computes the scalar (global 208), and
  `HeuristicEvaluator` already has the term. PBRS, so policy-invariant.
- **Lock potential** — `w = 1.0` for a base controlled *and* occupied, `0.6` for
  controlled-and-empty, replacing the flat `base_diff`. Justified by §3.3's 53-steals-from-occupied
  / 0-from-empty and by `is_valid_claim` making a parked unit an absolute lock. The current
  potential prices a lock and a walk-in-able base identically, which is not how the game works.

Note what R.0.3's fix does and does not do. Annealing the base term holds the base : material
ratio flat at **3.3 : 1** for the whole run instead of letting it widen to 33 : 1 — the whole
material axis of a game goes from ~2 % of the base axis to ~25 %. That is the big correction. But
`C_MAT` still prices only material *already boxed*; nothing prices material *hanging*, which is
the two-ply quantity §1's thesis is about. `Φ_risk` is that term.

### R.9.5 Is R.3b (Gumbel AlphaZero) worth doing before the launch? — No, during it

**It is the right next piece of work and it is not on this run's critical path.**

- Gumbel changes `PuctBot`'s search and ExIt's policy target. Neither touches PPO's gradient.
- The only way it could help *this* run is as a better training opponent — and R.9.2 says the
  right move for that slice is to **remove** it, because the cost is what is capping the run's
  size. Improving a 100 ms opponent does not fix a 100 ms opponent.
- ExIt has to come *after* a PPO run on the re-balanced reward anyway: you distil from a
  policy/critic trained on the objective you actually want, and every checkpoint on disk is
  trained on the old one. Running ExIt now would distil the 33 : 1 objective.
- The PPO run is 3–9 h of wall-clock in which nothing else is happening. That is exactly the
  window for a moderate, self-contained search change with its own tests.

What Gumbel buys, restated so it is not lost: it makes a **low-simulation** search a valid policy
improvement operator (completed-Q targets + sequential halving over Gumbel-sampled root actions),
which is the regime this project is structurally stuck in — 0.86 ms unbatched CPU forwards,
branching 10.5, ~14–30 expansions at 0.1 s. It also **dissolves** the `--dirichlet-frac`
question (R.3d) rather than answering it, since Gumbel root sampling replaces Dirichlet noise
outright. Build it against `preflight`, whose four gates are the acceptance test.

### R.9.6 The list, in order

| # | do | cost | why it is here |
|---|---|---|---|
| 1 | ~~decouple `HeuristicEvaluator` from `SHAPING_C`/`C_MAT`~~ | **done** | otherwise the reward change re-tunes every gauntlet baseline and the ExIt teacher's leaf (R.9.1) |
| 2 | `eval_episodes` 20 → 50; pin `--reference-policy` | trivial | R.8.2 — the current instrument misread a 0.55 as a 0.35 (R.9.3) |
| 3 | `p_lookahead_critic_finetune` 0.3 → 0.10, `p_puct_finetune` 0.2 → 0.0, into `p_pool` | trivial | ~3× the batches in the same wall-clock, on opponents measured weaker than the policy (R.9.2) |
| 4 | L4: `Φ_risk` + lock potential, in the same reward edit | low | the only source of prophylaxis; search provably does not supply it (R.9.4) |
| 5 | copy the newest critic into `data/lookahead_critic/lookahead_critic_v6.pth` | ~0 | the remaining search opponent otherwise runs a 2026-08-08 critic |
| 6 | raise the verb-entropy floor (`verb_entropy_coeff_final` 0.01 → ~0.015) | trivial | judgement call. `verb_ent` ends at 0.276 of a 2.40 max; a reward change meant to induce new behaviour needs exploration to find it, and the floor was tuned for the old reward. Bundling it does cost attribution — skip it if that matters more |
| 7 | `--dump-returns-dir` off unless another target A/B is planned | ~0 | 252 shards / 4.34 M samples last time, and the reader was OOM-prone (`decision.md` 2026-08-16) |
| 8 | R.3b (Gumbel) — **during** the run, not before | mod | R.9.5. **R.10 sharpens this:** Gumbel root sampling + sequential halving is the principled form of exactly the two things R.10.3 and R.10.6 do cheaply — it decouples "was this move considered" from "did the U term let us visit it" (R.10's M1: only 3.3 of 8 children are reachable), and its completed-Q target is not the over-peaked visit count R.10.1 indicts. Do R.10.3 first because it is one line; reach for Gumbel when that sweep runs out of room |

Standing caveat, unchanged: items 3, 4 and 6 all land on the same side of a reward boundary as
R.0.3, so `score`, `score_parts`, returns and `critic_mae` are **not** comparable across it.
Win rate and the gauntlet are. That is what item 2 exists to protect.

---

## R.10 ExIt, read symbolically (2026-08-18) — the budget fix holds up; the target's *inputs* are what is left

A symbol-level pass over the whole pipeline (`services/expert_iteration.py`,
`app/expert_iteration.py`, `services/selfplay_collector.py`, `bots/puct_bot.py`) plus four new
measurements. It was written to test R.8/R.9's assumption that ExIt's problem is teacher strength
and that raising the budget fixes it.

**The one-sentence version.** R.8.3's `--time-budget 1.0` is validated by two independent
instruments — the search now diverges from its own prior on **13.1 %** of moves (was 1.3 % at
0.02 s) and the target's entropy lands roughly balanced against the policy's — so there *is*
something to distil; what remains broken is upstream and downstream of the tree rather than in it:
the teacher **sees the opponent's hand** (15.4 % of its moves depend on it, as large as the whole
improvement signal), the prior it searches inside is a near point mass (82 % on one move, ~3 of 8
children reachable), and the distillation step has **no replay window, no promotion gate, no trust
region and no early stopping**.

**Read R.10.1 and R.10.2 as retractions.** This section's first draft argued that the target is
over-peaked and that raising the budget makes things worse. M4 was run to test that and inverted it.
The retraction is kept in full because the *reason* it was wrong — a signed comparison of two
entropies drawn from a single 103-sample game — is the same methodological trap this project has
now written down three times.

### R.10.0 Four measurements

**M1 — the prior the search has to work inside is nearly a point mass.** 2643 decision states,
`ckpt_20260817-2102`. "Children PUCT can reach" counts root children whose *first-visit*
exploration bonus `c_puct · P(a) · √N_total / 1` exceeds 0.05 at `c_puct = 1.5` and 97 expansions
(the 1.0 s budget) — i.e. moves the U term can actually lift into a visit at all. The 0.05 is a
stipulated Q-scale, not a derived one, but it is grounded: `eval_value_calibration.py puct` measured
Q's p5–p95 span at **1.74** (`decision.md` 2026-08-16), so 0.05 is ~3 % of the range the values
actually occupy — a small but real difference in evaluation. The 97 is the *contended* expansion count
from R.0.1 (12 workers); M4 measures **262** on an idle box at the same budget, which raises `√N` by
1.6× and so lowers the reachability threshold from `P > 0.0034` to `P > 0.0021`. The reachable-children
column would therefore be somewhat higher on an idle box — the table below is the pessimistic end, and
the qualitative point (a handful of children, not a dozen) holds either way:

| prior temperature τ | mean top-1 prior | entropy (nats) | of max (ln 13.7 = 2.62) | children PUCT can reach |
|---|---|---|---|---|
| **1.0 (shipped)** | **0.817** | **0.495** | **19 %** | **3.3** |
| 1.5 | 0.737 | 0.737 | 28 % | 4.4 |
| 2.0 | 0.668 | 0.956 | 37 % | 5.2 |
| 3.0 | 0.557 | 1.316 | 50 % | 6.3 |

**The teacher effectively considers ~3 moves, and already puts 82 % on one of them.** That is the
quantitative content of the recorded 0.94–0.95 agreement, and it is a property of the *prior*, not
of the tree.

**M2 — `max_branching = 8` is not the constraint, and widening it provably cannot help.** Same
states:

| cap | states truncated | legal moves kept | prior mass kept (mean / p10 / min) |
|---|---|---|---|
| 8 | 66.2 % | 69.2 % | **99.74 % / 99.58 % / 81.10 %** |
| 12 | 46.0 % | 83.4 % | 99.94 % / 99.97 % / 91.91 % |
| 16 | 31.7 % | 91.7 % | 99.99 % / 100.00 % / 96.25 % |

Even in the **widest decile** (≥ 24 legal moves, where the cap keeps only 29.4 % of them) it
retains **99.39 %** of the prior's mass. So the discarded moves carry ~0.26 % of the prior between
them — and a move with prior ~0.001 gets a U bonus of `1.5 · 0.001 · √97 ≈ 0.015`, far below any
real Q gap, so it would never be visited even if it *were* in the tree. R.0.2's aside
("widening it is **not** the fix") is hereby confirmed, but for a much stronger reason than was
given there: the cap is downstream of the collapsed prior, and so is everything else.

**M3 — the search *does* diverge from its prior at 1.0 s, and the cheating penalty grows with the
budget too.** `eval_move_agreement.py --games 10 --time-budget 1.0 --value-mode shaped
--sample-every 2`, run for this section: 428 probed positions across 10 self-play games driven by the
cheating teacher, with a blind search run on the same position at every probe. Against the only prior
record — a 2-game smoke at **0.02 s** (`search_under_uncertainty.md` §8.3):

| | 0.02 s smoke (n≈small) | **1.0 s (n = 428)** |
|---|---|---|
| search agrees with its own policy argmax | **98.7 %** | **86.9 %** (blind variant: 84.3 %) |
| cheat vs blind, top-1 agreement | 92 % | **84.6 %** |
| cheat vs blind, mean TV of visit distributions | 0.084 | **0.123** |

Two readings, pulling opposite ways, and both matter:

- **R.8.3's budget fix is validated on the axis it was aimed at.** The search now picks a different
  move from its own prior on **13.1 %** of decisions, against 1.3 % at 0.02 s — a ~10× increase in
  the quantity ExIt actually distils. §8.3's warning ("if this holds at the full budget, ExIt is
  distilling the policy into itself") **does not hold at 1.0 s.** There is real signal to teach.
- **But the privileged-information component grew with it, and is now as large as the signal.**
  Cheat and blind pick different moves on **15.4 %** of positions (TV 0.123) — *more* than the
  13.1 % on which the search improves on its prior at all. The divergence is flat across game phase
  (85.3 % early / 83.9 % late) and does not concentrate in wide positions (84.3 % at ≥ 8 legal), so
  it is not a handful of complex spots. See R.10.4: this is no longer a nicety.

**M4 — how the target's entropy moves with the budget, on identical states.** The claim this
section was originally built on (see the correction in R.10.1) needed a controlled sweep rather than
one preflight game, so: 45 probe states, `dirichlet_alpha = 0`, the same states scored at four
budgets, against the policy's own entropy on those states (0.456 nats).

| budget | expansions | visit_entropy | **Δ = visit − policy** |
|---|---|---|---|
| 0.1 s | 29.1 | 0.678 | **+0.222** |
| 0.5 s | 133.7 | 0.555 | +0.098 |
| **1.0 s** | **262.1** | **0.487** | **+0.031** |
| 2.0 s | 512.4 | 0.497 | +0.041 |

Two facts: the target **does** tighten as the budget rises (0.678 → 0.487) and then **plateaus** by
1.0–2.0 s; and **Δ is positive at every budget**, i.e. the visit distribution is *softer* than the
policy, not sharper. Note also the expansion counts on an idle box — 262 at 1.0 s, against the ~97
measured under 12-way worker contention in R.0.1. Both are real; which one applies depends on how
many workers are running.

### R.10.1 Correction — the target is *not* over-peaked; the budget fix lands it near the healthy point

**This subsection originally argued the opposite, and the retraction is the finding.** It read the
R.8.3 `preflight` line `policy_entropy = 0.626, visit_entropy = 0.450` (Δ = −0.176) as evidence that
the target is more peaked than the policy, concluded that a distillation round is therefore an
entropy-*reduction* operator, and predicted that raising the budget would make it worse. M4 was run
to check that prediction on identical states across four budgets and **it inverts**: Δ is **+0.222**
at 0.1 s and **+0.031** at 1.0 s — positive throughout, and *converging toward* zero as the budget
rises rather than away from it.

Why the preflight number was not evidence: it came from **one self-play game, 103 samples**, and both
of its halves sit on the opposite side of M4's values (policy 0.626 vs 0.456; visit 0.450 vs 0.678).
A one-game average of a per-state entropy cannot support a signed comparison of two numbers 0.18
apart. This is the trap this project has written down twice already — `next_iteration.md`'s
"reliability is not validity" and "in a paired measurement, precision is binding, not sample count" —
and it caught this section within a day of R.8 citing those very rules.

**What survives, and it is the useful half.** The structural reading of `distill` stands: 4 epochs of
plain cross-entropy to the visit distribution with **no trust region, no KL to the previous policy,
no entropy term and no early stopping** (`val_frac` is computed and *reported*, never used to stop).
On a 300-game round that is ~25 k samples / 256 = 88 minibatches × 4 ≈ **350 unconstrained Adam steps
at 3e-4**, against PPO's clipped, KL-gated ~171 per batch. Put that beside M3: the improvement signal
lives in **13.1 %** of samples. So the correct statement is not "sharpening dominates" but **"350
unconstrained steps are taken toward a target whose informative content is one sample in eight, with
nothing bounding how far the policy may move."** That argues for a trust region and for R.10.6's
weighting, and it holds regardless of Δ's sign.

**And the budget fix comes out better than R.8.3 claimed.** It buys teacher divergence (M3: prior
agreement 98.7 % → 86.9 %) *and* walks Δ from +0.222 to +0.031 — from clearly-flattening to roughly
balanced — with the plateau at 2.0 s (+0.041) saying 1.0 s is already at the useful end of that
curve. Two independent gates improve together; the claim that they pull opposite ways is withdrawn.

### R.10.2 The entropy guard: right side, no tolerance, and only one side ever observed

With M4 in hand the guards can be assessed properly, and the picture is different from R.10.1's
first draft in both directions.

`_run_distill` warns when `visit_entropy > policy_entropy`; `preflight` gate 4 (R.8.3) uses the same
polarity and scores that condition `FAIL`. **That orientation is correct for the regime M4 measures**
— Δ is positive at every budget, so "the target is flatter than the policy" is the side we are
actually on, and it is the side with the one recorded failure. So the "both guards watch the wrong
edge" claim from this section's first draft is withdrawn along with R.10.1's.

What is wrong with them is narrower and still worth fixing: **a bare inequality has no tolerance,
and Δ's natural resting point at a sane budget is a small positive number.** At 1.0 s, Δ = +0.031 —
both guards fire, and M4 gives no reason to think that value is unhealthy. As written they are
false-alarm generators at exactly the setting R.8.3 recommends.

The useful calibration falls out of putting the two data points on one axis:

| Δ = visit − policy | evidence |
|---|---|
| **+0.22 … +0.27** | the regime that *failed*. The recorded `--dirichlet-frac 0.25` arm measured visit 0.87 against policy ~0.6 (Δ ≈ +0.27) and flattened the policy every round; M4's **0.1 s** row sits at **+0.222** with noise switched off entirely |
| **≈ +0.03** | 1.0 s, M4. Untested in a real round, and the closest thing to balanced this project has measured |
| **< 0** | **never observed.** R.10.1's first draft inferred it from a 103-sample preflight line; M4 contradicts it |

Two consequences worth carrying:

- **Warn above ~+0.15, not above 0.** That keeps the alarm on the one regime with evidence behind
  it and stops it firing on the recommended configuration. Print signed Δ rather than two numbers
  the reader must subtract.
- **The historical `frac 0.25` failure may have been as much about the budget as about the noise.**
  Δ ≈ +0.27 then, and M4 reaches Δ = +0.222 at 0.1 s with `dirichlet_alpha = 0`. The two are the
  same regime, and the budget alone accounts for most of it. That is independent corroboration of
  R.8.3's fix, and it further weakens the case for treating `--dirichlet-frac` as the lever
  (R.3d).

### R.10.3 The one lever aimed at M1: soften the prior inside the search — and what it costs

Everything above is one quantity — the entropy of the prior the search runs on. The cheapest place
to change it is not the network and not the noise, it is a **policy softmax temperature applied
inside the search**: raise `P(a)` to `1/τ` and renormalise in `PuctBot._policy_priors` (or in
`_expand` after the top-`max_branching` cut, so the cut still uses the raw ranking).

This is not an invention. **Leela Chess Zero ships exactly this knob** (`--policy-softmax-temp`)
and runs it above 1.0 in its shipped configurations, for exactly the reason M1 measures: a trained
net's raw policy is too peaked for MCTS to explore. (Check the current LC0 default before quoting a
number — the point that matters here is the sign, not the value.) It is one line, it has no effect on the network, and by M1's table
τ = 2.0 takes the reachable-children count 3.3 → 5.2 and the prior's entropy 19 % → 37 % of max.

**But M4 changes what this costs, and the first draft of this item got that wrong too.** Softening the
prior raises the *search's* entropy without touching the network's, so it pushes **Δ up** — and Δ is
already +0.031 at 1.0 s, with the failing regime starting somewhere around +0.15…+0.22 (R.10.2). So
temperature is **not** a free win that raises reach and Δ together; it buys reach and spends the Δ
budget, and the two effects have to be weighed against each other rather than assumed to align.

What is not obvious a priori is the *size* of that cost: more reachable candidates means the visits
spread wider, but it also means Q has more to discriminate between, which concentrates them again.
The net effect on Δ is an empirical question and cheap to answer, which is the whole point of running
it as a sweep rather than shipping a value.

Two properties that survive unchanged and still make it the first thing to try:

1. It attacks the one quantity M1 measures directly — the search examines ~3 moves and its prior
   already puts 82 % on one of them — and nothing else on this list touches that.
2. It **sidelines the `--dirichlet-frac` question** (R.3d) rather than answering it. Noise tries to
   lift tail moves past the U threshold by adding mass, which is sparse at `frac = 0.03` (a move at
   `P = 0.001` reaches at best `0.03 · Dir(0.3)`, usually ~0) and overshoots into the failing Δ
   regime at `frac = 0.25`. Temperature reshapes the mass already there, so it has no such cliff —
   and R.10.2's second consequence suggests most of what `frac 0.25` was blamed for was the budget
   anyway.

Sweep τ ∈ {1.0, 1.5, 2.0, 3.0} and read **three** numbers per arm, not one: signed Δ (must stay well
under ~+0.15), teacher strength vs the raw policy (must hold ≥ 0.65), and search/own-prior divergence
from `eval_move_agreement.py` (13.1 % at τ = 1 — this is the number τ is being bought to raise). If Δ
blows past +0.15 before divergence improves meaningfully, τ is the wrong lever here and R.10.6 is the
fallback.

### R.10.4 The teacher cheats, and nothing in ExIt turns that off

`PuctBot.__init__` has `see_opponent_hand=True` as its **default**, `_build_bot` in
`app/expert_iteration.py` does not pass the argument, and `LookaheadBot._prepare_root` only
re-splits the opponent's hidden coins when it is `False`. So every ExIt sample is a target produced
by a search that **knew the opponent's actual hand**, distilled into a policy that structurally
cannot observe it.

`search_under_uncertainty.md` §8.1 measured that cheating is worth ~0 in *strength*, and §6.4-3
already made the point that strength is the wrong test: two searches can be equally strong and still
choose differently, and then the cheating one's targets encode a choice the student cannot derive
from its own observation. Equal win rates and unlearnable targets are perfectly compatible. The
measurement is now in, at the real budget (M3), and it is not marginal: cheat and blind disagree on
**15.4 %** of positions with a visit-distribution TV of **0.123**, against the **13.1 %** of
positions where the search improves on its prior at all. **The hidden-information component of the
target is at least as large as the entire improvement signal** — and unlike the signal, no amount of
training can reduce it, because the student's observation does not contain the variable it depends
on. It is irreducible label noise sitting exactly on top of the thing being learned.

Fix: pass `see_opponent_hand=False` for data-gen, and raise `n_determinizations` to 3–4 at the same
time — with a blind root, `n_determinizations=1` (the default) is single-determinization search,
which is the strategy-fusion failure `search_under_uncertainty.md` §2 is entirely about. The cost is
linear in determinizations and generation is offline. **The gate this item was waiting on has now
passed** — at TV 0.123 the cheating is worth removing, and the 3–4× data-gen cost of blind
determinization voting is the price. Expect blind search to be no weaker (§8.1 measured the strength
difference at ~0), so the only thing being bought is that the target becomes a function of what the
student can actually see.

### R.10.5 Four AlphaZero-family mechanisms the pipeline does not have

Each is standard, cheap, and maps onto a recorded symptom.

**(a) No replay window.** `cmd_loop` distils on `round{r}.npz` **only** — one round, ~25 k samples.
Every AlphaZero-family trainer samples a sliding window over many generations. The recorded symptom
names its absence precisely: each round the previous round's critic scored held-out MSE 1.2–1.55 on
the new data then re-fit to 0.03–0.05 within the round — *"it memorises each round's narrow
self-play distribution and generalises to the next round's barely at all: churn, not learning"*
(`independent_opponents.md` §1). Cost is RAM, not compute: ~0.5 GB per 300-game round
(`visit_targets` is `[N, 1875] f32` and dominates), and `SelfPlayDataset.concat` already exists.

**(b) No promotion gate.** `cur_policy = out_policy` unconditionally, so a bad round becomes the
next round's seed *and* the next round's teacher prior. AlphaGo Zero gated promotion at 55 %. The
recorded symptom is the ungated signature: *"the base policy beats every one of its 30
descendants … monotonically weaker for the first several rounds, then plateaued below the start."*
The post-round gauntlet already computes the number; nothing reads it. A gated loop can stall — an
ungated one spirals.

**(c) `val_frac` is a report, not a control, and its split leaks.** Two separate problems. The
held-out CE is never used to stop training (350 unconstrained steps regardless), and the split is
`np.random.permutation` over *samples* — but ~84 samples come from one game and share its trajectory
and outcome, so a random split puts near-duplicates on both sides and the held-out number is
optimistic. `eval_board_value.py` already fixed exactly this ("held out **by round**") and
`next_iteration.md`'s methodological rules list it. **Split by game, then early-stop on it.**

**(d) The search's own root value is computed, thrown away, and would be a better critic target
than `z`.** `LookaheadCriticBot.act` puts `last_stats['best_value']` — the visit-weighted root value
of the chosen move — on every decision, and `SelfPlayDataset` never records it. `z` is the noisiest
target available: one bit per game, shared by all ~84 of its samples. Row 2b measured that a
less-noisy target is worth ~1.3× *to everyone*, and MuZero's value target is exactly this
substitution (n-step bootstrapped return in place of the terminal outcome) once intermediate rewards
exist. Recording it costs one column and makes `λ·z + (1−λ)·V_search` available as a target —
relevant the moment `--freeze-critic` is turned back off.

### R.10.6 Weight the loss by how much the teacher disagreed

Follows directly from R.10.1: if ~95 % of samples carry no improvement signal and an actively
harmful sharpening gradient, stop paying full price for them. Weight each sample by the
teacher/prior divergence already available at record time — the TV distance between the visit
distribution and the raw prior, both of which `PuctBot` exposes (`last_stats['visit_counts']` and
`last_stats['policy_argmax']` come from `_combine_visit_counts` over `_search_visits` and
`_search_priors`, so the full prior vector is one line away).

This extracts the 5 % without discarding the 95 % outright (dropping them would let the policy
drift on everything the search agreed with). KataGo's *policy target pruning* is the nearest
published relative. It is also the natural fallback if R.10.3's temperature sweep cannot get Δ
positive without costing teacher strength — the two are alternative routes to the same end, and
either alone is enough.

### R.10.7 Two small things found while reading

- **`seed_base = args.seed or 0`**, so `--seed 0` and no `--seed` are the same run, and workers are
  seeded once per *collector* (`np.random.seed(seed_base + worker_id)` in `_worker_loop`) with the
  stream continuing across rounds. Rounds therefore differ, but **re-running `gen` on the same
  checkpoint reproduces the identical dataset** — good for reproducibility, and a trap for anyone
  trying to buy more data by re-running generation. Vary `--seed` per round if a replay window
  (R.10.5a) is used.
- **`eval_move_agreement.py` earns its keep and should be run at every budget change.** One pass
  reports all three quantities R.10 turns on: search/own-prior divergence (the ExIt go/no-go),
  cheat-vs-blind top-1 and TV (item 1's gate), and the visit-vs-prior TV distribution (R.10.6's
  weighting signal). `search_under_uncertainty.md` §8.4 had the command written down at the wrong
  budget (0.1 s) since 2026-08-02; at 1.0 s it changed two conclusions in this document. Cost: ~21
  min single-process for 10 games at `--sample-every 2`.

### R.10.8 Revised order for the ExIt attempt

Supersedes R.8.4's launch list. Item 0 is done; items 1–2 are the ones that changed most between this
section's first draft and its measurements.

| # | do | why |
|---|---|---|
| 0 | ~~run `eval_move_agreement.py` at 1.0 s~~ | **done (M3).** Prior divergence 98.7 % → **86.9 %**: the loop has something to teach at this budget. This was the go/no-go and it passed |
| 1 | **`see_opponent_hand=False` + `n_determinizations=3` for data-gen** | R.10.4, and M3 promoted it from "nice to have" to the largest single defect: **15.4 %** of the teacher's moves depend on hidden state (TV 0.123), against **13.1 %** of moves carrying the improvement signal. Irreducible label noise as large as the signal, and §8.1 says blind search costs ~0 in strength |
| 2 | **replay window + promotion gate** | R.10.5a/b — the two mechanisms whose absence matches the recorded 30-round collapse one-for-one, and neither depends on anything above |
| 3 | split `val` **by game**, early-stop on it | R.10.5c — the split currently leaks (84 correlated samples per game, random permutation), and 350 unconstrained steps run regardless of what it says |
| 4 | fix the entropy guard: print signed **Δ**, warn above ~**+0.15** | R.10.2 — as written both guards fire at Δ = +0.031, which is the recommended configuration. Currently a false-alarm generator, not a wrong-side guard |
| 5 | policy softmax temperature in the search, sweep τ ∈ {1, 1.5, 2, 3} | R.10.3 — still the only lever aimed at M1 (3.3 reachable children, 82 % top-1 prior), but M4 shows it **spends** the Δ budget rather than adding to it. Read Δ, teacher strength and prior-divergence per arm |
| 6 | `loop --rounds 3 --freeze-critic --time-budget 1.0` | R.8.4's command, with 1–4 in place. 1 and 2 are the ones that would change the outcome; 5 is a tuning pass |
| 7 | record `best_value`; weight the loss by teacher/prior divergence | R.10.5d, R.10.6 — the next round of improvements, and R.10.6 is the fallback if τ has no room |

**The honest summary of R.8 → R.10.** R.8 measured the teacher and concluded ExIt was justified;
R.10 measured the *channel* and the conclusion mostly held, with one real correction and one real
find. The correction: the channel is not over-peaked, and the budget increase improves teacher
divergence and target entropy **together** rather than trading them (R.10.1, R.10.2 — both
retractions of this section's own first draft). The find: the teacher is conditioning on the
opponent's hand, and M3 sizes that at the same order as the entire signal being distilled — which no
amount of budget, temperature or rounds can fix, because it is not noise in the search, it is
information the student does not have. **Item 1 is the change most likely to decide whether the run
works.**

### R.10.9 The run itself, and the correction R.10.1 needed after seeing it (2026-08-19)

`loop --rounds 3 --freeze-critic --time-budget 1.0` (item 6, item 2 not yet in place) ran overnight
2026-08-18/19, ~50 min/round on 8 CPU workers, `data/exit/round{0,1,2}_policy.pth`. Result — the base
beat every round, by a widening margin:

| round | base vs round | agreement (pre→post distill) | policy_entropy (pre→post distill) |
|---|---|---|---|
| 0 | **0.70–0.75** | 0.749 → 0.858 | 0.469 → **0.875** |
| 1 | 0.78 | 0.827 → 0.862 | 0.882 → 0.959 |
| 2 | 0.82 | 0.858 → 0.866 | 0.957 → 0.988 |

Same symptom the user reported before any of this section existed — "плays like an absolute
beginner" — reproduced end to end with the exact numbers behind it: one round of distillation raised
the policy's own entropy from 0.469 to 0.875 nats, i.e. it made the model's decisions almost twice as
*undecided*, and the next two rounds kept climbing (never re-sharpening), because `cur_policy =
out_policy` ran unconditionally regardless of the gauntlet report sitting right above it in the same
log saying the round had lost.

**This is the flattening failure the `evaluate_distillation` warning was written to catch, and it
fired every round** (`visit_entropy (0.720) > pre-distill policy_entropy (0.469)` etc.) — logged,
and then ignored, because a `logger.warning` is not a gate.

**The size of the gap needed a second look at R.10.1.** M4 (a *controlled* sweep re-using the same
handful of states across four budgets) measured Δ = visit_entropy − policy_entropy at **+0.031** at
1.0 s — small, "softer but not collapsed," the basis for retracting the original flattening claim.
The real run's *own* self-play data (23065 samples, the full state distribution a game actually
visits, not a curated identical-state set) measured Δ = **+0.251** at the same nominal 1.0 s budget —
eight times larger. Both measurements are correct; they answer different questions. M4 isolated "how
does more budget change the entropy gap, holding the state fixed" (answer: shrinks it, converges by
2 s). It says nothing about the *average* gap over a real trajectory, which includes many
near-forced, lopsided, or already-clearly-won positions where the trained policy is close to a
one-hot (entropy near 0) but the search still burns its ~97 expansions plus a flat 3 %
(`--dirichlet-frac`) root-noise floor across every child — a fixed absolute noise cost that reads as
enormous *relative* entropy exactly where the policy was already confident and correct. R.10.1's
retraction stands for the budget-vs-budget comparison it measured; it should not have been read as
"the production channel is fine," and this section did not say that explicitly enough. Corrected here.

**Shipped, both directly targeting what actually broke, not the two items R.10.8 had queued next:**

1. **`--visit-temp` (default 0.5), `src/services/expert_iteration.py:_sharpen_target`.** Raises the
   recorded visit distribution to `1/visit_temp` and renormalises before it is used as the CE target
   — in `distill()`'s training loop and in `evaluate_distillation()`'s before/after report, so the
   logged numbers reflect what training actually sees rather than the raw recording (a monotonic
   transform never changes the argmax, so `agreement` is unaffected). Verified on the real, already-
   recorded `data/exit/round0.npz` — no new self-play needed: `visit_temp=1.0` (old default) mean
   entropy 0.720 nats; `0.7` → 0.451; **`0.5` → 0.304**, chosen for a clear margin under the round's
   0.469 policy_entropy without going as far as `0.15`'s near-one-hot 0.090. Re-running `distill` on
   that exact dataset end to end at the new default: `policy_entropy` before/after **0.469 → 0.486**
   (essentially flat, not 0.469 → 0.875) while `agreement` still rises **0.749 → 0.861** — the real
   signal survives, the flattening does not.
2. **Promotion gate, `cmd_loop` (`--promote-threshold`, default 0.5).** Each round's post-round
   gauntlet result now gates advancement: the win rate of `round{r}_policy` against the checkpoint
   *that round's self-play was generated from* (not the run's original base — that comparison is what
   the existing report already showed, and it is not what decides whether to build the next round on
   top of this one) must clear the threshold or the round is REJECTED — logged, kept on disk for
   inspection, but the next round retries self-play from the same, still-current checkpoint instead of
   compounding a regression. This is R.10.8 item 2's promotion half (the replay-window half is still
   open); with it in place this exact run would have stopped after round 0 rather than running two
   more rounds on top of a checkpoint that had already measurably lost.

Both are covered by `tests/test_expert_iteration.py` (the pure sharpening function: no-op at
`visit_temp=1.0`, lowers entropy below 1.0, preserves zeros/argmax, raises entropy above 1.0 — the
promotion gate's branching is exercised by inspection, not unit-tested, since it is a thin
`win_rate[...]  >=  threshold` read off gauntlet output already covered by `gauntlet_parallel`'s own
tests).

**Still open, unchanged from R.10.8:** item 1 (blind teacher, `see_opponent_hand=False` +
`n_determinizations=3` for data-gen — 15.4 % of the teacher's moves depend on hidden state, as large
as the whole improvement signal) and item 3 (the `val` split leaks 84 correlated samples per game
under a random permutation, and is never used to early-stop). Neither was touched by this fix; both
still apply on top of it. **Re-run `preflight` (now `--visit-temp`-aware, gate 4 reports the
sharpened entropy) before the next `loop` invocation** — the two fixes above address why the last run
made the model worse, not whether this run's teacher is strong enough to make it better, which is
still items 0/3/4 in R.10.8's table.

### R.10.10 Replay window shipped (2026-08-19), and a recorded fallback if it still isn't enough

Both `loop --freeze-critic --time-budget 1.0` and `--time-budget 2.0` re-runs (2026-08-19, on
desktop-legion, `data/exit/20260819-111340` and `.../20260819-131927`) confirmed the two R.10.9 fixes
work exactly as designed — `policy_entropy` stays flat round to round instead of blowing up, and every
round that lost to its own base was correctly REJECTED, so nothing compounded — **and still netted
zero improvement**: every independently-generated round, always starting fresh from the untouched
base, still lost to it (0.40/0.33/0.23 at 1.0 s; 0.467/0.433 at 2.0 s before the run was stopped for
this diagnosis). Sharpening and the gate fixed the mechanism that made rounds *actively worse*; they
were never going to fix a mechanism that keeps rounds merely *not better*.

A clean split of one round's own recorded data (`data/exit/round0.npz`, agree vs. disagree with the
policy's own argmax) pinned that second mechanism down directly, independent of R.10.4's hidden-
information story: `disagree_only` distillation loses to base **0.28–0.29**, `agree_only` loses more
mildly at **0.40–0.41**, and `agree_only` beats `disagree_only` head-to-head **0.70** — confirmed at
both k=40 and k=80. Epoch count (1 vs 4) and learning rate (1e-4 vs 3e-4) were also swept across both
loop runs and made no reliable difference — every combination lost to base by a similar margin, ruling
out "distillation hyperparameters" as the lever. If the hidden-opponent-hand story (R.10.4) were the
dominant mechanism it would corrupt agree and disagree samples alike, not cleanly separate them this
way — this result is closer kin to R.10.5a's overfitting story: `distill()` was fitting `epochs=4` at
`lr=3e-4` to one round's ~25k-sample, single-network self-play slice with nothing from any earlier
round to anchor it, which is exactly the missing replay window R.10.8 item 2 already named as cheap
and already-scoped (`SelfPlayDataset.concat` existed before this fix; only the sliding-window
bookkeeping was missing).

**Shipped:** `ReplayWindow` (`src/services/expert_iteration.py`) and `--replay-rounds` (default 3,
`src/app/expert_iteration.py cmd_loop`). Each round still self-plays and saves its own
`round{r}.npz`, but `distill()` now trains on the concatenation of the last `--replay-rounds` rounds'
datasets rather than the current round alone — `--replay-rounds 1` reproduces the exact pre-fix
behaviour. A round's data stays in the window even if that round is later REJECTED by the promotion
gate: a rejected round's retry self-plays from the same checkpoint, so the data is still drawn from
the retry's own distribution rather than being stale. Covered by
`tests/test_expert_iteration.py`'s `ReplayWindow` tests (single-round no-op, concatenation within the
window, dropping the oldest round past the window size). Not yet re-run on desktop-legion — the two
loop runs above predate this fix.

**Recorded fallback if the replay window alone doesn't turn the loop net-positive — do not build this
yet, only measure it if the window fails:** retire CE-imitation of the search's move choice entirely
and use `PuctBot` as an on-policy PPO sparring opponent instead of a behavior-cloning teacher. The
premise behind ExIt — that the search is a genuinely stronger player than the raw policy — is not in
question: a clean matched-seed, matched-machine A/B this session (`preflight --time-budget 1.0`,
dirichlet settings identical to gate 3) measured PUCT at **0.76–0.79** against the raw policy,
consistent across two independent runs and refuting an earlier, weaker 0.567 remote read (n=60,
single seed — concluded to be sampling noise, not a real weak-teacher effect). What may not survive
the transfer is specifically *one-shot CE-to-visit-distribution imitation* of that strength: PUCT's
edge comes from multi-ply lookahead, and cloning the argmax move it happened to land on in one game is
a lossy, indirect way to transfer a lookahead advantage, especially on the ~26 % of positions where its
choice disagrees with the policy's own (the sub-batch measured most harmful above). PPO's existing
opponent-pool machinery (`src/services/opponent_pool.py`) already supports adding a fixed opponent
into self-play; adding `PuctBot` there lets the policy improve via its own on-policy PPO gradient
against a stronger opponent — with PPO's clip acting as the trust region the replay window only
approximates by diluting the update — rather than via imitation of its move choices at all. Untried.
