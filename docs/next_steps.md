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

## Step 1 — Round-robin gauntlet + diagnostics

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
   opponent-independent quality metrics. *(Restore the signal.)*
2. **Strengthen the opponent** (economy-aware greedy + shallow lookahead, optionally MCTS).
   *(Raise the bar.)*
3. **Add an exploitability metric**; optionally upgrade the pool toward **PSRO**. *(Nash
   direction, measured.)*
4. **Only then** run 1000–1500 batches against the harder opponent and read the plateau against
   the gauntlet.
5. **Last:** build online play as the final human validation, after a rules-parity audit.
