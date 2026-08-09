# Next iteration — the problem, and how to fix it

**Written 2026-08-02, restructured 2026-08-03** after a domain-expert objection that
invalidates part of it. Trigger: the policy plays "correct beginner" — spends coins in
tempo, walks at bases, trades units, but has **no strategy** (every unit is a generic
move/attack piece) and **never deliberately bolsters**. Three prior fixes came back
negative: a bolster-archetype opponent ([independent_opponents.md](independent_opponents.md)
Phase 1), the belief track ([search_under_uncertainty.md](search_under_uncertainty.md) §8),
and expert iteration (every round weaker than base).

> **Amended 2026-08-07, twice, and the plan changed.** Four retractions and one promotion:
>
> 1. **§3.1 is retracted.** The within-state metric pooled two disjoint sub-problems and
>    reported their average. Bucketed, the board ranks siblings where it alone can (§3.1a),
>    and `board_solo` — board only, no globals — out-predicts every non-board feature across
>    states (R² 0.1846 vs 0.1633). §4's first row is re-opened.
> 2. **§2 step 1's two consolations are retracted.** At 16 playouts instead of 4, the
>    "within-state signal grows under `lookahead`" claim inverts (std 0.208 → 0.158) and
>    `HeuristicEvaluator`'s Spearman falls 0.246 → 0.149 against a ceiling that rose to 0.778.
>    The hand-written leaf is the *worst* evaluator on clean labels, not the best.
> 3. **The binding constraint was label precision, not the playout bot.** Learned evaluators
>    all improved with better labels; the v10 critic reached 26 % of ceiling, the best result
>    in this investigation.
> 4. **New, and now the top lever (§3.3b): the critic's target beats its board pathway.** A
>    board-blind critic trained on shaped GAE returns outranks a board-reading network trained
>    on ExIt's `z` by ~2×. ExIt trains on `z`.
>
> Net effect on §5: **the critic's training, not the search, is the best-supported target.**
> Quiescence slipped from first to fourth. §3.1b lists what was verified in the instrument.
> Every interval in §3.1a is bootstrapped with **states** resampled, not pairs — an earlier
> pair-level "~5 σ" did not survive that.

Read §1 and §2. §3 is the evidence, §4 is what is ruled out so it is not re-proposed.
Supersedes the *sequencing* in [independent_opponents.md](independent_opponents.md) §7 and
[IDEAS.md](IDEAS.md) § *Recommended next steps*.

---

## 1. The problem

**Warchest value lives one to two moves ahead, and nothing in this system computes one to
two moves ahead.**

The domain fact, in the user's words: *this is a deeply positional game; you attack mostly
so that your unit is not attacked — that is the whole logic.* Prophylaxis is a two-ply
statement. So is bolstering ("your attack no longer kills me next turn"). So is unit
quality — a Berserker matters through *what it threatens*, not through what it is. So is
sacrificing a 3-stack to walk onto a base, which is correct only when it closes the game.

Now the three places that could supply that lookahead, and why none does:

| Mechanism | Status |
|---|---|
| **The policy** | Purely reactive: a function of the current position. Its only forward-looking input is a *one-turn* worst-case threat plane. No lookahead at inference. |
| **The critic** | A static evaluator. The shipped one is **board-blind**: its trunk is dead, so it ties 89–93 % of sibling pairs that differ only in position and cannot rank them by arithmetic (§3.4). *Amended: the original claim here — "measurably unable to distinguish the ~7 successors of a position (§3.1)" — is retracted. A healthy critic reaches 26 % of the achievable ceiling and 61 % pairwise on same-verb pairs (§3.1a); the failure is this checkpoint's training, not static evaluation as such.* |
| **The search** | `PuctBot`/`LookaheadCriticBot` rank children by that critic; the hand-written leaf ties **89 %** of board-only and **67 %** of same-verb pairs (§3.1a). A static leaf scoring a position **mid-exchange** is exactly the failure quiescence search exists to fix. |

That was framed as a closed loop: the policy cannot convert positional advantage → so
positional advantage does not show up in its returns → so nothing learns to convert it → and
search, the standard way out, is blind at the leaf for the same reason. **The loop is real but
it is not closed** — the 2026-08-07 measurements find the information present and learnable
(§3.1a), with two identified, fixable breaks in the critic: a dead trunk and the wrong target
(§3.3b, §3.4).

### 1.1 The pivot — every measurement so far was taken *inside* that loop

This is the correction, and it is mine to own. §3's within-state measurements all label a
successor with **the expected outcome under the current policy's continuation**. But a
position whose value is *a threat you must convert* is worth nothing to a player who never
converts threats. So the labels contain almost no positional information — and "no method
can find positional value in them" was never evidence about the game.

The trap has a name: I confused **reliability** with **validity**. Two independent halves of
the Monte-Carlo label agree with each other at `r = 0.61` (16 playouts), which is why the
harness prints a confident-looking ceiling of 0.87 — but agreement between two runs of the
same broken ruler says nothing about whether the ruler measures the right thing.

The tell: **everything fails equally.** Networks 0.13–0.15, a globals-only control 0.15, the
trained critic 0.11, the hand-written heuristic 0.11, a ridge over twelve hand-built scalars
0.11 — against a nominal ceiling of 0.67. When every method lands in one narrow band, the
usual explanation is the **target**, not the methods.

> **2026-08-07: there were two more explanations, and both beat this one.** (i) The *metric* —
> a pooled average over two disjoint sub-problems compresses every method into one band
> mechanically, whichever way each is actually doing; §3.1a splits it and the band disappears.
> (ii) **Label precision** — going 4 → 16 playouts raised every learned arm and lowered the
> hand-written one, reversing two of this document's conclusions (§2 step 1). §1.1's own
> diagnosis has been *tested* and is the weakest of the three: swapping the playout bot from
> `policy` to `lookahead` moved the numbers less than the playout count did. So the label was
> indeed a problem, but its **noise**, not its policy-conditioning, was the binding part.
> Third and fourth distinct ways this investigation has been fooled by an aggregate (after the
> two tie-rate artefacts in §3.2).

Two of those legs are weaker than they look and should not be leaned on: the heuristic and
the hand features are both *too coarse to express the difference at all* (they tie on 74 %
and 45.5 % of sibling pairs respectively — §3.1, §3.2), so their low scores are not
independent evidence. What carries the argument is the pair that *can* express arbitrary
differences — the networks and the trained critic — plus the fact that the label is
policy-conditioned by construction.

**So the conclusions that concern the *game* are downgraded to conclusions about *this
target*.** What survives unconditionally is only what does not depend on the label — the
dead critic trunk (§3.4), the `opp_onehot` offsets (§3.5), the base-occupancy facts (§3.3),
the policy's own verb marginals (§3.7). *Amended 2026-08-07: with clean labels the
conclusions about the game came back **positive** — positional information is present and
learnable (§3.1a). The downgrade stands as a description of what the 2026-08-02 evidence
could support; it is no longer where the plan is.*

---

## 2. The fix

Originally: break the loop by getting a lookahead reference that is not the policy. That step
has run (step 1 below), and it redirected the plan — **the loop's weak link is the critic's
training, not the absence of a non-policy reference.** Read step 1's result, then step 2,
which is now the first thing to do.

### Step 1 — re-label against a non-policy reference *(~30 min) — the gate*

Does positional value exist in this game as implemented? Label successors by playing them
out with **`lookahead`** — alpha-beta, 4–6 plies, no network, so it *does* convert threats —
instead of with the policy:

```bash
python src/app/eval_board_value.py siblings --states 120 --playouts 4 --max-siblings 8 \
    --relabel --playout-bot lookahead --labels data/la_labels.pt
python src/app/eval_board_value.py rank --labels data/la_labels.pt \
    --init-from data/board_value_probe.pt --epochs 12
```

#### Result (2026-08-04): the gate did not open, but the run was underpowered

120 states, 860 successors, 4 playouts per half → reliability `r = 0.267`, ceiling
corr ≤ 0.649. **Labels are 58 % noise** (residual var 0.1025 = signal 0.0431 + noise
0.0594).

Scored with no training and no epoch selection (`siblings` mode — the clean instrument):

| evaluator | corr | mean per-state Spearman | ties |
|---|---|---|---|
| `HEURISTIC` (`LookaheadBot` leaf) | 0.075 | **0.246** | 68 % |
| `board_xy` | 0.067 | 0.004 | 0 % |
| `polfeat` / `polfeat_xy` | 0.033 | 0.019 / 0.054 | 0 % |
| `board` | 0.018 | 0.021 | 0 % |
| trained critic | 0.016 | 0.016 | 5 % |
| `globals` | **−0.057** | −0.047 | 5 % |
| *ceiling* | **0.649** | | |

Retrained on these labels (`rank`, 90 train / 30 held-out states, se(corr) ≈ 0.069):
`globals` 0.149, `polfeat_xy` 0.055, `board` 0.032, `board_xy` 0.012, `board_solo` 0.013 —
**final-epoch** figures. Best-epoch figures were 0.162 / 0.077 / 0.129 / 0.110 / 0.013, and
the gap between the two columns is pure **selection bias**: the epoch was chosen on the same
30-state held-out set it is then reported on. Train corr sits at 0.30+ for the board arms
against held-out 0.01–0.03 — overfitting on 90 states. *(The tool now prints both columns
and warns below 100 held-out states.)*

**So nothing clears the band, and the only arm above noise is the one with no board at all.**
Two things stop that from being a verdict:

1. **The run is underpowered, and the spec was mine.** `--playouts 4` put 58 % noise in the
   labels. This is the *second* time this document has traded label precision for sample
   count and had the precision turn out to be the binding constraint (the previous set used
   `--playouts 8`). **New standing rule: in a paired within-state measurement, label
   precision is binding, not sample count** — the quantity is a small difference, so label
   noise enters it twice while sample count only shrinks the standard error.
2. **Two signals point the predicted way.** The true within-state signal **rose** from std
   0.170 under policy playouts to **0.208** under `lookahead` — the choice matters *more*
   when the continuation converts it, exactly as the positional thesis predicts. And the one
   evaluator that **improved** is the hand-written positional one: `HeuristicEvaluator`'s
   mean per-state Spearman went 0.166 → **0.246**, the best within-state ranking in the
   table (~38 % of ceiling) despite tying on 68 % of pairs. Everything learned went *down*.
   Both are ~2 σ on different state samples — suggestive, not conclusive.

#### Result of the re-run (2026-08-07): **both consolations were noise, and the verdict inverted**

150 states, 1070 successors, **16 playouts** per half → reliability `r = 0.434`, ceiling
corr ≤ **0.778** (residual var 0.0412 = signal 0.0249 + noise 0.0163 — 40 % noise, down from
58 %). Cache: `data/la16_labels.pt`.

Point 2 above does not survive better labels. Both of its legs were artefacts of the
4-playout run:

| | policy labels | `lookahead`, 4 playouts | `lookahead`, **16 playouts** |
|---|---|---|---|
| ceiling corr | 0.675 | 0.649 | **0.778** |
| true within-state signal, std | 0.170 | 0.208 | **0.158** |
| `HEURISTIC` mean per-state Spearman | 0.166 | 0.246 | **0.149** |
| `board` corr | 0.112 | 0.018 | **0.099** |
| trained critic corr (v10, alive) | 0.172 | 0.119 | **0.202** |

**Retract "the within-state signal grows under a lookahead continuation."** At 16 playouts
the true signal std is 0.158 — *below* the 0.170 measured under policy playouts. The 0.208
was noise inflating a variance estimate. **Retract "`HeuristicEvaluator` is the best
within-state ranker."** Its Spearman fell 0.246 → 0.149 against a ceiling that rose to 0.778,
so it went from ~38 % of ceiling to ~19 %.

**And the direction of the whole result flipped.** Cleaner labels made every *learned*
evaluator better and the *hand-written* one worse. The v10 critic reaches corr 0.202 at a
0.778 ceiling — **26 % of achievable**, the best any evaluator has managed in this
investigation, against the 4.8 % that opened §1. So the reading is the opposite of what this
step was set up to check: **distil into a network; do not build the plan around the
hand-written leaf.** Quiescence survives on a different argument (it ties 89 % of purely
positional pairs — §3.1a), not on the heuristic being a good ranker.

What §1.1 got right: the label *was* a large part of the problem. What it got wrong: the fix
is not a non-policy reference, it is **label precision**. Going 4 → 16 playouts moved
learned-arm corr more than switching the playout bot ever did.

### Step 2 — the critic's target, and the trunk *(~1 day) — now the best-supported change*

> **Implemented 2026-08-07.** `critic_v2` (GroupNorm + board-only auxiliary head), the
> trunk-health guard, and the shaped-return dump are all in. What remains is to *run* it:
> a training run for the trunk, and a short run + `fit` for the target A/B. See §5 rows 2a/3
> and the Appendix for the commands.

Promoted above quiescence by the 16-playout re-run, because two contrasts resolved under a
state-clustered bootstrap and both point at the critic's *training* rather than at search:

1. **Target.** A board-blind critic on shaped GAE returns beats a board-reading network on
   `z` by ~2× (§3.3b). ExIt trains on `z`. The test is one `fit` arm on shaped returns
   instead of `z`, scored on `data/la16_labels.pt` — it settles whether the target explains
   the gap or the `hidden_dim`/data confounds do.

   **Prerequisite, and it is not free:** no shaped-return target exists on disk.
   `data/exit/*.npz` stores only `z` (`boards, globals, masks, visit_targets, opp_onehots,
   privileged, z`), and `RolloutBuffer.returns` is computed in memory during PPO and never
   dumped. So this needs a collection step first — add a `(board, global, opp, priv,
   shaped_return)` dump to `ppo.py`'s rollout, then re-fit. Budget half a day for the dump
   plus a short PPO run to fill it, not "hours".
2. **Trunk.** GroupNorm removes the ReLU absorbing state structurally (§3.4), and
   `board_solo`'s healthy 43.2 % trunk shows the death is caused by absent gradient pressure,
   not by the stack. So pair it with pressure: a **board-only auxiliary value head** off the
   trunk, whose loss cannot be satisfied by globals. Plus a per-conv health guard logged
   every run.

### Step 3 — quiescence at the search leaf *(~1–2 days)*

Independent of steps 1–2. The argument has **changed basis** since this was written: it no
longer rests on the heuristic being a good positional ranker (it is not — step 1's re-run
retracts that), only on it being *unable to express a preference* where position is the whole
difference. It ties **89 % of board-only** and **67 % of same-verb** sibling pairs (§3.1a).
That is not "the heuristic is crude" — it is the signature of a static evaluator scoring
positions **in the middle of an exchange**, where the material about to change hands is
invisible. Every chess engine hit this and every one solved it the same way: **do not evaluate
a position with pending captures; resolve them first.**

Concretely, for `LookaheadBot`: at the depth limit, instead of calling the leaf evaluator,
continue searching *only* attack/tactic moves (and the bolster that answers a live threat)
until the position is quiet, then evaluate. This is the mechanism by which "bolster saves a
stack two plies later" becomes visible to a search that today prices bolster as a pure tempo
loss — and it is exactly what [independent_opponents.md](independent_opponents.md) Phase 3
proposed and never built.

Note the scope: quiescence does not improve the evaluator, it stops asking it the questions it
cannot answer. It is no longer the *first* thing to do — step 2 is.

Success test, all on existing infrastructure: `lookahead+quiescence` vs `lookahead` head to
head on paired seeds; the board-only tie rate falling well below 89 %; and the quiescent bot's
bolster/tactic rates rising off the floor without its win rate dropping.

### Step 4 — then, and only then, decide what to train

Step 1 has reported. Its answer is neither of the two branches this section originally
anticipated — it is a third:

- ~~If step 1 is positive~~ / ~~if step 1 is negative~~ — the gate was posed as a question
  about the *game* ("does positional value exist here"), and it came back as a question about
  the *critic's training*. Positional value exists and is learnable: the board alone
  out-predicts every non-board feature (§3.1a), and on same-verb pairs board-blind evaluators
  are at or **below** chance while board-reading ones are above. What fails is the pathway
  from that information to the value — a dead trunk and the wrong target.
- **So: fix the critic (step 2), then distil from a quiescent search.** The environment
  branch (§6's parity questions) is *not* triggered — the game does reward positional play at
  the ±8.5 pp-per-decision scale §3.1 measured. Keep §6 as parity hygiene, not as the plan.

**The first full training run is justified only after step 2 has reported.**

---

## 3. Evidence

Measured on `data/warchest_ppo_20260727-0506.pth` + `data/warchest_critic_20260727-0506.pth`.
Tool: `src/app/eval_board_value.py` unless noted.

### 3.1 ~~Nothing ranks the successors of a position~~ — **RETRACTED 2026-08-07**, see §3.1a

*The table below is kept as the record of what the pooled metric said. Its conclusion — "the
board adds nothing within a state" — does not survive bucketing and must not be cited.*

588 sibling sets, 4256 successors, 8 Monte-Carlo playouts per label half, paired seeds.
Within-state, per-state mean removed, held out by state where training is involved.

| evaluator | corr | notes |
|---|---|---|
| `globals` (no board at all) | 0.146 | nominally the **best** arm |
| `board` (HexConv trunk + flank pool) | 0.132 | |
| `board_xy` (location-preserving readout) | 0.135 | tests, and refutes, "the flank average is the culprit" |
| `polfeat_xy` (frozen policy trunk) | 0.131 | train corr 0.307 vs held-out 0.131 — overfitting |
| `board_solo` (board only, no globals) | 0.066 | train corr **0.003** — cannot fit even its training set |
| trained critic | 0.112 | no better than a 3-epoch globals-only MLP |
| `HeuristicEvaluator` | 0.079 | **74 % predictor ties** |
| 12 hand scalars, joint ridge | 0.105 | |
| *label ceiling* | **0.667** | |

Best 0.146 of a 0.667 ceiling ⇒ **~4.8 % of the achievable variance**. Paired bootstrap on
an earlier 80-state / 16-playout set: `board − globals` corr `+0.008`, 95 % CI
`[−0.138, +0.153]` — the board adds nothing *within* a state, while adding **+0.039 pooled
R²** *across* states (0.163 → 0.202). Different jobs; the board does the second.

The within-state signal itself is real and sizeable: **±8.5 pp of win probability per
decision** (588-state estimate, true-signal std 0.170). Every turn, ~37 times a game.

Two methodological notes for anyone re-running this: the first version of the probe scored
~half the sibling sets with an **inverted predictor** (observations are ego-centric to the
successor's mover, the label is player-1-relative) and reported corr ≈ 0 for everything; and
a cold-started ranking arm trains on ~3 k successors against a regression baseline's 120 k,
so `--init-from` is mandatory or a flat result is confounded with a 40× data gap.

**`board_solo`'s row above is void.** It was never fitted by `fit` — `data/board_value_probe.pt`
holds five arms, not six — so in `rank` it was silently cold-started while every arm it was
compared against was warm-started. Its "train corr **0.003** — cannot fit even its training
set" is the 40× data gap in the note above, not a fact about the board. `mode_rank` now
refuses this configuration up front instead of warning. Fitted properly (§3.1a) it reaches
pooled R² **0.1846**, *above* the globals-only control's 0.1633 — the row inverted completely.

### 3.1a The board does rank siblings — the pooled metric was averaging it away

The within-state question is **two disjoint sub-problems**, and ~5 % of sibling pairs decide
one of them:

- **~30 % of pairs have identical boards.** Recruit vs recruit vs pass: a coin leaves the
  hand, the board never moves. Only globals can rank these.
- **~5 % of pairs have identical non-board inputs and differing boards.** Only the board can
  rank these — a globals-only model assigns them the same value *by construction* and is
  pinned at 50 % with a ~90 % tie rate.

Pooling the two reports an average that understates both. `distinguish` mode always bucketed
its pairs and called "boards differ, same verb" *the sharpest bucket*; `siblings` and `rank`
never carried that through. That is the whole defect.

Re-scored on the **existing** label caches — no new playouts, no retraining. Pairwise
accuracy, ties = 0.5, tie rate in parentheses:

**`data/rank_labels.pt`** — 588 states, policy playouts, 8 per half, ceiling corr 0.675:

| bucket | n | `globals` | `board` | `board_xy` | `polfeat_xy` | HEURISTIC |
|---|---|---|---|---|---|---|
| all pairs *(what §3.1 reported)* | 10490 | 54.8 % | 55.9 % | 56.8 % | 55.9 % | 53.1 % (74 %) |
| board differs, **non-board SAME** | 518 | **49.2 % (87 %)** | **61.0 %** | 56.6 % | 56.8 % | 51.0 % (86 %) |
| board differs, same verb | 1390 | 52.4 % (32 %) | **58.6 %** | 57.9 % | 55.7 % | 53.0 % (67 %) |
| board IDENTICAL | 3062 | 54.0 % | 55.2 % | 55.7 % | 56.3 % | 50.4 % (95 %) |

**`data/la_labels.pt`** — 120 states, `lookahead` playouts, 4 per half (the §2-step-1 run):

| bucket | n | `globals` | `board` | `board_xy` | `polfeat_xy` | HEURISTIC |
|---|---|---|---|---|---|---|
| all pairs | 1796 | 46.8 % | 51.5 % | 50.4 % | 52.7 % | 52.3 % (68 %) |
| board differs, **non-board SAME** | 89 | 49.4 % (94 %) | **68.5 %** | 58.4 % | 62.9 % | 54.5 % (87 %) |
| board differs, same verb | 238 | 49.4 % (35 %) | **59.2 %** | 56.3 % | 56.7 % | 48.9 % (60 %) |
| board IDENTICAL | 514 | 47.5 % | 47.9 % | 46.9 % | 50.6 % | 49.8 % (90 %) |

Pooled, `board − globals` is +1.1 pp and the honest reading is "the board adds nothing". On
the pairs where the board is the *only* thing that differs it is +11.8 pp, 61.0 % against a
structural 49.2 %.

#### Confirmed on the 16-playout labels, with honest error bars

The two tables above were scored without clustering, which overstates precision: sibling
pairs inside one state are correlated, so the effective n is well below the nominal. Redone
on `data/la16_labels.pt` (150 states, 16 playouts, ceiling 0.778) with a **bootstrap
resampling states, not pairs**, 4000 draws:

| bucket | contrast | Δ | 95 % CI | verdict |
|---|---|---|---|---|
| board differs, non-board SAME (n=148) | `board` − `globals` | +7.4 pp | [−0.9, +16.0] | **not resolved** |
| " | v10 alive − v11 dead | +7.1 pp | [−3.3, +17.2] | not resolved |
| " | `board` − `board_xy` | −1.4 pp | [−12.3, +8.1] | not resolved |
| **board differs, same verb (n=396)** | `board` − `globals` | **+8.2 pp** | **[+1.3, +15.4]** | **resolved** |
| " | `board_solo` − `globals` | **+10.7 pp** | **[+3.7, +17.7]** | **resolved** |
| " | **v10 alive − v11 dead** | **+15.2 pp** | **[+4.8, +25.0]** | **resolved** |
| " | `board_solo` − `board` | +2.5 pp | [−2.2, +7.4] | not resolved |
| all pairs (n=2694) | `board` − `globals` | +0.2 pp | [−2.2, +2.6] | not resolved |

Absolute accuracies on the same-verb bucket: `globals` **44.3 % CI [38.8, 50.0]** — *below*
chance, the non-board features actively mislead there; `v11` dead critic 46.0 %; `board`
52.5 %; `board_solo` 55.1 %; **`v10` alive critic 61.1 % CI [53.9, 68.2]**.

**So the board-only bucket at 150 states cannot resolve anything — but the same-verb bucket
can, and it is the sharper test anyway.** Same-verb pairs are two ways to move the same unit:
the same coin leaves the hand, so the economy is fixed and only the position differs. That is
the closest thing in this sample to the decision §6's domain notes are about, and there
**every board-blind evaluator sits at or below chance while every board-reading one is
above it.**

Three further readings, and one non-result:

- **A frozen policy trunk is a wash with the critic's own** at matched readout. Sharing is a
  parameter-count argument, not a quality one — which is why it was dropped from §5's critic
  step.
- **The flank average is not the culprit.** `board` (`_split_pool`) ≥ `board_xy`
  (location-preserving) on every bucket, but no contrast resolves (best CI [−12.3, +8.1]).
  There is no case for replacing the pool, and none for keeping it on evidence either.
- **`HeuristicEvaluator` ties 89 % of board-only and 67 % of same-verb pairs.** This, not its
  ranking quality, is what supports quiescence (§2 step 3) — the leaf is nearest to blind
  exactly where position is the only difference.
- **Non-result worth recording so it is not re-derived:** `board` has strictly more inputs
  than `board_solo`, and its point estimates are *lower* on two of three buckets — which
  would be direct evidence that the head fails to exploit the board when globals are
  available. The contrast does not resolve (+1.4 pp [−2.6, +5.4] on board-differs, −2.0 pp on
  board-only). The mechanism stays plausible on §3.4's sensitivity split (76 % globals vs
  14 % board) and on the dead trunk itself, but it is **not** established here.

#### `board_solo`: the board alone out-predicts every non-board feature

Fitted properly for the first time (`data/board_value_probe_solo.pt`; §3.1's row was
cold-started). Pooled value regression, 120 k samples, held out **by round**:

| arm | pooled R² | trainable params |
|---|---|---|
| `globals` — no board at all | 0.1633 | 42.8 k |
| **`board_solo` — board only, no globals/opp/privileged** | **0.1846** | 177 k |
| `board` — both | 0.2023 | 186 k |

**The board on its own explains more of the outcome than all non-board features combined.**
Its trunk is 43.2 % alive, as it must be — it has nothing else to satisfy the loss with, which
is itself the cleanest demonstration that the trunk dies from lack of gradient pressure, not
from anything intrinsic to the architecture.

And `board_solo` is the exact mirror of `globals`, which validates the bucketing from the
other side: on **board-identical** pairs it scores 50.1 % with a **92 % tie rate**, pinned by
construction, just as `globals` is pinned at 50.0 % with 93 % ties on the board-only bucket.
Two complementary blocks, each blind on the other's bucket — and the pooled row averages
across both, which is why it shows nothing.

### 3.1b What was checked in the instrument, and what it is blind to

Before trusting §3.1a, four things that would have invalidated it were tested. All clean:

| check | result |
|---|---|
| Ego-centric obs vs player-1-frame label (the `sign` factor) | ✅ `corr(sign·pred, z)` = +0.50…+0.66 pooled, raw `pred` ≈ 0. `HEURISTIC` correctly needs no sign (+0.51 without it) |
| Are the eval's **own** board arms alive? | ✅ final-ReLU pre-activations > 0: `board` 25.9 %, `board_xy` 44.1 %, `polfeat` 19.0 %. §3.1 was **not** a dead-trunk artefact |
| What the sibling filter discards | ✅ 97.1 % of legal successors kept. Losses: **100 % of tactic initiations** (48/48 — they always leave a pending choice), 30 recruit, 3 control, 2 terminal |
| Label granularity (only multiples of 0.5 at 4 playouts) | ✅ artefact-free: with `lookahead` playouts draws are ~absent, so playout sums are always even |

Two bounded blind spots to state whenever these results are quoted. The instrument is
**structurally blind to the tactic verb** (the pending filter removes all of it), and its
successor mix is 40 % recruit / 26 % move / 13 % pass / 7 % deploy / 6 % bolster / 6 % claim
and only **0.7 % attack** — the last because attacks are rarely legal, not because of the
filter. So "ranking the successors of a position" here means mostly ranking economy and
manoeuvre choices. Also: `enumerate_siblings`' docstring claimed all successors leave the
same mover to act; ~17 % of sets actually mix both. Numerically fine (`sign` handles it), but
a sibling set is not a fixed ply parity.

Fixed in the tool at the same time: `playout()` swallowed every exception into a `0.0` label,
indistinguishable from a genuine draw — it now returns the outcome kind and `_label_sets`
reports the abort rate, so this class of silent label corruption is visible in future runs.

### 3.2 Hand-built features: weak, but too coarse to be evidence either way

Correlation of each hand-built scalar with the same within-state label:

```
units_diff  0.097   stack_diff 0.084   boxed_diff 0.066   base_diff  0.053
at_risk_diff 0.053  anchored_bases 0.050   opp_at_risk 0.049   dist_to_their_bases 0.037
their_dist_to_mine 0.023   unanswerable_threats 0.022   own_at_risk 0.020   hand_size 0.000
```

Joint ridge over all twelve, held out by state: corr **0.105** (0.104–0.106 across
`lam` 1/10/100), pooled Spearman 0.105, mean per-state Spearman 0.104.

> **Do not read more into this than it supports.** The twelve features **tie on 45.5 % of
> sibling pairs** — they assign two siblings the same vector, so the ridge cannot express a
> preference at all. Pairwise accuracy is 52.7 % with ties scored 0.5, and 55.0 % among the
> pairs it actually ranked. They are aggregate scalars: `own_at_risk` is a *sum* of
> at-risk material, which cannot tell "my Berserker is hanging" from "my Footman is hanging
> on the far flank". So *"crude material counts beat every threat term"* is true of these
> features and says little about whether threat structure carries the signal — a
> conclusion an earlier draft of this doc over-reached on.
>
> This is the third appearance of the same measurement trap: a coarse predictor scored as
> if a tie were a wrong answer. It produced `HeuristicEvaluator`'s spurious "20 %"
> pair-accuracy (§3.1 note) and a spurious "30 %" here. **Always score ties as 0.5 and
> report the tie rate** — the tie rate is itself the interesting number.

### 3.3 Bases change hands only through combat *(label-independent)*

30 policy-vs-policy games:

| | policy vs policy | vs `lookahead` |
|---|---|---|
| own controlled bases with a friendly unit on them | **45.0 %** | 42.2 % |
| plies where a *free* steal was available (base empty, own unit adjacent, matching coin) | 4.4 % | 6.8 % |
| steals from an **occupied** base (a kill came first) | **53** | 24 |
| steals from an **empty** base (walked in) | **0** | 1 |

Every steal in 30 games followed a kill. `is_valid_claim` makes an occupied base untakeable,
so a parked unit is a lock. The low steal rate (~1.8/game) is **correct play, not
passivity** — which killed the `ReclaimerBot` design and, with §1.1, removed the whole "the
opponent pool never contests" diagnosis.

Game shape: 11.1 rounds, 74.6 plies, peak stack 2.20; per player per game ~1.3 bolsters,
~1.6 tactics, ~5 recruits, ~1.7 initiative claims. **The mechanics are not unused — they are
unconditioned.** ([IDEAS.md](IDEAS.md) #R8/#13's "essentially never bolsters / may never
recruit" is out of date.)

### 3.3b The critic's **target** outranks its board pathway *(2026-08-07)*

The largest single effect in the 16-playout table is not about the board at all. Holding the
obs version fixed at v11:

| evaluator | target | board | within-state corr | ceiling |
|---|---|---|---|---|
| critic `20260727-0506` | shaped GAE returns (PPO) | **dead — outputs zero** | **0.184** | 0.778 |
| `board` arm | ExIt `z`, the game outcome | alive, 26 % | 0.099 | 0.778 |
| `globals` arm | ExIt `z` | none | 0.111 | 0.778 |

**A board-blind critic trained on shaped returns outranks a board-reading network trained on
the game outcome, by roughly 2×.** The mechanism is in the target: a shaped return carries the
per-action immediate reward, so the difference between two siblings is *in the label*; `z` is
a single terminal outcome shared by the whole trajectory, so within-state differences reach
the model only through the discounted tail.

This is a direct hit on **expert iteration**, whose critic objective is exactly
`MSE(critic_raw, z)` (`expert_iteration.py`, "the critic with `return_mean=0`/`return_std=1`,
since it now predicts z on the [-1,1] scale"). ExIt swapped the critic onto the worse of the
two targets for the one job — ranking siblings — that its search then depends on.

Caveats, stated because the comparison is not a clean ablation: the critic is `hidden_dim`
192 against the arms' 96, and it was trained on a full PPO run rather than 120 k samples. The
obs version is the same, and the gap is 2×, so treat the direction as solid and the factor as
soft. The clean test is one `fit` arm trained on shaped returns instead of `z` — not yet run.

### 3.4 The critic's board trunk dies irreversibly *(label-independent)*

`Policy` and `Critic` share the identical `HexConv2d` stack, yet:

| Checkpoint | obs | final-ReLU pre-activations > 0 |
|---|---|---|
| policy `20260727-0506` | v11 | **21.40 %** |
| critic `20260725-1737` | v10 | 19.84 % |
| critic `20260727-0506` | v11 | **0.00 %** |

Mechanism, settled: **not** dead at init (fresh trunks are 51 % alive, 0/8 seeds dead); the
dead trunks' weights sit within **10 %** of initialisation scale (live ones grew 3.6–12.8×),
so they froze within a few hundred updates; and once every pre-activation is ≤ 0 the ReLU
gradient is exactly 0 and Adam's moments stay 0 — frozen forever. The dead critic's conv3
pre-activation max is **−0.003** across 1085 states and all 192 channels.

**What it costs, measured (2026-08-07).** Both critics scored on the same cached labels with
the §3.1a buckets — pairwise accuracy, tie rate in parentheses:

| bucket | `20260725-1737` v10, trunk **alive** | `20260727-0506` v11, trunk **dead** |
|---|---|---|
| all pairs — `rank_labels` | 58.5 % | 55.8 % |
| all pairs — `la_labels` | 57.3 % | **50.7 %** |
| board differs, non-board SAME — `rank_labels` | 61.4 % | **50.6 % (89 %)** |
| board differs, non-board SAME — `la_labels` | 62.9 % | **48.9 % (93 %)** |
| board differs, same verb — `rank_labels` | 59.2 % | 54.2 % (33 %) |

The 89–93 % tie rate is not statistics, it is **arithmetic**: the pooled block is identically
zero, so two siblings with equal non-board inputs receive equal values and the critic cannot
express a preference at all. So the dead critic is structurally unable to rank **5 % of all
sibling pairs and 33 % of same-verb pairs** — and it is the critic driving the gauntlet, ExIt,
`PuctBot` and `LookaheadCriticBot` (`lookahead_critic_v4.pth` is byte-identical, md5
`fe3c43df`). Every search result on record has a board-blind leaf.

On the 16-playout labels with a state-clustered bootstrap, the alive−dead gap on the same-verb
bucket is **+15.2 pp, 95 % CI [+4.8, +25.0]** (§3.1a) — the largest resolved contrast in that
table. Caveat: the two checkpoints differ in obs version (v10/v11) and training run as well as
trunk health, same `hidden_dim` 192, so this is not a clean ablation of trunk health alone.
The board-only rows are, because the mechanism there is definitional.

**Why it dies, now with the positive control.** `board_solo` — the same HexConv+ReLU stack with
*no* globals or privileged inputs — trains to a 43.2 % alive trunk and pooled R² 0.1846
(§3.1a). Same architecture, same data, same optimiser; the only difference is that its head
has no alternative way to satisfy the loss. So the absorbing state is not a property of the
stack, it is what happens when the head can reach its target through 299 non-spatial inputs
and the board pathway is left with negligible gradient. That is the argument for creating
pressure (a board-only auxiliary head) alongside removing the trap (GroupNorm).

Why the critic and not the policy: the policy's trunk feeds a per-cell 1×1 conv and globals
are spatially constant, so the trunk is the *only* thing that can discriminate cells and must
stay alive. The critic's trunk feeds two numbers per channel into a head with 299
non-spatial inputs that already contain the dominant predictor. Even the *healthy* v10 critic
draws only **14.1 %** of head sensitivity from the board vs **76.2 %** from globals.

#### What the fix bought, measured (2026-08-08) — `critic_v2` trained and scored

First `critic_v2` run: `data/warchest_critic_20260808-0607.pth`, obs v11, hidden 192. Scored
against the shipped dead critic on `data/la16_labels.pt`, same instrument, same labels.

The trunk is alive and carrying information — the thing the fix existed to do:

| | arch | per-conv alive | `out_std` |
|---|---|---|---|
| `20260727-0506` | `critic_v1` | `[0.3876, 0.3568, **0.0000**]` | **0** |
| `20260808-0607` | `critic_v2` | `[0.4250, 0.2620, 0.1852]` | **0.116** |

**The decisive column is `tied`, not `corr`.** The dead critic could not express a preference
at all on the pairs the board alone separates; the new one ranks every one of them:

| bucket | n | old tie rate → new | old acc → new | Δ, 95 % CI (state-clustered) |
|---|---|---|---|---|
| **board differs, same verb** | 396 | 34 % → **0 %** | **46.0 % → 55.8 %** | **+9.8 pp [+1.3, +18.8] — resolved** |
| board differs, non-board SAME | 148 | 93 % → **0 %** | 50.3 % → 55.4 % | +5.1 pp [−3.7, +13.9] |
| all pairs | 2694 | 5 % → **0 %** | 56.9 % → 58.3 % | +1.4 pp [−2.3, +5.4] |
| board IDENTICAL | 824 | 0 % → 0 % | 59.0 % → 58.9 % | −0.1 pp [−5.1, +4.8] |

Three readings:

- **The gate passed.** Same-verb accuracy came off 46.0 % — which was *below chance*, i.e. the
  dead critic was actively misleading exactly where position is the only difference — to
  55.8 %, and the contrast resolves. `top1` rose 30.7 % → **38.7 %** against a 26.5 % chance
  rate, which is the quantity search consumes directly.
- **Nothing regressed.** The board-identical bucket (economy choices, where only globals can
  rank) is unchanged at −0.1 pp. The board pathway was added without costing the globals
  pathway — the main risk of the aux head, and it did not materialise.
- **Headroom remains.** The healthy v10 critic scores 61.1 % on the same-verb bucket and
  Spearman 0.222; `critic_v2` reaches 55.8 % and 0.198. Most of the gap to a healthy critic is
  closed, not all of it. (v10 is a different obs era and run, so this is a reference point,
  not a clean comparison.)

**One thing to watch: the new critic's output has much heavier tails.** Pooled `corr` *fell*
0.184 → 0.069 and `R²` worsened −0.175 → −0.369, while every rank-based metric rose
(Spearman 0.142 → 0.198, pair-acc 56.9 % → 58.3 %). That is not a ranking regression — it is
outliers, and the check is direct:

| | kurtosis (3 = normal) | max \|z\| | frac \|z\| > 4 | corr | corr minus top 1 % \|z\| |
|---|---|---|---|---|---|
| `critic_v1` | 8.2 | 6.2 | 0.65 % | +0.184 | +0.180 |
| `critic_v2` | **20.2** | **9.3** | **1.12 %** | +0.069 | **+0.119** |

Dropping the most extreme 1 % of predictions recovers most of v2's Pearson deficit
(+0.069 → +0.119) and barely moves v1's (+0.180). Pearson measures a *linear* fit and is
dominated by outliers; Spearman and pairwise accuracy are rank-based and are what
`PuctBot`/`LookaheadCriticBot` actually consume, so the operative metrics improved. The `R²`
move is pure calibration, as the tool's own HOW TO READ says. **But the tails are worth
tracking**: `LookaheadCriticBot` calibrates on the raw value scale, and a 9 σ output could
distort PUCT. If search behaves oddly after this change, look here first.

**A second failure mode, found while implementing the guard (2026-08-07).** GroupNorm makes
the absorbing state *unreachable*, which is the point — but it also makes the alive fraction
a **useless** guard on its own. Force the last conv to a constant −50 and:

| | per-conv alive | trunk output |
|---|---|---|
| `critic_v1` | `[0.50, 0.49, **0.00**]` | exactly 0 |
| `critic_v2` | `[0.50, 0.51, **1.00**]` | constant 6.1e−05 |

v2 reports **fully alive** while carrying no information at all: GroupNorm re-centres the
constant into a small *positive* one, so every pre-activation passes the `> 0` test. A constant
all-positive output is exactly as board-blind as an all-zero one. So `Critic.trunk_health`
returns **both** the per-conv alive fractions and `out_std` — the spread of the pooled block
across the batch — and `ppo.py` alarms on `min(alive) == 0 or out_std < 1e-6`. `out_std` is the
condition that covers both architectures; the alive fraction only diagnoses v1.

Fresh v11 trunks, 8 seeds — only two options remove the absorbing state:

| Variant | nonzero at init | \|out\|max | Removes it? |
|---|---|---|---|
| ReLU (current) | 51.05 % | 0.083 | **no** |
| GELU | 51.32 % | 0.029 | no — `GELU'(x) → 0`, and shrinks the signal 3× |
| LeakyReLU(0.01) | 51.06 % | 0.083 | **yes** — constant floor |
| ReLU + GroupNorm | 50.26 % | **5.03** | **yes, structurally** |

GroupNorm matters more than the activation. Not BatchNorm (`ppo.py` toggles
`critic.eval()/.train()` around the rollout, so batch statistics would desynchronise the
values the critic is fit to from the values used in the rollout).

### 3.5 `opp_onehot` moves the value more than the position does *(label-independent)*

`V(start)` over 40 fresh games, raw output: `random` +0.681, `greedy` −0.066, `pool` +0.089
(v11) — a **0.747 spread** against a 0.44 std of `V` across positions. Per-input weight norm
1.276, comparable to globals (1.798) and privileged (1.713). *(An earlier "0.0 % sensitivity"
reading was an artefact: `eval_privileged_ablation.py` hard-codes this block's input std to
`1e-9` because the opponent is constant within one eval run.)*

Three consequences. It is **correct for the PPO baseline** — win rates vs `random` 1.000, vs
`greedy_fast` 0.825, vs itself 0.525, so a value function blind to the opponent gives every
action against `random` a positive advantage and every action against a snapshot a negative
one, regardless of merit. It is **constant during finetune**
(`p_random/greedy_finetune = 0`, and `OPP_ONEHOT_SLOT` maps `lookahead_critic`/`puct` onto
`pool`), so all its usefulness is in the initial phase. And it makes the absolute output
**meaningless to every search bot**, each of which must pick a slot arbitrarily — on top of
return normalisation (`return_mean = 0.214`, `return_std = 0.690`), so the raw output is a
z-score of a *shaped* return, never a win probability. Clean replacement when the critic is
next touched: group `rollout_buffer.py:181`'s advantage z-scoring by opponent type and drop
the input.

#### Done 2026-08-09 (§5 row 6) — and why the pair is safe

`critic_v3` (now the default) is v2 minus the one-hot; `RolloutBuffer.compute_gae(adv_norm=
'per_opponent')` (now the default) subtracts each opponent group's own mean advantage before
applying one shared std. Full record in `docs/history.md`. Three points worth pinning here,
because each was a live question while implementing it:

- **Removing the input costs the critic nothing where it ranks.** The offset is constant across
  the siblings of a state — the opponent does not change within a decision — so it cancels in
  the sibling comparison *and* in `δ_t = r_t + γV(s_{t+1}) − V(s_t)`. What the one-hot bought
  was absolute level, and the advantage is the only place that level was doing work. This is
  also why the change composes with a lower `λ` rather than fighting it (`IDEAS.md` L2).
- **Mean-only, not per-group z-scoring.** Returns against `random` have far less spread than
  returns against a snapshot, so rescaling each group to unit variance would hand the
  near-deterministic group as much gradient weight as the group carrying the real signal. One
  shared std; per-group means only. Pinned by `tests/test_rollout_buffer.py`.
- **The grouping is finer than the one-hot ever was.** `OPP_GROUP_IDX` gives
  `lookahead_critic`/`puct` their own groups, which `OPP_ONEHOT_SLOT` cannot (it collapses them
  onto `pool` to keep v1/v2 checkpoints loadable). Finetune is 75 % `pool` / 25 %
  `lookahead_critic`; grouping those together would have left in exactly the offset being
  removed.

`adv_group_spread` is logged per batch — the max−min of the removed offsets, i.e. how much of
the raw advantage was opponent identity rather than action quality. A 6-episode smoke run at
`random`/`greedy` 50/50 measured **0.37–0.54** with `random` above `greedy`, the predicted sign.
Expect it to shrink in finetune as the pool narrows. **The gate (pooled R² ~0.20) has not been
run**; this is implemented, not measured.

### 3.6 The critic is not biased against any verb *(label-dependent)*

Where the critic enters the gradient: GAE telescopes to `Â_t = G^λ_t − V(s_t)`, where
`V(s_t)` is identical for every action at that state — so it cancels in the comparison and
only sets variance — and `V(s_{t+1})`, the only place ranking ability can act, enters at
`γ(1−λ) ≈ 0.03`. At `λ = 0.97`, **~97 % of the discriminative signal is the realised
return.** Corollary: `λ = 0.97` is accidentally the right setting for a critic that cannot
rank, and improving the critic buys PPO nothing unless `λ` drops with it.

Residual `y − V` by verb, demeaned per state, 4256 successors — negative would mean the
critic overrates the verb and systematically starves its advantage:

| verb | n | residual | sem |
|---|---|---|---|
| move | 1120 | +0.0008 | 0.0103 |
| pass | 550 | +0.0156 | 0.0125 |
| recruit | 1710 | −0.0005 | 0.0076 |
| **bolster** | 263 | **−0.0247** | 0.0224 |
| attack | 30 | −0.0341 | 0.0569 |

No verb is significant (bolster −1.1 σ). The critic's failure mode is **"adds no
information", not "adds wrong information"** — it is not why bolster is suppressed.

### 3.7 The policy suppresses bolster unconditionally *(label-independent)*

Over 526 decision states, bolster is legal in **63.9 %** of them with `P̄(verb) = 0.029`,
p10 `0.0000`, p90 `0.0804` — **never** the preferred verb in any state — and its within-verb
spatial spread (2.62) is the second-narrowest of all verbs, so there is no opinion about
*which* unit to bolster either. A collapsed mode, not a judgement. Both critics meanwhile
rank the best available bolster in the top 21–29 % of successors, so there is no critic-side
veto. This **corrects** [rewards.md](rewards.md) §1 and
[independent_opponents.md](independent_opponents.md) §2, which both assume bolstering is
unlearned because nothing pays for it.

Related and untested on the actor side: `P(verb)` is computed from
`verb_head(split_pool(feat) ++ global)` — a flank-averaged summary. The within-verb head is
fully spatial, but the decision "should I bolster at all" is made location-blind, which is
the wrong input for a verb whose value is entirely "this unit, this cell, right now".

---

## 4. Ruled out — with the caveat that matters

Everything marked *(label-dependent)* rests on the policy-continuation label that §1.1
downgrades. The 16-playout re-run (§2 step 1) shows the binding problem was **label
precision**, not the playout bot, and it re-opens several of these.

| Idea | Status | Basis |
|---|---|---|
| ~~Board-architecture work on the critic — not justified~~ | **RE-OPENED 2026-08-07** | the basis was §3.1's pooled metric. Bucketed, on same-verb pairs `board` − `globals` = +8.2 pp CI [+1.3, +15.4] and `board_solo` − `globals` = +10.7 pp CI [+3.7, +17.7]; the board alone out-predicts every non-board feature across states, R² 0.1846 vs 0.1633 (§3.1a) |
| Keeping the critic's board trunk at all | **required** | the dead-trunk critic ties 89–93 % of purely positional sibling pairs and cannot rank them by arithmetic; the alive−dead gap on same-verb pairs is +15.2 pp CI [+4.8, +25.0] (§3.4). Deleting the trunk would make that permanent |
| Location-preserving readout (`board_xy`) instead of the flank pool | **no support** | `board` ≥ `board_xy` on every bucket, but nothing resolves (best CI [−12.3, +8.1]). Neither replace the pool nor claim it is fine — this is untested, not settled |
| Shared policy/critic encoder + stop-gradient | **demoted to an optimisation** | a wash with the critic's own trunk at matched readout (§3.1a). It cuts parameters and removes the death mode, but buys no quality and caps the critic at the actor's representation |
| An explicit ranking objective for the critic | **superseded** | the objective that matters is not ranking-vs-regression but **shaped returns vs `z`**: a board-blind critic on shaped returns beats a board-reading net on `z` by ~2× (§3.3b). Fix the target before adding a term |
| The hand-written leaf as the positional reference | **dead** *(new)* | `HEURISTIC` Spearman fell 0.246 → **0.149** when labels went 4 → 16 playouts while the ceiling rose to 0.778, and it ties 89 % of board-only pairs (§2 step 1, §3.1a). It is the *worst* evaluator on the clean labels, not the best |
| `ReclaimerBot` / "the pool never contests bases" | **dead** *(label-independent)* | free steals available in 4.4 % of plies; 0 of 53 steals from an empty base (§3.3) |
| `BolsterBot` and archetype opponents generally | **dead** | an archetype *demonstrates*; a learner best-responds rather than imitates. Keep as a gauntlet entrant only |
| Belief / IS-MCTS / CFR | **parked** | cheat − blind ≈ 0, but measured with a leaf that cannot rank in *either* arm — uninformative both ways |
| Widening the networks | **dead** | `hidden_dim` 64→128 and `critic_hidden_dim` 128→192 both targeted an underfit with a different cause; that generation came back worse (BT-Elo 923, last of four) |
| Reward-table tuning | **not the constraint** | two hygiene items remain: zero `ATTACK_REWARD = 0.02` (double-pays what material PBRS covers, flagged since 2026-07-03), and re-derive `holding_reward_rate` from ~37 real main-actor turns rather than the assumed 150 — it is ~4× weaker than designed and is not a mechanism in practice |
| ExIt as currently configured | **gated, and now with a named mechanism** | two independent faults, neither addressable by Dirichlet/simulation/replay tuning: its teacher was the student (§1.1), and its critic objective is `MSE(critic, z)` — the worse of the two available targets for sibling ranking, by ~2× (§3.3b) |

Two methodological rules this investigation produced, worth keeping:

- **Pick the anchor by strength relative to the subject.** §8.1 had two of four measurements
  sitting at floor or ceiling.
- **Reliability is not validity.** Two runs of the same measurement agreeing tells you
  nothing about whether it measures the intended thing. This cost several days here.
- **In a paired within-state measurement, label precision is binding, not sample count.**
  The quantity is a small difference, so label noise enters it twice while sample count only
  shrinks the standard error. Two runs here were spec'd the wrong way round (`--playouts` 8
  and 4 against 588 and 120 states) before this was noticed. Confirmed in the strongest
  possible way by the 16-playout re-run: **two headline conclusions reversed sign** when only
  the playout count changed (§2 step 1). Do not report a within-state number below
  `r ≈ 0.4` half-to-half reliability.
- **Cluster the bootstrap by the unit that was sampled.** Sibling pairs inside one state are
  correlated, so a pair-level interval is far too tight. The `board − globals` gap on the
  board-only bucket read as "~5 σ" on pair counts and as **[−0.9, +16.0], not resolved** once
  states were resampled instead. Everything in §3.1a is state-clustered; nothing before it
  was.
- **Score ties as 0.5 and print the tie rate.** A coarse predictor scored as if a tie were a
  wrong answer produced a spurious "20 %" for `HeuristicEvaluator` and a spurious "30 %" for
  the hand-feature ridge. The tie rate is itself the interesting number (68–74 % for the
  heuristic, 45.5 % for the hand features).
- **Never pool two sub-problems whose answers come from different inputs.** If some pairs are
  separable only by block A and others only by block B, the pooled score is a mixture whose
  value depends on the mixing ratio, and *every* evaluator is dragged toward the middle —
  which reads as "all methods fail equally" and indicts the target. Bucket first, then
  average if you still want to. This one cost the §3.1 conclusion and, downstream of it,
  §4's first row. The related trap: **a control that is pinned at chance by construction is
  not a baseline.** `globals` ties 87–94 % of the board-only bucket; reading its 49 % there
  as "nothing works on these pairs" inverts the meaning.
- **Never let one arm be cold-started while its comparators are warm-started.** `board_solo`
  was missing from the `fit` work file, so `rank` silently trained it from scratch and its
  flat result was recorded as evidence about the board. The tool now refuses.
- **Never select a hyperparameter on the set you then report.** Epoch-selection on a
  30-state held-out split inflated `board` from 0.032 to 0.129.

---

## 5. Sequencing

| # | Work | Cost | Training run? | Gate |
|---|---|---|---|---|
| 0 | ~~Bucket the within-state metric; re-score both cached label sets~~ | done | — | **done 2026-08-07, positive** (§3.1a): the pooled metric was averaging two disjoint sub-problems. §3.1 retracted, §4 row 1 re-opened. Zero new compute |
| 1 | ~~Re-label against `lookahead`, 120 states @ 4 playouts~~ | done | — | **run, inconclusive and, it turned out, misleading** — 58 % label noise (§2 step 1) |
| 1b | ~~Re-run at 150 states @ 16 playouts~~ | done | — | **done 2026-08-07. Both of step 1's consolations reversed** (§2 step 1): `HEURISTIC` Spearman 0.246 → **0.149**, true signal std 0.208 → **0.158**, while every learned arm *rose* and the v10 critic reached 26 % of ceiling. Verdict: distil into a network; the hand-written leaf is the worst evaluator on clean labels |
| 1c | ~~Fit `board_solo` properly~~ | done | — | **done 2026-08-07:** pooled R² **0.1846** vs the globals-only control's 0.1633 — the board alone out-predicts every non-board feature. Trunk 43.2 % alive with no globals to hide behind (§3.1a, §3.4) |
| 2a | ~~**Dump shaped returns** from `ppo.py`'s rollout~~ | done | — | **shipped 2026-08-07:** `--dump-returns-dir`. Writes `round*.npz` with the shaped GAE return under the key `z`, so `eval_board_value.py fit --data` reads it with no code change (verified end to end). Still needs a short run to fill it |
| 2b | **Critic target A/B** — a `fit` arm on shaped returns vs `z`, scored on `data/la16_labels.pt` (§2 step 2, §3.3b) | hours after a run | no | does the ~2× gap survive at matched `hidden_dim` and data? If yes, ExIt's `MSE(critic, z)` is the single biggest identified defect |
| 3 | ~~**Critic trunk** — GroupNorm + board-only auxiliary head + health guard~~ | done | — | **shipped 2026-08-07, trained and verified 2026-08-08 (§3.4).** Trunk alive (`[0.425, 0.262, 0.185]`, `out_std` 0.116 vs the dead critic's 0.000); tie rate on positional pairs **93 % → 0 %**; same-verb accuracy **46.0 % → 55.8 %**, +9.8 pp CI [+1.3, +18.8]; no regression on economy pairs. `--critic-arch critic_v1` reproduces the old baseline. **Gate met — row closed.** |
| 3b | **Head-to-head: `critic_v2` vs `critic_v1` in the gauntlet** — the behavioural check the within-state metric cannot give (see the Appendix command) | ~1 h | no | does the ranking gain convert to wins? A 6-game smoke run went 6/6 to `critic_v2`; needs ~200 games before it means anything |
| 4 | **Quiescence in `LookaheadBot`** (§2 step 3) | 1–2 days | no | board-only tie rate well below 89 %; beats plain `lookahead` head to head; bolster/tactic rates rise without losing WR |
| 5 | **Conditional metrics** in `eval_bucketed`, baselined on today's checkpoint | hours | no | must exist before any run: `P(bolster \| own unit on my base ∧ stack 1 ∧ enemy in reach ∧ matching coin)`, base retention, steal-after-kill rate, forward progress by unit tier |
| 6 | ~~**Remaining critic hygiene** — drop `opp_onehot` + per-opponent advantage normalisation~~. Shared encoder + stop-gradient is **not** part of this (§4: demoted to an optimisation) | done | no | **shipped 2026-08-09** as `critic_v3` (default) + `--adv-norm per_opponent` (default); see §3.5 and `docs/history.md`. Landed as a *new* arch, not a mutation of v2 — `warchest_critic_20260808-0607.pth` is v2 and is the checkpoint that proved row 3. 181 tests pass; all five prior critic checkpoints still load. **Gate still open:** pooled R² holds ~0.20, which needs a run + `fit` |
| 7 | **Conditional-bolster overlay** on `SimGreedyBot` — the never-run clean test of the domain claim | hours | no | paired seeds vs the unmodified bot |
| 8 | **▶ FIRST TRAINING RUN** — contents decided by 2, 3 and 4 | a run | **YES** | gauntlet Elo holds/rises **and** step-5 conditional metrics move off the floor |
| 9 | `puct` vs `lookahead` vs the **raw policy** (never run) | ~1 h | no | if search over the critic does not beat the policy guiding it, search contributes nothing. Must run *after* row 3 (the trunk fix): every search result on record used a board-blind leaf |
| 10 | Verb-gate `logsumexp` residual (§3.7) | hours | prep | A/B against `beta = 0`, which reproduces today's model exactly |
| 11 | **▶ SECOND TRAINING RUN** / ExIt with a quiescent teacher **and a shaped-return critic** | days | **YES** | a round beats base **and** step-5 metrics move |

Rows 2a–7 need no training run. What steps 0–1c changed about the plan: **the critic's
training, not the search, is now the best-supported target.** Two contrasts resolved under a
state-clustered bootstrap — the target (~2×, §3.3b) and trunk health (+15.2 pp, §3.4) — while
the two findings this document was built on (the heuristic as positional reference; the board
as within-state dead end) both reversed. Quiescence slipped from first to fourth; it is still
justified, on a narrower argument.

**Row 3 is done and its gate is met (§3.4).** The head of the list is now row 2b — the
critic-target A/B — which needs one PPO run with `--dump-returns-dir` to fill its dataset. If
the 2026-08-08 run was launched without that flag, the dataset does not exist yet and the
cheapest way to get it is to set the flag on the next run rather than to run one for it alone.
Row 3b (head to head) is independent and costs an hour.

**Do not raise `--playouts` below 16 again for any within-state measurement.** Two headline
conclusions in this document flipped sign on that parameter alone.

---

## 6. Domain notes and parity questions

From the user, recorded because the environment has to match:

- **Attrition is real but partial.** Site agents typically box **4–6 coins of the 16–20**
  available. Some human games reach ~**10 of 20**, which is roughly two units fully dead or
  one dead and two badly weakened — already a large advantage, *provided you have more units
  left*. So the axis exists in real play, but as *material pressure*, never as elimination.
- **Attacking is prophylactic.** You attack so your unit is not attacked; that is the whole
  logic. Sacrificing a 3-stack to walk onto an enemy base and pick up two attacks on the way
  is correct **only** as a game-closing move.
- **The focus is positional**, not bases / attacks / attrition as separate objectives.

Parity questions, in order of how much each would change the game:

1. **Does material pressure have any terminal consequence?** The env has only 6-base control
   plus `max_rounds = 50` truncation. Killed coins go to the box permanently, but a player
   can be ground down without the game registering it — and per the numbers above the real
   game does not end on elimination either, so the honest question is whether the *pressure*
   pays at all. In the env it pays only instrumentally, by opening a path to a base.
2. **`max_on_board = 1` per unit type** (Footman 2, `roster.py:52`). If the real game allows
   several of a type, the deploy/bolster trade-off is completely different.
3. **Board & marker counts**: 10 bases (2+2 home, 6 neutral), win at 6, `HAND_SIZE = 3`,
   2 coins per type in the bag and 2–3 in supply. If any differ, the 11-round race length is
   an artefact.
4. **`max_rounds = 50` vs games ending at 11.** Not a rules bug, but every constant derived
   from `max_rounds` is calibrated for a game 4.5× longer than the one played.

---

## Appendix — tools

`src/app/eval_board_value.py`, four modes. `--playout-bot lookahead` is the §2-step-1
reference; `--init-from` is mandatory when comparing a ranking arm against a regression
baseline (and is now enforced — see §3.1's `board_solo` note).

**Read the bucketed table `siblings` prints, not the pooled one above it** (§3.1a). The
buckets that carry the answer are `board differs, non-board SAME` (the board's only
uncontested ground — but underpowered below ~400 states) and **`board differs, same verb`**
(the one that resolves at 150 states, and the closest thing here to a purely positional
choice). In `rank` mode the first is the `bd-only` column. A dead trunk announces itself as a
~90 % tie rate in either.

`--playouts 16` is the floor for any within-state claim — see §5's closing note.

**Running the shipped critic fix (§2 step 2).** `critic_v2` is the default, so a plain
`python src/app/ppo.py` already gets GroupNorm, the board-only auxiliary head and the guard.
Adding the dump makes the same run fill the dataset the target A/B needs:

```bash
# one run does both: verifies the trunk stays alive AND collects shaped-return targets
python src/app/ppo.py --dump-returns-dir data/ppo_returns
# baseline for comparison, reproducing the old (dying) trunk exactly
python src/app/ppo.py --critic-arch critic_v1

# watch these per batch, in the console or W&B: alive1..3 must stay ~0.2-0.5 and never hit
# 0.000; out_std must stay well above 0 (a constant trunk reads "alive" but is still blind)
grep -o 'alive[0-9]=[0-9.]*\|out_std=[0-9.]*' logs/ppo_<run_id>.log | tail -20

# then the target A/B — shaped returns vs ExIt's outcome z, same machinery both sides
python src/app/eval_board_value.py fit --data 'data/ppo_returns/round*.npz' \
    --arms globals board board_solo --work data/probe_shaped.pt
python src/app/eval_board_value.py siblings --labels data/la16_labels.pt \
    --work data/probe_shaped.pt
# compare the `board` row's corr against the 0.099 it scores trained on z; ~0.18-0.20
# confirms the target is the lever and ExIt's MSE(critic, z) is the defect.

# and score the new critic against the dead one on the same labels
python src/app/eval_board_value.py siblings --labels data/la16_labels.pt \
    --critic-path data/warchest_critic_<new>.pth data/warchest_critic_20260727-0506.pth
# read `board differs, same verb` and the `tied` column: dead v11 scores 46.0% at a 34% tie
# rate, critic_v2 scores 55.8% at 0%, the healthy v10 reference is 61.1%.
```

**Head to head in the gauntlet (§5 row 3b)** — the behavioural check the within-state metric
cannot give. `--lookahead-critic-checkpoints` builds one `LookaheadCriticBot` per path, so the
two critics play each other with everything else held identical; colours alternate per game:

```bash
python src/app/gauntlet.py --bots lookahead_critic \
    --lookahead-critic-checkpoints data/warchest_critic_20260808-0607.pth \
                                   data/warchest_critic_20260727-0506.pth \
    --k-games 200 --lookahead-critic-time-budget 0.1
```

`--k-games` is games **per pair**, so this is 200 games; at 0.1 s/move budget expect roughly
an hour on 8 workers. se(WR) at 200 games is ~3.5 pp, so read anything inside 50 % ± 7 pp as a
draw. Agent names come from the run stamp (`c0808_0607` vs `c0727_0506`) — before 2026-08-08
both would have rendered as the same `lac_warche` column, which made this exact comparison
unreadable.

```bash
python src/app/eval_board_value.py distinguish --games 40 --stride 3
python src/app/eval_board_value.py fit --max-samples 120000 --epochs 3
# The live label set (150 states, 16 playouts, ceiling 0.778) — ~3 h to regenerate
python src/app/eval_board_value.py siblings --states 150 --playouts 16 --max-siblings 8 \
    --relabel --playout-bot lookahead --labels data/la16_labels.pt
python src/app/eval_board_value.py siblings --labels data/la16_labels.pt \
    --critic-path data/warchest_critic_20260727-0506.pth data/warchest_critic_20260725-1737.pth
python src/app/eval_board_value.py rank --labels data/la16_labels.pt \
    --init-from data/board_value_probe.pt --epochs 12
```

Label caches on disk: `data/la16_labels.pt` (**the one to use** — 150 states, 16 playouts,
`lookahead`, ceiling 0.778), `data/rank_labels.pt` (588 states, 8 playouts, policy, ceiling
0.675), `data/la_labels.pt` (120 states, 4 playouts — 58 % noise, produced two conclusions
that later reversed; keep only as the record). Arm work files: `data/board_value_probe.pt`
(five arms, no `board_solo`), `data/board_value_probe_solo.pt` (`globals` / `board` /
`board_solo`).

Label caches keep the raw state, so re-scoring another critic — including one from a
different obs era, which is re-encoded on the fly — is instant, and a weight fingerprint is
printed per checkpoint (`lookahead_critic_v4.pth` and `warchest_critic_20260727-0506.pth` are
byte-identical, md5 `fe3c43df…`; the critic driving the gauntlet, ExIt and every §8.1
measurement *is* the dead-trunk critic from the last PPO run).

Eight one-off probes produced §3.2–§3.7 and live in a session scratchpad that will not
survive: `policy_trunk_health.py`, `why_dead.py`, `init_lottery.py`, `race_probe.py`,
`verb_gate_probe.py`, `bolster_value_probe.py`, `defence_probe.py`, `positional_probe.py`.
Each is ~60 lines; the numbers above are the record unless they are landed under `src/app/`.
