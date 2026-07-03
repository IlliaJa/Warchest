# Reward Improvements — cross-project analysis and a proposed plan

Companion to [rewards.md](rewards.md) (current reward table + §9/§10 proposals),
[future_steps.md](future_steps.md) (the live sequential plan), and
[IDEAS.md](IDEAS.md). This doc does the one thing those don't: it looks at how
**other strong game-playing agents** design their reward signal — with a
deliberate focus on **imperfect-information** games (Stratego, card games) that
share Warchest's hidden-bag structure — and turns that into a concrete,
prioritized plan.

> **Filename note:** the request asked for `rewards_imporvements.md`; this is
> the same doc under the corrected spelling `rewards_improvements.md`.

---

## 1. The problem, restated precisely

The intuition in the brief — *"most rewards are deferred, especially the win;
a faster reward signal is critically missing"* — is directionally correct, but
the project's own measurements sharpen it into something more specific than
"add denser reward." From `future_steps.md` Step 0 (200-game bucketed eval on
`warchest_ppo_20260702-1442.pth`):

- WR vs greedy is **0.89 ± 0.02**, and losses are **roughly uniform across
  compositions** — this is a *skill* gap, not a draft-variance ceiling.
- The policy **essentially never bolsters** (1 bolster in 200 games) and
  **never triggers a stack chain** (Berserker stack never reached 2 in the 50
  games it was drafted). Tactics are used in 11.5% of games and *correlate with
  losing*.

So the gap is not "the agent can't tell it's winning." The base signal already
works (holding reward + base-diff shaping). The gap is that **an entire skill
axis — the coin/material/tempo economy, and the survival-only actions that feed
it (bolster, stack chains) — carries no reward gradient at all.** The win is
the only thing that ever pays for good economy play, and it pays ~150 turns
later, filtered through a noisy critic. That is the "deferred reward" the brief
is pointing at, and it is real — but the fix is *targeted densification of the
missing axis*, not more reward on the base axis the agent has already
over-learned (see the over-shaping diagnosis referenced throughout
`future_steps.md`).

Two independent failure modes are stacked here, and they need different tools:

| Failure | Symptom | Right tool |
|---|---|---|
| **No gradient on the economy axis** | never bolsters, never chains, ignores material | dense reward *shaped along that axis* |
| **Over-optimizing the base axis** | plateau, high entropy, score/win decoupling | *remove distortion*, don't add more base reward |

The rest of this doc is about picking the right tool for each, informed by what
worked elsewhere.

---

## 2. How comparable agents design their reward

The single most useful axis for sorting these projects is **"does the learner
have search (or search-equivalent value propagation), or not?"** It predicts
almost perfectly whether they used reward shaping.

### 2a. Terminal-reward-only, relies on search / value bootstrapping

- **AlphaZero / MuZero (chess, shogi, Go).** Reward is *only* the game outcome:
  `z = +1 / 0 / −1`. **No reward shaping, no material heuristic** — the value
  head is trained by regression toward `z`, and MCTS propagates that sparse
  signal into a dense per-move training target (the search policy π). The dense
  signal is manufactured by *search*, not by reward engineering.
- **TD-Gammon (backgammon).** Self-play, no search (in its core version), a
  value net trained with **TD(λ)**. Reward fires only at game end (win
  probability). The densification comes entirely from **bootstrapping**: TD(λ)
  smears the terminal signal back across the trajectory. This is the historical
  proof that a *value function + eligibility traces* can learn expert play from
  a purely terminal reward — no shaping.
- **DeepNash (Stratego).** The most relevant comparison: **imperfect
  information, hidden material, no search at play time**, model-free. The
  dominant signal is the **terminal game outcome**, propagated by a value head
  (bootstrapping — see §5). Note the reward is **not purely terminal**: per
  secondary write-ups (deeplearning.ai) it also received a **small reward for
  capturing an opponent piece** plus the larger win reward — i.e. a light
  material-shaping term on top of the outcome. The "reward transformation" in
  R-NaD (Regularized Nash Dynamics) is a *separate* mechanism — a **regularizer**
  (a KL-style pull toward a previous policy for convergence to Nash), not a
  hand-crafted intermediate reward. Lesson for Warchest: even the RL flagship in
  a hidden-material game leaned mostly on outcome + a *small* material term —
  which is exactly the shape of the §9 material-PBRS proposal, and a datapoint
  *for* it rather than against.
- **AlphaStar (StarCraft II).** Imperfect information, no search. The true
  reward is **win/loss**. It used built-in-score "pseudo-rewards" and
  statistics `z` mainly to **drive exploration and strategic diversity**, plus
  human-data bootstrapping — not as the primary objective. Terminal outcome
  stays the objective.
- **DouZero (DouDizhu).** Imperfect-information card game with a **large,
  turn-varying action space** — structurally close to Warchest's factored
  action head. Deep Monte-Carlo, self-play, reward is **terminal win/loss**.
  Crucially, they **deliberately removed** the "spring" bonus and the bidding
  phase because those extra reward components **increase return variance and
  hurt convergence**. This is the clearest published warning that *adding*
  reward terms is not free — variance can cost you more than the signal buys.

### 2b. Heavy dense shaping — and the one that looks like Warchest

- **OpenAI Five (Dota 2).** PPO, actor-critic, **no search** — methodologically
  the closest match to Warchest. And it is exactly the project that used
  **heavy hand-crafted dense shaping**: net worth, XP, gold, kills/deaths/
  assists, last-hits, *staying alive* — "rewards for things that happen often,"
  explicitly to solve credit assignment over long games. It also introduced the
  **`team_spirit`** coefficient, **annealed 0 → 1** over training, to shift from
  selfish (easy-to-learn) rewards toward the true team objective.

### 2c. The synthesis for Warchest

The dividing line is search/bootstrapping capacity:

> Agents that can **propagate a sparse terminal reward** (via MCTS, TD(λ),
> R-NaD dynamics, or sheer value-net scale) use **terminal reward only** and
> refuse to shape. The one strong agent that is **PPO + actor-critic + no
> search** — OpenAI Five — is precisely the one that leaned on **dense shaping**.

Warchest is in the OpenAI Five bucket: PPO, actor-critic, no search, a critic
(`hidden_dim=64`) that must carry ~150-turn credit assignment on its own. **So
the case for dense shaping is genuinely stronger here than the AlphaZero/
DeepNash precedent suggests** — those agents had machinery Warchest doesn't.
The brief's instinct is sound.

But three constraints come straight out of the same literature:

1. **Shape along the axis that's missing, not the one that's saturated.**
   OpenAI Five's shaped terms (net worth, last-hits) targeted the economy —
   exactly the mechanic that only pays off much later. Warchest's analogue is
   the coin/material economy. Do **not** add more base-axis reward.
2. **Prefer variance-neutral shaping.** DouZero's removed-bonus result and the
   Ng et al. (1999) policy-invariance theorem both point the same way:
   **potential-based reward shaping (PBRS)** telescopes over a trajectory, so
   it densifies the *per-step* signal without changing the optimal policy and
   without inflating return variance the way raw per-event bonuses do. This is
   the right default, and it's what `rewards.md` §9/§10 already propose.
3. **Anneal shaping toward the true objective.** OpenAI Five's annealed
   `team_spirit` is the template: use the dense signal to bootstrap learning
   early, then decay its influence so the *final* policy optimizes the terminal
   outcome. This is the direct antidote to Warchest's measured over-shaping /
   score-win decoupling.

---

## 3. Proposed solution (prioritized, one A/B at a time)

This refines — it does not replace — the sequence in `future_steps.md`. Each
step obeys the **standing A/B rule** at the bottom of that doc: same
`n_batches` both sides, correct baseline log, compare *distributions over the
settled phase*, never endpoints.

> **Implementation status (2026-07-03).** Steps 1 (holding **annealed**, not
> removed), 2 (material PBRS, `C_MAT=0.015`), 5 (critic-only widening,
> `critic_hidden_dim=128`) and the C17 truncation smoothing are **implemented**
> in `ppo.py` / `warchest_env.py`; see `rewards.md` and `decision.md`
> (2026-07-03). `ATTACK_REWARD` was **kept** (not subsumed — Step 2's refinement
> deferred to the A/B). The controlled A/Bs are still owed. GAE-λ sweep and
> policy-width increase remain not started.

### Step 1 — Reward hygiene *before* adding anything (highest priority)

The literature's strongest consensus is "don't distort the objective." Warchest
has one clear distortion today:

- **`holding_reward` is not potential-based.** It fires every turn proportional
  to the base lead and does **not** telescope, so unlike the base-diff PBRS term
  it **changes the optimal policy** — it rewards *grabbing an early lead and
  stalling* rather than closing the game (`future_steps.md` Step 3). This is
  very likely a bigger contributor to the plateau/decoupling than any missing
  term. Two options, A/B'd:
  - **(a) Remove it** and lean on the base-diff PBRS + critic bootstrapping
    (the AlphaZero/TD-Gammon stance). Risk: reintroduces the base-flip exploit
    it was added to kill — watch for it.
  - **(b) Anneal its coefficient to ~0** over training (the OpenAI Five stance):
    keep the early-learning benefit, remove the late-training distortion.
    Recommended first try, because it's reversible and directly tests the
    "distortion, not signal" hypothesis.

Do this first so that any gain from Step 2 is attributable to Step 2, not to
having accidentally also fixed the holding term.

### Step 2 — Material PBRS on the coin economy (the missing axis)

Implement `rewards.md` §9 — the term that densifies the axis Step 0 proved is
dark:

```python
phi_material = C_MAT * (boxed_total(opp) - boxed_total(me))   # C_MAT ≈ 0.01–0.02
shaped_r    += gamma * phi_material(s_next) - phi_material(s)
```

Refinements beyond what §9 already says:

- **Subsume `ATTACK_REWARD` into this term — don't stack them.** `ATTACK_REWARD
  = 0.02` is a **raw, non-telescoping** per-attack bonus: it is exactly the
  farmable, variance-adding kind of reward the DouZero result warns against, and
  it fires on the *same event* (boxing an enemy coin) that `phi_material` would.
  Keeping both **double-pays** attacks and reintroduces a non-invariant term
  next to an invariant one. Recommended: **replace** `ATTACK_REWARD` with the
  material PBRS term. The PBRS version is strictly better — policy-invariant,
  symmetric on defense (losing a coin is penalized automatically), and
  win-correlated. A/B `material-PBRS-only` vs the current `ATTACK_REWARD-only`
  baseline so the swap is measured.
- **Measure `phi` after the opponent's move**, matching the existing base-diff
  shaping (`ppo.py:288-291`). The telescoping guarantee breaks otherwise — any
  new potential term must respect the same "evaluate from the main actor's
  perspective after the turn flips" invariant already coded there.
- **Track bolster/stack-chain rate as a first-class A/B metric, not just WR.**
  Step 0 warns material PBRS is a *plausible partial* fix: for an offensive
  chain unit (Berserker) the link stack→hits→`boxed(opp)` is direct, but for a
  purely defensive bolster the boxed-coin count can be identical whether you
  bolster or not. `eval_bucketed.py` already emits `bolster_count`,
  `chain_offered`, `chain_used` — gate success on those moving, not only on
  aggregate WR.

### Step 3 — Anneal the shaping coefficients (over-shaping antidote)

Add a **linear (or cosine) decay** on the dense-shaping coefficients
(`SHAPING_C`, `C_MAT`, and — if kept — the holding rate) from their start value
toward a small fraction (e.g. 0.2×) over the run. Rationale, directly from
OpenAI Five's `team_spirit` schedule: dense reward is most useful *early* (it
bootstraps exploration into the right region); late in training you want the
policy optimizing the terminal objective, not the proxy. PBRS is
policy-invariant in the infinite-data limit, but under finite PPO training the
proxy still visibly distorts behavior — annealing is the practical fix that
keeps the early benefit. Cheap to implement; A/B against constant coefficients.

### Step 4 — Optional: unit/board-presence PBRS (§10), low coefficient

Only if Steps 2–3 leave a measurable tempo gap. Use **stack height** ("committed
board strength"), not unit *count*, to avoid rewarding thin over-deploys; keep
`C_UNIT` well below `C_MAT`; watch the over-deploy failure mode explicitly.
Treat as a companion to §9, never a replacement.

### Step 5 — Strengthen the densifier itself, not just the reward

The AlphaZero/TD-Gammon lesson is that **the value function is the real
densifier** — it's what turns a terminal reward into a per-step signal. If the
critic is under-capacity, no amount of shaping fixes credit assignment; shaping
just papers over a weak critic. Two cheap moves, both already flagged:

- **`hidden_dim` 64 → 128** (`future_steps.md` Step 4) — controlled A/B, done
  last so the gain is attributable to capacity.
- **Smooth the truncation reward** (IDEAS.md C17): the current 0 / −0.5 / −1.0
  **step function** is a high-variance critic target at exactly the states the
  agent spends most of its time in. A base-diff-*proportional* terminal value
  (e.g. `LOSS_REWARD * (opp_bases − my_bases) / winning_base_count`, clipped)
  lowers critic target variance for free.

### What stays out of the reward (deliberately)

- **Draw-probability / bag-dilution features (`p_soon`, `p_mean`).** The brief
  raised this; the project already reasoned it correctly and the research backs
  that call: this belongs in the **observation, not the reward**. A fixed
  potential cannot encode "reliably draw *the unit I've chosen to play next*"
  because the target is **policy-defined** — any static choice (Herfindahl →
  monotype; fielded-average → misses 4-of-each) is wrong for some valid
  strategy, i.e. a **mis-specified proxy** of exactly the kind DouZero's
  variance result and the over-shaping diagnosis warn against. Ship it as an
  observation feature bundled with the next `OBS_VERSION` bump, not as reward.
- **Win-speed bonus, distance-to-objective, exploration bonus, raw claim
  reward, sequence rewards.** Already correctly ruled out in `rewards.md` §4–8;
  the cross-project view doesn't change those calls (γ-discounting handles
  speed; raw per-event bonuses are the variance/exploit trap).

---

## 4. Other improvement points noticed during this analysis

These are outside the strict "reward table" scope but surfaced while tracing the
reward path; kept separate as requested.

1. **`ATTACK_REWARD` is a latent exploit and a variance source (independent of
   Step 2).** Even if material PBRS is *not* adopted, a raw non-telescoping
   per-attack bonus is farmable in principle (any attack loop that boxes coins
   without progressing toward a win) and inflates return variance. It survived
   only because attacks are hard to farm in practice — but it's the same class
   of term as the removed `CLAIM_BASE_REWARD`. Convert to PBRS or justify
   explicitly.

2. **Reward sparsity is fundamentally a *credit-assignment* problem — tune the
   propagation, not only the reward.** Before (or alongside) adding terms, it's
   worth a **GAE λ sweep**. λ closer to 1 propagates the terminal reward further
   back with less bias (the TD-Gammon eligibility-trace lesson), which is a
   *reward-neutral* way to densify the effective per-step signal. Cheap A/B, and
   it doesn't risk distorting the objective at all.

3. **Return-scale interaction when adding terms.** Advantages are z-scored but
   returns are kept in original scale and fed through a return normalizer
   (`ppo.py`). Adding `phi_material` / annealing coefficients shifts the return
   distribution; re-verify the normalizer's running stats settle and the critic
   loss scale stays sane after each reward change, or a "no-effect" A/B might
   actually be a normalization artifact.

4. **The tactic-underuse finding may be an *exploration* problem, not a reward
   one.** Step 0 shows entropy annealed to a near-zero floor
   (`entropy_coeff_final=0.003`) before random exploration ever surfaced a case
   where bolstering paid off — the action was dropped from the repertoire before
   any reward could reinforce it. Material PBRS helps only if the behavior is
   *sometimes sampled*. Consider: (a) a higher/again-annealed entropy floor, or
   (b) a small, decaying **intrinsic/count-based bonus on rarely-used verbs**
   (BOLSTER, TACTIC) — the approach JP-DouZero used to break analogous
   collaboration-behavior gaps. Keep it decaying and tiny; the board is small,
   so this is about *action* coverage, not *spatial* coverage.

5. **`value_single` runs under `.train()` at rollout (IDEAS.md C18).** Harmless
   today, but a silent bug the moment BatchNorm/Dropout is added to the critic
   — relevant because Step 5 proposes touching critic capacity. Fix the
   `eval()`/`train()` bracketing pre-emptively if you widen the critic.

6. **Reverse-causation on tactics is still unresolved and blocks interpretation.**
   "Tactics correlate with losing" (Step 0) has two readings — reached-for-when-
   behind vs. executed-poorly — and they imply *opposite* reward responses. A
   cheap disambiguator: log tactic usage **conditioned on base-lead at time of
   use**. If tactics cluster in already-behind states, it's reverse causation
   and no reward change is warranted; if they're spread across lead states,
   it's execution and Step 2/4 are on target. Run this before committing to
   Step 4.

---

## 5. One-paragraph recommendation

The brief's instinct is right *for this architecture specifically*: the
search-based agents (AlphaZero) and the massive-scale search-free ones
(DeepNash, which used mostly outcome + a *small* capture reward) can lean on a
near-terminal signal because search and/or scale + a strong value head do the
densifying. Warchest has neither the search nor that scale — it's a compute-
limited search-free PPO agent, the same class as OpenAI Five, the one strong
system that needed dense shaping. But the measured problem
is **not** "too little reward"; it's (i) one **distortionary** dense term
(`holding_reward`, non-PBRS) and (ii) **zero** gradient on the coin/material
economy where the skill actually lives. So: **fix hygiene first (Step 1), add
policy-invariant material PBRS on the dark axis and fold `ATTACK_REWARD` into it
(Step 2), anneal all shaping toward the terminal objective (Step 3), and
strengthen the critic that does the real densifying (Step 5)** — measuring
bolster/chain rate, not just WR, at every A/B. Keep the bag-dilution signal in
the observation, never the reward.

---

## Sources

- [Mastering the Game of Stratego with Model-Free Multiagent RL (DeepNash) — arXiv](https://arxiv.org/abs/2206.15378) · [DeepMind blog](https://deepmind.google/blog/mastering-stratego-the-classic-game-of-imperfect-information/)
- [Dota 2 with Large Scale Deep Reinforcement Learning (OpenAI Five) — arXiv](https://arxiv.org/pdf/1912.06680) · [OpenAI Five](https://openai.com/index/openai-five/)
- [AlphaZero — reward = game outcome only; MCTS/value bootstrapping](https://cacm.acm.org/research/reimagining-chess-with-alphazero/)
- [TD-Gammon — Wikipedia](https://en.wikipedia.org/wiki/TD-Gammon) · [Tesauro, Temporal Difference Learning and TD-Gammon](https://www.csd.uwo.ca/~xling/cs346a/extra/tdgammon.pdf)
- [DouZero: Mastering DouDizhu with Self-Play Deep RL — arXiv](https://arxiv.org/abs/2106.06135) (terminal reward; removed spring/bidding bonuses to cut variance)
- [JP-DouZero: intrinsic rewards for peasant collaboration — IEEE](https://ieeexplore.ieee.org/document/10415578)
- Ng, Harada, Russell (1999), *Policy invariance under reward transformations* — the PBRS theorem the existing base/material shaping relies on.
</content>
</invoke>
