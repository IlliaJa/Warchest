# Reward Design

See [experiments.md](experiments.md) for the full training run history.

This doc is organised in four parts:

1. **[Current rewards](#1-current-rewards)** — what the agent is actually paid, and why.
2. **[Rejected ideas](#2-rejected-ideas)** — reward terms considered and deliberately not used.
3. **[Unrealized ideas](#3-unrealized-ideas)** — proposed but not yet implemented (currently one).
4. **[What is discussed](#4-what-is-discussed)** — how comparable strong game-playing agents design their reward signal; the cross-project literature grounding behind the choices above.

---

## 1. Current rewards

### Reward table

| Event | Reward |
|---|---|
| Win (6 bases controlled) | +1.0 |
| Claim base | 0.0 |
| Attack | +0.02 |
| Truncation — base lead | 0.0 |
| Truncation — tied bases | −0.5 |
| Truncation — base deficit | −1.0 (proportional, see below) |
| Invalid action attempt | −0.02 |
| Any move | −0.002 |

Constants are defined at the top of `src/services/environment/warchest_env.py`. Raw environment rewards are augmented with potential-based shaping (base-diff **and** material), a holding reward, and per-run **annealing** of the holding + material terms in `src/app/ppo.py` before being stored in the rollout buffer. Every move costs the same step penalty regardless of destination — directional pull toward bases is handled entirely by shaping and the holding reward.

> **Implemented 2026-07-03.** Material PBRS, linear annealing of the holding + material shaping (1.0 → 0.1 over the first half of the run), a base-diff-proportional truncation reward (C17), and a widened **critic-only** trunk (`critic_hidden_dim=128`, policy left at 64) all shipped together. Rationale and cross-project grounding: [decision.md](decision.md) (2026-07-03) and this doc's [What is discussed](#4-what-is-discussed) below. A/B still owed against the pre-change baseline — see [IDEAS.md](IDEAS.md) #3.

### Potential-based reward shaping (base differential)

Applied by the training loop, not the environment:

```python
shaped_r = r + gamma * phi(s_next) - phi(s)
phi(s)   = SHAPING_C * (my_bases - opp_bases)    # SHAPING_C = 0.05
```

A theoretically clean way to add dense rewards without changing the optimal policy (Ng et al. 1999): the `gamma * phi(s') - phi(s)` terms telescope and nearly cancel over a full trajectory, leaving only boundary terms. Fires a positive pulse (~+0.05) when the agent claims a neutral base and (~+0.10) when stealing an enemy base. Fires a negative pulse when the opponent claims. Advantages are z-scored batch-wide; returns are kept in the original reward scale. `SHAPING_C` is **not** annealed (only the holding + material terms are — see below).

### Material potential-based shaping (coin economy)

Applied by the training loop alongside base shaping:

```python
phi_material = C_MAT * (boxed_total(opp) - boxed_total(me))   # C_MAT = 0.015
shaped_r    += shaping_anneal * (gamma * phi_material(s') - phi_material(s))
```

Base rewards are otherwise base-centric, but `attack` sends an enemy coin **to the box** (`GameState.boxed`, permanent), and losing coins shrinks the opponent's bag → fewer coins drawn per round → fewer actions → a compounding tempo/material spiral. Without this term the agent is paid for that **only** if the attack eventually converts to a base — a very delayed, sparse signal across the ~150-turn midgame. This term rewards the coin-economy axis directly.

**Why this term exists — the measured gap.** A 200-game bucketed eval (`src/app/eval_bucketed.py`, checkpoint `warchest_ppo_20260702-1442.pth`) found the coin economy carried **zero** reward gradient before this term: the policy bolstered in 1 of 200 games and never triggered a Berserker stack chain (stack never reached 2 in the 50 games it was drafted), despite winning ~89% of games — a real skill gap, not draft variance. Tactics were used in only 11.5% of games and correlated with *losing* more (reverse-causation vs. poor execution is still undistinguished — see `docs/IDEAS.md`). Cross-project grounding for why a *small potential-based* term (not a raw bonus) is the right fix: this doc's [What is discussed](#4-what-is-discussed) below.

`WarChestEnv.boxed_total(pid) = sum(state.boxed[pid].values())` — coins only ever leave play into the box, so a player's live-coin count is `owned_total - boxed_total`; the owned offset is fixed per composition and cancels in the telescoping, leaving the boxed differential as the live term. It is keyed by **absolute** player id (no perspective flip needed, unlike base shaping).

Design rationale & caveats:

- **Fires** a positive pulse when the agent boxes an enemy coin (attack), a negative pulse when it loses one — exactly the mechanic base shaping never touches.
- **Coefficient.** `C_MAT` is kept well below `SHAPING_C = 0.05` (bases *win* the game; material is a means). A single killed coin is worth a fraction of a base swing.
- **Telescoping / measurement point.** `phi_material` is measured at the same points as base shaping (before / immediately after the main actor's coin play), i.e. the same convention already used for `SHAPING_C`. The original proposal called for measuring `phi` *after the opponent's move*; that is **not** what the current base shaping does, so the material term matches base shaping for consistency. If an A/B shows this leaks, the "after opponent's move" fix must be applied to **both** terms together.
- **Exploit safety.** `boxed` is monotonic per player, but PBRS is policy-invariant for *any* state function, so there is no farming loop (unlike a raw per-kill bonus, which the circular-claiming precedent warns against). Uses only `GameState.boxed` — no BFS, negligible cost.
- **Over-shaping caveat.** The `ppo_20260630-060400` run analysis found the agent already *over-optimizes* dense shaping relative to winning (score climbed while win rate stayed flat — see `docs/history.md`). This term is still worth it because (a) it is proper PBRS, unlike the non-telescoping holding reward that is the likelier decoupling culprit; (b) material advantage is **strongly win-correlated**, so it points the dense signal at something that predicts wins. This is the main reason it is annealed (see below) and A/B'd, not stacked blind.

> **Note — `ATTACK_REWARD` was intentionally kept, not subsumed.** Folding the raw `ATTACK_REWARD = 0.02` into this term is the recommended next move — it fires on the same box-a-coin event and is non-telescoping. That swap was **not** made in this pass to keep the change set to exactly what was requested; both currently fire. Flagged for the A/B (see `IDEAS.md` #3): zero `ATTACK_REWARD` and re-measure so attacks aren't double-paid.

### Shaping annealing

The holding reward and the material shaping term are multiplied by `shaping_anneal`, linearly decayed **1.0 → 0.1 over the first half of the run**, then held at the 0.1 floor:

```python
half = max(n_batches * 0.5, 1.0)          # tracks n_batches: 200 @400, 250 @500
anneal_frac = min((batch_num - 1) / half, 1.0)
shaping_anneal = 1.0 + anneal_frac * (0.1 - 1.0)
```

Base-diff PBRS (`SHAPING_C`) is deliberately left constant — it's already policy-invariant (proper PBRS), so it has nothing to anneal away. The holding reward and material term are annealed because dense guidance is most valuable *early* (bootstraps exploration while the critic is weak and entropy is high) and becomes a source of distortion *late*, once the policy is capable enough that the proxy signal and the true win objective can diverge (the measured symptom on `ppo_20260630-060400`: shaped `score` rose +50% over ~500 batches while `wr_greedy` stayed flat — see `docs/history.md`). This mirrors OpenAI Five's `team_spirit` coefficient, annealed 0→1 over training to shift from easy-to-learn selfish rewards toward the true team objective; Warchest anneals the *dense* terms down toward the *terminal* one instead, same idea in reverse direction. Cross-project grounding: this doc's [What is discussed](#4-what-is-discussed) below (*The synthesis for Warchest*). Logged per batch as `shaping_anneal`; the annealed holding/material contributions are logged as `score_holding` / `score_material`.

### Per-turn holding reward

Applied by the training loop alongside shaping:

```python
holding_reward = holding_reward_rate * (my_bases - opp_bases)
holding_reward_rate = WIN_REWARD / ((winning_base_count - 1) * (max_rounds * HAND_SIZE)) * 0.8
                    = 1.0 / (5 * (50 * 3)) * 0.8 = 1.0 / 750 * 0.8 ≈ 0.001067
```

Fires every agent turn proportional to the current base lead. Incentivises defending claimed bases and breaks the base-flip exploit (two policies claiming the same bases in a loop), where potential shaping alone produces near-zero net reward per cycle. The 0.8 factor is a safety margin ensuring worst-case accumulated holding (`0.001067 * 5 * 150 = 0.8`) never exceeds `WIN_REWARD = 1.0`. `max_rounds * HAND_SIZE` is used as the worst-case bound on main-actor turns per episode. `holding_reward_rate` is derived from env constants at script startup so it stays valid if `max_rounds`, `HAND_SIZE`, or `winning_base_count` change.

Shaping and holding are complementary: shaping delivers an immediate, one-step credit signal at the claim action; holding creates persistent per-turn pressure that requires no bootstrapping. The holding reward is **not** potential-based (it does not telescope), so unlike `SHAPING_C` it genuinely *changes* the optimal policy — rewarding grabbing an early lead and stalling rather than closing the game — which is the likeliest single contributor to the `ppo_20260630-060400` decoupling above. Two fixes were weighed: **(a) remove it outright** and lean on base-diff PBRS + critic bootstrapping (the AlphaZero/TD-Gammon stance), risking the base-flip exploit it was added to kill in the first place; or **(b) anneal its coefficient toward a small floor** (the OpenAI Five stance) — keep the early-training tie-break benefit, remove the late-training distortion, and stay reversible/cheap to A/B against removal later. **(b) was chosen** as the lower-risk first try.

### Truncation reward

When the episode truncates (`round_number >= max_rounds = 50`), a terminal reward is added to the last main-actor step based on the base lead. **Updated 2026-07-03 (C17)** from a 0 / −0.5 / −1.0 step function to a base-diff-*proportional* value, which lowers critic-target variance at the truncation states the agent spends most of its time near while preserving the old anchor values (0 for a winning draw, −0.5 for a true draw, −1.0 for a full-deficit rout):

```python
if diff > 0:                    # drew from a strong position
    trunc_reward = 0.0
else:                           # tie or deficit
    deficit_frac = min(-diff, winning_base_count) / winning_base_count  # 0 at a tie ... 1 at max deficit
    trunc_reward = LOSS_REWARD * (0.5 + 0.5 * deficit_frac)             # -0.5 (tie) ... -1.0 (rout)
```

### Implementation notes

- Every move returns `MOVE_NEG_REWARD_PER_TURN = -0.002` regardless of destination. The binary approach rewards (`MOVE_ON_BASE_REWARD`, `MOVE_NEAR_BASE_REWARD`) were removed: they created a bias toward neutral bases over enemy bases (which are strategically more valuable), and the holding reward + shaping now provide all directional pull needed.
- `ATTACK_REWARD = 0.02` is paid on every successful attack (kept small so a game's worth of attacks cannot rival `WIN_REWARD = 1.0`). Tracked separately in the training loop as `r_attack` / `score_attack` for logging, but it is a plain per-action reward, not shaping. (See the material-shaping note above: flagged for zeroing now that material PBRS pays the same event.)
- An exploration bonus (`MOVE_EXPLORE_REWARD_MAX_TURN = 5`, `MOVE_EXPLORE_REWARD_PER_TURN = 0.1`) is defined as constants but is not wired into `perform_move_action`. The exploration map is updated every step for use in the observation, but no reward is computed from it.
- `CLAIM_BASE_REWARD` is 0.0. A non-zero direct claim reward caused a circular-claiming exploit: the policy learned to claim bases back and forth with pool opponents, accumulating reward far in excess of `WIN_REWARD = 1.0`. Potential shaping and the holding reward handle base value correctly without this risk.

---

## 2. Rejected ideas

Reward terms that were considered and deliberately **not** used. (The base-lead bonus, base-differential PBRS, and coin/material PBRS were *accepted* and are documented in [Current rewards](#1-current-rewards).)

### Opponent-loss penalty (symmetric claiming) — *ruled out*

**Ruled out**: the holding reward already penalises losing a base (reduces per-turn income). An explicit penalty at the opponent's action step would create attribution problems since the defender had no direct control over that timestep.

### Distance-to-objective shaping — *not worth it*

```python
delta_dist = prev_min_dist_to_target - new_min_dist_to_target
proximity_reward = k * delta_dist
```

**Not worth it**: requires BFS on the hex grid every agent step (non-trivial cost on CPU). More importantly, `MOVE_ON_BASE_REWARD` and `MOVE_NEAR_BASE_REWARD` already provided a coarser version of this signal, and the holding reward creates pull toward bases via future value. The oscillation exploit risk (agent bouncing near a base to farm the reward) is real. Better to improve the critic so it bootstraps the holding reward correctly.

### Graduated distance reward — *superseded*

Was a proposed replacement for the binary `MOVE_ON_BASE_REWARD` / `MOVE_NEAR_BASE_REWARD`. Both the binary approach and this graduated version were removed in favour of letting the holding reward and shaping handle all directional pull. Avoids the neutral-vs-enemy base bias entirely.

### Re-enable exploration reward — *not worth it*

```python
explore_reward = max(0, MOVE_EXPLORE_REWARD_PER_TURN * (MOVE_EXPLORE_REWARD_MAX_TURN - visit_count[end]))
```

**Not worth it**: the board is small (7×7 hex, ~37 valid cells) and the policy needs to focus on objectives, not coverage. With only 2 units per player, coverage is not a bottleneck. An exploration bonus would distract from the objective-focused rewards already in place.

### Win-speed bonus — *not needed*

```python
win_reward = WIN_REWARD + speed_bonus * (max_turns - turn_count) / max_turns
```

**Not needed**: γ=0.99 discounting already makes winning sooner better than accumulating future holding rewards. The step penalty (`-0.002` per move) adds further pressure. Verified: winning at turn 70 gives `1.0` vs holding 65 more agent turns then winning gives `≈0.83` net present value.

### Base-capture sequence reward — *not worth it*

```python
MOVE_ADJACENT_TO_UNCLAIMED = +0.002
MOVE_ONTO_UNCLAIMED        = +0.008
CLAIM_UNCLAIMED            = +0.15
```

**Not worth it**: restoring a non-zero `CLAIM_BASE_REWARD` risks the circular-claiming exploit. The potential shaping and holding reward already incentivise the full capture sequence without the exploit risk.

---

## 3. Unrealized ideas

Proposed but not yet implemented.

### Unit / board-presence potential *(unit economy)*

A softer companion to the material term: reward having units **deployed and alive on the board** (you can't claim or hold a base with an empty board, and a dead unit's coins sit uselessly in the bag until re-deployed).

```python
phi_units = C_UNIT * (units_on_board(me) - units_on_board(opp))   # C_UNIT small
shaped_r += gamma * phi_units(s_next) - phi_units(s)
```

- **Rationale.** Encourages board tempo — deploying to contest locations and denying the opponent presence — which base shaping only rewards *after* a claim lands.
- **Overlap with material shaping.** Killing an enemy unit raises both `phi_material` and `phi_units`; deploying spends a coin (neutral to material until it dies) but raises presence. To avoid double-paying the same kill, keep the **material term as primary** and treat this as an optional low-coefficient add-on, or use *total on-board stack height* instead of unit *count* so it reads as "committed board strength" rather than "number of bodies."
- **Over-deploy risk.** A presence bonus can push wasteful deploys (dumping coins onto the board just for the count, leaving the bag thin). PBRS keeps the *optimal* policy unchanged in theory, but the coefficient must stay small and this should be A/B'd for exactly this failure mode. Consider gating on *controlled/contested* cells rather than any cell.
- **Cheaper alternative.** If it doesn't pay off, the deploy/recruit decisions may be better shaped by the *material* term alone plus letting the critic bootstrap board value.

---

## 4. What is discussed

Cross-project survey of how comparable strong game-playing agents design their reward
signal — deliberately focused on **imperfect-information** games (Stratego, card games) that
share Warchest's hidden-bag structure. This is the literature grounding behind the choices in
[Current rewards](#1-current-rewards) above (material PBRS, annealed shaping); remaining
action items derived from it live in `docs/IDEAS.md`.

The single most useful axis for sorting these projects is **"does the learner have search (or
search-equivalent value propagation), or not?"** It predicts almost perfectly whether they used
reward shaping.

### Terminal-reward-only, relies on search / value bootstrapping

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
  (bootstrapping). Note the reward is **not purely terminal**: per
  secondary write-ups (deeplearning.ai) it also received a **small reward for
  capturing an opponent piece** plus the larger win reward — i.e. a light
  material-shaping term on top of the outcome. The "reward transformation" in
  R-NaD (Regularized Nash Dynamics) is a *separate* mechanism — a **regularizer**
  (a KL-style pull toward a previous policy for convergence to Nash), not a
  hand-crafted intermediate reward. Lesson for Warchest: even the RL flagship in
  a hidden-material game leaned mostly on outcome + a *small* material term —
  which is exactly the shape of the material-PBRS term above, and a datapoint
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

### Heavy dense shaping — and the one that looks like Warchest

- **OpenAI Five (Dota 2).** PPO, actor-critic, **no search** — methodologically
  the closest match to Warchest. And it is exactly the project that used
  **heavy hand-crafted dense shaping**: net worth, XP, gold, kills/deaths/
  assists, last-hits, *staying alive* — "rewards for things that happen often,"
  explicitly to solve credit assignment over long games. It also introduced the
  **`team_spirit`** coefficient, **annealed 0 → 1** over training, to shift from
  selfish (easy-to-learn) rewards toward the true team objective.

### The synthesis for Warchest

The dividing line is search/bootstrapping capacity:

> Agents that can **propagate a sparse terminal reward** (via MCTS, TD(λ),
> R-NaD dynamics, or sheer value-net scale) use **terminal reward only** and
> refuse to shape. The one strong agent that is **PPO + actor-critic + no
> search** — OpenAI Five — is precisely the one that leaned on **dense shaping**.

Warchest is in the OpenAI Five bucket: PPO, actor-critic, no search, a critic that must carry
~150-turn credit assignment on its own (widened to `critic_hidden_dim=128` for exactly this
reason — see [Current rewards](#1-current-rewards)). **So the case for dense shaping is
genuinely stronger here than the AlphaZero/DeepNash precedent suggests** — those agents had
machinery Warchest doesn't.

But three constraints come straight out of the same literature, and are why the reward table
above looks the way it does:

1. **Shape along the axis that's missing, not the one that's saturated.**
   OpenAI Five's shaped terms (net worth, last-hits) targeted the economy —
   exactly the mechanic that only pays off much later. Warchest's analogue is
   the coin/material economy (why material PBRS exists). Do **not** add more base-axis reward.
2. **Prefer variance-neutral shaping.** DouZero's removed-bonus result and the
   Ng et al. (1999) policy-invariance theorem both point the same way:
   **potential-based reward shaping (PBRS)** telescopes over a trajectory, so
   it densifies the *per-step* signal without changing the optimal policy and
   without inflating return variance the way raw per-event bonuses do. This is
   why both the base-diff and material terms are PBRS, not raw bonuses (and why
   `ATTACK_REWARD`, a raw bonus, is flagged for removal above).
3. **Anneal shaping toward the true objective.** OpenAI Five's annealed
   `team_spirit` is the template: use the dense signal to bootstrap learning
   early, then decay its influence so the *final* policy optimizes the terminal
   outcome — why the holding + material terms are annealed above.

### Sources

- [Mastering the Game of Stratego with Model-Free Multiagent RL (DeepNash) — arXiv](https://arxiv.org/abs/2206.15378) · [DeepMind blog](https://deepmind.google/blog/mastering-stratego-the-classic-game-of-imperfect-information/)
- [Dota 2 with Large Scale Deep Reinforcement Learning (OpenAI Five) — arXiv](https://arxiv.org/pdf/1912.06680) · [OpenAI Five](https://openai.com/index/openai-five/)
- [AlphaZero — reward = game outcome only; MCTS/value bootstrapping](https://cacm.acm.org/research/reimagining-chess-with-alphazero/)
- [TD-Gammon — Wikipedia](https://en.wikipedia.org/wiki/TD-Gammon) · [Tesauro, Temporal Difference Learning and TD-Gammon](https://www.csd.uwo.ca/~xling/cs346a/extra/tdgammon.pdf)
- [DouZero: Mastering DouDizhu with Self-Play Deep RL — arXiv](https://arxiv.org/abs/2106.06135) (terminal reward; removed spring/bidding bonuses to cut variance)
- [JP-DouZero: intrinsic rewards for peasant collaboration — IEEE](https://ieeexplore.ieee.org/document/10415578)
- Ng, Harada, Russell (1999), *Policy invariance under reward transformations* — the PBRS theorem the existing base/material shaping relies on.
