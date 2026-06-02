# RL Algorithm Comparison

Reference for RL algorithm choices in Warchest. Current trainer: **PPO** (`src/app/ppo.py`). The legacy REINFORCE+GAE trainer is kept at `src/app/reinforce.py`.

---

## Current: REINFORCE + GAE

**Generalized Advantage Estimation** is not an algorithm on its own — it is a variance-reduction technique for computing the advantage signal used in the policy gradient update.

Standard REINFORCE estimates the advantage of action `a` in state `s` as the full discounted return minus a baseline:

```
A(s, a) = G_t - V(s)
```

The return `G_t` has high variance because it sums all future rewards including noise far in the future. GAE smooths this with a weighted average of k-step TD errors:

```
A_GAE = sum over t of (gamma * lambda)^t * delta_t
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

The `lambda` parameter interpolates between:
- `lambda=0`: one-step TD (low variance, high bias — critic error dominates)
- `lambda=1`: full Monte Carlo return (unbiased, but high variance)

**Benefits for Warchest:**
- Simple to implement and reason about
- GAE significantly reduces gradient variance compared to raw REINFORCE
- Works naturally with the current episode-at-a-time rollout structure

**Drawbacks:**
- REINFORCE is strictly on-policy: the policy gradient theorem requires trajectories sampled from the current policy, so each episode can only be used for one gradient update then discarded. This is a constraint of the update rule, not of GAE — GAE is just an advantage estimator and is also used inside PPO with importance sampling corrections.
- High variance even with GAE when episodes are long (~100 turns) and sparse rewards
- Self-play produces contradictory gradients without careful batching (→ idea #8)

---

## PPO (Proximal Policy Optimization)

PPO keeps the on-policy structure of REINFORCE + GAE but reuses each batch of rollouts for multiple gradient steps. It prevents destructive updates with a clipped surrogate objective:

```
L_CLIP = E[ min(r_t * A_t,  clip(r_t, 1-eps, 1+eps) * A_t) ]
r_t = pi(a|s) / pi_old(a|s)   # importance sampling ratio
```

When the new policy drifts too far from the one that collected the data (`r_t` leaves `[1-eps, 1+eps]`), the gradient is clipped to zero, preventing large destructive updates. In practice, `eps=0.2` and 4–10 inner epochs per rollout batch.

**Benefits for Warchest:**
- 4–10× more gradient steps per collected episode — biggest sample efficiency gain available without changing the algorithm family
- Directly compatible with the current actor-critic architecture: same network, same GAE advantage, same masking — only the loss function and the inner loop change
- Clips prevent the instability seen in the current runs (grad norm hitting 1.0 every step)
- Still on-policy in spirit: rollout buffer is discarded after a few epochs, so self-play data stays fresh
- Standard choice for turn-based game AI (OpenAI Five, AlphaStar both used PPO variants)

**Drawbacks:**
- Requires collecting a batch of N episodes before each update (can't update after every step)
- Hyperparameter sensitive: `eps`, inner epochs, and `lambda` all interact
- Still suffers from sparse terminal reward if dense shaping is not added (→ idea #3)

**Migration from current code:** change the loss function and add an inner epoch loop around the existing GAE block. The network, environment, and data collection loop stay the same.

---

## Architecture: shared vs separate actor-critic encoders

PPO with an actor-critic has two natural architectures: share the encoder trunk between the policy and value head, or give each network its own encoder. The literature default — and the AlphaZero-style choice — is to share. `docs/improvement_ideas.md` C10 nominally recommends sharing on those grounds.

**For this codebase, separate encoders are the correct choice.** The current design (`Policy` with its own `board_encoder + unit_encoder + actor_head`, `Critic` with independent copies plus a privileged `opp_onehot` input) was reached after explicitly moving away from a shared trunk. The reasons hold up; here is each, with what would break under a shared trunk.

### 1. Privileged critic input forbids a fully shared encoder

`Critic.value_batch` concatenates a 3-d `opp_onehot` (random / greedy / pool) at the head input. The actor never sees this signal — it must play the same opponent-agnostic policy at deployment. If the encoder were shared, two things go wrong:

- **Information leak.** Even though `opp_onehot` is only injected at the head, the encoder is updated by the critic loss, which depends on `opp_onehot`. The encoder will learn features that are useful for opponent-conditional value prediction, and the actor reads those same features. The actor's distribution becomes implicitly conditioned on an inferred opponent — defeating the whole point of giving the critic privileged information in the first place (the critic was supposed to absorb the opponent-distribution variance the actor cannot see).
- **Architectural mixing of two contracts.** "Actor: only public state → π" and "Critic: public state + opp_id → V" are different functions; their encoders should not be tied.

You could in principle share `board_encoder + unit_encoder` and fork only at the head with `opp_onehot` appended on the critic side. That removes some of the parameter savings of sharing, and the leak above still applies — the shared trunk is still trained by gradients that depend on `opp_onehot`.

### 2. Different learning rates and gradient scales

`lr_actor=1e-4`, `lr_critic=3e-4` (3× ratio). A single optimiser over a shared trunk can only honour one LR for those weights — parameter groups can patch this, but the trunk has to pick one. The historic gradient magnitudes you observed (critic 10–100× actor, pre-clip) tell you what would happen at any single trunk LR: the trunk would be driven almost entirely by critic signal, with actor gradients washed out. The framing of "shared trunk as auxiliary value-prediction task" assumes the two gradients are comparable in scale; when they aren't, the smaller signal effectively trains zero.

The recent flip (actor ≈ 5× critic post-changes) is consistent with the entropy bonus growing into the actor loss (`ENTROPY_COEFF=0.025`, raised from 0.005) and the critic loss tightening (value-loss clipping + return normalisation). Both gradients are clipped to norm 1.0 per `clip_grad_norm_` before stepping, so the downstream optimiser sees the same effective scale either way. The pre-clip ratio is informative about which head currently carries the richer signal — it is not a measure of which head is "winning" a shared trunk, because there is no shared trunk to win.

### 3. Independent KL early-stop is incompatible with a combined loss

`_update_actor` early-stops the inner PPO loop when approx-KL exceeds `KL_TARGET=0.015`; `_update_critic` runs all `ppo_epochs` epochs unconditionally. This is the right asymmetry: the actor's update is bounded by the trust-region argument that legitimises PPO, but the critic's MSE has no equivalent constraint and benefits from extra passes precisely in the regime where the actor wants to stop (returns shifting faster than the value head can track).

With a combined loss `L = L_π + c_v · L_V` and one `.backward()`, you cannot honour both budgets — stopping for actor KL also stops the critic, losing critic updates exactly when they are most useful.

### 4. The single-optimiser footgun

This is the concrete worry from your point 3. With a shared trunk and one `optim.Adam(model.parameters())`, the possible failure modes are:

- **Summed-gradient trunk.** `(L_π + c_v · L_V).backward()` populates `.grad` on trunk weights with the sum of both contributions. The trunk steps in the direction of `grad(L_π) + c_v · grad(L_V)`. When the two disagree — common in non-stationary self-play, where the actor wants to commit and the critic wants more data — the trunk oscillates instead of converging.
- **Two backwards without zero_grad.** Trunk `.grad` accumulates, and now the order of the two backwards (and any earlier residual gradient) determines the step. Easy to get subtly wrong, hard to notice from logs.
- **Two backwards with zero_grad and two steps.** Cleaner, but the second backward propagates through a trunk that has *already moved* relative to the forward that produced its logits. PPO's importance ratio handles small actor-side drift, but the critic has no analogous correction — its TD target is just stale by one trunk update.
- **"Some params" ambiguity.** With one optimiser over a model that contains both actor-side and value-side params, every `optimizer.step()` updates every param that has a non-None gradient. To make `actor_optimizer.step()` only touch actor weights, you must either (a) zero out the value head's grads before backward, (b) detach the value head's forward path, or (c) construct two optimisers over disjoint param lists. (a) and (b) are easy to forget; (c) is what the current code does, and it makes the contract typecheck-level visible.

The current code sidesteps all four by holding two `nn.Module`s with disjoint parameter sets and two `Adam` instances each constructed over its own params. `actor_optimizer.step()` cannot touch critic weights even by accident, and vice versa. Note that `_actor_side_params == list(self._policy.parameters())` in `src/app/ppo.py:139` — `Policy` contains *only* actor-side params, so the explicit name is documentation rather than a filter. That is the design making the partition unambiguous, which is exactly what was missing in the previous shared-trunk version.

### When shared encoders would make sense

Shared trunks are the right call when:

- both heads see the same inputs (no privileged critic info);
- both losses have comparable magnitudes after their coefficients, so neither head drowns the other;
- the actor and critic are updated jointly with one schedule (no asymmetric early-stop);
- you want the cheaper compute / fewer parameters (small models, very large batches);
- AlphaZero-style: the critic target is the MCTS-weighted outcome of the same trajectories the policy is trained on, so gradients align by construction.

If this codebase moves to AlphaZero (C15), a shared trunk becomes natural again — the privileged `opp_onehot` goes away (MCTS handles opponent variance via search), the value target aligns with the policy target, and the asymmetric KL budget disappears (no PPO ratio). Until then, the three structural reasons above are doing real work.

### Bottom line

You do not need shared encoders. The C10 recommendation in `docs/improvement_ideas.md` predates the privileged-critic / separate-KL / separate-LR design and should be treated as obsolete; sharing the trunk now would actively interact badly with all three. The ~30% sample-efficiency claim in that bullet assumes a setup this codebase no longer has.

The only refactor in this direction that is still defensible is sharing **just** `board_encoder + unit_encoder` (not the head, not the `opp_onehot` path) as a pure auxiliary-task experiment, with the critic loss scaled to make its trunk gradient comparable to the actor's. That is a separate, much smaller change than what C10 describes, and the expected upside is marginal compared to fixing the remaining items in `improvement_ideas.md`.

---

## Action head: flat spatial vs factored / autoregressive

The current policy emits a single spatial softmax over an `[A, 7, 7]` head (`A*49` logits). That works for the toy env — Swordsman only, verbs move / attack / control, all of them anchored to a board cell. It stops working once the **full Warchest** action set lands, for a reason that is structural, not just a matter of width.

### The split that forces the issue: spatial vs coin-only actions

Real Warchest actions divide into two kinds. The first kind points at a board cell; the second kind has **no cell to point at**.

| Verb | Needs a board cell? | Coin it spends |
|---|---|---|
| Move | yes (source + direction) | coin matching the moved unit |
| Attack | yes (source + target) | coin matching the attacker |
| Control | yes (source cell only) | coin matching the unit on the point |
| Deploy | yes (a control point) | a hand coin of the deployed type |
| Bolster | yes (an existing matching unit) | a hand coin of that type |
| **Recruit** | **no** | any hand coin (discarded to pay) |
| **Claim initiative** | **no** | any hand coin (discarded) |
| **Pass** | **no** | none |

A flat spatial grid head literally cannot represent "recruit a Swordsman" or "claim initiative" — there is no cell to attach them to. You would have to bolt on dummy cells or a second parallel head and hand-stitch the masks together. **This is the decisive argument for factoring in Warchest specifically**, independent of the action-space-width / sample-efficiency argument: the coin-only verbs need a first-class slot, and a top-level verb categorical gives them one.

### Head tree

```
verb  ∈ {move, attack, control, deploy, bolster, recruit, initiative, pass}   # 8-way softmax

├─ move / attack / control →
│     source_cell ∈ my units I hold a matching coin for        # the [7,7] map (the existing head)
│     ├─ move:    direction ∈ legal adjacent cells
│     ├─ attack:  target    ∈ attackable enemy cells
│     └─ control: (no further stage)
│
├─ deploy / bolster →
│     hand_coin ∈ distinct unit-types currently in hand        # small categorical
│     dest_cell ∈ control points (deploy) / matching units (bolster)   # the [7,7] map
│
├─ recruit →
│     recruit_stack ∈ supply stacks still available            # small categorical, non-spatial
│     [optional] pay_coin ∈ hand
│
├─ initiative →
│     [optional] pay_coin ∈ hand                               # else verb alone is the whole action
│
└─ pass → (terminal, no sub-decision)
```

Joint log-prob is the sum of the stages actually traversed:

```
log pi(a) = log pi(verb) + sum_i log pi(stage_i | earlier stages)
```

PPO's ratio, clipping, and entropy bonus all operate unchanged on that sum (entropy = sum of per-stage entropies). Mask at **every** stage, and the masks are conditional — legal sources given the verb, legal targets given the source. This is tighter and simpler than one flat mask over the full joint space. The "source" / "dest" stages reuse the existing `[7,7]` spatial head, so the board encoder is preserved, not discarded — it becomes one conditional stage among several. This is the AlphaStar autoregressive-head shape: a sequence of conditional categoricals where earlier picks gate later masks.

### The Warchest-specific wrinkle: the coin hand

Actions are gated by which coins are in hand (typically 3 drawn from a bag), not by the board alone. Two consequences the verb/source/target picture does not capture by itself:

1. **Masking depends on the hand, not just the board.** You can only move / attack / control a board unit if you currently hold a coin matching its type. So the `source_cell` mask for those verbs must be intersected with "do I hold a matching coin?" — which means **the hand must enter the observation** (it does not today). Add a hand encoding: counts per unit-type in hand, plus face-down / initiative state.

2. **"Which coin to spend" is itself strategic for the coin-only verbs.** For move / attack / control the coin type is *determined* once the source cell is picked (it must match the unit). But recruit and initiative discard *any* coin, and which one you throw away shapes next round's hand. Two options:
   - **First cut:** auto-pick the coin to discard with a heuristic (spend the least-useful coin), so verb alone fully specifies recruit / initiative. Fewer heads, faster to ship.
   - **Full version:** add the optional `pay_coin` head so the agent learns coin economy. Worth it eventually — hand management is a large part of real Warchest skill.

So the honest mapping is **verb → (coin selection) → (spatial placement) → (target)**, where the coin-selection stage is sometimes implicit (forced by the source cell) and sometimes an explicit head (recruit / initiative).

### What this buys you

- **Recruit / initiative / pass become representable at all** — the flat spatial head cannot hold them without kludges.
- **The verb head gets gradient on every move**, so the policy keeps learning as unit types are added (Archer, Cavalry, …). The per-unit-type growth lands on the small `hand_coin` / `source_cell` masks, not on a multiplicative joint head — parameters and compute scale as the *sum* of stage sizes, not their product.
- **Coin economy is modelable** once the `pay_coin` stage exists — something a flat head has no place for.

### Sequencing

Not worth building until the env actually has the bag / hand / recruit machinery. Order: (1) implement coin / hand state + recruit / initiative / pass in `warchest_env.py` and the observation; (2) *then* the factored head — its entire structure is dictated by the verb set and the hand masks, so it follows the env, not the other way around. At the current scale (~6 legal actions, Swordsman only) the flat head is already dense enough and factoring would add complexity for no gain.

This composes with AlphaZero (C15) rather than competing with it: a factored head is a fine prior network *inside* an MCTS loop. "Factor the head" (network output structure) and "add MCTS" (decision-time search) live in different layers and are orthogonal choices.

### Quick answers

**How do I know sample efficiency has become the bottleneck?** When WR-vs-greedy plateaus *while the machinery looks healthy* — `clip_frac` in 0.05–0.20, `critic_mae` low and flat, entropy not collapsed, grad norms stable — **and** the action space has actually grown. The tell: training longer or feeding more games still moves the plateau up (sample-starved), versus more games changing nothing (capability- or credit-assignment-capped, a different problem). A concrete diagnostic: log per-logit visitation; if a large fraction of action logits are hit <~1% of the time, the flat head is too wide to learn from limited games. Do the `improvement_ideas.md` fundamentals (C1/C2/C3/C6) first — only suspect the head *after* those are clean and the action set has expanded.

**Is "large-action PPO copes without MCTS" saying they're mutually exclusive?** No. They are two solutions at *different layers* to the same big-action-space problem: factoring restructures the **policy network's output** (pure learning, cheap at inference); MCTS adds a **decision-time search loop** (~500 lines, slower per move). The phrase only means you do not need the heavy search machinery merely to handle a wide action space — the cheap architectural trick suffices. They combine fine: a factored head makes a good prior network inside an AlphaZero loop.

**What does factoring actually buy?** (1) Each softmax stays small → the verb head gets dense gradient every move regardless of unit count; (2) params/compute scale as the *sum* of stage sizes, not the product; (3) clean conditional masking per stage instead of one giant joint mask; (4) reuses the existing `[7,7]` board encoder as the source stage; (5) correct joint log-prob for free, so PPO's ratio/entropy are unchanged. It buys **nothing** at the current ~6-action scale and does not add search depth or opponent reasoning — that is MCTS territory. Strictly an "action set has grown" investment.

---

## DQN (Deep Q-Network)

DQN learns a Q-function `Q(s, a)` — the expected discounted return from taking action `a` in state `s` and playing optimally afterwards. The policy is implicit: always pick `argmax_a Q(s, a)`. Transitions `(s, a, r, s')` are stored in a **replay buffer** and sampled randomly for training, and a frozen **target network** (Q updated every N steps) stabilises the TD targets:

```
L = E[ (r + gamma * max_a' Q_target(s', a') - Q(s, a))^2 ]
```

### Verdict

**DQN is fine for the current 2-unit / 14-action prototype. For the full Warchest (4 units, attack + ability actions, coin mechanics) PPO is the better choice. If sample efficiency later becomes the bottleneck, the right escalation is SAC-discrete or MuZero — not vanilla DQN.**

### Why DQN works for the current prototype

- 14 discrete actions is tiny and dense in valid choices — Q-learning has no problem here
- Off-policy replay is genuinely valuable: every transition is reused many times, which matters when self-play episodes are expensive
- Action masking is mechanical: set invalid actions to `-inf` in both the `argmax` and the TD target
- Replay buffer doubles as protection against "forgetting good strategies" when the opponent pool shifts: past transitions stay in the buffer and keep shaping Q

### Why PPO wins for the full game

1. **Action space is large and factored, not flat.** Full Warchest has roughly 4 units × (6 move + 6 attack + ~3 ability) + coin actions (recruit / bolster / deploy with sub-targets). That lands somewhere between 80 and 200 flat discrete actions. DQN can output that many heads, but the actions have natural structure (unit × verb × direction / target). Actor-critic expresses this cleanly with factored policy heads. DQN forces either a flat output (loses the structure) or an Action Branching architecture (fiddlier, less standard, harder to debug).

2. **Stochastic policy matters in hidden-information self-play.** Warchest has hidden state (opponent hand, bag, future draws). Optimal play in such games is often a *mixed* strategy, and the opponent pool means there are opponents actively learning to exploit predictable behaviour. PPO's stochastic policy expresses mixed strategies natively; DQN's `argmax` is deterministic and exploitable. Boltzmann action selection patches this but is not standard DQN and reintroduces the temperature-tuning problem PPO solves with entropy bonus.

3. **Self-play stability.** Policy-gradient methods with opponent pools (which we already have) are much better studied for self-play than DQN. DQN self-play is known to oscillate — the Q-function is a moving target against a moving opponent, and the target network only partially absorbs that.

4. **Sparse / delayed rewards.** GAE gives smooth credit assignment over long horizons via the `lambda` knob. DQN's 1-step TD bootstrapping under sparse rewards is unstable unless paired with n-step returns and most of the Rainbow stack (prioritised replay, distributional Q, dueling heads). At that point DQN is no longer simple.

5. **Sample efficiency is DQN's only real win — and there are better answers.** Replay reuse is real, but if simulation throughput becomes the bottleneck, **SAC-discrete** (replay + stochastic policy + entropy regularisation) or **MuZero / AlphaZero-style** (learns from search-improved policies) both dominate vanilla DQN for this setting.

### Factored head under DQN (Action Branching)

Can the factored head from the actor-critic section be reused with DQN? In theory yes, but it is strictly harder, and the reason is structural. A factored *policy* head works because probabilities factor via the chain rule — `log pi(a) = log pi(verb) + log pi(source|verb) + log pi(target|verb,source)`, an exact **sum** you can sample stage by stage. DQN has no distribution; it acts by `argmax_a Q(s,a)` and needs the joint argmax twice — to pick the greedy action and to compute the TD target `r + gamma * max_a' Q(s',a')`. The **max operator does not factor** the way a sum of log-probs does:

```
max_{verb,source,target} Q(...)  !=  max_verb(...) + max_source(...) + max_target(...)
```

The DQN-family answer is **Branching Dueling Q-Network** (BDQ, Tavakoli et al. 2018): shared trunk, one Q-branch per action dimension, dueling state-value plus per-branch advantages. Output grows as the *sum* of branch sizes, not the product — same combinatorial win as the factored head. It pays with two approximations:

- **Independent per-branch argmax.** BDQ maxes within each branch independently, assuming the branches' optimal choices don't depend on each other. In Warchest they do (best target depends on chosen source/verb), so both the greedy action and the bootstrapped max are approximate and DQN's optimality argument weakens.
- **Non-conditional branches.** Branches are computed in parallel from the shared state, not autoregressively, so a branched Q cannot natively express "value of this target *given* the chosen source" — exactly the conditional structure that makes the factored policy head correct, and exactly what conditional per-stage masking relies on.

The "do it properly" route — autoregressive `Q(target|source,verb)` evaluated sequentially — restores the conditioning, but then `max_a' Q(s',a')` becomes a search over the action tree at every bootstrap step. That reintroduces search into DQN's inner loop, loses its cheap one-shot argmax, and lands most of the way toward MuZero/AlphaZero anyway.

**Bottom line:** the factored structure is exact and free for a policy head (chain rule) and approximate-or-expensive for a value head (max doesn't factor). The conditional masking that is a clean win for the factored policy head is precisely what branched Q-networks struggle to express — which reinforces point 1 above: the factored action space is an argument *for* the actor-critic family, not a neutral feature both share.

### Practical implication

If PPO feels unstable on the current prototype, the right move is to debug PPO (reward shaping, advantage normalisation, entropy schedule) rather than swap algorithm families. Switching costs compound once coin mechanics land — the factored action heads and self-play machinery built around PPO do not transfer to DQN.

---

## R-NaD (Regularized Nash Dynamics) — idea, the "NaD" in WaRNaD

This is a proposal, not implemented. It is the algorithmic core of DeepMind's **DeepNash** (Perolat et al., *Science* 2022), the agent that reached expert-level Stratego with no search. The project name "WaRNaD" points at this method; the section records why it is the right long-term target and what it buys over the current PPO + opponent-pool setup.

### The problem it solves: self-play cycles, it does not converge

Current training is self-play (pool of frozen snapshots + greedy/random). The implicit goal of self-play in a two-player zero-sum game is the **Nash equilibrium** — the unexploitable strategy that secures the game's value against *any* opponent, including a best-responder. Warchest is genuinely zero-sum and genuinely **imperfect-information** (the critic is fed a privileged `PRIV_DIM` vector of the opponent's true hidden coin split that the policy never sees, `Critic` in `src/services/policy/policy.py`). Hidden information means the equilibrium is generally a **mixed** strategy — being predictable is exploitable, exactly as in Rock-Paper-Scissors.

The catch: plain policy-gradient dynamics in a zero-sum game **do not converge to the equilibrium — they orbit it**. This is a structural property of the dynamics, not a learning-rate bug: gradient/replicator dynamics in zero-sum games are conservative (energy-preserving), like a frictionless pendulum. The policy circles the equilibrium forever — beat A, drift to a policy that beats A but loses to B, drift back, repeat. Two consequences:

- **The last iterate is exploitable.** Whatever point on the orbit you stop at is *not* the center. Saving the final network (`torch.save(... policy.state_dict())`, `src/app/ppo.py`) saves a point on the orbit.
- **Only the time-average converges.** The average over the orbit lands near the equilibrium even though no single point does. The opponent pool is a practical approximation of "average over the orbit" — but it treats the symptom (average away the cycling) rather than the cause (the dynamics cycle at all). The `max_size`-capped snapshot deque evicting old snapshots (`OpponentPool`, `src/services/opponent_pool.py`) is a known cause of *re-forgetting* and re-introduces cycling.

### The mechanism: a moving KL-to-reference regularizer adds friction

R-NaD adds a reward transform that pulls the policy toward a **reference policy** `pi_ref` (a recent frozen copy):

```
r'(s, a) = r(s, a) - eta * ( log pi(a|s) - log pi_ref(a|s) )
```

This converts the frictionless orbit into a **damped spiral that contracts to a fixed point** — the pendulum now has friction and comes to rest. That gives **last-iterate convergence**: the single network you hold at the end is itself near-equilibrium, so you no longer need to average or ship a snapshot mixture to be unexploitable.

The reference must **move**, and this is the part that distinguishes it from the entropy bonus the code already has:

- A *fixed* pull (toward `pi_ref`, or toward uniform — which is what `entropy_coeff=0.025` in `src/app/ppo.py` does) converges to a **biased, smoothed** equilibrium (a quantal-response equilibrium), not the true Nash. The regularization permanently distorts the answer, the same way a non-potential reward bonus would.
- R-NaD instead solves the regularized game to its stable fixed point, then sets `pi_ref <- pi` and re-solves with the new anchor. Each solve is convergent (friction), and the **sequence of anchors walks to the true Nash**. This is a proximal-point / mirror-descent scheme: stable convergent sub-problems whose fixed points converge to the exact equilibrium, because `eta`'s pull vanishes as `pi` and `pi_ref` coincide at Nash.

Distinguish this from PPO's existing KL machinery. The `KL_TARGET=0.015` early-stop (`src/app/ppo.py`) is a pull toward the policy *one optimization step ago* — a trust region for **optimizer** stability (step-size control). It does nothing about the orbit. The R-NaD reference is a **slowly-moving anchor on a much longer timescale** that reshapes *which* equilibrium is sought and damps the cycle. Different layer, different purpose.

### What it buys for Warchest

- **The saved network is unexploitable on its own.** Last-iterate convergence removes the "did I save an exploitable point on the orbit?" problem at the root, rather than patching it by deploying a mixture.
- **No more forgetting / cycling.** Training progress becomes roughly monotone toward equilibrium instead of orbiting; the pool stops being load-bearing.
- **A well-defined target.** Plain self-play has no fixed point it is converging to (it orbits); R-NaD makes the Nash equilibrium an actual attractor.
- **Makes the exploitability metric meaningful.** With a best-responder exploitability eval (the work-in-progress: pool of size 10–20 playing a frozen policy), an orbiting policy would show exploitability *wobbling around a floor*; a converging one shows it *descending*. R-NaD turns that curve from a noise gauge into a progress curve.
- **Mixed strategies where the game requires them.** Because Warchest hides the opponent's coin split, equilibrium play is mixed; a method that targets Nash directly is the principled way to get unexploitable randomization rather than relying on the entropy bonus to keep the policy from collapsing to a predictable best-response.

### Drawbacks / cost

- **Heavier than the other items.** Requires holding a frozen `pi_ref`, adding the per-step log-ratio term to the reward before GAE, and a schedule for refreshing `pi_ref` and annealing `eta`. More moving parts than dropping `holding_reward` or annealing greedy out.
- **New hyperparameters.** `eta` (regularization strength), the reference-update period, and the anneal schedule all interact, on top of PPO's existing knobs.
- **Pairs with, does not replace, the cheaper fixes.** It assumes the reward is the true zero-sum payoff. The non-potential `holding_reward` term (`src/app/ppo.py`) and training against fixed greedy/random opponents both bias the game away from its true equilibrium, so R-NaD would converge to the *wrong* fixed point unless those are addressed first. Sequence it after: (1) exploitability metric, (2) deploy a mixture/average, (3) pure zero-sum reward, (4) anneal fixed bots out — *then* R-NaD.

### Relationship to the other ideas

The snapshot pool fights cycling by **averaging over** the orbit; R-NaD fights cycling by **killing** the orbit so there is nothing to average. They are complementary, but if R-NaD works the pool and the deploy-a-mixture workaround stop being necessary for unexploitability (the pool may still be useful as an opponent-diversity curriculum). It composes cleanly with the factored action head (the head is just the policy parameterization; R-NaD only changes the reward and adds the reference anchor). It is an alternative path to AlphaZero (C15) rather than a complement: both target the equilibrium of a two-player zero-sum game, but R-NaD is **search-free** (model-free, no MCTS at decision time), which is the whole reason DeepNash used it for Stratego — the game tree was too large and too imperfect-information for the MCTS that worked on Go/chess.

---

## Other Alternatives

### A2C / A3C (Synchronous / Asynchronous Advantage Actor-Critic)

Runs multiple environment instances in parallel, collecting transitions from all of them before each update. A3C uses asynchronous workers that push gradients independently.

**Relevant for Warchest:** parallel self-play rollouts would multiply effective episode throughput. But A3C's asynchronous updates cause stale gradients — PPO with parallel envs (the standard `VecEnv` setup) achieves the same throughput with cleaner theory. Use PPO + parallel envs instead.

---

### SAC-Discrete (Soft Actor-Critic for Discrete Actions)

Off-policy actor-critic with maximum-entropy objective. Learns a Q-function and a policy simultaneously, using the replay buffer for Q and a separate policy gradient for the actor.

```
J = E[ Q(s, a) - alpha * log pi(a|s) ]
```

The `alpha` (temperature) parameter automatically balances exploration and exploitation.

**Relevant for Warchest:** combines the replay buffer benefits of DQN with an explicit stochastic policy (useful for self-play diversity). More complex to implement than PPO or DQN. Worth considering if DQN exploration proves insufficient once the policy is non-trivial.

---

### AlphaZero / MCTS + Policy-Value Network

Replaces rollouts with Monte Carlo Tree Search guided by a learned policy and value head. At each move, MCTS runs hundreds of simulated playouts, using the policy network to focus search and the value network to evaluate leaf nodes. The improved MCTS policy is then used to train the network.

**Relevant for Warchest:** this is idea #19. The game is small enough (14 actions, 7×7 board, ~100 turns) that MCTS is computationally feasible. AlphaZero-style training is provably the strongest approach for two-player zero-sum games of this size. The downside is implementation complexity: MCTS, self-play game generation, and the training loop are all significantly more involved than PPO. Recommended as a long-term target once PPO converges to a non-trivial policy.

---

## Decision (2026-05-23) — PPO implemented

**PPO is the active training algorithm.** The other candidates were ruled out:

**DQN — ruled out.** See the "Why PPO wins for the full game" subsection above for the full reasoning. Summary: the full action space (~80–200 actions) is factored as unit × verb × direction/target and is much cleaner under a multi-head policy than under a flat Q-head; `argmax` removes the stochasticity that mixed strategies need in a hidden-information self-play setting; and DQN self-play is known to oscillate against an opponent pool. Off-policy replay would be a benefit, but the better answers to "we need replay" are SAC-discrete or MuZero — not vanilla DQN.

**SAC-Discrete — ruled out.** More complex than PPO with no clear advantage for this setup. The entropy regularization it provides is already available in the current actor-critic via the entropy bonus term.

**AlphaZero / MCTS — ruled out for now.** Fundamentally different training loop: requires MCTS simulation, a separate game-generation pipeline, and a policy-improvement operator. Strong long-term ceiling but the implementation cost is high and the benefit is uncertain before the fundamentals (reward shaping, stable gradient signal) are working. Revisit once PPO produces a non-trivial policy.

**Note from log analysis (run_20260523-101428, 665 episodes):** PPO alone will not unblock training. The current actor gradient is structurally zero — the critic converged to a constant (predicting truncation time penalty for every state), advantages are all near-zero, and advantage normalization kills the sparse win signal. PPO makes better use of signal; it cannot create signal from nothing. The prerequisite fixes are dense reward shaping and weaker advantage normalization. PPO comes after those.

## Implemented path

| Step | Change | Status |
|---|---|---|
| 1 | Dense reward shaping (potential-based) | ✅ Done |
| 2 | Z-score advantage normalisation | ✅ Done |
| 3 | Low entropy coefficient (0.001) | ✅ Done |
| 4 | Episode batching (16 eps per batch) | ✅ Done |
| 5 | PPO with clipped surrogate | ✅ Done |
| 6 | Opponent pool (random + greedy + snapshots) | ✅ Done |
