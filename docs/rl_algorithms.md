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

### Practical implication

If PPO feels unstable on the current prototype, the right move is to debug PPO (reward shaping, advantage normalisation, entropy schedule) rather than swap algorithm families. Switching costs compound once coin mechanics land — the factored action heads and self-play machinery built around PPO do not transfer to DQN.

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
