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

## DQN (Deep Q-Network)

DQN learns a Q-function `Q(s, a)` — the expected discounted return from taking action `a` in state `s` and playing optimally afterwards. The policy is implicit: always pick `argmax_a Q(s, a)`.

Uses a **replay buffer**: transitions `(s, a, r, s')` are stored and sampled randomly for training, breaking temporal correlations. A **target network** (frozen copy of Q updated every N steps) stabilises the TD targets:

```
L = E[ (r + gamma * max_a' Q_target(s', a') - Q(s, a))^2 ]
```

**Benefits for Warchest:**
- Off-policy: every transition is reused many times — orders of magnitude more sample efficient than REINFORCE for environments where experience is expensive to collect
- Replay buffer directly addresses the "forgetting good strategies" concern: past transitions remain in the buffer and continue to shape the Q-function
- 14 discrete actions is tiny — DQN is well-suited to small discrete action spaces
- No need for the GAE advantage computation; the critic (Q-network) trains directly on TD error
- Can train on a mix of self-play and random-opponent data without bias (off-policy handles it naturally)

**Drawbacks:**
- Requires a separate network architecture: a single head outputting Q-values for all 14 actions, rather than actor + critic heads
- Action masking requires care: invalid actions must be masked in the `argmax` and set to `-inf` in the target — doable but needs explicit handling
- Q-learning in two-player zero-sum games is theoretically less clean than policy gradient: opponent behaviour changes the Q-values, creating a non-stationary target. Mitigated by the target network and the opponent pool
- No built-in entropy regularisation — exploration requires epsilon-greedy or separate mechanisms
- Credit assignment over 100-turn episodes is harder for TD(1) Q-learning than for GAE with `lambda=0.95`

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

**DQN — ruled out.** DQN represents state-action value as `Q(s, a)` with one output per action. Today there are 14 actions, but the plan is to add more unit types, each with their own move and claim actions. The action space will grow with each unit type added, requiring the Q-network output layer to be rebuilt and retrained from scratch every time. A policy network outputs a distribution over whatever actions exist — it scales transparently. DQN also loses the natural stochasticity of a policy, which matters for self-play diversity. Off-policy replay would be a benefit, but not worth the scaling cost.

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
