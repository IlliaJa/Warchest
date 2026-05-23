# Ideas for improving training

Analysis from logs `run_20260522-200919.log` (123 episodes, latest meaningful run) plus full read of `reinforce.py`, `environment/warchest_env.py`, `environment/board.py`, `environment/action.py`, `policy.py`.

## Observed symptoms

- `actor_loss ≈ 0.0000` every single episode — the policy gradient produces effectively no signal.
- Entropy stuck at **2.2** (max for 14 actions ≈ 2.64) — policy stays near-uniform.
- `wr_self` starts at **0.5–0.67** (random p1-advantage from moving first), then **drops to 0.24** by ep123 — the network actively learns something that hurts player 1.
- `wr_rng` ends at **0.12** — the trained policy loses to a uniformly random opponent more often than it wins.
- ~80% of episodes truncate at 99 turns — the policy rarely finishes games.

The fact that `wr_rng < 0.5` after training is the smoking gun: the policy is provably worse than random. It is not "stuck near random" — it is being pushed *away* from a useful strategy. Below is the chain of causes, ordered by how much they contribute.

---

## Fixes (correctness bugs blocking learning)

### 1. ✅ Observation is absolute, action space is relative — the policy is trying to learn two contradictory functions

> **Done 2026-05-23.** `generate_observation` now always puts the active player's units in slot 0 and the opponent's in slot 1. `global_feats` drops `active_player` and uses `[turn_count, my_bases, opp_bases]` (3 features). `encode_board` swaps channels 3/4 based on `active_player` so channel 3 is always "my bases". `observation_space` updated to match. Network input shrunk from `hidden_dim+4+32` to `hidden_dim+3+32`.

This is the single biggest reason the policy can't beat random.

In `warchest_env.py:230-251`:
```python
units_obs[player_id - 1, i] = _unit.loc        # p1 always slot 0, p2 always slot 1
global_feats = [active_player - 1, ..., bases_p1, bases_p2]  # absolute order
```

But action IDs are **relative** to the active player (`warchest_env.py:340-360`):
```python
def get_move_info(self, action_id):
    unit_id = action_id // 6
    player_units = self.get_active_player_units()   # filtered to active player
    start = player_units[unit_id].loc
```

So `action_id=0` means **"move my first unit south-west"** — but "my first unit" is in observation slot 0 when p1 is active and slot 1 when p2 is active. The policy outputs the same logits given the same board state regardless of perspective, except for one scalar (`active_player - 1`). It is being asked to learn two different mappings from the same convolutional features, conditioned on a single bit.

In self-play this creates **directly contradictory gradients**: a winning action for p1 in some state and a winning action for p2 in a mirrored state push the same logit in opposite directions, because the action IDs reference different units. The net update cancels out — or worse, the larger initial p1-advantage produces a stronger gradient that the p2-side later overwrites.

**Fix**: encode the observation from the active player's perspective. Always put the active player's units in slot 0 and opponent's in slot 1; swap base counts; rotate/mirror the board if you want full symmetry. Drop the `active_player` feature entirely afterwards — the observation should mean the same thing whoever is acting.

```python
# In generate_observation, after building units_obs and global_feats:
if self.active_player == 2:
    units_obs = units_obs[[1, 0]]                          # swap slots
    global_feats[2], global_feats[3] = global_feats[3], global_feats[2]
    # Also swap p1/p2 channels in board encoding (in Policy.encode_board)
```

This is a **prerequisite** for everything below to actually work in self-play. Without it, the policy is fighting itself.

---

### 2. ✅ `observation_space` lies about its shape

> **Done 2026-05-23** (resolved as part of fix #1). `global_feats` now has 3 elements to match the declared shape. `active_player` added as a separate `Discrete(2)` key in the space.

`warchest_env.py:218-227`:
```python
global_features = 3
```
But `generate_observation` emits **4** features. Gymnasium consumers and any wrapper using `observation_space` get the wrong shape. Not crashing today only because the policy reads `obs['global']` directly without checking the space.

**Fix**: change to `4` (or `3` if you drop `active_player` per fix #1).

---

### 3. ~~Reward shaping fires once per `(base, player)` and then goes silent~~ ✅ DONE

`warchest_env.py:316-337`: `MOVE_ON_BASE_REWARD` and `MOVE_NEAR_BASE_REWARD` are gated on `unclaimed_bases_approach_reward[base][player]['on'/'near']` — once True, never reset. After the first ~10 moves of any episode all six bases have been "visited near" by both players, and the only reward signal left is the −0.002 time penalty (until a claim or a win). The critic has almost nothing to predict for the middle 80 turns of every game.

**Fix**: either drop these flags (let the reward fire every visit, capped per episode), or reset them on `env.reset` — they currently persist across episodes if reset() is fast-pathed. Better: replace with a denser shaping signal like *negative distance from nearest unclaimed base* on each move.

---

### 4. ~~`make_random_step` corrupts credit assignment when the policy emits an invalid action~~ ✅ DONE

`reinforce.py:89-95`:
```python
if not step_info['action'].is_valid:
    state, reward, terminated, truncated, step_info = env.make_random_step()
```

When the policy chooses an invalid action (rare since masking is correct, but the masking softmax can still sample masked actions due to numerical issues at -1e9), the env then runs a *random* action and the **policy's `log_prob` is paired with the *random* action's reward**. This is incorrect credit assignment — the policy is rewarded/punished for an action it never took.

**Fix**: after `make_random_step`, overwrite the appended `log_prob`, `value`, `entropy` to detached zeros (treat as a non-learning step), or skip appending entirely. Today the count is 0 in logs, so it's latent — but it's a footgun.

---

### 5. ~~Random player's trajectory still runs full GAE + critic loss~~ ✅ DONE

`reinforce.py:126-152`: the GAE loop iterates `for pid, p_info in info.items()` for both players, even when one is random. The random player's `log_probs`, `values`, `entropies` are constant tensors. GAE produces meaningless advantages, an MSE critic_loss of ~0.038 every time (just the squared `LOSS_REWARD` term), an actor_loss near zero, and an entropy of 0. The result is discarded (loss assembly excludes random side), so the only damage is wasted compute and confusing debug lines.

**Fix**: skip the GAE block for the random side. Optional cleanup but it removes a recurring `critic_loss=0.038` red herring from the logs.

---

### 6. ✅ `generate_observation` uses sorted units, action IDs use raw board order

> **Done 2026-05-23.** Removed `sorted(...)` from `generate_observation`; both paths now iterate `self.board.units` directly.

`warchest_env.py:233`: `all_units = sorted(self.board.units, key=lambda u: u.__class__.__name__)` for the observation, but `get_active_player_units` uses raw `self.board.units` order. These coincidentally agree today because every unit is a `Swordsman` and Python's sort is stable — the moment a second unit type is added (or a unit is removed and re-added), observation slot 0 and action's "unit 0" will refer to different units.

**Fix**: use one canonical ordering throughout. The deployment order in `place_default_units` already gives a stable order; just keep using `self.board.units` directly in both places.

---

## Small improvements (tuning and minor refactors)

### 7. ~~`gamma=0.9` is too aggressive for 99-turn episodes~~ ✅ DONE

`0.9^99 ≈ 3×10⁻⁵`. The final WIN_REWARD is essentially invisible to the critic for the first ~50 turns. Combined with the silent middle-game reward shaping (#3), the critic has almost no learning signal for early actions. **Try `gamma=0.99`** (so the win reward decays to ~0.37 after 99 turns).

### 8. Batch episodes before the optimizer step

Each update is one episode of ~100 transitions. With advantage normalization that's a tiny, high-variance batch. **Accumulate 4–8 episodes before `optimizer.step()`** — same compute, dramatically lower gradient noise. Most actor-critic implementations batch 32–128 environment steps per update for exactly this reason.

### 9. Critic LR is 5× actor LR through a shared backbone

`reinforce.py:282-285`: `lr_critic=5e-4`, `lr_actor=1e-4`. Because the backbone is shared, every critic update reshapes the features the actor reads. With the actor gradient already near zero (problem #1), the backbone is driven almost entirely by the critic, which then changes the actor output without the actor having any say. **Try `lr_critic = lr_actor = 1e-4`**, or decouple the backbones (#13).

### 10. Entropy bonus schedule still dominates the actor

Even at `entropy_coeff=0.005`, the bonus contributes `0.005 × 2.2 ≈ 0.011` to the loss while actor_loss is `< 0.0001`. It's ~100× the actor signal. **Try 0.001 from the start, or 0.005 → 0.0001 linearly** over training.

### 11. ~~`WIN_REWARD = 1.0` but `CLAIM_BASE_REWARD = 0.03` — claims contribute almost nothing~~ ✅ DONE

The agent needs 4 claims to win (6 bases total, starts with 2). Total claim reward = 0.12 vs the final 1.0 — claims look almost free relative to winning. But since most episodes truncate without a win, claims should be your dominant learning signal. **Try `CLAIM_BASE_REWARD = 0.15`** so the cumulative claim reward (~0.6) is comparable to the win signal.

### 12. ~~Normalize returns for the critic, not just advantages for the actor~~ ✅ DONE

`reinforce.py:139-141`: advantages are normalized but returns (the critic target) are not. With `gamma=0.99` and dense rewards the return scale will balloon; the critic will spend its capacity tracking that scale instead of the relative ordering. **Use a running mean/std on returns** when changing gamma.

---

## Architecture improvements (bigger restructuring)

### 13. Separate actor and critic networks

Currently `board_encoder` and `unit_encoder` are shared between actor and critic. Critic gradients overwhelm actor gradients through the shared layers. **Give each head its own encoder** (or at least decouple after one shared block). This is one line in `policy.py` and the most impactful single architecture change for the current setup.

### 14. Move from REINFORCE + GAE to PPO

REINFORCE uses each episode once. PPO does 4–10 gradient steps per episode with importance-sampling correction and a clipped surrogate, which is **4–10× more sample-efficient** for the same wall-clock. PPO also stabilises training when advantage signals are noisy (your case). Same model, same loss structure, two extra lines for the clip and the inner epoch loop.

### 15. Replace the squared-grid CNN with a hex-aware encoder

`policy.py:13-20` uses a standard 2D conv on a 7×7 array — but the board is hexagonal with the `(±1, ±1)` neighbour topology from `Board.offsets`. A 3×3 conv kernel sees neighbours that aren't actually adjacent in the hex grid and misses neighbours that are. For a 7×7 board you can sidestep this entirely with a **small MLP on the flattened board** (343 inputs after one-hot, ~20k params) which has no false topology assumption. Or implement hex convolution properly with the 6-direction offset pattern.

### 16. Self-play needs an opponent pool, not a mirror of the current policy

Right now both sides are always the *current* policy. This produces non-stationary co-adaptation: the policy can learn to exploit its own current weaknesses, then those weaknesses change, and the cycle continues — explaining the gradual *decline* you see in `wr_self`. **Keep a buffer of past policy snapshots and sample opponents from it** (a tiny "league"). Even a 10-deep snapshot buffer breaks the cycle.

### 17. The 30% random-opponent mixing is too high once the policy is non-trivial

Currently 30% of episodes have a random p1 or p2, and another ~9% are random-vs-random (no learning signal). Early on this is fine for exploration, but after a few hundred episodes you want most rollouts to be policy-vs-policy or policy-vs-past-snapshot. **Schedule the random ratio from 30% down to ~5%** over the run.

### 18. Replace dummy-step terminal handling with a value-bootstrap-only approach

The current terminal fix (`reinforce.py:101-112`) appends a fake transition to the loser's trajectory. This works but adds a dummy action to the policy gradient (with tiny log_prob) and a dummy critic target. A cleaner approach: **don't append a step — instead set `values[-1] = 0` for the loser and add `LOSS_REWARD` to their last real reward**. Mathematically equivalent (the bootstrap value at the terminal is 0), no fake transitions in the trajectory.

### 19. Consider MCTS + policy/value network (AlphaZero-style)

For a game this small (action space 14, board 7×7, ~100-turn games), pure RL is not the most efficient approach. **MCTS with a learned policy/value head** would dramatically outperform what you can extract from REINFORCE/PPO alone, at the cost of a more complex training loop. Worth doing once the fundamentals (#1, #13, #14) are in place.

---

## Recommended order

If you only do one thing: **fix #1 (relative observation)**. Without it, no amount of hyperparameter tuning will produce a policy that consistently beats random in self-play, because the network is structurally being asked to learn a contradiction.

After that, in order of impact-per-effort:
1. Fix #1 — relative observation
2. Improvement #13 — separate actor/critic encoders
3. Improvement #8 — batch episodes (4–8 per update)
4. Improvement #14 — PPO
5. Fix #3 — denser middle-game shaping
6. Improvement #7 — `gamma=0.99`
