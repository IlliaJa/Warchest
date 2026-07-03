# Reward Design

See [experiments.md](experiments.md) for the full training run history.

This doc is organised in three parts:

1. **[Current rewards](#1-current-rewards)** — what the agent is actually paid, and why.
2. **[Rejected ideas](#2-rejected-ideas)** — reward terms considered and deliberately not used.
3. **[Unrealized ideas](#3-unrealized-ideas)** — proposed but not yet implemented (currently one).

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

> **Implemented 2026-07-03.** Material PBRS, linear annealing of the holding + material shaping (1.0 → 0.1 over the first half of the run), a base-diff-proportional truncation reward (C17), and a widened **critic-only** trunk (`critic_hidden_dim=128`, policy left at 64) all shipped together. Rationale and cross-project grounding: [rewards_improvements.md](rewards_improvements.md) and [decision.md](decision.md) (2026-07-03). A/B still owed against the pre-change baseline per the standing rule in [future_steps.md](future_steps.md).

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

`WarChestEnv.boxed_total(pid) = sum(state.boxed[pid].values())` — coins only ever leave play into the box, so a player's live-coin count is `owned_total - boxed_total`; the owned offset is fixed per composition and cancels in the telescoping, leaving the boxed differential as the live term. It is keyed by **absolute** player id (no perspective flip needed, unlike base shaping).

Design rationale & caveats:

- **Fires** a positive pulse when the agent boxes an enemy coin (attack), a negative pulse when it loses one — exactly the mechanic base shaping never touches.
- **Coefficient.** `C_MAT` is kept well below `SHAPING_C = 0.05` (bases *win* the game; material is a means). A single killed coin is worth a fraction of a base swing.
- **Telescoping / measurement point.** `phi_material` is measured at the same points as base shaping (before / immediately after the main actor's coin play), i.e. the same convention already used for `SHAPING_C`. The original proposal called for measuring `phi` *after the opponent's move*; that is **not** what the current base shaping does, so the material term matches base shaping for consistency. If an A/B shows this leaks, the "after opponent's move" fix must be applied to **both** terms together (see [future_steps.md](future_steps.md)).
- **Exploit safety.** `boxed` is monotonic per player, but PBRS is policy-invariant for *any* state function, so there is no farming loop (unlike a raw per-kill bonus, which the circular-claiming precedent warns against). Uses only `GameState.boxed` — no BFS, negligible cost.
- **Over-shaping caveat.** `docs/analysis_ppo_20260630.md` found the agent already *over-optimizes* dense shaping relative to winning. This term is still worth it because (a) it is proper PBRS, unlike the non-telescoping holding reward that is the likelier decoupling culprit; (b) material advantage is **strongly win-correlated**, so it points the dense signal at something that predicts wins. This is the main reason it is annealed (see below) and A/B'd, not stacked blind.

> **Note — `ATTACK_REWARD` was intentionally kept, not subsumed.** `rewards_improvements.md` §2/Step 2 recommends folding the raw `ATTACK_REWARD = 0.02` into this term (it fires on the same box-a-coin event and is non-telescoping). That swap was **not** made in this pass to keep the change set to exactly what was requested; both currently fire. Flagged for the A/B (see `IDEAS.md`): zero `ATTACK_REWARD` and re-measure so attacks aren't double-paid.

### Shaping annealing

The holding reward and the material shaping term are multiplied by `shaping_anneal`, linearly decayed **1.0 → 0.1 over the first half of the run**, then held at the 0.1 floor:

```python
half = max(n_batches * 0.5, 1.0)          # tracks n_batches: 200 @400, 250 @500
anneal_frac = min((batch_num - 1) / half, 1.0)
shaping_anneal = 1.0 + anneal_frac * (0.1 - 1.0)
```

Base-diff PBRS (`SHAPING_C`) is deliberately left constant. Rationale (the over-shaping antidote — keep dense guidance while the critic is weak / entropy is high, hand the final policy back toward the terminal objective): [decision.md](decision.md), 2026-07-03, and OpenAI Five's annealed `team_spirit`. Logged per batch as `shaping_anneal`; the annealed holding/material contributions are logged as `score_holding` / `score_material`.

### Per-turn holding reward

Applied by the training loop alongside shaping:

```python
holding_reward = holding_reward_rate * (my_bases - opp_bases)
holding_reward_rate = WIN_REWARD / ((winning_base_count - 1) * (max_rounds * HAND_SIZE)) * 0.8
                    = 1.0 / (5 * (50 * 3)) * 0.8 = 1.0 / 750 * 0.8 ≈ 0.001067
```

Fires every agent turn proportional to the current base lead. Incentivises defending claimed bases and breaks the base-flip exploit (two policies claiming the same bases in a loop), where potential shaping alone produces near-zero net reward per cycle. The 0.8 factor is a safety margin ensuring worst-case accumulated holding (`0.001067 * 5 * 150 = 0.8`) never exceeds `WIN_REWARD = 1.0`. `max_rounds * HAND_SIZE` is used as the worst-case bound on main-actor turns per episode. `holding_reward_rate` is derived from env constants at script startup so it stays valid if `max_rounds`, `HAND_SIZE`, or `winning_base_count` change.

Shaping and holding are complementary: shaping delivers an immediate, one-step credit signal at the claim action; holding creates persistent per-turn pressure that requires no bootstrapping. Note the holding reward is **not** potential-based (it does not telescope), which is why it is now annealed toward a small floor rather than left constant — see [decision.md](decision.md) (2026-07-03) for the "why annealing rather than removal" reasoning.

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
