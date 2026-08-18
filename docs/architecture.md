# Warchest — Architecture Overview

Warchest is a two-player turn-based hex-grid strategy game paired with a reinforcement learning training framework. A PPO actor-critic policy learns to play the game.

## High-level component map

```
src/app/ppo.py  (PPOTrainer)        training entry point
│
├─ WarChestEnv                      Gymnasium env  (src/services/environment/)
│   ├─ Board                        hex grid + cell logic
│   ├─ GameState                    state snapshot for replay (bags/hands/discards/boxed)
│   ├─ roster.py                    single source of truth: all 16 unit types + Royal coin
│   ├─ obs_encoders/                versioned observation encoders; env delegates encode()
│   │                               /observation_space()/encode_privileged() to the chosen one
│   └─ Action                       action dataclass
│
├─ Policy                           actor network   (src/services/policy/policy.py)
│   │                               arch-versioned: policy_factored_v1 | v2 (default)
│   ├─ type_emb       (v2) shared unit-type table; contracts the per-type unit
│   │                      planes and coin vectors before the trunk  [A1]
│   ├─ conv_blocks + films  (v2) 3x HexConv2d, each followed by FiLM(globals)  [A3]
│   │                      (v1: board_encoder, a plain 3x HexConv2d stack)
│   ├─ policy_head    → per-cell verb logits (masked), 1x1 conv
│   │                      (v1 also takes globals broadcast onto all 49 cells)
│   └─ verb_head / facedown_head → non-spatial verb logits, from a flank-split pool
│
├─ Critic                           separate value network, wider trunk (critic_hidden_dim=192)
│   │                               arch-versioned: critic_v1 … v5 (default)
│   ├─ type_emb       (v5) the same unit-type table, own weights  [A1]
│   ├─ board_encoder  (3x HexConv2d + GroupNorm from v2; independent weights; NO FiLM)
│   ├─ head           → scalar state value, fed pooled board features + globals
│   │                    + a privileged (critic-only) hidden-coin vector
│   │                    (v1/v2 also take a 3-wide opponent one-hot)
│   └─ board_only_head  (v2+) auxiliary value from the board ALONE — the reason
│                        the critic must stay globals-free upstream of the pool
│
├─ OpponentPool                     opponent sampler  (src/services/opponent_pool.py)
│   ├─ RandomBot                    uniform random over valid actions
│   ├─ GreedyBot                    priority: attack → control → move-toward-base → deploy → pass
│   └─ frozen Policy snapshots      rolling window (snapshot every 15 batches, max 20)
│
└─ RolloutBuffer                    GAE buffer  (src/utils/rollout_buffer.py)
```

## Data flow per environment step

```
Observation dict (generate_observation() → obs_encoders/v11.py, OBS_VERSION=11)
  ├─ board[48,7,7]             base/terrain + per-type unit stacks + threat + coord + base-reach planes
  ├─ global[245]                round/base/initiative + per-type coin counts + draw-share
  │                              + material-at-risk + E_opp_hand + base-reach scalars
  │                              + pending one-hot
  └─ valid_action_mask[1875]   legal-action binary mask (factored verb x cell + face-down)
       │
       ├──► Policy.act()       → sampled action, log_prob
       ├──► Critic.value_single() → state value V(s), given a privileged opponent-hand vector too
       │
       ▼
WarChestEnv.step(action_id)
  → next obs, reward, terminated, truncated, info
  (turn does not advance while a multi-step tactic's `state.pending` is set)
```

For `active_player == 2`, the encoder returns the board/globals pre-rotated 180°
so the network always sees "my units" the same way; `WarChestEnv.remap_action` performs the
matching inverse remap on the chosen action id. Full schema: `docs/environment_api.md`.

**Observation encoding is versioned and pluggable** (`src/services/environment/obs_encoders/`).
The engine holds only the stable game rules and rules-queries
(`unit_threat_footprint`, `attack_enabler_coins`, `unit_base_reach_cells`); an encoder
(`v11.py` is the live one, `v10.py` is kept loadable for the gauntlet, both via the registry) owns
the version-varying part — the coin-availability
model, feature aggregation, normalizers, plane layout, and ego-rotation. `WarChestEnv(obs_encoder=…)`
picks the encoder (default: latest); `Policy`/`Critic` size their input layers from the paired
encoder's dims. Because `encode(view)` is a pure function of the game state, agents built for
different obs versions can encode one shared `WarChestEnv` independently — the basis for the
cross-era round-robin gauntlet (`src/services/gauntlet.py`; design rationale in `docs/history.md`
→ *Measurement + opponent infrastructure*).
`tests/test_obs_golden.py` guards the encoding byte-for-byte.

## How the network reads a unit type, and how the hand reaches the board

Two changes shipped together on 2026-08-16 (`docs/IDEAS.md` A1 + A3) and are the current
default pair, `policy_factored_v2` + `critic_v5`. They are documented together because they
answer the same question from opposite ends: the observation addresses a unit type by *index*
and hands the globals to the network as an afterthought, and both are wrong for this game.

### A1 — a shared unit-type table instead of 32 one-hot planes

The observation gives each unit type its own board plane (`base + id - 1`) and its own slot in
every per-type coin vector. An index carries no information about the unit: plane 6 (Swordsman)
and plane 7 (Knight) are as unrelated to a conv kernel as any two integers. Since only 4 of 16
types are drafted per side, `probe_costs.py` Table B measures **62 % of board planes and 82 % of
global dims exactly zero on any given forward pass** — and *which* ones changes every game. So
the first conv layer maintained 16 disjoint kernel slices per side, each trained on the ~1/4 of
games where its type appeared.

`src/services/policy/unit_embedding.py` replaces the index with a 16-wide row per type and
contracts the observation against it, `embedded[d] = Σ_t count[t] · E[t, d]`:

| | shape | trained | role |
|---|---|---|---|
| frozen | `[16, 10]` | never (a `register_buffer`) | rules attributes from `roster.py` |
| learned | `[16, 6]` | yes | per-type identity the attributes miss |

The frozen half is what transfers between types, and **every one of its 10 columns is shared by
at least two units by rule** — a column belonging to one unit shares nothing and would just be a
one-hot slot with a friendlier name. The columns are `coin_count`, `can_normal_attack`,
`gives_extra_tempo` (5 units), `has_defensive_trait` (3), `has_tactic` (9), and five tactic
columns decomposed by *behaviour* rather than one-hot by mechanic name. The `tactic_ranged_strike`
/ `tactic_charge_strike` split deliberately mirrors `v11.THREAT_KINDS`, which already spends six
planes separating "hits from two hexes away" (Archer, Crossbowman) from "moves in and hits"
(Cavalry, Lancer).

Three consequences worth knowing before touching it:

- **Freezing is load-bearing.** If those columns took gradient the parameterisation would drift
  back into a rotation of the per-type one-hot it replaces.
- **The frozen block is deliberately not injective.** Swordsman/Berserker/Mercenary,
  Knight/Pikeman and Ensign/Marshall share a frozen row on purpose; the learned block carries
  identity, and starting genuinely-similar types together is the prior being bought.
  `tests/test_unit_embedding.py` pins exactly which three collide, so a *new* collision — two
  unrelated types merged — fails loudly.
- **The royal coin has no row.** It is bag-only with no board unit, so there is no unit
  behaviour for the frozen columns to describe; its count passes through as a raw scalar.

Nothing is compressed: 10 + 6 = 16 = `NUM_UNIT_TYPES`, so `board_channels` (48) and `global_dim`
(245) come out unchanged, and the effective per-type weight `W_eff[o,t] = Σ_d W[o,d]·E[t,d]` is
still full rank. What changed is that 10 of each type's 16 degrees of freedom are tied to shared,
rules-derived columns that take gradient from *every* game. `ObsTypeEmbedding` recomputes both
widths rather than assuming them, because that equality is a coincidence of the chosen dims.

### A3 — FiLM the globals into the trunk

`policy_factored_v1` never let the trunk see the globals at all. `policy_head` was
`Conv2d(hidden + 245 → N_VERBS, k=1)` fed a `global_feats` tensor broadcast to all 49 cells, and
because the broadcast value is identical at every cell while a 1×1 conv applies the same weights
at every cell, that whole path collapses to one constant term `W_g @ g` added everywhere. The
globals contributed a position-independent bias and nothing else — 245 × 32 = 7840 weights to
compute one offset 49 times.

An additive bias cannot express what the game needs: *this channel does not matter with the hand
I am holding*. A Berserker threat cell is worth nothing when no Berserker coin is playable this
round, and switching a channel off requires multiplying it. So v2 applies
`MLP(globals) → (γ, β)` per channel after each conv block, `x ← x·(1 + γ) + β`, and `policy_head`
drops the global input block entirely. `facedown_head`/`verb_head` keep their globals: those read
a *pooled* vector, which is an ordinary MLP input rather than the per-cell broadcast at issue.

The conditioning MLP's output layer is zero-initialised, so at step 0 the module is exactly the
identity and a fresh v2 net starts from v1's trunk behaviour. The usual zero-init consequence
applies: on the first step only that output layer has a gradient and the layers below it start
moving from the second.

**The critic deliberately does not get FiLM.** `critic_v5` is `critic_v4` plus the A1 embedding
and nothing else. `board_only_head` reads `pool(trunk(board))` and exists precisely so its loss
cannot be satisfied from the globals — the `critic_v2` fix that took the positional-sibling tie
rate from 93 % to 0 % (`docs/next_iteration.md` §3.4). Conditioning the trunk on globals would
put them straight back inside that path and void it silently.
`test_v5_board_only_value_ignores_globals` is that invariant.

### No `OBS_VERSION` bump — and why the plan said otherwise

`IDEAS.md` A1 originally called for a new obs version. It was not needed and did not happen. The
contraction has to run *inside* the network, because the learned half of the table cannot exist in
the numpy encoder; and the raw 32 unit planes already *are* the per-type count tensor that
contraction consumes. So v11's output is byte-identical, `tests/test_obs_golden.py` is untouched,
pool snapshots stay compatible, and every prior checkpoint still loads on its recorded `arch`.
What v11 gained is purely descriptive metadata — `deck_block_offsets`, `unit_block_offsets`,
`deck_unit_positions`, `deck_royal_position` — so the network can read the flat 245-vector as
per-type blocks instead of 245 anonymous slots.

### Measured outcome (2026-08-18, `src/app/eval_a1_a3.py`)

First v2 run (`warchest_ppo_20260817-2102`) against the v1 baseline
(`warchest_ppo_20260810-0802`):

| claim | measurement | verdict |
|---|---|---|
| A3 conditions the board on the hand | 46.4 % of verbs change their top-1 cell when only the hand is substituted, vs **0.00 % for v1** | **confirmed** |
| FiLM escaped its zero init | mean \|γ\| 1.83 / 1.77 / 1.13; γ spread *across observations* 0.74 / 0.60 / 0.38 | **confirmed** |
| A1's table learns per-type identity | Knight/Pikeman separated to 2.24× init, Ensign/Marshall 1.47×, Swordsman/Berserker/Mercenary **1.15× (not separated)** | partial |
| either change buys strength | pooled head-to-head **+4.0 % [−3.9 %, +11.9 %]**, 300 decided games per arm | **not detected** |

The A3 row is unusually strong evidence because its control is arithmetic rather than
statistical: on v1 the answer is *provably* zero, since the broadcast globals shift all 49 logits
of a verb by the same constant and cancel in a softmax across cells. A non-zero v1 reading would
mean the measurement is broken, not the network.

Two measurement traps this run exposed, both of which produced confidently wrong answers before
being fixed, and both of which apply to any future head-to-head here:

1. **Ego-frame remap.** A policy's action id is in the rotated frame whenever P2 is to move and
   must go through `WarChestEnv.remap_action` (this is what `gauntlet.PolicyAgent.act` does).
   Skipping it does not raise — every P2 ply lands mirrored, fails validation and silently
   becomes a random legal move, which reads as "P1 wins nearly every game". `eval_a1_a3.py`
   now counts illegal plies and warns above 2 %.
2. **Forcing a composition onto one arm only** confounds "this net plays the deck worse" with
   "the deck is weak". Unmirrored, the `support` archetype read 0 % for v2; mirrored across both
   nets it showed v2 *ahead*. The `comps` subcommand deals each archetype to both arms.

## File reference

| File | Role |
|---|---|
| `src/app/ppo.py` | PPOTrainer class: collect, update, eval, log |
| `src/app/reinforce.py` | Legacy REINFORCE+GAE trainer (kept for reference, not the primary path) |
| `src/app/demo.py` | Evaluate saved model vs random + interactive replay |
| `src/app/eval_bucketed.py` | Per-composition eval bucketing (`docs/IDEAS.md` #R1) |
| `src/app/eval_a1_a3.py` | Did the A1 embedding / A3 FiLM pair change anything: `weights` (FiLM activity + embedding drift), `hand` (does the hand re-rank the board, with a provable v1 zero control), `comps` (mirrored forced-composition head-to-head) |
| `src/app/gauntlet.py` | Round-robin gauntlet CLI: load checkpoints + baselines, print WR matrix / Elo / transitivity |
| `src/app/main.py` | Minimal random-action smoke test |
| `src/services/environment/warchest_env.py` | Gymnasium env: reset, step, rewards, action encode/decode, rules-queries; delegates obs encoding |
| `src/services/environment/obs_encoders/` | Versioned observation encoders (`v10.py`, `v11.py` + registry); owns plane layout, normalizers, ego-rotation, feature derivation, and the per-type global block offsets the unit-type embedding reads |
| `src/services/environment/board.py` | Hex board, adjacency, base ownership |
| `src/services/environment/game_state.py` | State snapshot: bags/hands/discards/boxed/pending, used for replay |
| `src/services/environment/roster.py` | Single source of truth for all 16 unit types + Royal coin (id/icon/colour/total-coins) |
| `src/services/environment/game_renderer.py` | Matplotlib interactive game replay |
| `src/services/environment/coin_render.py` | Per-unit coin colours/glyphs for rendering |
| `src/services/environment/units/baseunit.py` | Unit class, generated per-type from `roster.py` |
| `src/services/environment/action.py` | Action dataclass |
| `src/services/environment/cell_ids.py` | Cell type constants |
| `src/services/policy/policy.py` | Policy (actor) and Critic networks, `HexConv2d`, `FiLM`; arch-versioned (`policy_factored_v1`/`v2`, `critic_v1`…`v5`); obs dims from the paired encoder |
| `src/services/policy/checkpoint.py` | Checkpoint (de)serialization with obs-version + arch metadata (legacy bare-`state_dict` fallback) |
| `src/services/policy/unit_embedding.py` | Shared unit-type table: 10 frozen `roster.py` attribute columns + 6 learned per type, and the board/global contractions that replace the one-hot type indices (`docs/IDEAS.md` A1) |
| `src/services/gauntlet.py` | Gauntlet agents (`act(env)` → absolute action), `round_robin`, Bradley-Terry/Elo, transitivity |
| `src/services/opponent_pool.py` | Weighted sampler: random / greedy / pool snapshots |
| `src/services/bots/base.py` | Bot ABC |
| `src/services/bots/random_bot.py` | Uniform-random valid-action bot |
| `src/services/bots/greedy_bot.py` | Priority attack → control → move → deploy → pass bot |
| `src/utils/rollout_buffer.py` | Transition storage + GAE computation |
| `src/utils/elo.py` | Elo rating tracker |
| `Dockerfile` | Container for cloud training |
| `launch-agent.yaml` | W&B Agents queue config |
