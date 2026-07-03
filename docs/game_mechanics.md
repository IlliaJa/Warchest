# Game Mechanics

This describes the current full base game (all 16 unit types + per-game disjoint drafting +
coin economy). For the exact observation/action schema see `docs/environment_api.md`; for
per-unit tactics/attributes see `docs/UNITS.md` and `docs/history.md` → "Tactics, attributes
& restrictions".

## Board

7×7 hexagonal grid (~37 valid cells). Cell types (`cell_ids.py`):

| Constant | Value | Meaning |
|---|---|---|
| `INVALID_CELL_ID` | -1 | Outside hex boundary |
| `EMPTY_CELL_ID` | 0 | Traversable, no base |
| `UNCONTROLLED_BASE_CELL_ID` | 1 | Unclaimed base |
| `CONTROLLED_BASE_PLAYER_1_CELL_ID` | 2 | Base owned by player 1 |
| `CONTROLLED_BASE_PLAYER_2_CELL_ID` | 3 | Base owned by player 2 |

### Initial setup

```
Player 1 bases  (control markers only): (1,0), (4,1)
Player 2 bases  (control markers only): (2,5), (5,6)
Unclaimed bases:                        (0,1), (2,2), (5,3), (1,3), (4,4), (6,5)
```

The board starts with **no units on it** — only the base control markers above. Before the
first turn, 8 distinct unit types are sampled and split 4/4 disjoint between the two players
(`set_init_state`), each player's bag/supply is built from their drafted composition, and each
player draws an opening hand of `HAND_SIZE = 3` coins. Units enter play by `deploy`-ing a hand
coin onto one of the player's own controlled bases.

## Hex movement

6 neighbour offsets:

```
(-1,-1)  top-left
(-1, 0)  top-right
( 0, 1)  right
( 1, 1)  bottom-right
( 1, 0)  bottom-left
( 0,-1)  left
```

A unit can move to any adjacent cell that is valid and unoccupied, unless its tactic overrides
normal movement (e.g. Lancer's `line_charge`, Light Cavalry/Royal Guard's `move_to`).

## The coin economy

Each player has a private **bag** (shuffled draw pool), **hand** (`HAND_SIZE = 3` coins drawn
per round), a **discard** (face-up + face-down), and a per-type **supply** (recruitable
reserve). A coin's lifecycle: `bag → hand → {board (deployed) | discard} → (reshuffle) → bag`.
Attacking a unit removes one of its coins to the **box** — a permanent, one-way exit from the
cycle (`GameState.boxed`, read via `WarChestEnv.boxed_total`). Units support a coin **stack**
(height = how many hits they can absorb); `bolster` adds a coin to a friendly stack.

Actions available with hand coins: `deploy` (place a new unit), `bolster`, `recruit` (buy a
supply coin, paying a hand coin), `claim_initiative`, `pass`, and — once a matching unit is on
the board — `move` / `attack` / `control` / `tactic`. See `docs/environment_api.md` for the
exact action-id layout.

## Action space

The action space is a **factored** verb × cell space (spatial) plus a small face-down block,
flattened to `Discrete(ACTION_SPACE_SIZE = 1875)`. Full layout, including the `tactic`/`select`
verbs that drive multi-step unit abilities: `docs/environment_api.md`.

Invalid actions are masked to −1e9 in the logits before softmax (`valid_action_mask`).

## Claiming bases (`control` verb)

Requirements:
- A unit must occupy the base cell.
- Base must be uncontrolled or enemy-controlled.
- A player cannot claim their own already-controlled base.

Effect: ownership transfers immediately (`CLAIM_BASE_REWARD = 0.0` — see `docs/rewards.md` for
why direct claim reward is deliberately zero).

## Win / end conditions

| Condition | Outcome |
|---|---|
| A player controls `winning_base_count = 6` bases | That player wins (`WIN_REWARD = +1.0`) |
| `round_number >= max_rounds` (50) | Truncation; main actor receives a base-diff-proportional terminal reward |

A **round** is one full pass where both players empty their hand (not a fixed step/action
count). The truncation terminal reward (added to the last main-actor step):

```python
if diff > 0:                    # drew from a strong position
    trunc_reward = 0.0
else:                           # tie or deficit
    deficit_frac = min(-diff, winning_base_count) / winning_base_count  # 0 at tie ... 1 at max deficit
    trunc_reward = LOSS_REWARD * (0.5 + 0.5 * deficit_frac)             # -0.5 (tie) ... -1.0 (rout)
```

## Rewards

See [Reward Design](rewards.md) for the full reward table, shaping terms, and unrealized ideas.

## Turn order

Players alternate, one coin-play at a time, with **initiative** determining who acts first
each round (randomised at setup; `claim_initiative` can transfer it at most once per round).
`active_player` is stored in `GameState` and flips after every non-pending `step()` — while a
multi-step tactic's `state.pending` is set, the turn does not pass until the continuation
resolves (see `docs/environment_api.md` → "Pending tactics").
