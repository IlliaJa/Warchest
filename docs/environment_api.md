# WarChestEnv API

`WarChestEnv` follows the [Gymnasium](https://gymnasium.farama.org/) interface. This
describes the current full-base-game schema (`OBS_VERSION = 10`, `ACTION_SPACE_SIZE = 1875`).
All named constants below are defined in `src/services/environment/warchest_env.py` unless
noted otherwise.

## Construction

```python
from src.services.environment.warchest_env import WarChestEnv

env = WarChestEnv(save_game_history=True, debug_mode=False)
```

| Argument | Default | Purpose |
|---|---|---|
| `save_game_history` | `False` | Store every `GameState` snapshot; required for `render_game()` |
| `debug_mode` | `False` | Print verbose step info to stdout |

## Gymnasium methods

### `reset() → (obs, info)`

Starts a new game: builds a fresh `Board` (control markers only, no units on it yet),
samples 8 distinct unit types and gives 4 disjoint types to each player (`set_init_state`),
builds each player's bag/supply from their drafted composition, randomly assigns initiative,
and draws each player's opening hand. Returns the observation dict and an empty info dict.

### `step(action_id) → (obs, reward, terminated, truncated, info)`

Executes the action for the current player. Multi-step tactics do **not** advance the turn
while `state.pending` is set — `step()` keeps returning to the same acting player with a
narrowed legal-action set until the continuation resolves (see "Pending tactics" below).

- `terminated=True` when a player controls `winning_base_count = 6` bases.
- `truncated=True` when `state.round_number >= max_rounds` (50). A round is one full pass
  where both players empty their 3-coin hand — **not** a fixed action count.

### `render()`

Displays the current board with matplotlib (non-blocking).

### `render_game()`

Opens an interactive matplotlib window with Previous / Next buttons and keyboard shortcuts
(`←`/`→` or `A`/`D`) to step through recorded game history. Requires `save_game_history=True`.

## Observation dict

```python
{
    'board':             np.ndarray (BOARD_CHANNELS, 7, 7)  # float32, [0,1] — see below
    'global':            np.ndarray (GLOBAL_DIM,)           # float32, [0,1] — coin/round/pending features
    'valid_action_mask': np.ndarray (ACTION_SPACE_SIZE,)    # 1.0 = legal action, else 0.0
    'active_player':     int                                # 1 or 2
}
```

`BOARD_CHANNELS = 48`, `GLOBAL_DIM = 211`, `ACTION_SPACE_SIZE = 1875` for the current schema
(bump `OBS_VERSION` — currently `10` — whenever any of these change). Board planes and global
layout are documented in full in `docs/policy_network.md` (board encoder / global features
sections); in short: 6 base/terrain planes, 16 own + 16 opponent per-unit-type stack planes,
6 threat planes (own/enemy × melee/ranged/charge), 2 static coordinate planes, and **2
base-control reach planes** (own/enemy: base cells a side could move onto and claim this turn).
The global vector adds (OBS_VERSION 10) 2 material-at-risk scalars, a 17-wide expected-opponent-hand
vector, and 3 base-control reach scalars — see `docs/observation_improvement.md`. A
`PRIV_DIM = 51`-wide privileged (critic-only) opponent hidden-coin vector is obtained separately
(not part of the public `generate_observation()` dict — see `Critic.value_single`).

For `active_player == 2` the whole observation is rotated 180° (board planes, `row_coord`/
`col_coord`, base/initiative feature order) so the network always sees "my units" as player 1
would — `WarChestEnv.remap_action` performs the matching inverse remap on any action id chosen
against a P2 observation before calling `step()`.

## Action space

The action space is **factored**: a spatial block (verb × cell) followed by a face-down
(non-spatial) block, flattened into a single `Discrete(ACTION_SPACE_SIZE)` for Gymnasium
compatibility. `get_possible_actions()` returns the list of currently legal flat ids (this is
what `valid_action_mask` is built from); `VERB_OF_ACTION[action_id]` gives the verb group (one
of `N_FACTORED_VERBS = 11`) that the factored policy head reads.

### Spatial block — `action_id = verb * 49 + r * 7 + q` (`SPATIAL_SIZE = 1568`)

| Verb(s) | Meaning |
|---|---|
| 0–5 | Move the unit on this cell in hex direction 0–5 |
| 6–11 | Attack from this cell in hex direction 0–5 |
| 12 | `control` — claim/steal the base at this cell |
| 13 | `bolster` — add a coin to the stack of the unit on this cell |
| 14–29 | `deploy` — one verb per deployable unit type (`DEPLOY_VERBS`, 16 types); target cell must be a controlled, empty base |
| 30 | `tactic` — the unit on this cell initiates its tactic (opens a pending sub-turn; see below) |
| 31 | `select` — pick cell `(r, q)` as a non-directional **target** (ranged-attack target, friendly-grant recipient); only ever legal mid-tactic |

### Face-down block (no board cell) — appended after the spatial block, over the full 17-coin universe (16 units + Royal)

| Offset (from `SPATIAL_SIZE`) | Meaning |
|---|---|
| `[0, 17)` | `claim_initiative`, paying hand coin `c` |
| `[17, 34)` | `pass`, discarding hand coin `c` |
| `[34, 34 + 16·17)` | `recruit` — take supply unit type `t`, paying hand coin `c` |
| last slot | `decline` — end an optional pending continuation (no coin) |

Only the player's actually-drafted coins/types are ever unmasked at runtime; the full
17/16-wide blocks exist so the schema doesn't change per composition.

### Pending tactics (multi-step actions)

Tactics with a follow-up (Cavalry's move-then-attack, Archer's ranged attack, Marshall's
grant, etc.) are **not** single atomic ids. `tactic` (verb 30) only *initiates* — the
follow-up click(s) reuse verbs 0–11/31 and are gated by `state.pending` inside
`get_possible_actions()`, so the turn does not pass until the continuation resolves. The
`global` observation carries a `PENDING_CTX_DIM = 15`-wide one-hot (14 named pending kinds +
"no pending") telling the policy which continuation, if any, is in progress — this is what
disambiguates "a Cavalry follow-up move" from an ordinary maneuver using the same verb 0–5 ids.
See `docs/history.md` → "Tactics, attributes & restrictions" for the full per-unit mechanic
list and `PENDING_KINDS` in `warchest_env.py` for the exact kind names.

## Useful properties

| Property | Type | Description |
|---|---|---|
| `board` | `Board` | Current board object (`state.board`) |
| `active_player` | `int` (1 or 2) | Whose turn it is |
| `action_count` | `int` | Total valid actions taken this episode |
| `boxed_total(player_id)` | `int` | Coins `player_id` has permanently lost to the box (used by the material PBRS term — `docs/rewards.md`) |
| `get_observation_space()` | `gym.spaces.Dict` | Full observation schema (see above) |

## Board API

```python
board.get_adjacent_cells(r, q)           # → list[(r,q)] of valid neighbours
board.get_free_adjacent_cells(r, q)      # → only unoccupied neighbours
board.get_controlled_bases(player_id)    # → list[(r,q)] of player's bases
board.deploy_unit(unit, place)           # place unit on a controlled base
board.change_base_control(player_id, loc)   # transfer base ownership
board.is_valid_claim(player_id, loc)     # bool
```
