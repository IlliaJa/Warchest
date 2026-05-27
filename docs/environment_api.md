# WarChestEnv API

`WarChestEnv` follows the [Gymnasium](https://gymnasium.farama.org/) interface.

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

Resets board to initial state, deploys units, clears exploration maps.
Returns observation dict and empty info dict.

### `step(action_id) → (obs, reward, terminated, truncated, info)`

Executes the action for the current player, swaps `active_player`, returns next observation.

- `terminated=True` when a player reaches 6 bases.
- `truncated=True` when `action_count >= max_actions` (200).

### `render()`

Displays the current board with matplotlib (non-blocking).

### `render_game()`

Opens an interactive matplotlib window with Previous / Next buttons and keyboard shortcuts (`←`/`→` or `A`/`D`) to step through recorded game history. Requires `save_game_history=True`.

## Observation dict

```python
{
    'board':             np.ndarray (7, 7)    # raw cell-id grid (INVALID=-1, EMPTY=0, ...)
    'exploration_map':   np.ndarray (7, 7)    # visit counts for active player
    'units':             np.ndarray (2, 2, 2) # [player_slot, unit_idx, (row, col)]
                                              # slot 0 = active player, slot 1 = opponent
    'global':            np.ndarray (3,)      # [turn // 2, my_bases, opp_bases]
    'valid_action_mask': np.ndarray (14,)     # 1 = legal action
    'active_player':     int                  # 1 or 2
}
```

The board is encoded into 6 channels by `Policy.encode_board()` before being fed to the network (the raw grid is returned in the obs; encoding happens in the policy forward pass).

## Action IDs

```
0– 5   unit 0 → move in hex directions 0–5
6–11   unit 1 → move in hex directions 0–5
12     unit 0 → claim base at its current location
13     unit 1 → claim base at its current location
```

`get_possible_actions()` returns the list of currently legal action IDs.

## Useful properties

| Property | Type | Description |
|---|---|---|
| `board` | `Board` | Current board object |
| `active_player` | `int` (1 or 2) | Whose turn it is |
| `action_count` | `int` | Total valid actions taken this episode |
| `action_space` | `gym.spaces.Discrete(14)` | Action space |
| `observation_space` | `gym.spaces.Dict` | Full observation schema |

## Board API

```python
board.get_adjacent_cells(r, q)           # → list[(r,q)] of valid neighbours
board.get_free_adjacent_cells(r, q)      # → only unoccupied neighbours
board.get_controlled_bases(player_id)    # → list[(r,q)] of player's bases
board.deploy_unit(unit, place)           # place unit on a controlled base
board.change_base_control(player_id, loc)   # transfer base ownership
board.is_valid_claim(player_id, loc)     # bool
```
