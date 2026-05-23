# Game Mechanics

## Board

7×7 hexagonal grid (~37 valid cells). Cell types:

| Constant | Value | Meaning |
|---|---|---|
| `INVALID` | -1 | Outside hex boundary |
| `EMPTY` | 0 | Traversable, no base |
| `UNCONTROLLED_BASE` | 1 | Unclaimed base |
| `PLAYER_1_BASE` | 2 | Base owned by player 1 |
| `PLAYER_2_BASE` | 3 | Base owned by player 2 |

### Initial positions

```
Player 1 bases  (yellow): (1,0), (4,1)
Player 2 bases  (blue):   (2,5), (5,6)
Unclaimed bases (green):  (0,1), (2,2), (5,3), (1,3), (4,4), (6,5)
Player 1 units: Swordsman @ (1,0) and (4,1)
Player 2 units: Swordsman @ (2,5) and (5,6)
```

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

A unit can move to any adjacent cell that is valid and unoccupied.

## Action space

Total 14 actions: 2 units × (6 move directions + 1 claim).

| Action IDs | Meaning |
|---|---|
| 0–5 | Unit 0 moves in directions 0–5 |
| 6–11 | Unit 1 moves in directions 0–5 |
| 12 | Unit 0 claims the base it stands on |
| 13 | Unit 1 claims the base it stands on |

Invalid actions are masked to −1e9 in the logits before softmax.

## Claiming bases

Requirements:
- Unit must occupy the base cell.
- Base must be uncontrolled or enemy-controlled.
- A player cannot claim their own base.

Effect:
- Ownership transfers immediately.

## Win / end conditions

| Condition | Outcome |
|---|---|
| Player controls 6 bases | That player wins (`WIN_REWARD = +1.0`) |
| Total actions reach 200 | Truncation; both players receive `LOSS_REWARD = −1.0` |

## Reward table

| Event | Reward |
|---|---|
| Win | +1.0 |
| Claim unclaimed base | +0.15 |
| Truncation (draw) | −1.0 |
| Invalid action attempt | −0.02 |
| Move onto unclaimed base | +0.005 |
| Move adjacent to unclaimed base | +0.001 |
| Each action taken | −0.002 |

Base approach rewards (`MOVE_ON_BASE_REWARD`, `MOVE_NEAR_BASE_REWARD`) fire on every qualifying move for the duration of the episode — the once-per-base flag was removed (fix #3).

## Turn order

Players alternate strictly: player 1 → player 2 → player 1 → …
`active_player` is stored in `GameState` and flips after every `step()`.
