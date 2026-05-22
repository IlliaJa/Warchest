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
- Opponent receives a −15 penalty to maintain zero-sum balance.

## Win / end conditions

| Condition | Outcome |
|---|---|
| Player controls 6 bases | That player wins (+500 reward) |
| Total actions reach 500 | Truncation; both players receive −500 |

## Reward table

| Event | Reward |
|---|---|
| Win | +500 |
| Claim unclaimed base | +15 |
| Opponent claims base | −15 |
| Truncation (draw) | −500 |
| Invalid action attempt | −10 |
| First move onto unclaimed base | +2.5 |
| First move adjacent to unclaimed base | +0.5 |
| Each action taken | −0.002 |

Exploration rewards (`MOVE_ON_BASE_REWARD`, `MOVE_NEAR_BASE_REWARD`) are only issued for the first `max_rewardable_moving_action = 30` steps to encourage early spread.

## Turn order

Players alternate strictly: player 1 → player 2 → player 1 → …
`active_player` is stored in `GameState` and flips after every `step()`.
