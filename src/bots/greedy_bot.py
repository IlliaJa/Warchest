from collections import deque

import numpy as np

from .base import Bot

OFFSETS = [(-1, -1), (-1, 0), (0, 1), (1, 1), (1, 0), (0, -1)]
BOARD_SIZE = 7
INVALID = -1
UNCONTROLLED_BASE = 1
PLAYER1_BASE = 2
PLAYER2_BASE = 3
CLAIM_START = 12  # move actions occupy IDs 0-11, claims start at 12


class GreedyBot(Bot):
    """Claims immediately if possible; otherwise BFS-moves toward the nearest unclaimed or enemy base."""

    # --- temporary handicap: remove this block once the game grows more complex ---
    RANDOM_ACTION_PROB = 0.30
    # ------------------------------------------------------------------------------

    def act(self, obs: dict) -> tuple[int, None, None]:
        board = obs['board']
        units = obs['units']  # (2, max_units, 2): [player_slot, unit_slot, (row, col)]
        active = int(obs['active_player'])
        valid = list(np.where(obs['valid_action_mask'] == 1)[0])

        # --- temporary handicap ---
        if np.random.random() < self.RANDOM_ACTION_PROB:
            return np.random.choice(valid), None, None
        # --------------------------

        claims = [a for a in valid if a >= CLAIM_START]
        if claims:
            return claims[0], None, None

        enemy_base = PLAYER2_BASE if active == 1 else PLAYER1_BASE
        rows, cols = np.where((board == UNCONTROLLED_BASE) | (board == enemy_base))
        targets = list(zip(rows.tolist(), cols.tolist()))

        if not targets:
            return np.random.choice(valid), None, None

        my_units = [tuple(units[0, i]) for i in range(units.shape[1])]

        best_action = None
        best_dist = float('inf')

        for unit_idx, unit_pos in enumerate(my_units):
            dists = [self._bfs(unit_pos, t, board) for t in targets]
            nearest = targets[int(np.argmin(dists))]

            move_base = unit_idx * 6
            for action in (a for a in valid if move_base <= a < move_base + 6):
                dr, dc = OFFSETS[action - move_base]
                new_pos = (unit_pos[0] + dr, unit_pos[1] + dc)
                d = self._bfs(new_pos, nearest, board)
                if d < best_dist:
                    best_dist = d
                    best_action = action

        if best_action is None:
            best_action = np.random.choice(valid)

        return best_action, None, None

    def _bfs(self, start, end, board):
        if start == end:
            return 0
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            pos, dist = queue.popleft()
            for dr, dc in OFFSETS:
                npos = (pos[0] + dr, pos[1] + dc)
                if npos == end:
                    return dist + 1
                if (
                    npos not in visited
                    and 0 <= npos[0] < BOARD_SIZE
                    and 0 <= npos[1] < BOARD_SIZE
                    and board[npos] != INVALID
                ):
                    visited.add(npos)
                    queue.append((npos, dist + 1))
        return float('inf')
