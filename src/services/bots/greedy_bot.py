from collections import deque

import numpy as np

from .base import Bot
from ..environment.warchest_env import (
    N_VERBS, BOARD_DIM, WarChestEnv,
    MOVE_ACTION, ATTACK_ACTION, CLAIM_BASE_ACTION, DEPLOY_ACTION,
)

# Verb ranges in the spatial action scheme
_VERB_MOVE_END = 5
_VERB_ATTACK_START = 6
_VERB_ATTACK_END = 11
_VERB_CONTROL = 12
_VERB_DEPLOY = 13

OFFSETS = [(-1, -1), (-1, 0), (0, 1), (1, 1), (1, 0), (0, -1)]


class GreedyBot(Bot):
    """Priority: attack → control → move toward nearest base → random.

    Operates entirely in the ego-centric (rotated) observation frame it receives.
    Block boundaries are derived from N_VERBS/BOARD_DIM constants, not hardcoded
    action IDs, so they remain valid across action-space changes.
    """

    RANDOM_ACTION_PROB = 0.30

    def act(self, obs: dict) -> tuple[int, None, None]:
        valid = list(np.where(obs['valid_action_mask'] == 1)[0])
        board = obs['board']  # [8, 7, 7] ego-centric encoded

        if np.random.random() < self.RANDOM_ACTION_PROB:
            return int(np.random.choice(valid)), None, None

        # Attack: verb 6-11
        attacks = [a for a in valid if _VERB_ATTACK_START <= self._verb(a) <= _VERB_ATTACK_END]
        if attacks:
            return int(attacks[0]), None, None

        # Control (claim): verb 12
        controls = [a for a in valid if self._verb(a) == _VERB_CONTROL]
        if controls:
            return int(controls[0]), None, None

        # Move toward nearest target base (uncontrolled ch2 or opponent ch4)
        moves = [a for a in valid if self._verb(a) <= _VERB_MOVE_END]
        if moves:
            best = self._best_move_toward_base(moves, board)
            if best is not None:
                return int(best), None, None

        return int(np.random.choice(valid)), None, None

    @staticmethod
    def _verb(action_id: int) -> int:
        return action_id // (BOARD_DIM * BOARD_DIM)

    @staticmethod
    def _decode(action_id: int):
        verb = action_id // (BOARD_DIM * BOARD_DIM)
        cell = action_id % (BOARD_DIM * BOARD_DIM)
        return verb, cell // BOARD_DIM, cell % BOARD_DIM

    def _best_move_toward_base(self, moves, board):
        # Targets: uncontrolled (ch2) or opponent base (ch4)
        target_mask = (board[2] + board[4]) > 0
        rows, cols = np.where(target_mask)
        targets = list(zip(rows.tolist(), cols.tolist()))
        if not targets:
            return None

        best_action = None
        best_dist = float('inf')
        for action in moves:
            verb, r, q = self._decode(action)
            dr, dq = OFFSETS[verb]
            new_pos = (r + dr, q + dq)
            d = min(self._bfs(new_pos, t, board) for t in targets)
            if d < best_dist:
                best_dist = d
                best_action = action
        return best_action

    @staticmethod
    def _bfs(start, end, board) -> float:
        if start == end:
            return 0
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            pos, dist = queue.popleft()
            for dr, dq in OFFSETS:
                npos = (pos[0] + dr, pos[1] + dq)
                if npos == end:
                    return dist + 1
                r, q = npos
                if (
                    npos not in visited
                    and 0 <= r < BOARD_DIM
                    and 0 <= q < BOARD_DIM
                    and board[0, r, q] == 0  # not invalid
                ):
                    visited.add(npos)
                    queue.append((npos, dist + 1))
        return float('inf')
