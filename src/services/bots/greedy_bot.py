from collections import deque

import numpy as np

from .base import Bot
from .board_geometry import N_CELLS, UNREACHABLE, distance_to
from ..environment.warchest_env import (
    BOARD_DIM, SPATIAL_SIZE, N_COIN_TYPES, TACTIC_VERB,
    CONTROL_VERB, BOLSTER_VERB, DEPLOY_VERB_BASE, ROYAL_COIN_IDX,
)

# Verb ranges in the spatial action scheme
_VERB_MOVE_END = 5
_VERB_ATTACK_START = 6
_VERB_ATTACK_END = 11
_VERB_CONTROL = CONTROL_VERB        # 12
_VERB_DEPLOY_START = DEPLOY_VERB_BASE  # 14
_VERB_DEPLOY_END = TACTIC_VERB - 1  # 29 — deploy block ends just before the tactic verb
# GreedyBot is a myopic yardstick: it never *initiates* a tactic (no V_TACTIC branch),
# but if random exploration parks it mid-tactic the continuation clicks reuse the
# move/attack verbs above, so it still plays them through.

# Face-down block layout (offsets from SPATIAL_SIZE): [0:C) claim, [C:2C) pass.
_PASS_OFFSET = N_COIN_TYPES         # pass block starts after the claim block
_ROYAL_IDX = ROYAL_COIN_IDX         # royal coin's contiguous index in DECK

OFFSETS = [(-1, -1), (-1, 0), (0, 1), (1, 1), (1, 0), (0, -1)]


class GreedyBot(Bot):
    """Priority: attack → control → move toward nearest base → deploy → pass.

    Operates entirely in the ego-centric (rotated) observation frame it receives.
    Face-down actions are a last resort; when forced to discard it prefers to pass
    with the Royal coin so unit coins are not wasted.
    """

    RANDOM_ACTION_PROB = 0.0

    def act(self, obs: dict) -> tuple[int, None, None]:
        valid = list(np.where(obs['valid_action_mask'])[0])
        board = obs['board']  # [C, 7, 7] ego-centric encoded

        if np.random.random() < self.RANDOM_ACTION_PROB:
            return int(np.random.choice(valid)), None, None

        spatial = [a for a in valid if a < SPATIAL_SIZE]

        attacks = [a for a in spatial if _VERB_ATTACK_START <= self._verb(a) <= _VERB_ATTACK_END]
        if attacks:
            return int(attacks[0]), None, None

        controls = [a for a in spatial if self._verb(a) == _VERB_CONTROL]
        if controls:
            return int(controls[0]), None, None

        moves = [a for a in spatial if self._verb(a) <= _VERB_MOVE_END]
        if moves:
            best = self._best_move_toward_base(moves, board)
            if best is not None:
                return int(best), None, None

        deploys = [a for a in spatial if _VERB_DEPLOY_START <= self._verb(a) <= _VERB_DEPLOY_END]
        if deploys:
            return int(deploys[0]), None, None

        # Only face-down actions remain — prefer passing the Royal coin.
        royal_pass = SPATIAL_SIZE + _PASS_OFFSET + _ROYAL_IDX
        if royal_pass in valid:
            return int(royal_pass), None, None
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
        """Lowest-id move whose destination is closest to a claimable base.

        One multi-source BFS from every target, then an index per candidate — the
        original ran a whole-board BFS per candidate move x per target, which is what
        made this bot as expensive as a policy forward (docs/IDEAS.md Table A). The
        selection is unchanged, including its tie-break (strict `<` over ascending
        action ids) and its "no reachable target -> None" behaviour, which the caller
        reads as "fall through to deploy"; `_bfs` below is kept as the reference
        implementation both properties are tested against.
        """
        # Targets: uncontrolled (ch2) or opponent base (ch4)
        target_mask = ((board[2] + board[4]) > 0).reshape(N_CELLS)
        if not target_mask.any():
            return None
        dist = distance_to(target_mask, board[0].reshape(N_CELLS) == 0)

        best_action = None
        best_dist = UNREACHABLE
        for action in moves:
            verb, r, q = self._decode(action)
            dr, dq = OFFSETS[verb]
            new_pos = (r + dr) * BOARD_DIM + (q + dq)
            d = dist[new_pos]
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
