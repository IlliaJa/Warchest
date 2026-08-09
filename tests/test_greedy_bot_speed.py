"""GreedyBot's march was a whole-board BFS per candidate move x per target, which is
what made it cost as much as a policy forward (docs/IDEAS.md Table A). It now runs one
multi-source BFS and indexes it.

GreedyBot is a training-pool opponent, a gauntlet entrant and the yardstick every
historical number is quoted against, so the refactor is only admissible if it changes
*nothing* about which action comes out. These tests pin that against the original
implementation, which is kept in the class as `_bfs`.
"""
import numpy as np

from src.services.environment.warchest_env import WarChestEnv, BOARD_DIM
from src.services.bots.board_geometry import N_CELLS, UNREACHABLE, distance_to
from src.services.bots.greedy_bot import GreedyBot, OFFSETS


class ReferenceGreedyBot(GreedyBot):
    """GreedyBot with the pre-refactor march: one BFS per candidate per target."""

    def _best_move_toward_base(self, moves, board):
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


def _decision_states(games=6, seed=11):
    """Real decision observations, reached by random legal play."""
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(games):
        env = WarChestEnv(save_game_history=False)
        obs, _ = env.reset()
        for _ in range(1000):
            states.append(obs)
            legal = env.get_possible_actions()
            obs, _, terminated, truncated, _ = env.step(legal[rng.integers(len(legal))])
            if terminated or truncated:
                break
    return states


def test_picks_the_same_action_as_the_reference_everywhere():
    states = _decision_states()
    assert len(states) > 500
    fast, reference = GreedyBot(), ReferenceGreedyBot()
    for i, obs in enumerate(states):
        np.random.seed(i)  # the face-down fallback may draw randomly; pin both draws
        a, _, _ = fast.act(obs)
        np.random.seed(i)
        b, _, _ = reference.act(obs)
        assert a == b, f'state {i}: {a} != {b}'


def test_multi_source_bfs_matches_the_per_target_bfs():
    """The distance metric itself, cell by cell — including the unreachable case, which
    the caller reads as 'no target, fall through to deploy'."""
    env = WarChestEnv()
    env.reset()
    board = env.generate_observation()['board']
    targets = ((board[2] + board[4]) > 0).reshape(N_CELLS)
    dist = distance_to(targets, board[0].reshape(N_CELLS) == 0)

    target_cells = [(int(i) // BOARD_DIM, int(i) % BOARD_DIM) for i in np.flatnonzero(targets)]
    assert target_cells
    for i in range(N_CELLS):
        cell = (i // BOARD_DIM, i % BOARD_DIM)
        if board[0][cell] > 0:  # invalid cells are never a move destination
            continue
        expected = min(GreedyBot._bfs(cell, t, board) for t in target_cells)
        got = dist[i] if dist[i] < UNREACHABLE else float('inf')
        assert got == expected, f'{cell}: {got} != {expected}'


def test_no_targets_still_falls_through():
    """With every base already controlled by the mover there is nothing to march to, and
    the rung must return None rather than a distance-0 nonsense move."""
    env = WarChestEnv()
    env.reset()
    board = env.generate_observation()['board'].copy()
    board[2] = 0
    board[4] = 0
    assert GreedyBot()._best_move_toward_base([0, 1, 2], board) is None
