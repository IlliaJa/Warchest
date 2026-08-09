"""Hex-grid geometry over the observation's board planes, shared by the obs-only bots.

The 7x7 layout and its neighbour table are identical for every board ever built, so
they are computed once at import instead of per decision. `distance_to` is the piece
that matters: a bot that wants "how far is this cell from the nearest claimable base"
for *many* candidate cells pays one BFS here, not one per candidate per target — the
difference between `GreedyBot`'s original 0.90 ms per decision and 0.16 ms
(docs/IDEAS.md Table A, and B5 § Measured for the same fix in `ThreatAwareGreedyBot`).
"""
from collections import deque

import numpy as np

from ..environment.board import Board
from ..environment.warchest_env import BOARD_DIM

N_CELLS = BOARD_DIM * BOARD_DIM
# Larger than any real distance, and finite so it can sit in an int array. Callers
# comparing against it should treat >= UNREACHABLE as "no path".
UNREACHABLE = 10 ** 6

# `STEP[d, i]` is the flat index of cell i's neighbour in direction d, or -1 if that
# step leaves the 7x7 grid. The direction table is the board's own, so a spatial verb
# means here exactly what it means to the env.
STEP = np.full((len(Board.offsets), N_CELLS), -1, dtype=np.int64)
for _d, (_dr, _dq) in enumerate(Board.offsets):
    for _r in range(BOARD_DIM):
        for _q in range(BOARD_DIM):
            _nr, _nq = _r + _dr, _q + _dq
            if 0 <= _nr < BOARD_DIM and 0 <= _nq < BOARD_DIM:
                STEP[_d, _r * BOARD_DIM + _q] = _nr * BOARD_DIM + _nq

# Flat cell index -> its in-grid neighbours, for the BFS.
NEIGHBOURS = [[int(STEP[d, i]) for d in range(len(Board.offsets)) if STEP[d, i] >= 0]
              for i in range(N_CELLS)]


def distance_to(targets, passable):
    """Hex-step distance from every cell to the nearest target, in one BFS.

    `targets` and `passable` are flat [N_CELLS] boolean arrays; steps may only pass
    through cells that are `passable` (in practice: not the INVALID plane). Cells with
    no path read `UNREACHABLE`.

    Equivalent to `min over targets of a per-pair BFS` — `tests/test_greedy_bot_speed.py`
    pins that against `GreedyBot._bfs`, which is the reference implementation this
    replaced and is kept for exactly that comparison.
    """
    dist = np.full(N_CELLS, UNREACHABLE, dtype=np.int32)
    queue = deque()
    for i in np.flatnonzero(targets):
        dist[i] = 0
        queue.append(int(i))
    while queue:
        i = queue.popleft()
        d = dist[i] + 1
        for j in NEIGHBOURS[i]:
            if passable[j] and dist[j] > d:
                dist[j] = d
                queue.append(j)
    return dist
