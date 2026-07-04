"""OBS_VERSION 10 observation features (docs/observation_improvement.md +
docs/IDEAS.md "base-control reach planes"):

  - material-at-risk scalars   — min(hits, stack) reduction over on-board units
  - E_opp_hand                 — expected opponent hand composition (formula)
  - base-control reach grids   — _base_reach_grids / _maneuver_range / _is_claimable_base

The end-to-end wiring + normalization + placement of all three into obs['global']
is covered exhaustively by test_obs_global_vectorized (independent reimplementation,
full random game). These tests pin the helper semantics against hand-computed values,
since the equivalence reference calls the same helpers and would not catch a shared bug.
"""
from collections import Counter

import numpy as np

from src.services.environment.warchest_env import (
    THREAT_KINDS, UNCONTROLLED_BASE_CELL_ID,
    CONTROLLED_BASE_PLAYER_1_CELL_ID, CONTROLLED_BASE_PLAYER_2_CELL_ID,
)
from _helpers import (
    blank_env, place,
    SWORDSMAN, KNIGHT, BERSERKER, LIGHT_CAV, ROYAL_GUARD,
)

# Base cells (Board.default_bases)
P1_BASE = (1, 0)
P2_BASE = (2, 5)
UNCONTROLLED = (2, 2)


def _free_adjacent(env, cell):
    """A valid, currently-unoccupied neighbour of `cell`."""
    occupied = {u.loc for u in env.board.units}
    for adj in env.board.get_adjacent_cells(*cell):
        if adj not in occupied:
            return adj
    raise AssertionError(f'no free neighbour of {cell}')


# --------------------------------------------------------------------------- #
# _is_claimable_base — mirrors Board.is_valid_claim's cell test
# --------------------------------------------------------------------------- #

def test_is_claimable_base_by_side():
    env = blank_env()
    assert env.board.board[UNCONTROLLED] == UNCONTROLLED_BASE_CELL_ID
    # uncontrolled: claimable by both
    assert env._is_claimable_base(1, UNCONTROLLED)
    assert env._is_claimable_base(2, UNCONTROLLED)
    # P1's base: claimable only by P2
    assert env.board.board[P1_BASE] == CONTROLLED_BASE_PLAYER_1_CELL_ID
    assert not env._is_claimable_base(1, P1_BASE)
    assert env._is_claimable_base(2, P1_BASE)
    # P2's base: claimable only by P1
    assert env.board.board[P2_BASE] == CONTROLLED_BASE_PLAYER_2_CELL_ID
    assert env._is_claimable_base(1, P2_BASE)
    assert not env._is_claimable_base(2, P2_BASE)
    # a non-base empty cell is not claimable by anyone
    assert not env._is_claimable_base(1, (3, 3))
    assert not env._is_claimable_base(2, (3, 3))


# --------------------------------------------------------------------------- #
# _maneuver_range — normal 1, move_to/royal_move max_dist, Berserker stack
# --------------------------------------------------------------------------- #

def test_maneuver_range_per_unit():
    env = blank_env()
    assert env._maneuver_range(place(env, SWORDSMAN, 1, (3, 3))) == 1
    assert env._maneuver_range(place(env, LIGHT_CAV, 1, (0, 0))) == 2   # move_to max_dist=2
    assert env._maneuver_range(place(env, ROYAL_GUARD, 1, (6, 6))) == 2  # royal_move max_dist=2
    assert env._maneuver_range(place(env, BERSERKER, 1, (5, 5), stack=4)) == 4  # stack chain


# --------------------------------------------------------------------------- #
# _base_reach_grids
# --------------------------------------------------------------------------- #

def test_base_reach_adjacent_and_gated():
    env = blank_env(active=1)
    adj = _free_adjacent(env, UNCONTROLLED)
    place(env, SWORDSMAN, 1, adj, stack=1)
    # holding the coin: the adjacent uncontrolled base is reachable this turn
    grids = env._base_reach_grids(1, Counter({SWORDSMAN: 1}), Counter())
    assert grids[1][UNCONTROLLED] == 1.0
    # no coin in hand: cannot activate the unit, so not reachable
    grids_nocoin = env._base_reach_grids(1, Counter(), Counter())
    assert grids_nocoin[1][UNCONTROLLED] == 0.0


def test_base_reach_claim_in_place():
    """A unit already standing on a claimable base (dist 0) counts as reachable."""
    env = blank_env(active=1)
    place(env, SWORDSMAN, 1, UNCONTROLLED, stack=1)
    grids = env._base_reach_grids(1, Counter({SWORDSMAN: 1}), Counter())
    assert grids[1][UNCONTROLLED] == 1.0


def test_base_reach_own_base_not_claimable():
    env = blank_env(active=1)
    adj = _free_adjacent(env, P1_BASE)
    place(env, SWORDSMAN, 1, adj, stack=1)
    grids = env._base_reach_grids(1, Counter({SWORDSMAN: 1}), Counter())
    assert grids[1][P1_BASE] == 0.0  # already mine — nothing to claim


def test_base_reach_enemy_flip_gated_by_hidden():
    """Enemy reach onto my base is gated by the opponent hidden-pool count."""
    env = blank_env(active=1)
    adj = _free_adjacent(env, P1_BASE)
    place(env, KNIGHT, 2, adj, stack=1)
    # opponent holds a Knight somewhere hidden -> can flip my base
    grids = env._base_reach_grids(1, Counter(), Counter({KNIGHT: 1}))
    assert grids[2][P1_BASE] == 1.0
    # opponent has no hidden Knight -> cannot
    grids_none = env._base_reach_grids(1, Counter(), Counter())
    assert grids_none[2][P1_BASE] == 0.0


def test_base_reach_blocked_by_occupant():
    """A base reachable only through an occupied cell is not reachable (move blocked)."""
    env = blank_env(active=1)
    # Put my mover 2 hexes from the base with the single intervening cell blocked.
    adj = _free_adjacent(env, UNCONTROLLED)          # neighbour of the base
    beyond = _free_adjacent(env, adj)                # a neighbour of that neighbour
    if beyond == UNCONTROLLED:                       # ensure we're strictly farther out
        beyond = [c for c in env.board.get_adjacent_cells(*adj)
                  if c != UNCONTROLLED and c not in {u.loc for u in env.board.units}][0]
    place(env, KNIGHT, 2, adj, stack=1)              # blocker on the only 1-step path
    mover = place(env, SWORDSMAN, 1, beyond, stack=1)
    assert env._maneuver_range(mover) == 1           # 1-step mover, base is 2 away now
    grids = env._base_reach_grids(1, Counter({SWORDSMAN: 1}), Counter())
    assert grids[1][UNCONTROLLED] == 0.0


# --------------------------------------------------------------------------- #
# material-at-risk reduction: min(hits, stack), capped per unit
# --------------------------------------------------------------------------- #

def test_material_at_risk_caps_at_stack():
    env = blank_env(active=1)
    origin = (3, 3)
    fragile = place(env, SWORDSMAN, 1, origin, stack=1)
    adj = _free_adjacent(env, origin)
    place(env, BERSERKER, 2, adj, stack=3)  # could land 3 melee hits on `origin`

    threat = env._threat_grids(1, Counter({SWORDSMAN: 1}), Counter({BERSERKER: 1}))
    enemy_hits = sum(threat[(2, k)] for k in THREAT_KINDS)
    assert enemy_hits[origin] >= 3, 'berserker stack-3 should threaten >=3 hits adjacent'

    own_at_risk = sum(min(enemy_hits[u.loc], u.stack)
                      for u in env.board.units if u.player_id == 1)
    assert own_at_risk == 1.0  # capped at the stack-1 unit's own height

    # Same threat, a stack-3 defender loses all 3 (min(3, 3)).
    fragile.stack = 3
    own_at_risk = sum(min(enemy_hits[u.loc], u.stack)
                      for u in env.board.units if u.player_id == 1)
    assert own_at_risk == 3.0


# --------------------------------------------------------------------------- #
# E_opp_hand formula (hypergeometric mean)
# --------------------------------------------------------------------------- #

def test_e_opp_hand_formula_properties():
    hidden = np.array([3.0, 2.0, 1.0, 1.0, 0.0])  # total 7
    total = hidden.sum()
    # sums to hand size, and decays to 0 when the hand is empty
    for hand_size in (0, 1, 2, 3):
        e = hidden * hand_size / total
        assert abs(e.sum() - hand_size) < 1e-9
    # single-copy type: expected count == probability that coin is in hand
    hand_size = 3
    e = hidden * hand_size / total
    assert abs(e[2] - hand_size / total) < 1e-9  # the "1 copy" entry
    # equals the hidden vector (scaled) when hand size == pool size
    e_full = hidden * total / total
    assert np.allclose(e_full, hidden)
