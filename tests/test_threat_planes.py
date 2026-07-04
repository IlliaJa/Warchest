"""Threat/reach planes (docs/IDEAS.md "the agent can't see the board as one
position", Part A). Covers the Berserker closed-form reach, the Cavalry/Lancer
charge geometry, ranged targeting, and the Marshall grant-activation path,
mostly against `_threat_contributions`/`_threat_berserker_reach`/`_threat_grids`
directly (bypassing bag/supply bookkeeping) since those are the novel,
error-prone pieces; one end-to-end test checks the `generate_observation` wiring.
"""
from collections import Counter

from src.services.environment.warchest_env import (
    BOARD_CHANNELS, THREAT_KINDS, OWN_THREAT_PLANE_BASE, ENEMY_THREAT_PLANE_BASE,
    ROW_COORD_PLANE, COL_COORD_PLANE, THREAT_NORM,
    OWN_BASE_REACH_PLANE, ENEMY_BASE_REACH_PLANE,
)
from _helpers import (
    blank_env, place, cavalry_scenario, archer_scenario,
    LANCER, CROSSBOW, BERSERKER, MARSHALL, SWORDSMAN,
)


def test_threat_channel_layout():
    assert BOARD_CHANNELS == 48
    assert THREAT_KINDS == ('melee', 'ranged', 'charge')
    assert OWN_THREAT_PLANE_BASE == 38
    assert ENEMY_THREAT_PLANE_BASE == 41
    assert ROW_COORD_PLANE == 44
    assert COL_COORD_PLANE == 45
    assert OWN_BASE_REACH_PLANE == 46
    assert ENEMY_BASE_REACH_PLANE == 47


# --------------------------------------------------------------------------- #
# Berserker closed-form: hits(D) = max(0, stack - D + 1)
# --------------------------------------------------------------------------- #

def test_berserker_reach_formula_unblocked():
    env = blank_env()
    origin = (0, 0)
    be = place(env, BERSERKER, player=1, loc=origin, stack=3)
    dists = env._hex_distances(origin, 4)
    reach = env._threat_berserker_reach(be)

    # Straight line (0,0)-(0,1)-(0,2)-(0,3) via offsets[2]=(0,1): distances 1,2,3.
    d1, d2, d3 = (0, 1), (0, 2), (0, 3)
    assert dists[d1] == 1 and dists[d2] == 2 and dists[d3] == 3
    assert reach[d1] == 3   # max(0, 3-1+1)
    assert reach[d2] == 2   # max(0, 3-2+1)
    assert reach[d3] == 1   # max(0, 3-3+1)
    # Nothing at distance 4 or beyond should carry a hit (stack=3 caps reach at D=3).
    assert all(reach.get(c, 0) == 0 for c, d in dists.items() if d >= 4)


def test_berserker_reach_reduced_by_blocked_path():
    env = blank_env()
    origin = (0, 0)
    be = place(env, BERSERKER, player=1, loc=origin, stack=3)
    # Block the only direct route toward (0,3) with an enemy sitting adjacent.
    place(env, SWORDSMAN, player=2, loc=(0, 1), stack=1)
    reach = env._threat_berserker_reach(be)
    # (0,3) needed a 3-step direct path; the detour around the block exceeds the
    # stack's move budget (stack-1=2 steps), so it must drop out entirely.
    assert reach.get((0, 3), 0) == 0


def test_berserker_contribution_kind_split_by_adjacency():
    env = blank_env()
    origin = (3, 3)
    be = place(env, BERSERKER, player=1, loc=origin, stack=3)
    contributions = env._threat_contributions(be)
    kinds_by_cell = {cell: kind for cell, kind, _ in contributions}
    adjacent = set(env.board.get_adjacent_cells(*origin))
    assert all(kinds_by_cell[c] == 'melee' for c in kinds_by_cell if c in adjacent)
    assert all(kinds_by_cell[c] == 'charge' for c in kinds_by_cell if c not in adjacent)
    assert any(kind == 'charge' for kind in kinds_by_cell.values())  # stack 3 reaches D=2,3


# --------------------------------------------------------------------------- #
# Cavalry / Lancer charge geometry
# --------------------------------------------------------------------------- #

def test_cavalry_cells_more_permissive_than_straight_line_charge():
    env, cav, _ = cavalry_scenario()
    A = cav.loc
    cavalry_cells = set(env._threat_cavalry_cells(A))
    straight_line_only = set(env._threat_charge_cells(A, max_dist=1))
    assert straight_line_only.issubset(cavalry_cells)
    assert len(cavalry_cells) > len(straight_line_only)  # off-axis attacks included

    contributions = env._threat_contributions(cav)
    charge_cells = {c for c, kind, _ in contributions if kind == 'charge'}
    melee_cells = {c for c, kind, _ in contributions if kind == 'melee'}
    assert charge_cells == cavalry_cells
    assert melee_cells == set(env.board.get_adjacent_cells(*A))  # Cavalry keeps its normal attack


def test_lancer_charge_cells():
    env = blank_env()
    loc = (3, 3)
    lancer = place(env, LANCER, player=1, loc=loc, stack=1)
    cells = env._threat_charge_cells(loc, max_dist=2)
    assert (3, 5) in cells  # straight line via offsets[2]=(0,1) twice: distance 3
    contributions = env._threat_contributions(lancer)
    assert all(kind == 'charge' for _, kind, _ in contributions)
    assert not any(kind == 'melee' for _, kind, _ in contributions)  # can_normal_attack=False


# --------------------------------------------------------------------------- #
# Ranged (Archer / Crossbowman)
# --------------------------------------------------------------------------- #

def test_archer_ranged_cells_any_direction():
    env, archer, _ = archer_scenario()
    contributions = env._threat_contributions(archer)
    assert all(kind == 'ranged' for _, kind, _ in contributions)
    dists = env._hex_distances(archer.loc, 2)
    expected = {c for c, d in dists.items() if d == 2}
    assert {c for c, _, _ in contributions} == expected


def test_crossbow_blocked_straight_line_excluded():
    env = blank_env()
    loc = (3, 3)
    cb = place(env, CROSSBOW, player=1, loc=loc, stack=1)
    far = (3, 5)  # distance 2 via offsets[2]=(0,1) twice
    cells_clear = env._threat_ranged_cells(loc, distance=2, straight_line=True)
    assert far in cells_clear
    place(env, SWORDSMAN, player=2, loc=(3, 4), stack=1)  # blocks the line
    cells_blocked = env._threat_ranged_cells(loc, distance=2, straight_line=True)
    assert far not in cells_blocked


# --------------------------------------------------------------------------- #
# Marshall grant_attack: activation via a different coin
# --------------------------------------------------------------------------- #

def test_marshall_grant_enables_unit_without_its_own_coin():
    env = blank_env()
    marshall_loc, sword_loc = (3, 3), (3, 5)  # hex-distance 2
    place(env, MARSHALL, player=1, loc=marshall_loc, stack=1)
    sword = place(env, SWORDSMAN, player=1, loc=sword_loc, stack=1)

    own_hand = Counter({MARSHALL: 1})  # no SWORDSMAN coin in hand
    grids = env._threat_grids(active=1, own_hand=own_hand, opp_hidden=Counter())
    melee = grids[(1, 'melee')]
    for cell in env.board.get_adjacent_cells(*sword_loc):
        assert melee[cell] >= 1

    grids_no_marshall = env._threat_grids(active=1, own_hand=Counter(), opp_hidden=Counter())
    assert grids_no_marshall[(1, 'melee')].sum() == 0  # neither own nor grant path available


def test_marshall_grant_triggers_full_berserker_chain():
    env = blank_env()
    marshall_loc, berserker_loc = (3, 3), (3, 5)
    place(env, MARSHALL, player=1, loc=marshall_loc, stack=1)
    place(env, BERSERKER, player=1, loc=berserker_loc, stack=3)

    own_hand = Counter({MARSHALL: 1})  # no Berserker coin in hand
    grids = env._threat_grids(active=1, own_hand=own_hand, opp_hidden=Counter())
    total_hits = sum(grids[(1, k)].sum() for k in THREAT_KINDS)
    # A single granted attack would be 1 hit; the full chain (stack=3) is more.
    assert total_hits > 1


# --------------------------------------------------------------------------- #
# End-to-end wiring: generate_observation rotation/indices/normalisation
# --------------------------------------------------------------------------- #

def test_generate_observation_wires_enemy_ranged_threat():
    env, archer, far_enemy = archer_scenario()  # P1 archer, active player 1
    obs = env.generate_observation()
    ranged_idx = THREAT_KINDS.index('ranged')
    own_ranged = obs['board'][OWN_THREAT_PLANE_BASE + ranged_idx]
    assert own_ranged.max() > 0.0
    assert own_ranged.max() <= 1.0 + 1e-6

    # Same scenario from P2's ego-centric view: the archer's threat now reads as
    # "enemy", and the board must be point-rotated (raw_board/expl treatment).
    env.state.active_player = 2
    obs2 = env.generate_observation()
    enemy_ranged = obs2['board'][ENEMY_THREAT_PLANE_BASE + ranged_idx]
    assert enemy_ranged.max() > 0.0
    # Rotation must move values, not create/destroy them: same total either way.
    assert own_ranged.sum() == enemy_ranged.sum()
