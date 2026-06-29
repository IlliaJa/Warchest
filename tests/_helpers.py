"""Shared test helpers: env drivers, coin accounting, and tactic scenario builders.

The domain-organized test files import from here so scenario setup and invariants
live in one place. The tactic scenario builders especially multiply as Phase 4
adds more units, so they are kept here rather than copied per file.
"""
from collections import Counter

import numpy as np
import torch

from src.services.environment.warchest_env import WarChestEnv
from src.services.environment.units import UNIT_CLASS_BY_ID

# Coin ids (roster.py)
SWORDSMAN = 1
KNIGHT = 2
CAV = 3            # Cavalry
LIGHT_CAV = 4
LANCER = 5
ARCHER = 6
CROSSBOW = 7
BERSERKER = 8
FOOTMAN = 9
PIKEMAN = 10
ENSIGN = 11
MARSHALL = 12
MERCENARY = 13
SCOUT = 14
ROYAL_GUARD = 15
WARRIOR_PRIEST = 16
ROYAL = 17         # the Royal coin (no board unit)


def blank_env(active=1, initiative=1):
    """An env with an empty board and all coin zones cleared, ready to place a scenario.

    Compositions are left empty; set them in the test if you need owned()/observation
    consistency. active/initiative default to player 1.
    """
    env = WarChestEnv()
    env.reset()
    s = env.state
    s.active_player = active
    s.initiative_owner = initiative
    s.pending = None
    s.hands = {1: Counter(), 2: Counter()}
    s.bags = {1: Counter(), 2: Counter()}
    s.discard_faceup = {1: Counter(), 2: Counter()}
    s.discard_facedown = {1: Counter(), 2: Counter()}
    s.supply = {1: Counter(), 2: Counter()}
    s.boxed = {1: Counter(), 2: Counter()}
    env.board.units = []
    return env


def place(env, unit_id, player, loc, stack=1):
    """Place a fresh unit of `unit_id` for `player` at `loc` with the given stack height."""
    u = UNIT_CLASS_BY_ID[unit_id](player_id=player, board=env.board)
    u.place_on_board(loc)
    u.stack = stack
    env.board.units.append(u)
    return u


# --------------------------------------------------------------------------- #
# Generic env drivers / accounting
# --------------------------------------------------------------------------- #

def drive(env, n, seed=0):
    """Take up to n random steps; return True if the game ended."""
    np.random.seed(seed)
    for _ in range(n):
        _, _, t, tr, _ = env.make_random_step()
        if t or tr:
            return True
    return False


def find_action(env, kind):
    """First legal action of a given action-type, with its decoded args (or None, None)."""
    for a in env.get_possible_actions():
        k, args = env.get_action_info(a)
        if k == kind:
            return a, args
    return None, None


def zone_plus_board(env, pid):
    """Per-type coin count across every zone + on-board stacks (conservation check)."""
    c = Counter()
    s = env.state
    for z in (s.hands[pid], s.bags[pid], s.discard_faceup[pid],
              s.discard_facedown[pid], s.supply[pid], s.boxed[pid]):
        c += z
    for u in env.board.units:
        if u.player_id == pid:
            c[u.id] += u.stack
    return c


def obs_after(steps, seed=0):
    """An observation after `steps` uniform-random actions (resets on terminal)."""
    env = WarChestEnv()
    env.reset()
    np.random.seed(seed)
    for _ in range(steps):
        _, _, t, tr, _ = env.step(int(np.random.choice(env.get_possible_actions())))
        if t or tr:
            env.reset()
    return env.generate_observation()


def batch_from_obs(obs, action, device):
    """Single-row batch dict for Policy.evaluate_actions_batch."""
    return {
        'board': torch.tensor(obs['board'], dtype=torch.float32).unsqueeze(0).to(device),
        'global': torch.tensor(obs['global'], dtype=torch.float32).unsqueeze(0).to(device),
        'mask': torch.tensor(obs['valid_action_mask'].astype(bool)).unsqueeze(0).to(device),
        'actions': torch.tensor([action], dtype=torch.long).to(device),
    }


# --------------------------------------------------------------------------- #
# Cavalry scenario: move-then-attack on a deterministic board.
#   A=(3,3) cavalry (P1)   B=(3,4) free move target   C=(2,4) enemy adjacent to B
# --------------------------------------------------------------------------- #

A, B, C = (3, 3), (3, 4), (2, 4)
MOVE_DIR_A_TO_B = 2   # offsets[2] = (0, +1):  (3,3) -> (3,4)
ATK_DIR_B_TO_C = 1    # offsets[1] = (-1, 0):  (3,4) -> (2,4)


def cavalry_scenario():
    env = WarChestEnv()
    env.reset()
    s = env.state
    s.compositions = {1: (CAV,), 2: (1,)}
    s.active_player = 1
    s.initiative_owner = 1
    s.pending = None
    s.hands = {1: Counter({CAV: 1}), 2: Counter()}
    # Non-empty bags so the round restart after the tactic redraws a playable hand.
    s.bags = {1: Counter({CAV: 1}), 2: Counter({1: 1})}
    s.discard_faceup = {1: Counter(), 2: Counter()}
    s.discard_facedown = {1: Counter(), 2: Counter()}
    s.supply = {1: Counter(), 2: Counter()}
    s.boxed = {1: Counter(), 2: Counter()}

    env.board.units = []
    cav = UNIT_CLASS_BY_ID[CAV](player_id=1, board=env.board)
    cav.place_on_board(A)
    env.board.units.append(cav)
    enemy = UNIT_CLASS_BY_ID[1](player_id=2, board=env.board)
    enemy.place_on_board(C)
    enemy.stack = 1
    env.board.units.append(enemy)
    return env, cav, enemy


# --------------------------------------------------------------------------- #
# Archer scenario: the SELECT primitive driving a one-step ranged attack.
#   AR=(3,3) archer (P1)   FAR=(3,5) enemy exactly 2 away   ADJ=(3,4) adjacent enemy
# Archer targets any enemy at distance 2 (no straight-line/blocker constraint) and
# has the "no normal attack" restriction.
# --------------------------------------------------------------------------- #

AR, FAR, ADJ = (3, 3), (3, 5), (3, 4)
ATK_DIR_AR_TO_ADJ = 2  # offsets[2] = (0,+1): (3,3) -> (3,4), the adjacent enemy


def archer_scenario(adjacent_enemy=False):
    env = WarChestEnv()
    env.reset()
    s = env.state
    s.compositions = {1: (ARCHER,), 2: (1, 2)}
    s.active_player = 1
    s.initiative_owner = 1
    s.pending = None
    s.hands = {1: Counter({ARCHER: 1}), 2: Counter()}
    s.bags = {1: Counter({ARCHER: 1}), 2: Counter({1: 1})}
    s.discard_faceup = {1: Counter(), 2: Counter()}
    s.discard_facedown = {1: Counter(), 2: Counter()}
    s.supply = {1: Counter(), 2: Counter()}
    s.boxed = {1: Counter(), 2: Counter()}

    env.board.units = []
    archer = UNIT_CLASS_BY_ID[ARCHER](player_id=1, board=env.board)
    archer.place_on_board(AR)
    env.board.units.append(archer)
    far = UNIT_CLASS_BY_ID[1](player_id=2, board=env.board)
    far.place_on_board(FAR)
    far.stack = 1
    env.board.units.append(far)
    if adjacent_enemy:
        adj = UNIT_CLASS_BY_ID[2](player_id=2, board=env.board)
        adj.place_on_board(ADJ)
        adj.stack = 1
        env.board.units.append(adj)
    return env, archer, far
