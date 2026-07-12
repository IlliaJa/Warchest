"""Triggered / passive unit attributes (no TACTIC verb to initiate them):

  Pikeman        — when attacked by an adjacent unit, the attacker also loses a coin
  Swordsman      — after it attacks, an optional free move
  Berserker      — after it maneuvers, pay a stack coin to maneuver again (repeatable)
  Warrior Priest — after it attacks/controls, draw a coin and use it immediately

These fire from the normal maneuver paths and resolve as pending sub-turns (except
Pikeman's, which is an instantaneous counter). Directions use Board.offsets:
offsets[2] = (0,+1), so a unit at (3,3) acts toward (3,4).
"""
from collections import Counter

from src.services.environment.warchest_env import (
    WarChestEnv, TACTIC_VERB, SELECT_VERB, CONTROL_VERB, DECLINE_ACTION_ID, SPATIAL_SIZE,
)
from _helpers import (
    blank_env, place, SWORDSMAN, KNIGHT, CAV, ARCHER, BERSERKER, PIKEMAN, WARRIOR_PRIEST,
)

ATK2 = 6 + 2   # attack toward (3,4) from (3,3)
MV2 = 2        # move toward (3,4) from (3,3)


# --------------------------------------------------------------------------- #
# Pikeman — on-defense counter
# --------------------------------------------------------------------------- #

def test_pikeman_counters_an_adjacent_attacker():
    env = blank_env(active=1)
    env.state.compositions = {1: (KNIGHT,), 2: (PIKEMAN,)}
    env.state.hands[1] = Counter({KNIGHT: 1})
    env.state.bags = {1: Counter({KNIGHT: 1}), 2: Counter({PIKEMAN: 1})}
    place(env, KNIGHT, 1, (3, 3), stack=1)        # attacker, single coin
    pike = place(env, PIKEMAN, 2, (3, 4), stack=2)

    env.step(WarChestEnv.encode_action(ATK2, 3, 3))
    assert pike.stack == 1                          # Pikeman took the hit
    assert env.board.get_unit_at(3, 3) is None      # the single-coin attacker died to the counter
    assert env.state.boxed[2][PIKEMAN] == 1
    assert env.state.boxed[1][KNIGHT] == 1


def test_pikeman_does_not_counter_a_ranged_attacker():
    env = blank_env(active=1)
    env.state.compositions = {1: (ARCHER,), 2: (PIKEMAN,)}
    env.state.hands[1] = Counter({ARCHER: 1})
    env.state.bags = {1: Counter({ARCHER: 1}), 2: Counter({PIKEMAN: 1})}
    archer = place(env, ARCHER, 1, (3, 3), stack=1)
    pike = place(env, PIKEMAN, 2, (3, 5), stack=2)  # two spaces away

    env.step(WarChestEnv.encode_action(TACTIC_VERB, 3, 3))
    env.step(WarChestEnv.encode_action(SELECT_VERB, 3, 5))
    assert pike.stack == 1
    assert env.board.get_unit_at(3, 3) is archer    # archer unharmed (not adjacent)
    assert env.state.boxed[1][ARCHER] == 0


# --------------------------------------------------------------------------- #
# Swordsman — optional free move after attacking
# --------------------------------------------------------------------------- #

def test_swordsman_free_move_after_attack():
    env = blank_env(active=1)
    env.state.compositions = {1: (SWORDSMAN,), 2: (CAV,)}
    env.state.hands[1] = Counter({SWORDSMAN: 1})
    env.state.bags = {1: Counter({SWORDSMAN: 1}), 2: Counter({CAV: 1})}
    sw = place(env, SWORDSMAN, 1, (3, 3), stack=1)
    place(env, CAV, 2, (3, 4), stack=1)              # plain victim (Knight needs a bolstered attacker)

    env.step(WarChestEnv.encode_action(ATK2, 3, 3))   # kills the victim, vacating (3,4)
    assert env.board.get_unit_at(3, 4) is None
    assert env.state.pending is not None and env.state.pending.kind == 'bonus_move'
    assert env.active_player == 1

    cont = env.get_possible_actions()
    assert DECLINE_ACTION_ID in cont                  # the free move is optional
    move = WarChestEnv.encode_action(MV2, 3, 3)
    assert move in cont
    env.step(move)
    assert env.board.get_unit_at(3, 4) is sw          # moved for free
    assert env.state.pending is None


def test_swordsman_bonus_move_can_be_declined():
    env = blank_env(active=1)
    env.state.compositions = {1: (SWORDSMAN,), 2: (CAV,)}
    env.state.hands[1] = Counter({SWORDSMAN: 1})
    env.state.bags = {1: Counter({SWORDSMAN: 1}), 2: Counter({CAV: 1})}
    sw = place(env, SWORDSMAN, 1, (3, 3), stack=1)
    place(env, CAV, 2, (3, 4), stack=1)

    env.step(WarChestEnv.encode_action(ATK2, 3, 3))
    env.step(DECLINE_ACTION_ID)
    assert env.state.pending is None
    assert env.board.get_unit_at(3, 3) is sw          # stayed put


# --------------------------------------------------------------------------- #
# Berserker — pay stack coins for extra maneuvers
# --------------------------------------------------------------------------- #

def test_berserker_chains_extra_maneuvers_until_one_coin():
    env = blank_env(active=1)
    env.state.compositions = {1: (BERSERKER,), 2: (KNIGHT,)}
    env.state.hands[1] = Counter({BERSERKER: 1})
    env.state.bags = {1: Counter({BERSERKER: 1}), 2: Counter({KNIGHT: 1})}
    ber = place(env, BERSERKER, 1, (3, 3), stack=3)
    place(env, KNIGHT, 2, (3, 4), stack=1)

    # First maneuver: a normal attack paid by the hand coin; kills the knight.
    env.step(WarChestEnv.encode_action(ATK2, 3, 3))
    assert env.board.get_unit_at(3, 4) is None
    assert env.state.pending.kind == 'extra_maneuver'  # stack 3 >= 2
    assert env.active_player == 1

    # Extra maneuver 1: move into (3,4), paid by one stack coin (3 -> 2).
    env.step(WarChestEnv.encode_action(MV2, 3, 3))
    assert ber.stack == 2 and env.board.get_unit_at(3, 4) is ber
    assert env.state.pending.kind == 'extra_maneuver'

    # Extra maneuver 2: move on to (3,5), paid again (2 -> 1); now it cannot pay more.
    env.step(WarChestEnv.encode_action(MV2, 3, 4))
    assert ber.stack == 1 and env.board.get_unit_at(3, 5) is ber
    assert env.state.pending is None
    assert env.state.boxed[1][BERSERKER] == 2          # two payments to the box


def test_berserker_can_decline_extra_maneuvers():
    env = blank_env(active=1)
    env.state.compositions = {1: (BERSERKER,), 2: (KNIGHT,)}
    env.state.hands[1] = Counter({BERSERKER: 1})
    env.state.bags = {1: Counter({BERSERKER: 1}), 2: Counter({KNIGHT: 1})}
    ber = place(env, BERSERKER, 1, (3, 3), stack=3)
    place(env, KNIGHT, 2, (3, 4), stack=1)

    env.step(WarChestEnv.encode_action(ATK2, 3, 3))
    env.step(DECLINE_ACTION_ID)
    assert env.state.pending is None
    assert ber.stack == 3                              # never paid


# --------------------------------------------------------------------------- #
# Warrior Priest — draw a coin and use it immediately
# --------------------------------------------------------------------------- #

def test_warrior_priest_draws_and_acts_after_attack():
    env = blank_env(active=1)
    env.state.compositions = {1: (WARRIOR_PRIEST,), 2: (CAV,)}
    env.state.hands[1] = Counter({WARRIOR_PRIEST: 1})
    # P2 keeps a hand coin so the round doesn't end (and reshuffle the discard) after
    # the bonus; WP draws a Swordsman from its own bag.
    env.state.hands[2] = Counter({CAV: 1})
    env.state.bags = {1: Counter({SWORDSMAN: 1}), 2: Counter({CAV: 1})}
    place(env, WARRIOR_PRIEST, 1, (3, 3), stack=1)
    place(env, CAV, 2, (3, 4), stack=1)

    env.step(WarChestEnv.encode_action(ATK2, 3, 3))    # WP attacks → triggers the bonus
    assert env.state.pending is not None and env.state.pending.kind == 'bonus_action'
    assert env.state.pending.data['coin'] == SWORDSMAN
    assert env.state.bags[1][SWORDSMAN] == 0           # drawn out of the bag
    assert env.state.hands[1][SWORDSMAN] == 1          # into the hand
    assert env.active_player == 1

    cont = env.get_possible_actions()
    pass_id = WarChestEnv.encode_facedown(1, SWORDSMAN)  # spending the drawn coin is always possible
    assert pass_id in cont
    env.step(pass_id)
    assert env.state.pending is None
    assert env.state.discard_facedown[1][SWORDSMAN] == 1


def test_warrior_priest_triggers_on_control():
    env = blank_env(active=1)
    env.state.compositions = {1: (WARRIOR_PRIEST,), 2: (KNIGHT,)}
    env.state.hands[1] = Counter({WARRIOR_PRIEST: 1})
    env.state.bags = {1: Counter({SWORDSMAN: 1}), 2: Counter()}
    place(env, WARRIOR_PRIEST, 1, (2, 2), stack=1)     # (2,2) is an uncontrolled base by default

    env.step(WarChestEnv.encode_action(CONTROL_VERB, 2, 2))
    assert env.state.pending is not None and env.state.pending.kind == 'bonus_action'


def test_warrior_priest_bonus_coin_can_start_a_tactic():
    """A drawn coin whose unit has a tactic may spend the bonus on that tactic; the
    tactic's own pending sub-turn replaces the bonus-action pending (no clobber)."""
    env = blank_env(active=1)
    env.state.compositions = {1: (WARRIOR_PRIEST, CAV), 2: (SWORDSMAN,)}
    env.state.hands[1] = Counter({WARRIOR_PRIEST: 1})
    env.state.hands[2] = Counter({SWORDSMAN: 1})       # keep the round alive after the bonus
    env.state.bags = {1: Counter({CAV: 1}), 2: Counter({SWORDSMAN: 1})}
    place(env, WARRIOR_PRIEST, 1, (3, 3), stack=1)
    place(env, SWORDSMAN, 2, (3, 4), stack=1)          # WP's attack target
    cav = place(env, CAV, 1, (5, 5), stack=1)
    place(env, SWORDSMAN, 2, (5, 6), stack=1)          # enemy adjacent to the Cavalry's step target

    env.step(WarChestEnv.encode_action(ATK2, 3, 3))    # WP attacks → draws the Cavalry coin
    assert env.state.pending.kind == 'bonus_action'
    assert env.state.pending.data['coin'] == CAV

    tactic_id = WarChestEnv.encode_action(TACTIC_VERB, 5, 5)
    assert tactic_id in env.get_possible_actions()      # tactic-initiate is now a legal bonus
    env.step(tactic_id)
    # The bonus pending gave way to the Cavalry's move-then-attack sub-turn.
    assert env.state.pending is not None and env.state.pending.kind == 'move_then_attack:move'
    assert env.state.hands[1][CAV] == 0                 # the drawn coin paid for the tactic


# --------------------------------------------------------------------------- #
# Cluster 4 — restrictions / deploy / on-defense attributes
# --------------------------------------------------------------------------- #

from src.services.environment.warchest_env import DEPLOY_VERB_BASE  # noqa: E402
from _helpers import (  # noqa: E402
    LIGHT_CAV, FOOTMAN, MERCENARY, SCOUT, ROYAL_GUARD,
)

DEPLOY = lambda coin, r, q: WarChestEnv.encode_action(DEPLOY_VERB_BASE + (coin - 1), r, q)


def test_knight_only_attackable_when_attacker_is_bolstered():
    env = blank_env(active=1)
    env.state.compositions = {1: (CAV,), 2: (KNIGHT,)}
    env.state.hands[1] = Counter({CAV: 1})
    env.state.bags = {1: Counter({CAV: 1}), 2: Counter({KNIGHT: 1})}
    attacker = place(env, CAV, 1, (3, 3), stack=1)   # unbolstered
    place(env, KNIGHT, 2, (3, 4), stack=1)

    assert WarChestEnv.encode_action(ATK2, 3, 3) not in env.get_possible_actions()
    attacker.stack = 2                               # bolster the attacker
    assert WarChestEnv.encode_action(ATK2, 3, 3) in env.get_possible_actions()


def test_scout_deploys_adjacent_to_a_friendly_unit():
    env = blank_env(active=1)
    env.state.compositions = {1: (SCOUT, CAV), 2: ()}
    env.state.hands[1] = Counter({SCOUT: 1})
    place(env, CAV, 1, (3, 3), stack=1)              # a friendly unit off-base
    # The Scout may deploy onto an empty cell adjacent to the friendly unit; a normal
    # unit could only deploy onto controlled bases.
    assert (3, 4) in env._deploy_targets(SCOUT)
    assert (3, 4) not in env._deploy_targets(CAV)
    env.step(DEPLOY(SCOUT, 3, 4))
    assert env.board.get_unit_at(3, 4).id == SCOUT


def test_footman_allows_two_copies_on_board():
    env = blank_env(active=1)
    env.state.compositions = {1: (FOOTMAN,), 2: ()}
    env.state.hands[1] = Counter({FOOTMAN: 2})
    place(env, FOOTMAN, 1, (1, 0), stack=1)          # one Footman on a controlled base
    env.step(DEPLOY(FOOTMAN, 4, 1))                  # deploy a second onto the other base
    assert sum(1 for u in env.board.units if u.id == FOOTMAN) == 2
    # A third Footman is not allowed: max_on_board == 2 (also no empty base remains).
    deploy_ids = [a for a in env.get_possible_actions()
                  if a < SPATIAL_SIZE and WarChestEnv.decode_action(a)[0] == DEPLOY_VERB_BASE + (FOOTMAN - 1)]
    assert deploy_ids == []


def test_royal_guard_absorbs_a_hit_from_supply():
    env = blank_env(active=1)
    env.state.compositions = {1: (CAV,), 2: (ROYAL_GUARD,)}
    env.state.hands[1] = Counter({CAV: 1})
    env.state.bags = {1: Counter({CAV: 1}), 2: Counter({ROYAL_GUARD: 1})}
    place(env, CAV, 1, (3, 3), stack=1)
    rg = place(env, ROYAL_GUARD, 2, (3, 4), stack=1)
    env.state.supply[2] = Counter({ROYAL_GUARD: 1})  # a supply coin to absorb the hit

    env.step(WarChestEnv.encode_action(ATK2, 3, 3))
    assert rg.stack == 1                             # the on-board stack is unharmed
    assert env.board.get_unit_at(3, 4) is rg
    assert env.state.supply[2][ROYAL_GUARD] == 0     # the hit came from supply
    assert env.state.boxed[2][ROYAL_GUARD] == 1


def test_mercenary_free_maneuver_after_recruit():
    env = blank_env(active=1)
    env.state.compositions = {1: (MERCENARY,), 2: ()}
    env.state.hands[1] = Counter({MERCENARY: 1})     # pay coin for the recruit
    env.state.supply[1] = Counter({MERCENARY: 1})    # a Mercenary coin to recruit
    place(env, MERCENARY, 1, (3, 3), stack=1)        # Mercenary already on the board

    recruit = WarChestEnv.encode_recruit(MERCENARY, MERCENARY)  # take, pay
    assert recruit in env.get_possible_actions()
    env.step(recruit)
    assert env.state.pending is not None and env.state.pending.kind == 'free_maneuver'
    assert env.state.pending.unit_loc == (3, 3)
