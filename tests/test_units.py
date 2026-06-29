"""Unit roster data: counts, coin totals/supply split, and per-unit tactic and
restriction flags. Static facts about the roster, independent of game flow.
"""
from src.services.environment import roster
from _helpers import CAV, ARCHER


def test_roster_has_16_units_plus_royal():
    assert roster.NUM_UNIT_TYPES == 16
    assert [u.id for u in roster.UNIT_TYPES] == list(range(1, 17))
    assert roster.ROYAL_ID == 17
    assert roster.TOTAL_COINS[roster.ROYAL_ID] == 1


def test_roster_totals_and_supply_split():
    # Every unit owns 4 or 5 coins; bag keeps 2, supply gets the rest (>= 2).
    for u in roster.UNIT_TYPES:
        assert u.total_coins in (4, 5)
        assert roster.SUPPLY_CAP[u.id] == u.total_coins - roster.BAG_PER_UNIT
        assert roster.SUPPLY_CAP[u.id] >= 2
    # A couple of known cards from docs/UNITS.md.
    assert roster.TOTAL_COINS[1] == 5  # Swordsman x5
    assert roster.TOTAL_COINS[2] == 4  # Knight x4


def test_roster_tactics_are_named_by_mechanic():
    # Tactic ids name the mechanic, not the unit, so they're reusable across DLC
    # (e.g. Archer and Crossbowman both use 'ranged_attack').
    assert roster.UNIT_BY_ID[CAV].tactic == 'move_then_attack'
    assert roster.UNIT_BY_ID[ARCHER].tactic == 'ranged_attack'
    assert roster.UNIT_BY_ID[ARCHER].tactic_params == {'distance': 2, 'straight_line': False}
    assert roster.UNIT_BY_ID[7].tactic == 'ranged_attack'  # Crossbowman shares the mechanic
    assert roster.UNIT_BY_ID[7].tactic_params == {'distance': 2, 'straight_line': True}
    # Every unit with a printed tactic has one; the rest are vanilla.
    tactic_units = {u.id for u in roster.UNIT_TYPES if u.tactic is not None}
    assert tactic_units == {3, 4, 5, 6, 7, 9, 11, 12, 15}


def test_roster_restrictions_and_attributes():
    # Restrictions: Archer and Lancer cannot make a normal attack.
    assert roster.UNIT_BY_ID[6].can_normal_attack is False   # Archer
    assert roster.UNIT_BY_ID[5].can_normal_attack is False   # Lancer
    # Footman is the only unit allowed two copies on the board.
    assert roster.UNIT_BY_ID[9].max_on_board == 2
    assert all(u.max_on_board == 1 for u in roster.UNIT_TYPES if u.id != 9)
    # Spot-check the triggered/passive attribute flags.
    assert roster.UNIT_BY_ID[1].move_after_attack                      # Swordsman
    assert roster.UNIT_BY_ID[2].only_attackable_when_bolstered         # Knight
    assert roster.UNIT_BY_ID[8].extra_maneuvers_from_stack             # Berserker
    assert roster.UNIT_BY_ID[10].counter_when_attacked                 # Pikeman
    assert roster.UNIT_BY_ID[13].maneuver_after_recruit                # Mercenary
    assert roster.UNIT_BY_ID[14].deploy_adjacent_to_friendly           # Scout
    assert roster.UNIT_BY_ID[15].absorb_from_supply                    # Royal Guard
    assert roster.UNIT_BY_ID[16].bonus_action_after_attack_or_control  # Warrior Priest
