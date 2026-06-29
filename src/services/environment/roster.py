"""Single source of truth for the 16-unit roster + the Royal coin.

Phase 3 implements every unit as *vanilla* — identical move/attack/control/
deploy/bolster behaviour, differing only by coin identity (id / icon / colour /
how many coins of that type a player owns). Tactics & passive attributes arrive
in Phase 4; the rules text lives in docs/UNITS.md.

Coin ids: 1..16 are the unit types, 17 is the Royal coin (no board unit, a single
bag-only coin every player holds). Per-type coin totals follow the cards: a unit
keeps 2 coins in the bag at setup and the rest (total - 2) in the supply.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class UnitType:
    id: int
    name: str
    icon: str       # short look-alike glyph for the renderer (monochrome, see UNITS.md)
    color: str      # coin-face colour approximated from the card art
    total_coins: int  # x4 or x5 — coins a player owns of this type
    # Phase 4: the MECHANIC of this unit's active tactic (the face-up special
    # maneuver), or None for a vanilla unit. Named by mechanic so it is reused
    # across the roster and DLC rather than tied to one unit — e.g. 'ranged_attack'
    # is shared by Archer and Crossbowman (and any DLC ranged unit), differentiated
    # only by tactic_params. The env reads it to expose the TACTIC verb and drive
    # the matching pending sub-turn.
    tactic: str = None
    # Parameters for the tactic mechanic, letting units that share a mechanic differ
    # (e.g. ranged_attack: distance, whether a clear straight line is required).
    tactic_params: dict = None
    # Phase 4 restriction: some units cannot use the normal (adjacent) attack action
    # and may only attack via their tactic (e.g. Archer, Lancer). Defaults to True.
    can_normal_attack: bool = True
    # Phase 4 triggered/passive attributes (named by mechanic, reusable across DLC):
    #   counter_when_attacked              — Pikeman: an adjacent attacker also loses a coin
    #   move_after_attack                  — Swordsman: an optional free move after it attacks
    #   extra_maneuvers_from_stack         — Berserker: maneuver again by spending its own coins
    #   bonus_action_after_attack_or_control — Warrior Priest: draw a coin and use it at once
    counter_when_attacked: bool = False
    move_after_attack: bool = False
    extra_maneuvers_from_stack: bool = False
    bonus_action_after_attack_or_control: bool = False
    # Cluster 4 attributes:
    #   only_attackable_when_bolstered  — Knight: can only be hit by a bolstered attacker
    #   deploy_adjacent_to_friendly     — Scout: deploys next to any friendly unit, not just bases
    #   max_on_board                    — Footman: how many copies may be on the board at once
    #   maneuver_after_recruit          — Mercenary: a free maneuver when its coin is recruited
    #   absorb_from_supply              — Royal Guard: a hit may be taken from supply instead
    only_attackable_when_bolstered: bool = False
    deploy_adjacent_to_friendly: bool = False
    max_on_board: int = 1
    maneuver_after_recruit: bool = False
    absorb_from_supply: bool = False


# Bag holds 2 coins of each owned unit at setup; the remainder goes to the supply.
BAG_PER_UNIT = 2

UNIT_TYPES = (
    UnitType(1, 'Swordsman', 'Sw', '#2b3f6b', 5, move_after_attack=True),
    UnitType(2, 'Knight', 'Kn', '#3fa6dc', 4, only_attackable_when_bolstered=True),
    UnitType(3, 'Cavalry', 'Ca', '#c87a2c', 4, tactic='move_then_attack'),
    UnitType(4, 'Light Cavalry', 'Lc', '#8fae3e', 5,
             tactic='move_to', tactic_params={'max_dist': 2}),
    UnitType(5, 'Lancer', 'La', '#c0392b', 4,
             tactic='line_charge', tactic_params={'max_dist': 2}, can_normal_attack=False),
    UnitType(6, 'Archer', 'Ar', '#5a9ea0', 4,
             tactic='ranged_attack', tactic_params={'distance': 2, 'straight_line': False},
             can_normal_attack=False),
    UnitType(7, 'Crossbowman', 'Cb', '#7d4f5e', 5,
             tactic='ranged_attack', tactic_params={'distance': 2, 'straight_line': True}),
    UnitType(8, 'Berserker', 'Be', '#2f5d3b', 5, extra_maneuvers_from_stack=True),
    UnitType(9, 'Footman', 'Fo', '#2c8090', 5, tactic='maneuver_each', max_on_board=2),
    UnitType(10, 'Pikeman', 'Pk', '#d4a72c', 4, counter_when_attacked=True),
    UnitType(11, 'Ensign', 'En', '#9aa83c', 5,
             tactic='grant_move', tactic_params={'range': 2}),
    UnitType(12, 'Marshall', 'Ma', '#bf5a2f', 5,
             tactic='grant_attack', tactic_params={'range': 2}),
    UnitType(13, 'Mercenary', 'Me', '#8c2f2f', 5, maneuver_after_recruit=True),
    UnitType(14, 'Scout', 'Sc', '#3a6ea8', 5, deploy_adjacent_to_friendly=True),
    UnitType(15, 'Royal Guard', 'Rg', '#d98c9c', 5,
             tactic='royal_move', tactic_params={'max_dist': 2}, absorb_from_supply=True),
    UnitType(16, 'Warrior Priest', 'Wp', '#6e4a8c', 4,
             bonus_action_after_attack_or_control=True),
)

ROYAL_ID = 17
ROYAL = UnitType(ROYAL_ID, 'Royal', '♚', '#d4a72c', 1)

# Lookups keyed by coin id (units + royal).
UNIT_BY_ID = {u.id: u for u in UNIT_TYPES}
ALL_COINS = UNIT_TYPES + (ROYAL,)
COIN_BY_ID = {c.id: c for c in ALL_COINS}

UNIT_IDS = tuple(u.id for u in UNIT_TYPES)        # 1..16 (deployable)
ALL_COIN_IDS = UNIT_IDS + (ROYAL_ID,)             # 1..17
NUM_UNIT_TYPES = len(UNIT_TYPES)                  # 16

# Per-type coin totals and the bag/supply split, by coin id.
TOTAL_COINS = {u.id: u.total_coins for u in UNIT_TYPES}
TOTAL_COINS[ROYAL_ID] = ROYAL.total_coins         # 1
SUPPLY_CAP = {u.id: u.total_coins - BAG_PER_UNIT for u in UNIT_TYPES}  # 2 or 3
MAX_TOTAL = max(TOTAL_COINS.values())             # 5 — stack/owned normaliser
