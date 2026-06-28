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
    # Phase 4: name of this unit's active tactic (the face-up special maneuver), or
    # None for a vanilla unit. The env reads it to expose the TACTIC verb and drive
    # the matching pending sub-turn. Passive attributes are tracked separately.
    tactic: str = None


# Bag holds 2 coins of each owned unit at setup; the remainder goes to the supply.
BAG_PER_UNIT = 2

UNIT_TYPES = (
    UnitType(1, 'Swordsman', 'Sw', '#2b3f6b', 5),
    UnitType(2, 'Knight', 'Kn', '#3fa6dc', 4),
    UnitType(3, 'Cavalry', 'Ca', '#c87a2c', 4, tactic='cavalry'),
    UnitType(4, 'Light Cavalry', 'Lc', '#8fae3e', 5),
    UnitType(5, 'Lancer', 'La', '#c0392b', 4),
    UnitType(6, 'Archer', 'Ar', '#5a9ea0', 4),
    UnitType(7, 'Crossbowman', 'Cb', '#7d4f5e', 5),
    UnitType(8, 'Berserker', 'Be', '#2f5d3b', 5),
    UnitType(9, 'Footman', 'Fo', '#2c8090', 5),
    UnitType(10, 'Pikeman', 'Pk', '#d4a72c', 4),
    UnitType(11, 'Ensign', 'En', '#9aa83c', 5),
    UnitType(12, 'Marshall', 'Ma', '#bf5a2f', 5),
    UnitType(13, 'Mercenary', 'Me', '#8c2f2f', 5),
    UnitType(14, 'Scout', 'Sc', '#3a6ea8', 5),
    UnitType(15, 'Royal Guard', 'Rg', '#d98c9c', 5),
    UnitType(16, 'Warrior Priest', 'Wp', '#6e4a8c', 4),
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
