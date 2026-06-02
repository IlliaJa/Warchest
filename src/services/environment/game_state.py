from dataclasses import dataclass, field
from collections import Counter
from .board import Board
from .roster import (
    UNIT_IDS, ALL_COIN_IDS, ROYAL_ID, NUM_UNIT_TYPES,
    TOTAL_COINS, SUPPLY_CAP, BAG_PER_UNIT,
)

# Coin universe: unit coins 1..16 share the matching unit's id; the Royal coin
# (17) has no board unit. DECK is the full universe; per-game each player only
# holds coins of their drafted types (+ the Royal coin every player gets).
DECK = ALL_COIN_IDS  # (1..16, 17)
COIN_ROYAL = ROYAL_ID  # kept as a named constant; the royal coin is bag-only

HAND_SIZE = 3  # coins drawn into the hand each round
UNITS_PER_PLAYER = 4  # each player drafts this many distinct unit types


def _empty_counters():
    return {1: Counter(), 2: Counter()}


def build_bag(unit_ids) -> Counter:
    """A player's starting bag: 2 coins of each drafted unit + 1 Royal coin."""
    bag = Counter({t: BAG_PER_UNIT for t in unit_ids})
    bag[ROYAL_ID] = TOTAL_COINS[ROYAL_ID]  # 1
    return bag


def build_supply(unit_ids) -> Counter:
    """A player's starting supply: the non-bag coins of each drafted unit."""
    return Counter({t: SUPPLY_CAP[t] for t in unit_ids if SUPPLY_CAP[t] > 0})


@dataclass
class GameState:
    board: Board
    active_player: int
    action_count: int = 0
    # Each player's drafted unit-type ids (4 distinct, disjoint across players).
    compositions: dict = field(default_factory=lambda: {1: (), 2: ()})
    # Coin economy, per player. Board units are committed coins (not held here).
    bags: dict = field(default_factory=_empty_counters)
    hands: dict = field(default_factory=_empty_counters)
    discard_faceup: dict = field(default_factory=_empty_counters)
    discard_facedown: dict = field(default_factory=_empty_counters)
    supply: dict = field(default_factory=_empty_counters)  # recruitable coins (public)
    boxed: dict = field(default_factory=_empty_counters)  # coins removed from the game
    initiative_owner: int = 1
    # Initiative may transfer at most once per round.
    initiative_transferred_this_round: bool = False
    round_number: int = 0
    # Last action taken (for rendering); set on each valid coin play.
    last_action_type: str = None
    last_coin: int = None
    last_coin_player: int = None
    is_terminated = False
    is_truncated = False

    def owned(self, player: int) -> Counter:
        """Total coins this player owns per type (fixed by their composition).

        Used for opponent coin-counting: owned = on_board + faceup + supply +
        hidden_pool. The Royal coin is owned by both players.
        """
        c = Counter({t: TOTAL_COINS[t] for t in self.compositions[player]})
        c[ROYAL_ID] = TOTAL_COINS[ROYAL_ID]
        return c
