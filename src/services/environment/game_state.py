from dataclasses import dataclass, field
from collections import Counter
from typing import Optional, Tuple
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


@dataclass
class Pending:
    """A continuation owed by the active player mid-tactic (Phase 4 sub-turn).

    A tactic (or a triggered attribute) that spans several decisions parks its
    state here instead of inflating the action space: while `pending` is set the
    turn does NOT pass, `get_possible_actions` returns only the legal next clicks
    for this `kind`, and those clicks reuse the existing move/attack verbs (the
    policy disambiguates via the pending-context one-hot in the global features).

    kind:      context label; drives both the legal-continuation mask and the
               observation context one-hot (must be listed in PENDING_KINDS).
    unit_loc:  the on-board cell the continuation acts on/from (updated as a unit
               moves through a multi-step tactic).
    optional:  whether DECLINE is a legal continuation (ends the tactic early).
    data:      per-tactic scratch (e.g. a locked line direction) for future kinds.
    """
    kind: str
    unit_loc: Tuple[int, int]
    optional: bool = False
    data: dict = field(default_factory=dict)


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
    # Owed mid-tactic continuation (Phase 4); None during normal play.
    pending: Optional[Pending] = None
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
