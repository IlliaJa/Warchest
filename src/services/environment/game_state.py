from collections import Counter
from dataclasses import dataclass, field
from .board import Board

# Coin types. Unit coins share the matching unit's `id`; the Royal coin has no unit.
COIN_SWORD = 1
COIN_KNIGHT = 2
COIN_ROYAL = 3
DECK = (COIN_SWORD, COIN_KNIGHT, COIN_ROYAL)  # all coin types

# Per-player coin allotment (a bag-builder, not a deck-builder).
INITIAL_BAG = {COIN_SWORD: 2, COIN_KNIGHT: 2, COIN_ROYAL: 1}  # starts in the bag
SUPPLY = {COIN_SWORD: 2, COIN_KNIGHT: 2}  # recruitable coins beside the cards (no royal)
# Total owned per type = bag + supply; used to normalize coin-count features.
INITIAL_OWNED = {c: INITIAL_BAG.get(c, 0) + SUPPLY.get(c, 0) for c in DECK}  # {S:4, K:4, R:1}
HAND_SIZE = 3  # coins drawn into the hand each round


def _new_bags():
    return {1: Counter(INITIAL_BAG), 2: Counter(INITIAL_BAG)}


def _new_supply():
    return {1: Counter(SUPPLY), 2: Counter(SUPPLY)}


def _empty_counters():
    return {1: Counter(), 2: Counter()}


@dataclass
class GameState:
    board: Board
    active_player: int
    action_count: int = 0
    # Coin economy, per player. Board units are committed coins (not held here).
    bags: dict = field(default_factory=_new_bags)
    hands: dict = field(default_factory=_empty_counters)
    discard_faceup: dict = field(default_factory=_empty_counters)
    discard_facedown: dict = field(default_factory=_empty_counters)
    supply: dict = field(default_factory=_new_supply)  # recruitable coins (public)
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
