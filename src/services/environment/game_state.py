from dataclasses import dataclass, field
from .board import Board

# Coin types. Unit coins share the matching unit's `id`; the Royal coin has no unit.
COIN_SWORD = 1
COIN_KNIGHT = 2
COIN_ROYAL = 3
DECK = (COIN_SWORD, COIN_KNIGHT, COIN_ROYAL)


def _full_hands():
    return {1: set(DECK), 2: set(DECK)}


@dataclass
class GameState:
    board: Board
    active_player: int
    action_count: int = 0
    # Coin types still unspent in the current round, per player.
    hands: dict = field(default_factory=_full_hands)
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
