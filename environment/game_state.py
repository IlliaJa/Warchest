from dataclasses import dataclass
from environment.board import Board


@dataclass
class GameState:
    """
    Class representing the state of the game.
    """
    board: Board
    active_player: int
    action_count: int = 0
    is_terminated = False
    is_truncated = False
