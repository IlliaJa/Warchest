from dataclasses import dataclass, field
from .board import Board


@dataclass
class GameState:
    board: Board
    active_player: int
    action_count: int = 0
    deploys_used: dict = field(default_factory=lambda: {1: 0, 2: 0})
    is_terminated = False
    is_truncated = False
