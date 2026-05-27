from dataclasses import dataclass
from typing import Any


@dataclass
class Action:
    reward: float
    finishes_game: bool
    is_valid: bool
    id: int = None
    type: str = None
    player_id: int = None
    txt_result: str = ''
    additional_info: Any = None
