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
    # The part of `reward` that is the per-turn tempo cost (`TURN_TEMPO_REWARD`), already
    # included in `reward`. Non-zero only on the action that ends a turn. Broken out so
    # consumers that price tempo themselves — the reward decomposition in the training
    # logs, LookaheadBot's depth-bounded search — can subtract it without pattern-matching
    # on the reward value, which stopped being unique once every turn pays it.
    tempo_cost: float = 0.0
