from abc import ABC

class BaseUnit(ABC):
    id = -1
    icon = '.'

    def __init__(self, player_id: int, board: "Board"):
        self.player_id = player_id
        self.board = board
        self.loc = None

    def place_on_board(self, loc):
        self.loc = loc

    def move(self, loc):
        self.loc = loc
