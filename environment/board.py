import numpy as np
from environment.cell_ids import *
from environment.units.baseunit import BaseUnit
from typing import List


class Board:
    offsets = [
        (-1, -1),  # bottom-left
        (-1, 0),  # top-left
        (0, 1),   # top
        (1, 1),   # top-right
        (1, 0),  # bottom-right
        (0, -1)   # bottom
    ]
    default_bases = {
        CONTROLLED_BASE_PLAYER_1_CELL_ID: [(1, 0), (4, 1)],
        CONTROLLED_BASE_PLAYER_2_CELL_ID: [(2, 5), (5, 6)],
        UNCONTROLLED_BASE_CELL_ID: [(0, 1), (2, 2), (5, 3), (1, 3), (4, 4), (6, 5)],
    }

    def __init__(self):
        self._create_hex_board()
        self.units: List[BaseUnit] = []

    def _create_hex_board(self):
        self.size = 4  # Hexagon with sides of 4 tiles
        size = self.size
        board_size = 2 * size - 1
        self.board_size = board_size

        board = np.full(shape=(board_size, board_size), fill_value=INVALID_CELL_ID, dtype=int)
        for r in range(board_size):
            for q in range(max(0, r - size + 1), min(board_size, r + size)):
                board[r, q] = EMPTY_CELL_ID

        for cell_id, loc_list in self.default_bases.items():
            for r, q in loc_list:
                board[r, q] = cell_id

        self.board = board

    def get_controlled_bases(self, player_id: int):
        cell_id = CONTROLLED_BASE_PLAYER_1_CELL_ID if player_id == 1 else CONTROLLED_BASE_PLAYER_2_CELL_ID
        return list(zip(*np.where(self.board == cell_id)))

    def change_base_control(self, player_id, base_loc):
        cell_id = CONTROLLED_BASE_PLAYER_1_CELL_ID if player_id == 1 else CONTROLLED_BASE_PLAYER_2_CELL_ID
        self.board[base_loc] = cell_id

    def deploy_unit(self, unit, place):
        controlled_bases = self.get_controlled_bases(unit.player_id)
        if place not in controlled_bases:
            raise Exception(f'Unit {self.__class__.__name__} cannot be deployed outside of a controlled base')
        self.units.append(unit)
        unit.place_on_board(place)

    def get_adjacent_cells(self, r: int, q: int):
        valid_cells = []
        for r_offset, q_offset in self.offsets:
            new_r = r + r_offset
            new_q = q + q_offset
            if 0 <= new_r < self.board.shape[0] \
                and 0 <= new_q < self.board.shape[1] \
                and self.board[new_r, new_q] != INVALID_CELL_ID:
                valid_cells.append((new_r, new_q))
        return valid_cells

    def get_free_adjacent_cells(self, r: int, q: int):
        current_units_loc = [u.loc for u in self.units]
        adj_cells = self.get_adjacent_cells(r, q)
        return [cell for cell in adj_cells if cell not in current_units_loc]

    @property
    def all_cells_list(self):
        return list(zip(*np.where(self.board != INVALID_CELL_ID)))

    def is_valid_claim(self, player_id, cell_loc):
        available_cells_for_claim = {
            1: (CONTROLLED_BASE_PLAYER_2_CELL_ID, UNCONTROLLED_BASE_CELL_ID),
            2: (CONTROLLED_BASE_PLAYER_1_CELL_ID, UNCONTROLLED_BASE_CELL_ID)
        }
        is_cell_uncontrolled_or_claimed_by_other_player = self.board[cell_loc] in available_cells_for_claim.get(player_id, ())

        player_unit_locations = [unit.loc for unit in self.units if unit.player_id == player_id]
        unit_is_present_on_cell = cell_loc in player_unit_locations

        return is_cell_uncontrolled_or_claimed_by_other_player and unit_is_present_on_cell
