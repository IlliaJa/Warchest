import numpy as np
from .cell_ids import *
from .units.baseunit import BaseUnit
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
    # {(r, q): [neighbor, ...]} — the hex grid's shape (which cells are INVALID vs.
    # in-bounds) is identical for every Board ever constructed, so this is computed
    # once, lazily, and shared across all instances instead of redoing the 6-neighbour
    # scan (with per-cell numpy scalar indexing) on every get_adjacent_cells() call —
    # a hot path called tens of thousands of times per search (docs/lookahead_bot_plan.md).
    _adjacency_cache = None
    # [(r, q), ...] of every non-INVALID cell — same invariant as above, computed once
    # instead of an `np.where` full-array scan on every `all_cells_list` access.
    _all_cells_cache = None

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
        if Board._adjacency_cache is None:
            Board._adjacency_cache = self._compute_adjacency()
        if Board._all_cells_cache is None:
            Board._all_cells_cache = list(zip(*np.where(self.board != INVALID_CELL_ID)))

    def _compute_adjacency(self):
        rows, cols = self.board.shape
        adjacency = {}
        for r in range(rows):
            for q in range(cols):
                cells = []
                for r_offset, q_offset in self.offsets:
                    new_r, new_q = r + r_offset, q + q_offset
                    if (0 <= new_r < rows and 0 <= new_q < cols
                            and self.board[new_r, new_q] != INVALID_CELL_ID):
                        cells.append((new_r, new_q))
                adjacency[(r, q)] = cells
        return adjacency

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
        return Board._adjacency_cache.get((r, q), [])

    def get_free_adjacent_cells(self, r: int, q: int):
        current_units_loc = {u.loc for u in self.units}
        adj_cells = self.get_adjacent_cells(r, q)
        return [cell for cell in adj_cells if cell not in current_units_loc]

    @property
    def all_cells_list(self):
        return Board._all_cells_cache

    def get_unit_at(self, r: int, q: int):
        for u in self.units:
            if u.loc == (r, q):
                return u
        return None

    def remove_unit(self, unit):
        self.units.remove(unit)

    def is_valid_claim(self, player_id, cell_loc):
        available_cells_for_claim = {
            1: (CONTROLLED_BASE_PLAYER_2_CELL_ID, UNCONTROLLED_BASE_CELL_ID),
            2: (CONTROLLED_BASE_PLAYER_1_CELL_ID, UNCONTROLLED_BASE_CELL_ID)
        }
        is_cell_uncontrolled_or_claimed_by_other_player = self.board[cell_loc] in available_cells_for_claim.get(player_id, ())

        player_unit_locations = [unit.loc for unit in self.units if unit.player_id == player_id]
        unit_is_present_on_cell = cell_loc in player_unit_locations

        return is_cell_uncontrolled_or_claimed_by_other_player and unit_is_present_on_cell
