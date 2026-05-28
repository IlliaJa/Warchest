import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .units import *
from .board import Board
from .cell_ids import *
from .game_renderer import GameRenderer
from typing import Tuple, Dict
from .action import Action
from .game_state import GameState
from copy import deepcopy


BASE_COLORS = {
    EMPTY_CELL_ID: 'snow',
    UNCONTROLLED_BASE_CELL_ID: 'mediumspringgreen',
    CONTROLLED_BASE_PLAYER_1_CELL_ID: 'gold',
    CONTROLLED_BASE_PLAYER_2_CELL_ID: 'skyblue'
}
UNIT_COLORS = {
    1: 'darkred',
    2: 'midnightblue'
}
CLAIM_BASE_ACTION = 'claim_base'
MOVE_ACTION = 'move'

MOVE_EXPLORE_REWARD_MAX_TURN = 5
MOVE_EXPLORE_REWARD_PER_TURN = 0.1
MOVE_NEG_REWARD_PER_TURN = -0.002
INVALID_ACTION_REWARD = -0.02
CLAIM_BASE_REWARD = 0.0  # direct claim reward removed; potential shaping handles base value
WIN_REWARD = 1.0
LOSS_REWARD = -1.0

NUM_PLAYERS = 2
MAX_UNITS_PER_PLAYER = 2  # will be 4 when game will be more developed


class WarChestEnv(gym.Env):
    max_actions = 200
    winning_base_count = 6
    max_rewardable_moving_action = 30

    # Under 180° rotation offset i maps to its opposite: 0↔3, 1↔4, 2↔5.
    # Self-inverse: _OFFSET_FLIP[_OFFSET_FLIP[i]] == i for all i.
    _OFFSET_FLIP = [3, 4, 5, 0, 1, 2]

    @staticmethod
    def remap_action(action_id: int) -> int:
        """Translate between rotated-observation action space and env action space.

        When active_player==2 the observation is rotated 180° and the action
        mask is remapped so direction semantics are consistent with the rotated
        view.  Any action the policy (or a bot) returns from that observation
        must be passed through this function before env.step(), and vice-versa.
        Self-inverse: remap_action(remap_action(a)) == a.
        Claim actions (>= MAX_UNITS_PER_PLAYER*6) are unchanged.
        """
        n_move = MAX_UNITS_PER_PLAYER * 6
        if action_id < n_move:
            unit_id = action_id // 6
            offset_id = action_id % 6
            return unit_id * 6 + WarChestEnv._OFFSET_FLIP[offset_id]
        return action_id

    def __init__(self, save_game_history: bool = False, debug_mode: bool = False):
        super().__init__()
        self.debug_mode = debug_mode

        self.state = None
        self.history = [] if save_game_history else None
        self.exploration_map_dict = None
        self.set_init_state()

        self.observation_space = self.get_observation_space()
        self.action_dict, self.total_actions = self.get_all_actions_ids()
        self.action_space = spaces.Discrete(self.total_actions)

    def reset(self, seed=None, options=None):
        """
        Resets the state of the environment, returning an initial observation and info.

        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        info : a dictionary containing other diagnostic information
        """
        super().reset(seed=seed)
        self.set_init_state()
        return self.generate_observation(), {}

    def set_init_state(self):
        board = Board()
        map = np.where(board.board == INVALID_CELL_ID, INVALID_CELL_ID, 0)
        self.exploration_map_dict = {1: map.copy(), 2: map.copy()}
        self.place_default_units(board)
        state = GameState(board=board, active_player=1, action_count=0)
        self.state = state
        if self.history is not None:
            self.history = [deepcopy(state)]

    def set_state(self, state: GameState):
        """Set the state of the environment to a specific state in the history."""
        self.state = state

    @property
    def board(self):
        return self.state.board

    @property
    def action_count(self):
        return self.state.action_count

    @action_count.setter
    def action_count(self, value):
        self.state.action_count = value

    @property
    def active_player(self):
        return self.state.active_player

    def swap_active_player(self):
        """Swap the active player between 1 and 2."""
        self.state.active_player = 1 if self.state.active_player == 2 else 2

    def step(self, action_id):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.

        Input
        -----
        action_id : an action id provided by the environment

        Outputs
        -------
        (observation, reward, terminated, truncated, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        terminated : a boolean, indicating whether the episode has terminated, for example player 1 won
        truncated : a boolean, indicating whether the episode has truncated, for example max steps reached
        info : a dictionary containing other diagnostic information from the previous action
        """
        action_type, action_info = self.get_action_info(action_id)

        action = self.action_dict[action_type]['act_function'](*action_info)
        action.id = action_id
        action.player_id = self.active_player
        action.type = action_type
        action.additional_info = action_info

        if action.is_valid:
            self.action_count += 1
            self.swap_active_player()
            if self.history is not None:
                self.history.append(deepcopy(self.state))
        truncated = self.action_count >= self.max_actions
        if self.debug_mode:
            print(f"Got action_id {action.id} with type {action.type} and info {action.additional_info}")
        terminated = action.finishes_game
        return self.generate_observation(), action.reward, terminated, truncated, {'action': action}

    def render(self, ax=None):
        created_ax = False

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            created_ax = True
        else:
            ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        hex_radius = 0.5
        board = self.board.board

        for r in range(board.shape[0]):
            for q in range(board.shape[1]):
                if board[r, q] != INVALID_CELL_ID:
                    x, y = self.convert_hex_grid_to_cartesian(r, q, hex_radius=hex_radius)
                    hexagon = patches.RegularPolygon((x, y), numVertices=6, radius=hex_radius, orientation=np.pi/2,
                                                     edgecolor='black', facecolor=BASE_COLORS[int(board[r, q])])
                    ax.add_patch(hexagon)
                    ax.text(x, y-0.3, f'r={r} q={q}', ha='center', va='center', fontsize=10, color='black')

        for _unit in self.board.units:
            x, y = self.convert_hex_grid_to_cartesian(*_unit.loc, hex_radius=hex_radius)
            ax.text(x, y, s=_unit.icon, ha='center', va='center', fontsize=30, color=UNIT_COLORS[_unit.player_id])

        ax.set_aspect("equal")
        ax.autoscale_view()
        plt.margins(0)

        if created_ax:
            plt.show()

    def render_game(self):
        if self.history is None:
            raise ValueError("Game history is not available. Set save_game_history=True when creating the environment.")

        GameRenderer(env=self, history=self.history).draw()

    @staticmethod
    def convert_hex_grid_to_cartesian(row, column, hex_radius=0.5):
        hex_height = (3 ** 0.5) * hex_radius
        x = (row * hex_height)
        y = column - row / 2
        return x, y

    def place_default_units(self, board):
        default_units = [
            (Swordsman(player_id=1, board=board), (1, 0)),
            (Swordsman(player_id=1, board=board), (4, 1)),
            (Swordsman(player_id=2, board=board), (2, 5)),
            (Swordsman(player_id=2, board=board), (5, 6))
        ]

        for _unit, loc in default_units:
            board.deploy_unit(unit=_unit, place=loc)
        return board

    def get_observation_space(self):
        board_channels = 5  # all types of cell ids
        unit_features = 2  # row, column
        global_features = 3  # turn count, my bases, opponent bases

        return gym.spaces.Dict({
            "board": gym.spaces.Box(low=-1, high=3, shape=(board_channels, self.board.board_size, self.board.board_size), dtype=np.int32),
            "units": gym.spaces.Box(low=0, high=7, shape=(NUM_PLAYERS, MAX_UNITS_PER_PLAYER, unit_features), dtype=np.int32),
            "global": gym.spaces.Box(low=0.0, high=1.0, shape=(global_features,), dtype=np.float32),
            "active_player": gym.spaces.Discrete(2),
        })

    def generate_observation(self):
        unit_features = 2
        units_obs = np.zeros((NUM_PLAYERS, MAX_UNITS_PER_PLAYER, unit_features), dtype=np.int32)
        active = self.active_player
        opponent = 3 - active
        for slot, pid in enumerate([active, opponent]):
            player_units = [u for u in self.board.units if u.player_id == pid]
            for i, _unit in enumerate(player_units[:MAX_UNITS_PER_PLAYER]):
                units_obs[slot, i] = _unit.loc

        my_bases = len(self.board.get_controlled_bases(active))
        opp_bases = len(self.board.get_controlled_bases(opponent))
        global_feats = np.array([
            self.action_count / self.max_actions,
            my_bases / self.winning_base_count,
            opp_bases / self.winning_base_count,
        ], dtype=np.float32)

        valid_action_ids = self.get_possible_actions()

        if active == 2:
            s = self.board.board_size - 1
            board_obs = np.rot90(self.board.board, 2).copy()
            expl_obs = np.rot90(self.exploration_map_dict[active], 2).copy()
            for slot in range(NUM_PLAYERS):
                for i in range(MAX_UNITS_PER_PLAYER):
                    r, c = units_obs[slot, i]
                    units_obs[slot, i] = [s - r, s - c]
            valid_action_mask = np.zeros(self.total_actions)
            for a in valid_action_ids:
                valid_action_mask[self.remap_action(a)] = 1
            if self.debug_mode:
                orig_corner = self.board.board[0, 0]
                rot_corner = board_obs[0, 0]
                print(
                    f'[obs_rotate] P2 active: board[0,0]={rot_corner} (was board[{s},{s}]={orig_corner}) '
                    f'unit0={units_obs[0,0].tolist()} '
                    f'valid_orig={sorted(valid_action_ids)} '
                    f'valid_rot={sorted(int(a) for a in np.where(valid_action_mask)[0])}'
                )
        else:
            board_obs = self.board.board
            expl_obs = self.exploration_map_dict[active]
            valid_action_mask = np.zeros(self.total_actions)
            valid_action_mask[valid_action_ids] = 1

        return {
            'board': board_obs,
            'exploration_map': expl_obs,
            'units': units_obs,
            'global': global_feats,
            'valid_action_mask': valid_action_mask,
            'active_player': active,
        }

    def get_all_actions_ids(self) -> Tuple[Dict, int]:
        actions_ids = {}

        player_unit_moves = MAX_UNITS_PER_PLAYER * 6
        player_unit_claims = MAX_UNITS_PER_PLAYER * 1

        actions_ids_list = [
            (MOVE_ACTION, player_unit_moves, self.get_move_info, self.perform_move_action),
            (CLAIM_BASE_ACTION, player_unit_claims, self.get_claim_base_info, self.perform_claim_base_action)
        ]
        last_id = 0
        for action_type, action_number, decryptor, act_func in actions_ids_list:
            actions_ids[action_type] = {
                'start': last_id,
                'end': last_id + action_number - 1,
                'decrypt_func': decryptor,
                'act_function': act_func
            }
            last_id += action_number

        return actions_ids, last_id

    def make_random_step(self):
        possible_actions = self.get_possible_actions()
        action_id = np.random.choice(possible_actions)
        return self.step(action_id)

    def get_possible_actions(self):
        possible_actions_id = []
        players_units = self.get_active_player_units()
        for _unit in players_units:
            for cell in self.board.get_free_adjacent_cells(*_unit.loc):
                possible_actions_id.append(self.get_move_action_id(_unit.loc, cell))
            if self.board.is_valid_claim(_unit.player_id, _unit.loc):
                possible_actions_id.append(self.get_claim_base_action_id(_unit.loc))
        return possible_actions_id

    def get_action_info(self, action_id):
        for action_type, info in self.action_dict.items():
            if info['start'] <= action_id <= info['end']:
                offset_id = action_id - info['start']
                return action_type, info['decrypt_func'](offset_id)

    def perform_move_action(self, start, end) -> Action:
        try:
            moving_unit = [u for u in self.get_active_player_units() if u.loc == start][0]
        except IndexError:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False, txt_result='Attempt to move unit from empty cell', is_valid=False)

        wrong_move_reason = ''
        if (end[0] < 0) or (end[0] >= self.board.board_size) \
                or (end[1] < 0) or (end[1] >= self.board.board_size) \
                or (self.board.board[end] == INVALID_CELL_ID):
            wrong_move_reason = 'Attempt to move to invalid cell'
        if end not in self.board.get_free_adjacent_cells(*moving_unit.loc):
            wrong_move_reason = 'Attempt to move to cell with a unit'
        if moving_unit.player_id != self.active_player:
            wrong_move_reason = 'Attempt to move unit from another player'
        if wrong_move_reason != '':
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False, txt_result=wrong_move_reason, is_valid=False)
        moving_unit.move(loc=end)

        self.exploration_map_dict[self.active_player][end] += 1
        return Action(reward=MOVE_NEG_REWARD_PER_TURN, finishes_game=False, txt_result='Move successful', is_valid=True)

    def get_move_action_id(self, start, end):
        unit_id = [u.loc for u in self.get_active_player_units()].index(start)
        try:
            move_id = unit_id * 6 + self.board.offsets.index((end[0] - start[0], end[1] - start[1]))
            return self.action_dict[MOVE_ACTION]['start'] + move_id
        except ValueError:
            print(f'Error: Invalid move from {start} to {end}')

    def get_move_info(self, action_id) -> Tuple:
        unit_id = action_id // 6
        offset_id = action_id % 6

        player_units = self.get_active_player_units()
        if unit_id >= len(player_units):
            raise ValueError("Invalid action_id")

        start = player_units[unit_id].loc
        offset = self.board.offsets[offset_id]
        end = (start[0] + offset[0], start[1] + offset[1])

        return start, end

    def perform_claim_base_action(self, base_loc) -> Action:
        if not self.board.is_valid_claim(player_id=self.active_player, cell_loc=base_loc):
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False, txt_result='Attempt to claim invalid base', is_valid=False)

        self.board.change_base_control(player_id=self.active_player, base_loc=base_loc)
        if len(self.board.get_controlled_bases(self.active_player)) < self.winning_base_count:
            return Action(reward=CLAIM_BASE_REWARD, finishes_game=False, is_valid=True, txt_result='Successfully claimed to claim owned base')
        else:
            return Action(reward=WIN_REWARD, finishes_game=True, is_valid=True, txt_result=f'Player {self.active_player} won')

    def get_claim_base_action_id(self, base_loc: Tuple[int, int]):
        units_loc = [u.loc for u in self.get_active_player_units()]
        if base_loc not in units_loc:
            raise ValueError("Invalid base_loc")
        unit_id = units_loc.index(base_loc)
        return self.action_dict[CLAIM_BASE_ACTION]['start'] + unit_id

    def get_claim_base_info(self, action_id) -> Tuple:
        player_units = self.get_active_player_units()
        if action_id >= len(player_units):
            raise ValueError("Invalid action_id")
        unit = player_units[action_id]

        return unit.loc,

    def get_active_player_units(self):
        return [unit for unit in self.board.units if unit.player_id == self.active_player]
