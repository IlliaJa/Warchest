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

MOVE_ACTION = 'move'
ATTACK_ACTION = 'attack'
CLAIM_BASE_ACTION = 'claim_base'
DEPLOY_ACTION = 'deploy'

# Spatial action constants
# action_id = verb * BOARD_DIM^2 + r * BOARD_DIM + q
# verb 0-5:  move in direction d
# verb 6-11: attack in direction d
# verb 12:   control (claim) current cell
# verb 13:   deploy onto this cell
N_VERBS = 14
BOARD_DIM = 7
ACTION_SPACE_SIZE = N_VERBS * BOARD_DIM * BOARD_DIM  # 686

GLOBAL_DIM = 5  # [turn_frac, my_bases, opp_bases, my_deploys_left, opp_deploys_left]

MOVE_EXPLORE_REWARD_MAX_TURN = 5
MOVE_EXPLORE_REWARD_PER_TURN = 0.1
MOVE_NEG_REWARD_PER_TURN = -0.002
ATTACK_REWARD = 0.1
INVALID_ACTION_REWARD = -0.02
CLAIM_BASE_REWARD = 0.0
WIN_REWARD = 1.0
LOSS_REWARD = -1.0

NUM_PLAYERS = 2
MAX_UNITS_PER_PLAYER = 2
MAX_DEPLOYS = 4  # lifetime deploy cap per player


class WarChestEnv(gym.Env):
    max_actions = 200
    winning_base_count = 6
    max_rewardable_moving_action = 30

    # Under 180° rotation direction d maps to its opposite.
    # Self-inverse: _OFFSET_FLIP[_OFFSET_FLIP[i]] == i.
    _OFFSET_FLIP = [3, 4, 5, 0, 1, 2]

    @staticmethod
    def encode_action(verb: int, r: int, q: int) -> int:
        return verb * BOARD_DIM * BOARD_DIM + r * BOARD_DIM + q

    @staticmethod
    def decode_action(action_id: int) -> Tuple[int, int, int]:
        verb = action_id // (BOARD_DIM * BOARD_DIM)
        cell = action_id % (BOARD_DIM * BOARD_DIM)
        r = cell // BOARD_DIM
        q = cell % BOARD_DIM
        return verb, r, q

    @staticmethod
    def remap_action(action_id: int) -> int:
        """Translate a spatial action between ego-centric and absolute frames.

        When active_player==2 the observation (and valid_action_mask) is rotated
        180°. Any action the policy returns from that rotated observation must be
        passed through this function before env.step(), and vice-versa.
        Self-inverse: remap_action(remap_action(a)) == a.

        Cell rotation: (r,q) → (s-r, s-q) where s = BOARD_DIM - 1 = 6.
        Direction flip: applied only to move (verb 0-5) and attack (verb 6-11).
        Control (12) and deploy (13) rotate spatially only — no verb change.
        """
        s = BOARD_DIM - 1
        verb, r, q = WarChestEnv.decode_action(action_id)
        r_rot = s - r
        q_rot = s - q
        if 0 <= verb <= 5:
            verb_rot = WarChestEnv._OFFSET_FLIP[verb]
        elif 6 <= verb <= 11:
            verb_rot = 6 + WarChestEnv._OFFSET_FLIP[verb - 6]
        else:
            verb_rot = verb
        return WarChestEnv.encode_action(verb_rot, r_rot, q_rot)

    def __init__(self, save_game_history: bool = False, debug_mode: bool = False):
        super().__init__()
        self.debug_mode = debug_mode

        self.state = None
        self.history = [] if save_game_history else None
        self.exploration_map_dict = None
        self.set_init_state()

        self.observation_space = self.get_observation_space()
        self.action_dict = self._build_action_dict()
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

    def _build_action_dict(self) -> Dict:
        return {
            MOVE_ACTION: {'act_function': self.perform_move_action},
            ATTACK_ACTION: {'act_function': self.perform_attack_action},
            CLAIM_BASE_ACTION: {'act_function': self.perform_claim_base_action},
            DEPLOY_ACTION: {'act_function': self.perform_deploy_action},
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.set_init_state()
        return self.generate_observation(), {}

    def set_init_state(self):
        board = Board()
        map_ = np.where(board.board == INVALID_CELL_ID, INVALID_CELL_ID, 0)
        self.exploration_map_dict = {1: map_.copy(), 2: map_.copy()}
        self.place_default_units(board)
        self.state = GameState(board=board, active_player=1, action_count=0)
        if self.history is not None:
            self.history = [deepcopy(self.state)]

    def set_state(self, state: GameState):
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
        self.state.active_player = 1 if self.state.active_player == 2 else 2

    def step(self, action_id):
        action_type, action_args = self.get_action_info(action_id)
        action = self.action_dict[action_type]['act_function'](*action_args)
        action.id = action_id
        action.player_id = self.active_player
        action.type = action_type
        action.additional_info = action_args

        if action.is_valid:
            self.action_count += 1
            self.swap_active_player()
            if self.history is not None:
                self.history.append(deepcopy(self.state))
            # If the newly active player has no valid actions the previous mover wins.
            if not action.finishes_game and not self.get_possible_actions():
                action.finishes_game = True
                action.reward += WIN_REWARD

        truncated = self.action_count >= self.max_actions
        if self.debug_mode:
            print(f'Got action_id {action.id} type={action.type} args={action.additional_info}')
        return self.generate_observation(), action.reward, action.finishes_game, truncated, {'action': action}

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
                    hexagon = patches.RegularPolygon(
                        (x, y), numVertices=6, radius=hex_radius, orientation=np.pi / 2,
                        edgecolor='black', facecolor=BASE_COLORS[int(board[r, q])]
                    )
                    ax.add_patch(hexagon)
                    ax.text(x, y - 0.3, f'r={r} q={q}', ha='center', va='center', fontsize=10)

        for _unit in self.board.units:
            x, y = self.convert_hex_grid_to_cartesian(*_unit.loc, hex_radius=hex_radius)
            ax.text(x, y, s=_unit.icon, ha='center', va='center', fontsize=30,
                    color=UNIT_COLORS[_unit.player_id])

        ax.set_aspect('equal')
        ax.autoscale_view()
        plt.margins(0)
        if created_ax:
            plt.show()

    def render_game(self):
        if self.history is None:
            raise ValueError('Game history not available. Set save_game_history=True.')
        GameRenderer(env=self, history=self.history).draw()

    @staticmethod
    def convert_hex_grid_to_cartesian(row, column, hex_radius=0.5):
        hex_height = (3 ** 0.5) * hex_radius
        x = row * hex_height
        y = column - row / 2
        return x, y

    def place_default_units(self, board):
        default_units = [
            (Swordsman(player_id=1, board=board), (1, 0)),
            (Swordsman(player_id=1, board=board), (4, 1)),
            (Swordsman(player_id=2, board=board), (2, 5)),
            (Swordsman(player_id=2, board=board), (5, 6)),
        ]
        for _unit, loc in default_units:
            board.deploy_unit(unit=_unit, place=loc)

    def get_observation_space(self):
        return gym.spaces.Dict({
            'board': gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(8, BOARD_DIM, BOARD_DIM),
                dtype=np.float32,
            ),
            'global': gym.spaces.Box(low=0.0, high=1.0, shape=(GLOBAL_DIM,), dtype=np.float32),
            'valid_action_mask': gym.spaces.Box(
                low=0, high=1, shape=(ACTION_SPACE_SIZE,), dtype=np.float32
            ),
            'active_player': gym.spaces.Discrete(2),
        })

    def generate_observation(self):
        active = self.active_player
        opponent = 3 - active
        s = self.board.board_size - 1  # 6

        raw_board = self.board.board
        expl = self.exploration_map_dict[active]
        if active == 2:
            raw_board = np.rot90(raw_board, 2).copy()
            expl = np.rot90(expl, 2).copy()

        # Build 8-channel encoded board (ego-centric, already rotated for P2)
        board_enc = np.zeros((8, BOARD_DIM, BOARD_DIM), dtype=np.float32)
        board_enc[0] = (raw_board == INVALID_CELL_ID)
        board_enc[1] = (raw_board == EMPTY_CELL_ID)
        board_enc[2] = (raw_board == UNCONTROLLED_BASE_CELL_ID)
        my_base_id = CONTROLLED_BASE_PLAYER_1_CELL_ID if active == 1 else CONTROLLED_BASE_PLAYER_2_CELL_ID
        opp_base_id = CONTROLLED_BASE_PLAYER_2_CELL_ID if active == 1 else CONTROLLED_BASE_PLAYER_1_CELL_ID
        board_enc[3] = (raw_board == my_base_id)
        board_enc[4] = (raw_board == opp_base_id)
        visits = np.clip(expl, 0, None).astype(np.float32)
        board_enc[5] = visits / (visits.max() + 1e-5)
        # Unit planes: ego-centric. Rotate coords for P2.
        for u in self.board.units:
            r, q = u.loc
            if active == 2:
                r, q = s - r, s - q
            if u.player_id == active:
                board_enc[6, r, q] = 1.0
            else:
                board_enc[7, r, q] = 1.0

        # Global features [5]
        my_bases = len(self.board.get_controlled_bases(active))
        opp_bases = len(self.board.get_controlled_bases(opponent))
        global_feats = np.array([
            self.action_count / self.max_actions,
            my_bases / self.winning_base_count,
            opp_bases / self.winning_base_count,
            (MAX_DEPLOYS - self.state.deploys_used[active]) / MAX_DEPLOYS,
            (MAX_DEPLOYS - self.state.deploys_used[opponent]) / MAX_DEPLOYS,
        ], dtype=np.float32)

        # Valid action mask [686]
        valid_ids = self.get_possible_actions()
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        if active == 2:
            for a in valid_ids:
                mask[self.remap_action(a)] = 1.0
            if self.debug_mode:
                print(
                    f'[obs_rotate] P2: valid_orig={sorted(valid_ids)[:5]} '
                    f'valid_rot={sorted(int(a) for a in np.where(mask)[0])[:5]}'
                )
        else:
            mask[valid_ids] = 1.0

        return {
            'board': board_enc,
            'global': global_feats,
            'valid_action_mask': mask,
            'active_player': active,
        }

    def get_possible_actions(self):
        """Return list of valid action IDs in absolute (non-rotated) frame."""
        active = self.active_player
        ids = []

        for u in self.get_active_player_units():
            r, q = u.loc
            # Move: step into any free adjacent cell
            for d, (dr, dq) in enumerate(self.board.offsets):
                target = (r + dr, q + dq)
                if target in self.board.get_free_adjacent_cells(r, q):
                    ids.append(self.encode_action(d, r, q))
            # Attack: adjacent enemy unit
            for d, (dr, dq) in enumerate(self.board.offsets):
                target = (r + dr, q + dq)
                enemy = self.board.get_unit_at(*target)
                if enemy is not None and enemy.player_id != active:
                    ids.append(self.encode_action(6 + d, r, q))
            # Control (claim)
            if self.board.is_valid_claim(active, (r, q)):
                ids.append(self.encode_action(12, r, q))

        # Deploy: onto any controlled empty location within budget
        if (len(self.get_active_player_units()) < MAX_UNITS_PER_PLAYER
                and self.state.deploys_used[active] < MAX_DEPLOYS):
            for loc in self.board.get_controlled_bases(active):
                if self.board.get_unit_at(*loc) is None:
                    ids.append(self.encode_action(13, *loc))

        return ids

    def get_action_info(self, action_id: int) -> Tuple[str, Tuple]:
        verb, r, q = self.decode_action(action_id)
        if 0 <= verb <= 5:
            return MOVE_ACTION, (verb, r, q)
        elif 6 <= verb <= 11:
            return ATTACK_ACTION, (verb, r, q)
        elif verb == 12:
            return CLAIM_BASE_ACTION, (verb, r, q)
        elif verb == 13:
            return DEPLOY_ACTION, (verb, r, q)
        raise ValueError(f'Unknown verb {verb} in action_id {action_id}')

    def make_random_step(self):
        possible = self.get_possible_actions()
        if not possible:
            # No valid actions: active player forfeits, opponent wins.
            dummy = Action(
                reward=LOSS_REWARD, finishes_game=True, is_valid=True,
                txt_result='No valid actions — forfeit',
            )
            dummy.player_id = self.active_player
            dummy.type = None
            dummy.additional_info = None
            return self.generate_observation(), LOSS_REWARD, True, False, {'action': dummy}
        return self.step(np.random.choice(possible))

    # ------------------------------------------------------------------
    # Action perform functions
    # ------------------------------------------------------------------

    def perform_move_action(self, verb: int, r: int, q: int) -> Action:
        start = (r, q)
        offset = self.board.offsets[verb]
        end = (r + offset[0], q + offset[1])

        try:
            moving_unit = next(u for u in self.get_active_player_units() if u.loc == start)
        except StopIteration:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='No unit at source cell', is_valid=False)

        if end not in self.board.get_free_adjacent_cells(r, q):
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='Target cell not free', is_valid=False)

        moving_unit.move(loc=end)
        self.exploration_map_dict[self.active_player][end] += 1
        return Action(reward=MOVE_NEG_REWARD_PER_TURN, finishes_game=False,
                      txt_result='Move successful', is_valid=True)

    def perform_attack_action(self, verb: int, r: int, q: int) -> Action:
        direction = verb - 6
        start = (r, q)
        offset = self.board.offsets[direction]
        target = (r + offset[0], q + offset[1])

        try:
            next(u for u in self.get_active_player_units() if u.loc == start)
        except StopIteration:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='No own unit at source cell', is_valid=False)

        enemy = self.board.get_unit_at(*target)
        if enemy is None or enemy.player_id == self.active_player:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='No enemy unit at target cell', is_valid=False)

        self.board.remove_unit(enemy)
        return Action(reward=ATTACK_REWARD, finishes_game=False,
                      txt_result='Attack successful', is_valid=True)

    def perform_claim_base_action(self, verb: int, r: int, q: int) -> Action:
        base_loc = (r, q)
        if not self.board.is_valid_claim(player_id=self.active_player, cell_loc=base_loc):
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='Invalid claim', is_valid=False)

        self.board.change_base_control(player_id=self.active_player, base_loc=base_loc)
        if len(self.board.get_controlled_bases(self.active_player)) < self.winning_base_count:
            return Action(reward=CLAIM_BASE_REWARD, finishes_game=False, is_valid=True,
                          txt_result='Claimed base')
        return Action(reward=WIN_REWARD, finishes_game=True, is_valid=True,
                      txt_result=f'Player {self.active_player} won')

    def perform_deploy_action(self, verb: int, r: int, q: int) -> Action:
        active = self.active_player
        target = (r, q)

        if self.state.deploys_used[active] >= MAX_DEPLOYS:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='Deploy cap reached', is_valid=False)
        if len(self.get_active_player_units()) >= MAX_UNITS_PER_PLAYER:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='Max concurrent units reached', is_valid=False)
        if self.board.get_unit_at(*target) is not None:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='Target cell occupied', is_valid=False)

        try:
            new_unit = Swordsman(player_id=active, board=self.board)
            self.board.deploy_unit(unit=new_unit, place=target)
        except Exception as e:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result=f'Deploy failed: {e}', is_valid=False)

        self.state.deploys_used[active] += 1
        return Action(reward=0.0, finishes_game=False, is_valid=True,
                      txt_result='Unit deployed')

    def get_active_player_units(self):
        return [u for u in self.board.units if u.player_id == self.active_player]
