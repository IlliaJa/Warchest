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
from .game_state import GameState, COIN_SWORD, COIN_KNIGHT, COIN_ROYAL, DECK
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
COIN_ICONS = {
    COIN_SWORD: '♖',
    COIN_KNIGHT: '♞',
    COIN_ROYAL: '♚',
}

MOVE_ACTION = 'move'
ATTACK_ACTION = 'attack'
CLAIM_BASE_ACTION = 'claim_base'
DEPLOY_ACTION = 'deploy'
CLAIM_INITIATIVE_ACTION = 'claim_initiative'
PASS_ACTION = 'pass'

# ---------------------------------------------------------------------------
# Action encoding (Phase 1a — temporary flat head)
#
# Spatial actions: id = verb * BOARD_DIM^2 + r * BOARD_DIM + q
#   verb 0-5:   move in direction d
#   verb 6-11:  attack in direction d
#   verb 12:    control (claim) current cell
#   verb 13:    deploy a Swordsman onto this cell
#   verb 14:    deploy a Knight onto this cell
# Face-down actions (no board cell): appended after the spatial block.
#   SPATIAL_SIZE + 0..2: claim_initiative paying {sword, knight, royal}
#   SPATIAL_SIZE + 3..5: pass paying {sword, knight, royal}
# Deploy is per-type because an empty target cell does not determine the unit;
# move/attack/control are not, because the occupied source cell does.
# ---------------------------------------------------------------------------
N_VERBS = 15
BOARD_DIM = 7
SPATIAL_SIZE = N_VERBS * BOARD_DIM * BOARD_DIM  # 735
FACEDOWN_KINDS = 2  # claim_initiative, pass
FACEDOWN_SIZE = FACEDOWN_KINDS * len(DECK)  # 6
ACTION_SPACE_SIZE = SPATIAL_SIZE + FACEDOWN_SIZE  # 741

# Verbs that deploy, and which unit type each deploys.
DEPLOY_VERBS = {13: COIN_SWORD, 14: COIN_KNIGHT}
UNIT_CLASS_BY_COIN = {COIN_SWORD: Swordsman, COIN_KNIGHT: Knight}

# Coin <-> contiguous index, used for the face-down action block and obs encoding.
COIN_TO_IDX = {COIN_SWORD: 0, COIN_KNIGHT: 1, COIN_ROYAL: 2}
IDX_TO_COIN = list(DECK)

BOARD_CHANNELS = 10  # see Policy docstring for the channel map
GLOBAL_DIM = 8  # [turn_frac, my_bases, opp_bases, hand_sword, hand_knight, hand_royal, my_initiative, opp_coins_left]

MOVE_EXPLORE_REWARD_MAX_TURN = 5
MOVE_EXPLORE_REWARD_PER_TURN = 0.1
MOVE_NEG_REWARD_PER_TURN = -0.002
ATTACK_REWARD = 0.1
INVALID_ACTION_REWARD = -0.02
CLAIM_BASE_REWARD = 0.0
WIN_REWARD = 1.0
LOSS_REWARD = -1.0

NUM_PLAYERS = 2

OBS_VERSION = 1


class WarChestEnv(gym.Env):
    max_rounds = 50  # a round = both players empty a hand (~6 coin-plays); truncate here
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
        """Decode a *spatial* action id into (verb, r, q). Spatial ids only."""
        verb = action_id // (BOARD_DIM * BOARD_DIM)
        cell = action_id % (BOARD_DIM * BOARD_DIM)
        r = cell // BOARD_DIM
        q = cell % BOARD_DIM
        return verb, r, q

    @staticmethod
    def encode_facedown(kind: int, coin: int) -> int:
        """kind 0 = claim_initiative, 1 = pass."""
        return SPATIAL_SIZE + kind * len(DECK) + COIN_TO_IDX[coin]

    @staticmethod
    def decode_facedown(action_id: int) -> Tuple[int, int]:
        """Return (kind, coin) for a face-down action id."""
        off = action_id - SPATIAL_SIZE
        kind = off // len(DECK)
        coin = IDX_TO_COIN[off % len(DECK)]
        return kind, coin

    @staticmethod
    def remap_action(action_id: int) -> int:
        """Translate a spatial action between ego-centric and absolute frames.

        When active_player==2 the observation (and valid_action_mask) is rotated
        180°. Any action the policy returns from that rotated observation must be
        passed through this function before env.step(), and vice-versa.
        Self-inverse: remap_action(remap_action(a)) == a.

        Face-down actions are non-spatial and map to themselves. Cell rotation:
        (r,q) → (s-r, s-q) where s = BOARD_DIM - 1 = 6. Direction flip applies only
        to move (0-5) and attack (6-11); control (12) and deploy (13,14) rotate
        spatially only — no verb change.
        """
        if action_id >= SPATIAL_SIZE:
            return action_id
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
            CLAIM_INITIATIVE_ACTION: {'act_function': self.perform_claim_initiative_action},
            PASS_ACTION: {'act_function': self.perform_pass_action},
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.set_init_state()
        return self.generate_observation(), {}

    def set_init_state(self):
        # Board starts with control markers on the starting locations but NO units;
        # units are deployed from hand during play. Initiative (and thus who acts
        # first) is assigned randomly, mirroring the setup Initiative Marker flip.
        board = Board()
        map_ = np.where(board.board == INVALID_CELL_ID, INVALID_CELL_ID, 0)
        self.exploration_map_dict = {1: map_.copy(), 2: map_.copy()}
        owner = int(np.random.choice([1, 2]))
        self.state = GameState(board=board, active_player=owner, action_count=0,
                               initiative_owner=owner)
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

    # ------------------------------------------------------------------
    # Round / turn controller
    # ------------------------------------------------------------------

    def _advance_turn(self):
        """Pass the turn after a coin has been spent.

        Players alternate one coin per turn; a player with an empty hand is skipped
        while the other finishes. When both hands are empty the round ends: both
        hands refresh and the initiative owner acts first.
        """
        active = self.state.active_player
        other = 3 - active
        if self.state.hands[other]:
            self.state.active_player = other
        elif self.state.hands[active]:
            pass  # opponent is out of coins; keep playing
        else:
            self._start_new_round()

    def _start_new_round(self):
        self.state.hands = {1: set(DECK), 2: set(DECK)}
        self.state.initiative_transferred_this_round = False
        self.state.active_player = self.state.initiative_owner
        self.state.round_number += 1

    def _spend_coin(self, coin: int):
        self.state.last_coin = coin
        self.state.last_coin_player = self.active_player
        self.state.hands[self.active_player].discard(coin)

    def step(self, action_id):
        action_type, action_args = self.get_action_info(action_id)
        action = self.action_dict[action_type]['act_function'](*action_args)
        action.id = action_id
        action.player_id = self.active_player
        action.type = action_type
        action.additional_info = action_args

        if action.is_valid:
            self.action_count += 1
            self.state.last_action_type = action_type
            if not action.finishes_game:
                self._advance_turn()
            if self.history is not None:
                self.history.append(deepcopy(self.state))
            # If the newly active player has no valid actions the previous mover wins.
            # With pass always legal this is a safety net rather than a normal path.
            if not action.finishes_game and not self.get_possible_actions():
                action.finishes_game = True
                action.reward += WIN_REWARD

        truncated = self.state.round_number >= self.max_rounds
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

        # Initiative mark (top-left) and the coin spent on the action that produced
        # this state (top-right). Drawn as axes text so they survive even when the
        # replay renderer overwrites the title.
        owner = self.state.initiative_owner
        ax.text(0.02, 0.98, f'★ initiative P{owner}', transform=ax.transAxes,
                ha='left', va='top', fontsize=12, fontweight='bold',
                color=UNIT_COLORS[owner])
        if self.state.last_coin is not None:
            player = self.state.last_coin_player
            ax.text(0.98, 0.98, COIN_ICONS[self.state.last_coin], transform=ax.transAxes,
                    ha='right', va='top', fontsize=28, color=UNIT_COLORS[player])
            ax.text(0.98, 0.85, f'P{player} {self.state.last_action_type or ""}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    color=UNIT_COLORS[player])

        ax.set_aspect('equal')
        ax.autoscale_view()
        ax.set_title(self._render_status_text(), fontsize=10)
        plt.margins(0)
        if created_ax:
            plt.show()

    def _render_status_text(self) -> str:
        coin_names = {COIN_SWORD: 'S', COIN_KNIGHT: 'K', COIN_ROYAL: 'R'}

        def hand_str(pid):
            return ''.join(coin_names[c] for c in DECK if c in self.state.hands[pid]) or '-'

        return (
            f'round={self.state.round_number} active=P{self.active_player} '
            f'init=P{self.state.initiative_owner} | '
            f'P1 hand=[{hand_str(1)}] P2 hand=[{hand_str(2)}] | '
            f'P1 bases={len(self.board.get_controlled_bases(1))} '
            f'P2 bases={len(self.board.get_controlled_bases(2))}'
        )

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

    def get_observation_space(self):
        return gym.spaces.Dict({
            'board': gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(BOARD_CHANNELS, BOARD_DIM, BOARD_DIM),
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

        # Build encoded board (ego-centric, already rotated for P2).
        board_enc = np.zeros((BOARD_CHANNELS, BOARD_DIM, BOARD_DIM), dtype=np.float32)
        board_enc[0] = (raw_board == INVALID_CELL_ID)
        board_enc[1] = (raw_board == EMPTY_CELL_ID)
        board_enc[2] = (raw_board == UNCONTROLLED_BASE_CELL_ID)
        my_base_id = CONTROLLED_BASE_PLAYER_1_CELL_ID if active == 1 else CONTROLLED_BASE_PLAYER_2_CELL_ID
        opp_base_id = CONTROLLED_BASE_PLAYER_2_CELL_ID if active == 1 else CONTROLLED_BASE_PLAYER_1_CELL_ID
        board_enc[3] = (raw_board == my_base_id)
        board_enc[4] = (raw_board == opp_base_id)
        visits = np.clip(expl, 0, None).astype(np.float32)
        board_enc[5] = visits / (visits.max() + 1e-5)
        # Unit planes: per type per owner, ego-centric. Channels:
        #   6 own sword  7 own knight  8 opp sword  9 opp knight
        type_plane = {COIN_SWORD: 0, COIN_KNIGHT: 1}
        for u in self.board.units:
            r, q = u.loc
            if active == 2:
                r, q = s - r, s - q
            owner_offset = 6 if u.player_id == active else 8
            board_enc[owner_offset + type_plane[u.id], r, q] = 1.0

        # Global features [GLOBAL_DIM]
        my_bases = len(self.board.get_controlled_bases(active))
        opp_bases = len(self.board.get_controlled_bases(opponent))
        hand = self.state.hands[active]
        global_feats = np.array([
            min(self.state.round_number / self.max_rounds, 1.0),
            my_bases / self.winning_base_count,
            opp_bases / self.winning_base_count,
            float(COIN_SWORD in hand),
            float(COIN_KNIGHT in hand),
            float(COIN_ROYAL in hand),
            float(self.state.initiative_owner == active),
            len(self.state.hands[opponent]) / len(DECK),
        ], dtype=np.float32)

        # Valid action mask [ACTION_SPACE_SIZE]
        valid_ids = self.get_possible_actions()
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        if active == 2:
            for a in valid_ids:
                mask[self.remap_action(a)] = 1.0
        else:
            mask[valid_ids] = 1.0

        return {
            'board': board_enc,
            'global': global_feats,
            'valid_action_mask': mask,
            'active_player': active,
        }

    def get_possible_actions(self):
        """Return valid action IDs in absolute (non-rotated) frame."""
        active = self.active_player
        hand = self.state.hands[active]
        ids = []

        units = self.get_active_player_units()
        on_board_types = {u.id for u in units}

        # Maneuvers: gated by holding a coin matching the unit's type.
        for u in units:
            if u.id not in hand:
                continue
            r, q = u.loc
            for d, (dr, dq) in enumerate(self.board.offsets):
                target = (r + dr, q + dq)
                if target in self.board.get_free_adjacent_cells(r, q):
                    ids.append(self.encode_action(d, r, q))
            for d, (dr, dq) in enumerate(self.board.offsets):
                target = (r + dr, q + dq)
                enemy = self.board.get_unit_at(*target)
                if enemy is not None and enemy.player_id != active:
                    ids.append(self.encode_action(6 + d, r, q))
            if self.board.is_valid_claim(active, (r, q)):
                ids.append(self.encode_action(12, r, q))

        # Deploy: a coin in hand whose unit type is not already on the board, onto
        # any controlled empty location. (One unit of each type at a time.)
        controlled_empty = [
            loc for loc in self.board.get_controlled_bases(active)
            if self.board.get_unit_at(*loc) is None
        ]
        for verb, coin in DEPLOY_VERBS.items():
            if coin in hand and coin not in on_board_types and controlled_empty:
                for loc in controlled_empty:
                    ids.append(self.encode_action(verb, *loc))

        # Face-down actions: any coin in hand may pass; claim_initiative needs the
        # marker held by the opponent and untransferred this round.
        can_claim = (
            active != self.state.initiative_owner
            and not self.state.initiative_transferred_this_round
        )
        for coin in hand:
            ids.append(self.encode_facedown(1, coin))  # pass
            if can_claim:
                ids.append(self.encode_facedown(0, coin))  # claim_initiative

        return ids

    def get_action_info(self, action_id: int) -> Tuple[str, Tuple]:
        if action_id >= SPATIAL_SIZE:
            kind, coin = self.decode_facedown(action_id)
            if kind == 0:
                return CLAIM_INITIATIVE_ACTION, (coin,)
            return PASS_ACTION, (coin,)

        verb, r, q = self.decode_action(action_id)
        if 0 <= verb <= 5:
            return MOVE_ACTION, (verb, r, q)
        elif 6 <= verb <= 11:
            return ATTACK_ACTION, (verb, r, q)
        elif verb == 12:
            return CLAIM_BASE_ACTION, (verb, r, q)
        elif verb in DEPLOY_VERBS:
            return DEPLOY_ACTION, (DEPLOY_VERBS[verb], r, q)
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

        if moving_unit.id not in self.state.hands[self.active_player]:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='No matching coin in hand', is_valid=False)

        if end not in self.board.get_free_adjacent_cells(r, q):
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='Target cell not free', is_valid=False)

        moving_unit.move(loc=end)
        self.exploration_map_dict[self.active_player][end] += 1
        self._spend_coin(moving_unit.id)
        return Action(reward=MOVE_NEG_REWARD_PER_TURN, finishes_game=False,
                      txt_result='Move successful', is_valid=True)

    def perform_attack_action(self, verb: int, r: int, q: int) -> Action:
        direction = verb - 6
        start = (r, q)
        offset = self.board.offsets[direction]
        target = (r + offset[0], q + offset[1])

        try:
            attacker = next(u for u in self.get_active_player_units() if u.loc == start)
        except StopIteration:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='No own unit at source cell', is_valid=False)

        if attacker.id not in self.state.hands[self.active_player]:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='No matching coin in hand', is_valid=False)

        enemy = self.board.get_unit_at(*target)
        if enemy is None or enemy.player_id == self.active_player:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='No enemy unit at target cell', is_valid=False)

        self.board.remove_unit(enemy)
        self._spend_coin(attacker.id)
        return Action(reward=ATTACK_REWARD, finishes_game=False,
                      txt_result='Attack successful', is_valid=True)

    def perform_claim_base_action(self, verb: int, r: int, q: int) -> Action:
        base_loc = (r, q)
        if not self.board.is_valid_claim(player_id=self.active_player, cell_loc=base_loc):
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='Invalid claim', is_valid=False)

        unit = self.board.get_unit_at(r, q)
        if unit is None or unit.id not in self.state.hands[self.active_player]:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='No matching coin in hand', is_valid=False)

        self.board.change_base_control(player_id=self.active_player, base_loc=base_loc)
        self._spend_coin(unit.id)
        if len(self.board.get_controlled_bases(self.active_player)) < self.winning_base_count:
            return Action(reward=CLAIM_BASE_REWARD, finishes_game=False, is_valid=True,
                          txt_result='Claimed base')
        return Action(reward=WIN_REWARD, finishes_game=True, is_valid=True,
                      txt_result=f'Player {self.active_player} won')

    def perform_deploy_action(self, coin: int, r: int, q: int) -> Action:
        active = self.active_player
        target = (r, q)

        if coin not in self.state.hands[active]:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='No matching coin in hand', is_valid=False)
        if any(u.id == coin for u in self.get_active_player_units()):
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='Unit of this type already on board', is_valid=False)
        if self.board.get_unit_at(*target) is not None:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='Target cell occupied', is_valid=False)

        try:
            new_unit = UNIT_CLASS_BY_COIN[coin](player_id=active, board=self.board)
            self.board.deploy_unit(unit=new_unit, place=target)
        except Exception as e:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result=f'Deploy failed: {e}', is_valid=False)

        self._spend_coin(coin)
        return Action(reward=0.0, finishes_game=False, is_valid=True,
                      txt_result='Unit deployed')

    def perform_claim_initiative_action(self, coin: int) -> Action:
        active = self.active_player
        if coin not in self.state.hands[active]:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='No matching coin in hand', is_valid=False)
        if active == self.state.initiative_owner or self.state.initiative_transferred_this_round:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='Cannot claim initiative', is_valid=False)

        self.state.initiative_owner = active
        self.state.initiative_transferred_this_round = True
        self._spend_coin(coin)
        return Action(reward=0.0, finishes_game=False, is_valid=True,
                      txt_result='Claimed initiative')

    def perform_pass_action(self, coin: int) -> Action:
        if coin not in self.state.hands[self.active_player]:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='No matching coin in hand', is_valid=False)
        self._spend_coin(coin)
        return Action(reward=0.0, finishes_game=False, is_valid=True, txt_result='Passed')

    def get_active_player_units(self):
        return [u for u in self.board.units if u.player_id == self.active_player]
