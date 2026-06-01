import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .units import *
from .board import Board
from .cell_ids import *
from .game_renderer import GameRenderer
from .coin_render import draw_coin, draw_zone
from typing import Tuple, Dict
from .action import Action
from .game_state import (
    GameState, COIN_SWORD, COIN_KNIGHT, COIN_ROYAL, DECK,
    INITIAL_BAG, SUPPLY, INITIAL_OWNED, HAND_SIZE,
)
from collections import Counter
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
BOLSTER_ACTION = 'bolster'
RECRUIT_ACTION = 'recruit'

# ---------------------------------------------------------------------------
# Action encoding (Phase 1c — temporary flat head)
#
# Spatial actions: id = verb * BOARD_DIM^2 + r * BOARD_DIM + q
#   verb 0-5:   move in direction d
#   verb 6-11:  attack in direction d
#   verb 12:    control (claim) current cell
#   verb 13:    deploy a Swordsman onto this cell
#   verb 14:    deploy a Knight onto this cell
#   verb 15:    bolster the matching unit on this cell (type from the unit there)
# Face-down actions (no board cell): appended after the spatial block.
#   +0..2:  claim_initiative paying {S, K, R}
#   +3..5:  pass            paying {S, K, R}
#   +6..11: recruit take-type {S, K} × pay-coin {S, K, R}
# Deploy is per-type because an empty target cell does not determine the unit;
# move/attack/control/bolster are not, because the occupied source cell does.
# ---------------------------------------------------------------------------
N_VERBS = 16
BOARD_DIM = 7
SPATIAL_SIZE = N_VERBS * BOARD_DIM * BOARD_DIM  # 784

# Verbs that deploy, and which unit type each deploys; bolster is a single verb.
DEPLOY_VERBS = {13: COIN_SWORD, 14: COIN_KNIGHT}
BOLSTER_VERB = 15
UNIT_CLASS_BY_COIN = {COIN_SWORD: Swordsman, COIN_KNIGHT: Knight}

# Coin <-> contiguous index, used for the face-down action block and obs encoding.
COIN_TO_IDX = {COIN_SWORD: 0, COIN_KNIGHT: 1, COIN_ROYAL: 2}
IDX_TO_COIN = list(DECK)

# Recruitable unit types (those with a supply); the royal has none.
RECRUIT_TYPES = tuple(c for c in DECK if SUPPLY.get(c, 0) > 0)  # (S, K)
_RECRUIT_BLOCK = 2 * len(DECK)  # claim (3) + pass (3) precede recruit in the block
FACEDOWN_SIZE = _RECRUIT_BLOCK + len(RECRUIT_TYPES) * len(DECK)  # 6 + 6 = 12
ACTION_SPACE_SIZE = SPATIAL_SIZE + FACEDOWN_SIZE  # 796

# ---------------------------------------------------------------------------
# Verb grouping for the factored policy head (Phase 2).
# Each flat action id belongs to exactly one top-level verb. The factored head
# learns P(verb) explicitly and P(action | verb) over that verb's legal actions;
# the joint stays a single distribution over ACTION_SPACE_SIZE flat ids.
# ---------------------------------------------------------------------------
(V_MOVE, V_ATTACK, V_CONTROL, V_DEPLOY, V_BOLSTER, V_CLAIM, V_PASS, V_RECRUIT) = range(8)
N_FACTORED_VERBS = 8


def verb_of_action(action_id: int) -> int:
    if action_id < SPATIAL_SIZE:
        sv = action_id // (BOARD_DIM * BOARD_DIM)
        if sv <= 5:
            return V_MOVE
        if sv <= 11:
            return V_ATTACK
        if sv == 12:
            return V_CONTROL
        if sv in DEPLOY_VERBS:
            return V_DEPLOY
        return V_BOLSTER  # sv == BOLSTER_VERB
    off = action_id - SPATIAL_SIZE
    if off < len(DECK):
        return V_CLAIM
    if off < _RECRUIT_BLOCK:
        return V_PASS
    return V_RECRUIT


# Static map flat action id -> verb index; consumed by the factored policy head.
VERB_OF_ACTION = np.array([verb_of_action(a) for a in range(ACTION_SPACE_SIZE)], dtype=np.int64)

BOARD_CHANNELS = 10  # see Policy docstring for the channel map

# Unit coin types (deployable); the royal coin has no board unit.
UNIT_COINS = (COIN_SWORD, COIN_KNIGHT)
OWNED_TOTAL = sum(INITIAL_OWNED.values())  # 9 (used to normalize bag size)
STACK_NORM = max(INITIAL_OWNED.values())  # 4 (max coins of one type on one stack)

# Global feature layout (ego-centric, OBS_VERSION 3). Counts normalized by initial owned.
#   [0] round fraction   [1] my bases   [2] opp bases   [3] my initiative
#   own (known), per coin {S,K,R} unless noted:
#     [4:7] hand   [7:10] bag   [10:13] discard   [13:15] supply {S,K}   [15] bag_size/OWNED_TOTAL
#   opponent (public):
#     [16:18] on_board {S,K}   [18:21] faceup_discard {S,K,R}   [21:23] supply {S,K}
#     [23:26] hidden_pool {S,K,R}   [26] opp hand_size/HAND_SIZE
#   [27] initiative already transferred this round
GLOBAL_DIM = 28
OBS_VERSION = 3

# Privileged critic-only features: the opponent's true hidden split, per coin {S,K,R}.
#   [0:3] opp hand   [3:6] opp bag   [6:9] opp face-down discard
PRIV_DIM = 9

MOVE_EXPLORE_REWARD_MAX_TURN = 5
MOVE_EXPLORE_REWARD_PER_TURN = 0.1
MOVE_NEG_REWARD_PER_TURN = -0.002
ATTACK_REWARD = 0.1
INVALID_ACTION_REWARD = -0.02
CLAIM_BASE_REWARD = 0.0
WIN_REWARD = 1.0
LOSS_REWARD = -1.0

NUM_PLAYERS = 2


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
        """kind 0 = claim_initiative, 1 = pass, paying `coin`."""
        return SPATIAL_SIZE + kind * len(DECK) + COIN_TO_IDX[coin]

    @staticmethod
    def encode_recruit(take: int, pay: int) -> int:
        """Recruit a supply coin of type `take`, paying hand coin `pay`."""
        take_idx = RECRUIT_TYPES.index(take)
        return SPATIAL_SIZE + _RECRUIT_BLOCK + take_idx * len(DECK) + COIN_TO_IDX[pay]

    @staticmethod
    def decode_facedown(action_id: int) -> Tuple[str, Tuple]:
        """Return (action_type, args) for a face-down action id."""
        off = action_id - SPATIAL_SIZE
        if off < len(DECK):
            return CLAIM_INITIATIVE_ACTION, (IDX_TO_COIN[off],)
        if off < _RECRUIT_BLOCK:
            return PASS_ACTION, (IDX_TO_COIN[off - len(DECK)],)
        r = off - _RECRUIT_BLOCK
        take = RECRUIT_TYPES[r // len(DECK)]
        pay = IDX_TO_COIN[r % len(DECK)]
        return RECRUIT_ACTION, (pay, take)

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
            BOLSTER_ACTION: {'act_function': self.perform_bolster_action},
            CLAIM_INITIATIVE_ACTION: {'act_function': self.perform_claim_initiative_action},
            PASS_ACTION: {'act_function': self.perform_pass_action},
            RECRUIT_ACTION: {'act_function': self.perform_recruit_action},
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
        for pid in (1, 2):
            self._draw_hand(pid)
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
        self.state.initiative_transferred_this_round = False
        self.state.active_player = self.state.initiative_owner
        self.state.round_number += 1
        for pid in (1, 2):
            self._draw_hand(pid)

    def _draw_hand(self, player: int):
        """Draw up to HAND_SIZE coins from the bag into the player's hand.

        Reshuffles the discard (face-up and face-down) into the bag when it empties.
        If fewer than HAND_SIZE coins are available in total, draws what there is
        (the rulebook's 'not enough coins' case).
        """
        self.state.hands[player] = Counter()
        for _ in range(HAND_SIZE):
            if not self.state.bags[player]:
                self._reshuffle(player)
                if not self.state.bags[player]:
                    break  # nothing left to draw
            coin = self._draw_one(player)
            self.state.hands[player][coin] += 1

    def _reshuffle(self, player: int):
        """Move the whole discard pile back into the bag; face-up info is lost."""
        self.state.bags[player] += self.state.discard_faceup[player]
        self.state.bags[player] += self.state.discard_facedown[player]
        self.state.discard_faceup[player] = Counter()
        self.state.discard_facedown[player] = Counter()

    def _draw_one(self, player: int) -> int:
        """Remove and return one coin chosen uniformly from the bag's contents."""
        coins = list(self.state.bags[player].elements())
        coin = int(np.random.choice(coins))
        self.state.bags[player][coin] -= 1
        if self.state.bags[player][coin] == 0:
            del self.state.bags[player][coin]
        return coin

    def _play_coin(self, coin: int, dest: str):
        """Remove a coin from the active player's hand and route it to `dest`.

        dest: 'faceup' / 'facedown' discard, or 'board' (the coin becomes a unit).
        """
        active = self.active_player
        self.state.hands[active][coin] -= 1
        if self.state.hands[active][coin] == 0:
            del self.state.hands[active][coin]
        if dest == 'faceup':
            self.state.discard_faceup[active][coin] += 1
        elif dest == 'facedown':
            self.state.discard_facedown[active][coin] += 1
        # dest == 'board': the coin is now the deployed unit; nothing to store here.
        self.state.last_coin = coin
        self.state.last_coin_player = active

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
            fig, ax = plt.subplots(figsize=(9, 11))
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            created_ax = True
        else:
            ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        hex_radius = 0.5
        board = self.board.board

        xs, ys = [], []
        for r in range(board.shape[0]):
            for q in range(board.shape[1]):
                if board[r, q] != INVALID_CELL_ID:
                    x, y = self.convert_hex_grid_to_cartesian(r, q, hex_radius=hex_radius)
                    xs.append(x)
                    ys.append(y)
                    hexagon = patches.RegularPolygon(
                        (x, y), numVertices=6, radius=hex_radius, orientation=np.pi / 2,
                        edgecolor='black', facecolor=BASE_COLORS[int(board[r, q])]
                    )
                    ax.add_patch(hexagon)
                    ax.text(x, y - 0.34, f'{r},{q}', ha='center', va='center',
                            fontsize=6, color='silver')

        # On-board units as coin discs (face = type color, rim = player) plus a stack badge.
        for _unit in self.board.units:
            x, y = self.convert_hex_grid_to_cartesian(*_unit.loc, hex_radius=hex_radius)
            draw_coin(ax, x, y, _unit.id, _unit.player_id, radius=0.3, fontsize=22)
            ax.text(x + 0.22, y + 0.22, s=str(_unit.stack), ha='center', va='center',
                    fontsize=11, fontweight='bold', color=UNIT_COLORS[_unit.player_id],
                    zorder=5)

        # Coin-economy bands: P1 below the board, P2 above it. Each player's six
        # zones sit in a single row spanning the window width, so every coin pile
        # (incl. the normally-hidden bag and face-down discard) is visible in replay.
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        board_cx = (x_min + x_max) / 2
        col_w = 2.0
        zone_names = ('Hand', 'Bag', 'Supply', 'Discard ↑', 'Discard ↓', 'Eliminated')
        # Center the row of zones on the board.
        block_left = board_cx - (len(zone_names) - 1) * col_w / 2 - 0.3

        def draw_band(pid, row_y):
            star = ' ★' if self.state.initiative_owner == pid else ''
            ax.text(block_left, row_y + 0.62, f'Player {pid}{star}', ha='left',
                    va='center', fontsize=12, fontweight='bold', color=UNIT_COLORS[pid])
            counters = (
                self.state.hands[pid], self.state.bags[pid], self.state.supply[pid],
                self.state.discard_faceup[pid], self.state.discard_facedown[pid],
                self.state.boxed[pid],
            )
            for k, (label, counter) in enumerate(zip(zone_names, counters)):
                draw_zone(ax, block_left + k * col_w, row_y, label, counter, pid)

        draw_band(2, y_max + 1.0)
        draw_band(1, y_min - 1.0)

        # Round / turn banner and the coin spent on the action that produced this
        # state — axes text so they survive the replay renderer overwriting the title.
        ax.text(0.5, 0.995, f'round {self.state.round_number}  ·  P{self.active_player} to act',
                transform=ax.transAxes, ha='center', va='top', fontsize=11, color='black')
        # Last action played — in the empty gap right of the board, clear of the
        # top band's Eliminated zone and the nav buttons.
        if self.state.last_coin is not None:
            player = self.state.last_coin_player
            ax.text(0.985, 0.78, COIN_ICONS[self.state.last_coin], transform=ax.transAxes,
                    ha='right', va='top', fontsize=24, color=UNIT_COLORS[player])
            ax.text(0.985, 0.70, f'P{player} {self.state.last_action_type or ""}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    color=UNIT_COLORS[player])

        # Limits derived from actual content, symmetric about the board so nothing
        # clips and the board stays centered; datalim lets the wide axis fill space.
        rightmost = block_left + (len(zone_names) - 1) * col_w + 1.8
        half = max(board_cx - block_left, rightmost - board_cx, (x_max - x_min) / 2) + 0.4
        ax.set_aspect('equal')  # default adjustable='box': shrink the box, never the limits
        ax.set_xlim(board_cx - half, board_cx + half)
        ax.set_ylim(y_min - 1.0 - 0.7, y_max + 1.0 + 0.62 + 0.5)
        ax.set_title(self._render_status_text(), fontsize=10)
        if created_ax:
            plt.show()

    def _render_status_text(self) -> str:
        return (
            f'round={self.state.round_number} active=P{self.active_player} '
            f'init=P{self.state.initiative_owner} | '
            f'bases {len(self.board.get_controlled_bases(1))}-{len(self.board.get_controlled_bases(2))}'
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
            board_enc[owner_offset + type_plane[u.id], r, q] = u.stack / STACK_NORM

        # Global features [GLOBAL_DIM] — ego-centric coin-counting (OBS_VERSION 3).
        my_bases = len(self.board.get_controlled_bases(active))
        opp_bases = len(self.board.get_controlled_bases(opponent))

        own_hand = self.state.hands[active]
        own_bag = self.state.bags[active]
        own_discard = self.state.discard_faceup[active] + self.state.discard_facedown[active]
        own_supply = self.state.supply[active]

        # On-board counts are stack heights (committed coins), one unit per type.
        opp_on_board = Counter()
        for u in self.board.units:
            if u.player_id == opponent:
                opp_on_board[u.id] += u.stack
        opp_faceup = self.state.discard_faceup[opponent]
        opp_supply = self.state.supply[opponent]
        opp_owned = {c: INITIAL_OWNED[c] - self.state.boxed[opponent][c] for c in DECK}
        # Hidden cycle = owned minus all public zones (on board, face-up discard, supply).
        opp_hidden = {
            c: opp_owned[c] - opp_on_board[c] - opp_faceup[c] - opp_supply[c] for c in DECK
        }

        def norm(counter):  # per-coin counts normalized by initial owned
            return [counter[c] / INITIAL_OWNED[c] for c in DECK]

        def norm_supply(counter):
            return [counter[c] / SUPPLY[c] for c in RECRUIT_TYPES]

        global_feats = np.array(
            [
                min(self.state.round_number / self.max_rounds, 1.0),
                my_bases / self.winning_base_count,
                opp_bases / self.winning_base_count,
                float(self.state.initiative_owner == active),
            ]
            + norm(own_hand)
            + norm(own_bag)
            + norm(own_discard)
            + norm_supply(own_supply)
            + [sum(own_bag.values()) / OWNED_TOTAL]
            + [opp_on_board[c] / INITIAL_OWNED[c] for c in UNIT_COINS]
            + norm(opp_faceup)
            + norm_supply(opp_supply)
            + [opp_hidden[c] / INITIAL_OWNED[c] for c in DECK]
            + [sum(self.state.hands[opponent].values()) / HAND_SIZE]
            + [float(self.state.initiative_transferred_this_round)],
            dtype=np.float32,
        )

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

    def get_privileged_features(self):
        """Opponent's true hidden coin split — critic-only (never given to the policy).

        Ego-centric: the opponent is 3 - active. Per coin {S,K,R}: hand, bag, face-down
        discard counts, normalized by initial owned. These are exactly the quantities the
        policy can only estimate via `hidden_pool`.
        """
        opp = 3 - self.active_player
        hand = self.state.hands[opp]
        bag = self.state.bags[opp]
        fd = self.state.discard_facedown[opp]
        feats = (
            [hand[c] / INITIAL_OWNED[c] for c in DECK]
            + [bag[c] / INITIAL_OWNED[c] for c in DECK]
            + [fd[c] / INITIAL_OWNED[c] for c in DECK]
        )
        return np.array(feats, dtype=np.float32)

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
            # Bolster: add a matching coin onto this unit's stack (any number of times).
            ids.append(self.encode_action(BOLSTER_VERB, r, q))

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
            # Recruit: pay this coin face-down, take a supply coin of any available type.
            for take in RECRUIT_TYPES:
                if self.state.supply[active][take] > 0:
                    ids.append(self.encode_recruit(take, coin))

        return ids

    def get_action_info(self, action_id: int) -> Tuple[str, Tuple]:
        if action_id >= SPATIAL_SIZE:
            return self.decode_facedown(action_id)

        verb, r, q = self.decode_action(action_id)
        if 0 <= verb <= 5:
            return MOVE_ACTION, (verb, r, q)
        elif 6 <= verb <= 11:
            return ATTACK_ACTION, (verb, r, q)
        elif verb == 12:
            return CLAIM_BASE_ACTION, (verb, r, q)
        elif verb in DEPLOY_VERBS:
            return DEPLOY_ACTION, (DEPLOY_VERBS[verb], r, q)
        elif verb == BOLSTER_VERB:
            return BOLSTER_ACTION, (r, q)
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
        self._play_coin(moving_unit.id, 'faceup')
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

        # Remove one coin from the target's stack; it leaves the game (to the box),
        # not the discard. The unit is destroyed only when its last coin is removed.
        enemy.stack -= 1
        self.state.boxed[enemy.player_id][enemy.id] += 1
        if enemy.stack <= 0:
            self.board.remove_unit(enemy)
        self._play_coin(attacker.id, 'faceup')
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
        self._play_coin(unit.id, 'faceup')
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

        self._play_coin(coin, 'board')  # the coin becomes the deployed unit
        return Action(reward=0.0, finishes_game=False, is_valid=True,
                      txt_result='Unit deployed')

    def perform_bolster_action(self, r: int, q: int) -> Action:
        active = self.active_player
        unit = self.board.get_unit_at(r, q)
        if unit is None or unit.player_id != active:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='No own unit to bolster', is_valid=False)
        if unit.id not in self.state.hands[active]:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='No matching coin in hand', is_valid=False)

        unit.stack += 1  # the coin joins the stack on the board
        self._play_coin(unit.id, 'board')
        return Action(reward=0.0, finishes_game=False, is_valid=True,
                      txt_result='Unit bolstered')

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
        self._play_coin(coin, 'facedown')
        return Action(reward=0.0, finishes_game=False, is_valid=True,
                      txt_result='Claimed initiative')

    def perform_pass_action(self, coin: int) -> Action:
        if coin not in self.state.hands[self.active_player]:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='No matching coin in hand', is_valid=False)
        self._play_coin(coin, 'facedown')
        return Action(reward=0.0, finishes_game=False, is_valid=True, txt_result='Passed')

    def perform_recruit_action(self, pay: int, take: int) -> Action:
        active = self.active_player
        if pay not in self.state.hands[active]:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='No matching coin in hand', is_valid=False)
        if self.state.supply[active][take] <= 0:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='No supply coin of that type', is_valid=False)

        self._play_coin(pay, 'facedown')  # the payment is discarded face-down
        # The recruited coin is shown to the opponent and enters the discard face-up.
        self.state.supply[active][take] -= 1
        if self.state.supply[active][take] == 0:
            del self.state.supply[active][take]
        self.state.discard_faceup[active][take] += 1
        return Action(reward=0.0, finishes_game=False, is_valid=True,
                      txt_result='Recruited')

    def get_active_player_units(self):
        return [u for u in self.board.units if u.player_id == self.active_player]
