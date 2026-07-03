import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .units import UNIT_CLASS_BY_ID
from .board import Board
from .cell_ids import *
from .game_renderer import GameRenderer
from .coin_render import draw_coin, draw_zone
from typing import Tuple, Dict
from .action import Action
from .game_state import (
    GameState, Pending, DECK, COIN_ROYAL, HAND_SIZE, UNITS_PER_PLAYER,
    build_bag, build_supply,
)
from .roster import (
    UNIT_IDS, ROYAL_ID, NUM_UNIT_TYPES, TOTAL_COINS, SUPPLY_CAP, MAX_TOTAL,
    COIN_BY_ID, UNIT_BY_ID,
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
# Coin id -> renderer glyph (units + royal), from the roster.
COIN_ICONS = {c.id: c.icon for c in COIN_BY_ID.values()}

MOVE_ACTION = 'move'
ATTACK_ACTION = 'attack'
CLAIM_BASE_ACTION = 'claim_base'
DEPLOY_ACTION = 'deploy'
CLAIM_INITIATIVE_ACTION = 'claim_initiative'
PASS_ACTION = 'pass'
BOLSTER_ACTION = 'bolster'
RECRUIT_ACTION = 'recruit'
TACTIC_ACTION = 'tactic'      # initiate a unit's tactic (Phase 4); opens a pending sub-turn
DECLINE_ACTION = 'decline'    # end an optional pending continuation early

# ---------------------------------------------------------------------------
# Action encoding (flat ids under the factored head).
#
# Spatial actions: id = verb * BOARD_DIM^2 + r * BOARD_DIM + q
#   verb 0-5:    move in direction d
#   verb 6-11:   attack in direction d
#   verb 12:     control (claim) the current cell
#   verb 13:     bolster the matching unit on this cell (type from the unit there)
#   verb 14..29: deploy unit type UNIT_IDS[verb-14] onto this cell
#   verb 30:     tactic — the unit on this cell uses its tactic (Phase 4); the unit
#                type determines which tactic, which then opens a pending sub-turn
#   verb 31:     select — pick the board cell (r,q) as a non-directional target
#                (ranged-attack target, friendly-grant recipient). Only ever legal
#                as a pending continuation click; (r,q) is the TARGET, not a source.
# Face-down actions (no board cell): appended after the spatial block, over the
# full 17-coin universe (only the player's drafted coins are ever unmasked).
#   +[0:C):           claim_initiative paying coin c
#   +[C:2C):          pass            paying coin c
#   +[2C:2C+U*C):     recruit take-type t (a unit) × pay-coin c
#   +[2C+U*C]:        decline — end an optional pending continuation (no coin)
# Deploy is per-type because an empty target cell does not determine the unit;
# move/attack/control/bolster/tactic are not, because the occupied source cell does.
#
# Phase 4 sub-turns: multi-step tactics (and triggered attributes) are NOT encoded
# as one atomic id. The TACTIC verb only *initiates*; the follow-up clicks (a move
# direction, an attack direction) reuse verbs 0-11 and are gated by `state.pending`
# via get_possible_actions. This keeps the action space near-flat as the roster's
# tactic variety grows — the variety lives in the env's pending state machine.
# ---------------------------------------------------------------------------
N_COIN_TYPES = len(DECK)  # 17 (16 units + royal)
BOARD_DIM = 7
CONTROL_VERB = 12
BOLSTER_VERB = 13
DEPLOY_VERB_BASE = 14
# Deploy verb -> unit id (one verb per deployable unit type).
DEPLOY_VERBS = {DEPLOY_VERB_BASE + i: t for i, t in enumerate(UNIT_IDS)}
TACTIC_VERB = DEPLOY_VERB_BASE + NUM_UNIT_TYPES  # 30 — sits just past the deploy block
SELECT_VERB = TACTIC_VERB + 1  # 31 — non-directional target selection (ranged, grants)
N_VERBS = SELECT_VERB + 1  # 32
SPATIAL_SIZE = N_VERBS * BOARD_DIM * BOARD_DIM  # 1568

UNIT_CLASS_BY_COIN = UNIT_CLASS_BY_ID

# Coin <-> contiguous index, used for the face-down action block and obs encoding.
COIN_TO_IDX = {c: i for i, c in enumerate(DECK)}
IDX_TO_COIN = list(DECK)
ROYAL_COIN_IDX = COIN_TO_IDX[ROYAL_ID]

# Recruitable types = all unit types (each owns >= 4 coins, so supply >= 2).
RECRUIT_TYPES = UNIT_IDS  # 16 types
_RECRUIT_BLOCK = 2 * N_COIN_TYPES  # claim (C) + pass (C) precede recruit
_DECLINE_OFFSET = _RECRUIT_BLOCK + len(RECRUIT_TYPES) * N_COIN_TYPES  # past recruit
FACEDOWN_SIZE = _DECLINE_OFFSET + 1  # 34 + 272 + 1 = 307
ACTION_SPACE_SIZE = SPATIAL_SIZE + FACEDOWN_SIZE  # 1875
DECLINE_ACTION_ID = SPATIAL_SIZE + _DECLINE_OFFSET

# ---------------------------------------------------------------------------
# Verb grouping for the factored policy head (Phase 2/4). The factoring is by verb,
# independent of how many unit types exist — the type choice lives inside the
# within-verb softmax, so the head scales with the roster for free. Phase 4 adds
# TACTIC (initiate) and DECLINE (end an optional continuation); tactic follow-up
# moves/attacks stay under V_MOVE/V_ATTACK (their flat ids are ordinary spatial
# ids), disambiguated for the policy by the pending-context one-hot in globals.
# ---------------------------------------------------------------------------
(V_MOVE, V_ATTACK, V_CONTROL, V_DEPLOY, V_BOLSTER, V_CLAIM, V_PASS, V_RECRUIT,
 V_TACTIC, V_DECLINE, V_SELECT) = range(11)
N_FACTORED_VERBS = 11


def verb_of_action(action_id: int) -> int:
    if action_id < SPATIAL_SIZE:
        sv = action_id // (BOARD_DIM * BOARD_DIM)
        if sv <= 5:
            return V_MOVE
        if sv <= 11:
            return V_ATTACK
        if sv == CONTROL_VERB:
            return V_CONTROL
        if sv == BOLSTER_VERB:
            return V_BOLSTER
        if sv == TACTIC_VERB:
            return V_TACTIC
        if sv == SELECT_VERB:
            return V_SELECT
        return V_DEPLOY  # 14..29
    off = action_id - SPATIAL_SIZE
    if off < N_COIN_TYPES:
        return V_CLAIM
    if off < _RECRUIT_BLOCK:
        return V_PASS
    if off < _DECLINE_OFFSET:
        return V_RECRUIT
    return V_DECLINE


# Static map flat action id -> verb index; consumed by the factored policy head.
VERB_OF_ACTION = np.array([verb_of_action(a) for a in range(ACTION_SPACE_SIZE)], dtype=np.int64)

# Board planes: 6 terrain/base/exploration + one stack-valued plane per unit type
# per side (own then opp). Only the ~4 drafted types per side are ever nonzero.
N_BASE_PLANES = 6
OWN_UNIT_PLANE_BASE = N_BASE_PLANES               # 6
OPP_UNIT_PLANE_BASE = N_BASE_PLANES + NUM_UNIT_TYPES  # 22

# Threat planes (C16/architectural note in docs/IDEAS.md): graded hit-count a
# side could land on each cell *this turn*, split by delivery mechanism so the
# CNN doesn't have to re-derive tactic geometry from raw unit-type identity.
#   melee  — adjacent (hex-distance 1) attacks, incl. Marshall-granted ones
#            and the Berserker chain's distance-1 contribution
#   ranged — Archer/Crossbowman's ranged_attack tactic (no movement)
#   charge — a move-then-strike mechanic landing at hex-distance >= 2
#            (Lancer's line_charge, Cavalry's move_then_attack, and the
#            Berserker chain's distance->=2 contribution)
# See WarChestEnv._threat_contributions / _threat_grids for the derivation.
THREAT_KINDS = ('melee', 'ranged', 'charge')
N_THREAT_KINDS = len(THREAT_KINDS)
N_THREAT_PLANES = 2 * N_THREAT_KINDS  # own_* + enemy_* per kind = 6
OWN_THREAT_PLANE_BASE = OPP_UNIT_PLANE_BASE + NUM_UNIT_TYPES     # 38
ENEMY_THREAT_PLANE_BASE = OWN_THREAT_PLANE_BASE + N_THREAT_KINDS  # 41
THREAT_NORM = MAX_TOTAL  # 5 — same normaliser as unit stacks; clipped to 1.0

# Coordinate planes: ego-centric row/col index, static (no per-step compute).
# Board geometry (Board.default_bases) confirms `q` (column) is the flank axis
# — P1's bases sit at low q, P2's at high q — so col_coord is what lets the
# pooled/verb-level features distinguish "which side" a threat is on.
N_COORD_PLANES = 2
COORD_PLANE_BASE = ENEMY_THREAT_PLANE_BASE + N_THREAT_KINDS  # 44
ROW_COORD_PLANE = COORD_PLANE_BASE       # 44
COL_COORD_PLANE = COORD_PLANE_BASE + 1   # 45

BOARD_CHANNELS = COORD_PLANE_BASE + N_COORD_PLANES  # 6 + 32 + 6 + 2 = 46

# Unit coin types (deployable); the royal coin has no board unit.
UNIT_COINS = UNIT_IDS
STACK_NORM = MAX_TOTAL  # 5 — max coins of one type on one stack
# Max coins a player can hold across hand+bag (4 units × total + 1 royal); bag-size norm.
OWNED_TOTAL = UNITS_PER_PLAYER * MAX_TOTAL + 1  # 21

# Global feature layout (ego-centric, OBS_VERSION 4). Per-type vectors run over
# the full coin universe (DECK, len C=17) or the unit types (U=16); absent types
# are zero. Counts normalized per type by total owned (or supply capacity).
#   [0] round fraction  [1] my bases  [2] opp bases  [3] my initiative
#   own (known): hand[C] bag[C] discard[C] supply[U] bag_size[1] owned[C]
#   opponent (public): on_board[U] faceup[C] supply[U] hidden_pool[C] owned[C]
#                      opp_hand_size[1]
#   [last] initiative already transferred this round
#   + pending-context one-hot (PENDING_CTX_DIM): which mid-tactic continuation, if any
#
# Pending-context one-hot (Phase 4): slot 0 = no pending (normal play); the rest are
# one slot per kind in PENDING_KINDS. This is what lets the policy read an otherwise
# ambiguous move/attack click as "a Cavalry follow-up" vs "a normal maneuver". Adding
# a kind extends this vector → bump OBS_VERSION (a schema change), as for any phase.
PENDING_KINDS = (
    'move_then_attack:move',   # Cavalry: the move stage
    'move_then_attack:attack',  # Cavalry: the (optional) attack stage
    'ranged_attack',            # Archer/Crossbowman: SELECT a ranged target
    'bonus_move',               # Swordsman: an optional free move after attacking
    'extra_maneuver',           # Berserker: pay a stack coin to maneuver again
    'bonus_action',             # Warrior Priest: spend a freshly drawn coin at once
    'move_to',                  # Light Cavalry / Royal Guard: SELECT a destination ≤N away
    'line_charge',              # Lancer: SELECT an in-line enemy, then move+attack
    'grant_attack:select',      # Marshall: SELECT a friendly unit within range
    'grant_attack:strike',      # Marshall: the chosen unit makes a normal attack
    'grant_move:select',        # Ensign: SELECT a friendly unit within range
    'grant_move:step',          # Ensign: the chosen unit makes a normal move
    'free_maneuver',            # Mercenary: a free maneuver after its coin is recruited
    'footman_maneuver',         # Footman: one maneuver with each Footman on the board
)
PENDING_CTX_DIM = 1 + len(PENDING_KINDS)  # 15
PENDING_KIND_IDX = {k: i for i, k in enumerate(PENDING_KINDS)}
GLOBAL_DIM = 7 * N_COIN_TYPES + 3 * NUM_UNIT_TYPES + 7 + PENDING_CTX_DIM  # 189
OBS_VERSION = 9  # bumped for threat + coordinate board planes (BOARD_CHANNELS 38 -> 46)

# Privileged critic-only features: the opponent's true hidden split, per coin (C).
#   [0:C] opp hand   [C:2C] opp bag   [2C:3C] opp face-down discard
PRIV_DIM = 3 * N_COIN_TYPES  # 51

MOVE_EXPLORE_REWARD_MAX_TURN = 5
MOVE_EXPLORE_REWARD_PER_TURN = 0.1
MOVE_NEG_REWARD_PER_TURN = -0.002
ATTACK_REWARD = 0.02  # kept small so a game's worth of attacks cannot rival a win (+1.0)
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
        if off == _DECLINE_OFFSET:
            return DECLINE_ACTION, ()
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

        Face-down actions (incl. decline) are non-spatial and map to themselves.
        Cell rotation: (r,q) → (s-r, s-q) where s = BOARD_DIM - 1 = 6. Direction
        flip applies only to move (0-5) and attack (6-11); control, bolster, deploy
        and tactic rotate spatially only — no verb change. Tactic follow-up clicks
        are ordinary move/attack ids, so they flip exactly like normal maneuvers.
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
            TACTIC_ACTION: {'act_function': self.perform_tactic_action},
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

        # Draft: sample 8 distinct unit types, assign 4 to each player so the two
        # compositions never share a unit (mirrors a drafted War Chest setup).
        drafted = np.random.choice(UNIT_COINS, size=2 * UNITS_PER_PLAYER, replace=False)
        compositions = {
            1: tuple(int(t) for t in drafted[:UNITS_PER_PLAYER]),
            2: tuple(int(t) for t in drafted[UNITS_PER_PLAYER:]),
        }
        self.state = GameState(board=board, active_player=owner, action_count=0,
                               initiative_owner=owner, compositions=compositions)
        for pid in (1, 2):
            self.state.bags[pid] = build_bag(compositions[pid])
            self.state.supply[pid] = build_supply(compositions[pid])
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

    def boxed_total(self, player_id: int) -> int:
        """Total coins ``player_id`` has permanently lost to the box.

        Coins only ever leave the cycle into ``boxed`` (monotonic per player), so
        this is the live-material signal used by the material PBRS shaping term in
        the training loop. Keyed by absolute player id, so it is perspective-free.
        """
        return sum(self.state.boxed[player_id].values())

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
        # During a pending sub-turn (a tactic mid-resolution) the same player keeps
        # acting and the action is a continuation click, dispatched separately from
        # the normal verb table; the turn only passes once `pending` clears.
        if self.state.pending is not None:
            # Capture the acting unit before it might move/die during the continuation,
            # so we can record the correct coin in last_coin for the history display.
            _cont_unit = self.board.get_unit_at(*self.state.pending.unit_loc)
            _cont_kind = self.state.pending.kind
            action = self._perform_continuation(action_id)
            action.type = TACTIC_ACTION
            action_args = None
        else:
            _cont_unit = None
            _cont_kind = None
            action_type, action_args = self.get_action_info(action_id)
            action = self.action_dict[action_type]['act_function'](*action_args)
            action.type = action_type
        action.id = action_id
        action.player_id = self.active_player
        action.additional_info = action_args

        if action.is_valid:
            self.action_count += 1
            self.state.last_action_type = action.type
            # For non-bonus continuations, overwrite last_coin with the acting unit so
            # the history shows "tactic Me" instead of whatever coin was played earlier.
            if _cont_unit is not None and _cont_kind != 'bonus_action':
                self.state.last_coin = _cont_unit.id
                self.state.last_coin_player = self.active_player
            if not action.finishes_game and self.state.pending is None:
                self._advance_turn()
            if self.history is not None:
                self.history.append(deepcopy(self.state))
            # Safety net: if the newly active player has no valid actions, previous mover wins.
            # Pass is always legal so this path is nearly unreachable; skip in normal training.
            if self.debug_mode and not action.finishes_game and not self.get_possible_actions():
                action.finishes_game = True
                action.reward += WIN_REWARD
        truncated = self.state.round_number >= self.max_rounds
        if self.debug_mode:
            print(f'Got action_id {action.id} type={action.type} args={action.additional_info}')
        return self.generate_observation(), action.reward, action.finishes_game, truncated, {'action': action}

    def render(self, ax=None, player_labels=None):
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
            label = player_labels.get(pid, '') if player_labels else ''
            label_str = f'  [{label}]' if label else ''
            ax.text(block_left, row_y + 0.62,
                    f'Player {pid}{star}{label_str}', ha='left',
                    va='center', fontsize=12, fontweight='bold', color=UNIT_COLORS[pid])
            counters = (
                self.state.hands[pid], self.state.bags[pid], self.state.supply[pid],
                self.state.discard_faceup[pid], self.state.discard_facedown[pid],
                self.state.boxed[pid],
            )
            for k, (label, counter) in enumerate(zip(zone_names, counters)):
                draw_zone(ax, block_left + k * col_w, row_y, label, counter, pid,
                          zone_width=col_w)

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
        pending = ''
        if self.state.pending is not None:
            p = self.state.pending
            pending = f' | tactic={p.kind}@{p.unit_loc}'
        return (
            f'round={self.state.round_number} active=P{self.active_player} '
            f'init=P{self.state.initiative_owner} | '
            f'bases {len(self.board.get_controlled_bases(1))}-{len(self.board.get_controlled_bases(2))}'
            f'{pending}'
        )

    def render_game(self, player_labels=None):
        if self.history is None:
            raise ValueError('Game history not available. Set save_game_history=True.')
        GameRenderer(env=self, history=self.history, player_labels=player_labels)

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
        # Unit planes: one stack-valued plane per unit type per owner, ego-centric.
        #   own types -> planes [OWN_UNIT_PLANE_BASE : +NUM_UNIT_TYPES]
        #   opp types -> planes [OPP_UNIT_PLANE_BASE : +NUM_UNIT_TYPES]
        # plane index within a side = unit id - 1 (ids run 1..16).
        for u in self.board.units:
            r, q = u.loc
            if active == 2:
                r, q = s - r, s - q
            owner_base = OWN_UNIT_PLANE_BASE if u.player_id == active else OPP_UNIT_PLANE_BASE
            board_enc[owner_base + (u.id - 1), r, q] = u.stack / STACK_NORM
        # Coordinate planes: static, ego-centric row/col index (see COORD_PLANE_BASE).
        board_enc[ROW_COORD_PLANE] = _ROW_COORD_ROT if active == 2 else _ROW_COORD_RAW
        board_enc[COL_COORD_PLANE] = _COL_COORD_ROT if active == 2 else _COL_COORD_RAW
        # Global features [GLOBAL_DIM] — ego-centric coin-counting (OBS_VERSION 4).
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

        def in_play(pid):
            """Coins still in the cycle = owned-by-composition minus boxed."""
            o = self.state.owned(pid)
            b = self.state.boxed[pid]
            return Counter({c: o[c] - b[c] for c in DECK})

        own_owned = in_play(active)
        opp_owned = in_play(opponent)

        # Threat planes: opponent coin-availability is bounded (hand+bag+facedown
        # discard, unknown split) since only the aggregate is observable; own
        # availability is exact (own_hand, above). See _threat_grids.
        opp_hidden = Counter({
            c: opp_owned[c] - opp_on_board[c] - opp_faceup[c] - opp_supply[c] for c in UNIT_IDS
        })
        threat_grids = self._threat_grids(active, own_hand, opp_hidden)
        for i, kind in enumerate(THREAT_KINDS):
            own_grid = threat_grids[(active, kind)]
            enemy_grid = threat_grids[(opponent, kind)]
            if active == 2:
                own_grid = np.rot90(own_grid, 2)
                enemy_grid = np.rot90(enemy_grid, 2)
            board_enc[OWN_THREAT_PLANE_BASE + i] = np.clip(own_grid / THREAT_NORM, 0.0, 1.0)
            board_enc[ENEMY_THREAT_PLANE_BASE + i] = np.clip(enemy_grid / THREAT_NORM, 0.0, 1.0)

        # Convert all counters to dense numpy vectors; remaining ops are vectorised.
        hand_v    = _counter_to_deck_vec(own_hand)
        bag_v     = _counter_to_deck_vec(own_bag)
        discard_v = _counter_to_deck_vec(own_discard)
        supply_v  = _counter_to_unit_vec(own_supply)
        owned_v   = _counter_to_deck_vec(own_owned)
        onboard_v = _counter_to_unit_vec(opp_on_board)
        faceup_v  = _counter_to_deck_vec(opp_faceup)
        opp_sup_v = _counter_to_unit_vec(opp_supply)
        opp_own_v = _counter_to_deck_vec(opp_owned)

        # opp_hidden: expand unit-only vectors to DECK-length, then subtract.
        onboard_deck = np.zeros(N_COIN_TYPES, dtype=np.float32)
        onboard_deck[_UNIT_IN_DECK] = onboard_v
        opp_sup_deck = np.zeros(N_COIN_TYPES, dtype=np.float32)
        opp_sup_deck[_UNIT_IN_DECK] = opp_sup_v
        hidden_v = opp_own_v - onboard_deck - faceup_v - opp_sup_deck

        global_feats = np.concatenate([
            np.array([
                min(self.state.round_number / self.max_rounds, 1.0),
                my_bases / self.winning_base_count,
                opp_bases / self.winning_base_count,
                float(self.state.initiative_owner == active),
            ], dtype=np.float32),
            hand_v / _TOTAL_COINS_VEC,
            bag_v / _TOTAL_COINS_VEC,
            discard_v / _TOTAL_COINS_VEC,
            supply_v / _SUPPLY_CAP_VEC,
            np.array([sum(own_bag.values()) / OWNED_TOTAL], dtype=np.float32),
            owned_v / _TOTAL_COINS_VEC,
            onboard_v / _TOTAL_COINS_UNIT_VEC,
            faceup_v / _TOTAL_COINS_VEC,
            opp_sup_v / _SUPPLY_CAP_VEC,
            hidden_v / _TOTAL_COINS_VEC,
            opp_own_v / _TOTAL_COINS_VEC,
            np.array([
                sum(self.state.hands[opponent].values()) / HAND_SIZE,
                float(self.state.initiative_transferred_this_round),
            ], dtype=np.float32),
        ])

        # Pending-context one-hot: which mid-tactic continuation (if any) is being
        # asked for. Slot 0 = normal play; this disambiguates a reused move/attack
        # click as a Cavalry follow-up vs an ordinary maneuver.
        ctx = np.zeros(PENDING_CTX_DIM, dtype=np.float32)
        if self.state.pending is None:
            ctx[0] = 1.0
        else:
            ctx[1 + PENDING_KIND_IDX[self.state.pending.kind]] = 1.0
        global_feats = np.concatenate([global_feats, ctx])
        # Valid action mask [ACTION_SPACE_SIZE]
        valid_ids = self.get_possible_actions()
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        if active == 2:
            mask[_REMAP_TABLE[np.array(valid_ids, dtype=np.int64)]] = True
        else:
            mask[valid_ids] = True
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
            [hand[c] / TOTAL_COINS[c] for c in DECK]
            + [bag[c] / TOTAL_COINS[c] for c in DECK]
            + [fd[c] / TOTAL_COINS[c] for c in DECK]
        )
        return np.array(feats, dtype=np.float32)

    def get_possible_actions(self):
        """Return valid action IDs in absolute (non-rotated) frame."""
        # Mid-tactic: only the legal continuation clicks for the owed sub-turn.
        if self.state.pending is not None:
            return self._continuation_actions()
        return self._normal_actions()

    def _normal_actions(self):
        """Legal action IDs for a normal (non-pending) turn, given the current hand."""
        active = self.active_player
        hand = self.state.hands[active]
        ids = []

        units = self.get_active_player_units()
        on_board_types = {u.id for u in units}

        # Maneuvers: move/attack/control/bolster are gated by holding the unit's coin;
        # a tactic is gated separately because it may be paid by another coin (e.g. the
        # Royal Guard pays with the Royal coin).
        for u in units:
            r, q = u.loc
            if u.id in hand:
                for d, (dr, dq) in enumerate(self.board.offsets):
                    target = (r + dr, q + dq)
                    if target in self.board.get_free_adjacent_cells(r, q):
                        ids.append(self.encode_action(d, r, q))
                if UNIT_BY_ID[u.id].can_normal_attack:
                    for d, (dr, dq) in enumerate(self.board.offsets):
                        enemy = self.board.get_unit_at(r + dr, q + dq)
                        if self._can_attack(u, enemy):
                            ids.append(self.encode_action(6 + d, r, q))
                if self.board.is_valid_claim(active, (r, q)):
                    ids.append(self.encode_action(12, r, q))
                # Bolster: add a matching coin onto this unit's stack (any number of times).
                ids.append(self.encode_action(BOLSTER_VERB, r, q))
            # Tactic: once initiated the follow-up clicks are gated by `pending`.
            if self._tactic_startable(u):
                ids.append(self.encode_action(TACTIC_VERB, r, q))

        # Deploy: a coin in hand whose type has room on the board (one per type, except
        # Footman allows two), onto a legal cell (a controlled empty base; the Scout may
        # also deploy onto any empty cell adjacent to a friendly unit).
        type_counts = Counter(u.id for u in units)
        for verb, coin in DEPLOY_VERBS.items():
            if coin not in hand or type_counts[coin] >= UNIT_BY_ID[coin].max_on_board:
                continue
            for loc in self._deploy_targets(coin):
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
        elif verb == TACTIC_VERB:
            return TACTIC_ACTION, (r, q)
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
        self._fire_maneuver_triggers(moving_unit, 'move')
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

        if not UNIT_BY_ID[attacker.id].can_normal_attack:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='Unit cannot make a normal attack', is_valid=False)

        enemy = self.board.get_unit_at(*target)
        if not self._can_attack(attacker, enemy):
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='No legal enemy at target cell', is_valid=False)

        # Remove one coin from the target's stack (to the box, not the discard), apply
        # any defender's counter, then fire the attacker's post-attack attribute.
        self._resolve_attack(attacker, enemy)
        self._play_coin(attacker.id, 'faceup')
        self._fire_maneuver_triggers(attacker, 'attack')
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
            self._fire_maneuver_triggers(unit, 'control')
            return Action(reward=CLAIM_BASE_REWARD, finishes_game=False, is_valid=True,
                          txt_result='Claimed base')
        return Action(reward=WIN_REWARD, finishes_game=True, is_valid=True,
                      txt_result=f'Player {self.active_player} won')

    def _deploy_targets(self, coin: int):
        """Empty cells where `coin` may be deployed: controlled empty bases, plus — for
        the Scout — any empty cell adjacent to a friendly unit."""
        active = self.active_player
        targets = [loc for loc in self.board.get_controlled_bases(active)
                   if self.board.get_unit_at(*loc) is None]
        if UNIT_BY_ID[coin].deploy_adjacent_to_friendly:
            seen = set(targets)
            for u in self.get_active_player_units():
                for cell in self.board.get_free_adjacent_cells(*u.loc):
                    if cell not in seen:
                        seen.add(cell)
                        targets.append(cell)
        return targets

    def perform_deploy_action(self, coin: int, r: int, q: int) -> Action:
        active = self.active_player
        target = (r, q)

        if coin not in self.state.hands[active]:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='No matching coin in hand', is_valid=False)
        on_board = sum(1 for u in self.get_active_player_units() if u.id == coin)
        if on_board >= UNIT_BY_ID[coin].max_on_board:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='No room on board for another of this type', is_valid=False)
        if target not in self._deploy_targets(coin):
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='Not a legal deploy cell', is_valid=False)

        new_unit = UNIT_CLASS_BY_COIN[coin](player_id=active, board=self.board)
        new_unit.place_on_board(target)
        self.board.units.append(new_unit)
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
        self.state.last_recruited_coin = take
        # The recruited coin is shown to the opponent and enters the discard face-up.
        self.state.supply[active][take] -= 1
        if self.state.supply[active][take] == 0:
            del self.state.supply[active][take]
        self.state.discard_faceup[active][take] += 1
        # Mercenary attribute: recruiting its coin grants the on-board Mercenary a free maneuver.
        if UNIT_BY_ID[take].maneuver_after_recruit and self.state.pending is None:
            merc = next((u for u in self.get_active_player_units() if u.id == take), None)
            if merc is not None and self._can_free_maneuver(merc):
                self.state.pending = Pending('free_maneuver', unit_loc=merc.loc, optional=True)
        return Action(reward=0.0, finishes_game=False, is_valid=True,
                      txt_result='Recruited')

    def _can_free_maneuver(self, unit) -> bool:
        """Whether `unit` has any legal no-coin maneuver (move / attack / control)."""
        r, q = unit.loc
        if self.board.get_free_adjacent_cells(r, q):
            return True
        if UNIT_BY_ID[unit.id].can_normal_attack and any(
                self._can_attack(unit, self.board.get_unit_at(*adj))
                for adj in self.board.get_adjacent_cells(r, q)):
            return True
        return self.board.is_valid_claim(self.active_player, (r, q))

    # ------------------------------------------------------------------
    # Combat resolution + triggered attributes (Phase 4)
    # ------------------------------------------------------------------

    def _can_attack(self, attacker, target) -> bool:
        """Whether `attacker` may legally damage `target` (Knight bolster restriction).

        A Knight 'can only be attacked by units that are bolstered' — the attacker's
        stack must be > 1. Applies to every attack path (normal, tactic, granted).
        """
        if target is None or target.player_id == attacker.player_id:
            return False
        if UNIT_BY_ID[target.id].only_attackable_when_bolstered and attacker.stack <= 1:
            return False
        return True

    def _damage_unit(self, unit):
        """Remove one coin from a unit's stack to the box; remove the unit if it dies."""
        unit.stack -= 1
        self.state.boxed[unit.player_id][unit.id] += 1
        if unit.stack <= 0:
            self.board.remove_unit(unit)

    def _apply_hit(self, target):
        """Apply one attack hit to `target`, honoring the Royal Guard's absorb-from-supply
        option (auto-used when a supply coin of its type is available — strictly defensive)."""
        info = UNIT_BY_ID[target.id]
        supply = self.state.supply[target.player_id]
        if info.absorb_from_supply and supply[target.id] > 0:
            supply[target.id] -= 1
            if supply[target.id] == 0:
                del supply[target.id]
            self.state.boxed[target.player_id][target.id] += 1  # the coin came from supply
            return
        self._damage_unit(target)

    def _resolve_attack(self, attacker, target):
        """Apply one hit to `target`, plus a Pikeman counter on an adjacent attacker.

        Pikeman's attribute ('when attacked by an adjacent unit, remove a coin from
        that unit') is simultaneous and not itself an attack, so it can even kill an
        attacking Knight (and is not absorbable). Ranged attacks (attacker not adjacent)
        do not trigger it. Used by every attack path (normal, tactic, granted, bonus).
        """
        adjacent = attacker.loc in self.board.get_adjacent_cells(*target.loc)
        self._apply_hit(target)
        if (UNIT_BY_ID[target.id].counter_when_attacked
                and adjacent and attacker in self.board.units):
            self._damage_unit(attacker)

    def _draw_bonus_coin(self):
        """Draw one coin from the active player's bag (reshuffling if needed); None if empty."""
        active = self.active_player
        if not self.state.bags[active]:
            self._reshuffle(active)
            if not self.state.bags[active]:
                return None
        return self._draw_one(active)

    def _fire_maneuver_triggers(self, unit, kind: str):
        """After a PRIMARY maneuver by `unit`, open any owed triggered-attribute sub-turn.

        No-op while a pending sub-turn is already active (triggers never nest) or if the
        unit died during resolution (e.g. to a Pikeman counter). `kind` is one of
        'move' / 'attack' / 'control'.
        """
        if self.state.pending is not None:
            return
        if unit not in self.board.units:
            return
        attrs = UNIT_BY_ID[unit.id]
        if attrs.extra_maneuvers_from_stack and unit.stack >= 2:
            # Berserker: spend a stack coin to maneuver again, repeatable down to 1 coin.
            self.state.pending = Pending('extra_maneuver', unit_loc=unit.loc, optional=True)
        elif attrs.move_after_attack and kind == 'attack' \
                and self.board.get_free_adjacent_cells(*unit.loc):
            # Swordsman: an optional free move after attacking.
            self.state.pending = Pending('bonus_move', unit_loc=unit.loc, optional=True)
        elif attrs.bonus_action_after_attack_or_control and kind in ('attack', 'control'):
            # Warrior Priest: draw a coin and immediately use it for one action.
            coin = self._draw_bonus_coin()
            if coin is not None:
                self.state.hands[self.active_player][coin] += 1
                self.state.pending = Pending('bonus_action', unit_loc=unit.loc,
                                             optional=False, data={'coin': int(coin)})

    def _bonus_actions(self, coin: int):
        """Legal actions for a Warrior-Priest bonus turn: spend exactly the drawn coin
        on one normal action. Tactic-initiation is excluded so the bonus cannot nest a
        second pending sub-turn (a documented simplification of 'use it for any action').
        """
        active = self.active_player
        saved = self.state.hands[active]
        self.state.hands[active] = Counter({coin: saved[coin]})
        try:
            ids = [a for a in self._normal_actions() if VERB_OF_ACTION[a] != V_TACTIC]
        finally:
            self.state.hands[active] = saved
        return ids

    def _resolve_free_maneuver(self, loc, verb: int):
        """Resolve a no-coin move / attack / control by the unit at `loc` (used by
        granted, free, and Footman-tactic maneuvers). Returns (action, new_loc); an
        invalid action leaves the board untouched. Does not touch `pending`.
        """
        active = self.active_player
        r, q = loc
        unit = self.board.get_unit_at(r, q)
        invalid = Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                         txt_result='Illegal maneuver', is_valid=False)
        if unit is None:
            return invalid, loc
        if 0 <= verb <= 5:
            offset = self.board.offsets[verb]
            end = (r + offset[0], q + offset[1])
            if end not in self.board.get_free_adjacent_cells(r, q):
                return invalid, loc
            unit.move(loc=end)
            self.exploration_map_dict[active][end] += 1
            return Action(reward=MOVE_NEG_REWARD_PER_TURN, finishes_game=False,
                          txt_result='Maneuver: move', is_valid=True), end
        if 6 <= verb <= 11:
            offset = self.board.offsets[verb - 6]
            enemy = self.board.get_unit_at(r + offset[0], q + offset[1])
            if not self._can_attack(unit, enemy):
                return invalid, loc
            self._resolve_attack(unit, enemy)
            return Action(reward=ATTACK_REWARD, finishes_game=False,
                          txt_result='Maneuver: attack', is_valid=True), (r, q)
        if verb == CONTROL_VERB:
            if not self.board.is_valid_claim(active, (r, q)):
                return invalid, loc
            self.board.change_base_control(player_id=active, base_loc=(r, q))
            if len(self.board.get_controlled_bases(active)) >= self.winning_base_count:
                return Action(reward=WIN_REWARD, finishes_game=True, is_valid=True,
                              txt_result=f'Player {active} won'), (r, q)
            return Action(reward=CLAIM_BASE_REWARD, finishes_game=False, is_valid=True,
                          txt_result='Maneuver: control'), (r, q)
        return invalid, loc

    # ------------------------------------------------------------------
    # Tactics (Phase 4): a tactic initiates here, then resolves over one or
    # more pending sub-turn clicks. The coin is paid once, at initiation; the
    # follow-up clicks reuse the move/attack verbs and are masked by `pending`.
    # ------------------------------------------------------------------

    def _tactic_startable(self, unit) -> bool:
        """Can this unit begin its tactic right now (pay coin in hand + a legal target)?"""
        info = UNIT_BY_ID[unit.id]
        tac = info.tactic
        if tac is None or self._tactic_pay_coin(unit) not in self.state.hands[self.active_player]:
            return False
        params = info.tactic_params or {}
        loc = unit.loc
        if tac == 'move_then_attack':
            return bool(self.board.get_free_adjacent_cells(*loc))
        if tac == 'ranged_attack':
            return bool(self._ranged_targets(loc, **params))
        if tac in ('move_to', 'royal_move'):
            return bool(self._move_to_targets(loc, params.get('max_dist', 2),
                                              controlled=(tac == 'royal_move')))
        if tac == 'line_charge':
            return bool(self._line_charge_targets(loc, params.get('max_dist', 2)))
        if tac == 'grant_attack':
            return bool(self._grant_attack_targets(loc, params.get('range', 2)))
        if tac == 'grant_move':
            return bool(self._grant_move_targets(loc, params.get('range', 2)))
        if tac == 'maneuver_each':
            return bool(self._footman_queue(unit))
        return False

    def perform_tactic_action(self, r: int, q: int) -> Action:
        active = self.active_player
        unit = self.board.get_unit_at(r, q)
        if unit is None or unit.player_id != active:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='No own unit to use a tactic', is_valid=False)
        info = UNIT_BY_ID[unit.id]
        tac = info.tactic
        if tac is None or not self._tactic_startable(unit):
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='Tactic not available', is_valid=False)

        params = info.tactic_params or {}
        # The pay coin (the unit's, or the Royal coin for the Royal Guard) is spent
        # face-up now; the tactic then resolves over its pending sub-turn clicks.
        self._play_coin(self._tactic_pay_coin(unit), 'faceup')

        if tac == 'move_then_attack':
            self.state.pending = Pending('move_then_attack:move', unit_loc=(r, q), optional=False)
        elif tac == 'ranged_attack':
            self.state.pending = Pending('ranged_attack', unit_loc=(r, q), optional=False,
                                         data=dict(params))
        elif tac in ('move_to', 'royal_move'):
            self.state.pending = Pending('move_to', unit_loc=(r, q), optional=False,
                                         data={'max_dist': params.get('max_dist', 2),
                                               'controlled': tac == 'royal_move'})
        elif tac == 'line_charge':
            self.state.pending = Pending('line_charge', unit_loc=(r, q), optional=False,
                                         data={'max_dist': params.get('max_dist', 2)})
        elif tac == 'grant_attack':
            self.state.pending = Pending('grant_attack:select', unit_loc=(r, q), optional=False,
                                         data={'origin': (r, q), 'range': params.get('range', 2)})
        elif tac == 'grant_move':
            self.state.pending = Pending('grant_move:select', unit_loc=(r, q), optional=False,
                                         data={'origin': (r, q), 'range': params.get('range', 2)})
        elif tac == 'maneuver_each':
            queue = self._footman_queue(unit)
            self.state.pending = Pending('footman_maneuver', unit_loc=queue[0], optional=True,
                                         data={'queue': queue})
        return Action(reward=0.0, finishes_game=False, is_valid=True,
                      txt_result=f'Tactic {tac} initiated')

    def _hex_distances(self, start: Tuple[int, int], max_dist: int) -> Dict[Tuple[int, int], int]:
        """BFS hex distance from `start` over valid cells (occupancy ignored).

        On the gap-free hexagon board, graph distance equals hex distance, so this
        gives the 'N spaces away' count the rulebook uses for ranged tactics.
        """
        dist = {start: 0}
        frontier = [start]
        for d in range(1, max_dist + 1):
            nxt = []
            for cell in frontier:
                for adj in self.board.get_adjacent_cells(*cell):
                    if adj not in dist:
                        dist[adj] = d
                        nxt.append(adj)
            frontier = nxt
        return dist

    def _ranged_targets(self, loc: Tuple[int, int], distance: int = 2, straight_line: bool = False):
        """Enemy-occupied cells a ranged_attack tactic from `loc` may legally select.

        distance:       how many spaces away the target sits.
        straight_line:  if True the target must be exactly `distance` along one of
                        the 6 hex directions with every intervening cell empty
                        (Crossbowman); if False any enemy at hex-distance `distance`
                        is eligible (Archer). The SELECT mask is built from this, so
                        each ranged unit supplies its own targeting rule via
                        tactic_params while reusing the one SELECT primitive.
        """
        attacker = self.board.get_unit_at(*loc)
        targets = []
        if straight_line:
            r, q = loc
            for dr, dq in self.board.offsets:
                blocked = any(
                    self.board.get_unit_at(r + dr * step, q + dq * step) is not None
                    for step in range(1, distance)
                )
                if blocked:
                    continue
                far = (r + dr * distance, q + dq * distance)
                if self._can_attack(attacker, self.board.get_unit_at(*far)):
                    targets.append(far)
        else:
            for cell, d in self._hex_distances(loc, distance).items():
                if d == distance and self._can_attack(attacker, self.board.get_unit_at(*cell)):
                    targets.append(cell)
        return targets

    # --- targeting helpers for the remaining tactics ---

    def _is_empty_cell(self, cell) -> bool:
        r, q = cell
        return (0 <= r < BOARD_DIM and 0 <= q < BOARD_DIM
                and self.board.board[r, q] != INVALID_CELL_ID
                and self.board.get_unit_at(r, q) is None)

    def _reachable(self, start, max_dist):
        """Cells reachable from `start` by moving up to `max_dist` steps through empty
        cells (BFS over free neighbours); returns {cell: distance}, distance >= 1."""
        dist = {start: 0}
        frontier = [start]
        for d in range(1, max_dist + 1):
            nxt = []
            for cell in frontier:
                for adj in self.board.get_free_adjacent_cells(*cell):
                    if adj not in dist:
                        dist[adj] = d
                        nxt.append(adj)
            frontier = nxt
        return {c: d for c, d in dist.items() if d >= 1}

    def _move_to_targets(self, loc, max_dist, controlled):
        """Empty destinations for a move-to tactic; if `controlled`, restricted to the
        active player's own control locations (Royal Guard)."""
        cells = list(self._reachable(loc, max_dist))
        if controlled:
            own = set(self.board.get_controlled_bases(self.active_player))
            cells = [c for c in cells if c in own]
        return cells

    def _line_charge_targets(self, loc, max_dist):
        """Lancer: {enemy_cell: move_destination}. Move 1..max_dist in a straight hex
        line through empty cells, then attack the enemy immediately beyond."""
        attacker = self.board.get_unit_at(*loc)
        out = {}
        for dr, dq in self.board.offsets:
            for k in range(1, max_dist + 1):
                path = [(loc[0] + dr * s, loc[1] + dq * s) for s in range(1, k + 1)]
                if not all(self._is_empty_cell(c) for c in path):
                    break  # blocked; no farther charge in this direction
                enemy_cell = (loc[0] + dr * (k + 1), loc[1] + dq * (k + 1))
                if self._can_attack(attacker, self.board.get_unit_at(*enemy_cell)):
                    out[enemy_cell] = path[-1]  # end adjacent to the enemy, then strike
        return out

    def _grant_attack_targets(self, origin, rng):
        """Friendly units within `rng` that could make a normal attack (Marshall)."""
        active = self.active_player
        out = []
        for cell, d in self._hex_distances(origin, rng).items():
            if d == 0:
                continue
            u = self.board.get_unit_at(*cell)
            if u is None or u.player_id != active or not UNIT_BY_ID[u.id].can_normal_attack:
                continue
            if any(self._can_attack(u, self.board.get_unit_at(*adj))
                   for adj in self.board.get_adjacent_cells(*cell)):
                out.append(cell)
        return out

    def _grant_move_targets(self, origin, rng):
        """Friendly units within `rng` with a free move to a cell also within `rng` (Ensign)."""
        active = self.active_player
        dists = self._hex_distances(origin, rng)
        out = []
        for cell, d in dists.items():
            if d == 0:
                continue
            u = self.board.get_unit_at(*cell)
            if u is None or u.player_id != active:
                continue
            if any(adj in dists for adj in self.board.get_free_adjacent_cells(*cell)):
                out.append(cell)
        return out

    def _footman_queue(self, unit):
        """Locations of friendly units sharing `unit`'s type that have a legal maneuver."""
        active = self.active_player
        out = []
        for u in self.get_active_player_units():
            if u.id != unit.id:
                continue
            r, q = u.loc
            can_move = bool(self.board.get_free_adjacent_cells(r, q))
            can_attack = UNIT_BY_ID[u.id].can_normal_attack and any(
                self._can_attack(u, self.board.get_unit_at(*adj))
                for adj in self.board.get_adjacent_cells(r, q))
            can_control = self.board.is_valid_claim(active, (r, q))
            if can_move or can_attack or can_control:
                out.append((r, q))
        return out

    def _tactic_pay_coin(self, unit) -> int:
        """Coin spent to use `unit`'s tactic — normally its own, but the Royal coin
        for the Royal Guard (whose tactic is the Royal coin's only face-up use)."""
        return ROYAL_ID if UNIT_BY_ID[unit.id].tactic == 'royal_move' else unit.id

    # ------------------------------------------------------------------
    # Threat map (observation-only). Graded "hits this cell could take this
    # turn" planes — see docs/IDEAS.md "the agent can't see the board as one
    # position" for the motivation. These helpers are side-effect-free and,
    # unlike the legal-action targeting helpers above, must NOT read
    # `self.active_player`: they are evaluated for units on both sides
    # regardless of whose turn it actually is. `_ranged_targets`,
    # `_line_charge_targets`, `_hex_distances`, `_reachable`, `_can_attack`
    # are player-agnostic and safe to mirror; `_grant_attack_targets` /
    # `_grant_move_targets` bake in `self.active_player` and are not reused
    # here. Target-cell validity also can't reuse `_can_attack` (it requires
    # a live enemy and returns False for empty cells) — a threat map must
    # answer "would a unit standing here get hit", independent of whether
    # anyone is standing there right now.
    # ------------------------------------------------------------------

    def _is_valid_cell(self, cell) -> bool:
        r, q = cell
        return (0 <= r < BOARD_DIM and 0 <= q < BOARD_DIM
                and self.board.board[r, q] != INVALID_CELL_ID)

    def _threat_ranged_cells(self, loc, distance, straight_line):
        """Cells a ranged_attack tactic from `loc` could hit (validity, not legality)."""
        targets = []
        if straight_line:
            r, q = loc
            for dr, dq in self.board.offsets:
                blocked = any(
                    self.board.get_unit_at(r + dr * step, q + dq * step) is not None
                    for step in range(1, distance)
                )
                if blocked:
                    continue
                far = (r + dr * distance, q + dq * distance)
                if self._is_valid_cell(far):
                    targets.append(far)
        else:
            for cell, d in self._hex_distances(loc, distance).items():
                if d == distance:
                    targets.append(cell)
        return targets

    def _threat_charge_cells(self, loc, max_dist):
        """Lancer: cells reachable by charging 1..max_dist through empty cells in a
        straight hex line, then striking immediately beyond."""
        out = []
        for dr, dq in self.board.offsets:
            for k in range(1, max_dist + 1):
                path = [(loc[0] + dr * s, loc[1] + dq * s) for s in range(1, k + 1)]
                if not all(self._is_empty_cell(c) for c in path):
                    break  # blocked; no farther charge in this direction
                enemy_cell = (loc[0] + dr * (k + 1), loc[1] + dq * (k + 1))
                if self._is_valid_cell(enemy_cell):
                    out.append(enemy_cell)
        return out

    def _threat_cavalry_cells(self, loc):
        """Cavalry: move exactly 1 step in any direction, then a normal adjacent
        attack from the new cell — unlike the Lancer's charge, the follow-up attack
        is not constrained to continue in the move's direction."""
        out = []
        for cell in self.board.get_free_adjacent_cells(*loc):
            out.extend(self.board.get_adjacent_cells(*cell))
        return out

    def _threat_berserker_reach(self, unit):
        """{cell: hits} a Berserker of `unit.stack` could land this turn.

        Each *extra* maneuver (move/attack/control) costs 1 stack coin, checked
        and paid before the maneuver (see the 'extra_maneuver' pending handling);
        the initial hand-coin activation is free. Chaining continues while
        stack >= 2. Spending the minimum moves needed to close to hex-distance D
        then converting all remaining chain capacity into attacks gives a closed
        form: hits(D) = max(0, stack - D + 1) for 1 <= D <= stack.
        """
        stack = unit.stack
        reach = dict(self._reachable(unit.loc, stack - 1))
        reach[unit.loc] = 0
        out = {}
        for cell in self.board.all_cells_list:
            if cell == unit.loc:
                continue
            neighbor_dists = [reach[a] for a in self.board.get_adjacent_cells(*cell) if a in reach]
            if not neighbor_dists:
                continue
            d = 1 + min(neighbor_dists)
            if d <= stack:
                out[cell] = stack - d + 1
        return out

    def _threat_contributions(self, unit):
        """[(cell, kind, hits), ...] this unit could produce with a single
        enabling coin this turn. kind is one of THREAT_KINDS."""
        info = UNIT_BY_ID[unit.id]
        origin = unit.loc
        out = []
        if info.tactic == 'ranged_attack':
            p = info.tactic_params
            out += [(c, 'ranged', 1)
                    for c in self._threat_ranged_cells(origin, p['distance'], p['straight_line'])]
            return out  # Archer/Crossbowman: can_normal_attack=False
        if info.tactic == 'line_charge':
            out += [(c, 'charge', 1)
                    for c in self._threat_charge_cells(origin, info.tactic_params['max_dist'])]
            return out  # Lancer: can_normal_attack=False
        if info.tactic == 'move_then_attack':
            out += [(c, 'charge', 1) for c in self._threat_cavalry_cells(origin)]
            # falls through: Cavalry also keeps its normal adjacent attack option
        if info.extra_maneuvers_from_stack:
            for cell, hits in self._threat_berserker_reach(unit).items():
                is_adjacent = cell in self.board.get_adjacent_cells(*origin)
                out.append((cell, 'melee' if is_adjacent else 'charge', hits))
            return out  # Berserker formula already covers its D=1 (melee) case
        if info.can_normal_attack:
            out += [(c, 'melee', 1) for c in self.board.get_adjacent_cells(*origin)]
        return out

    def _threat_grids(self, active, own_hand, opp_hidden):
        """{(player_id, kind): np.ndarray[BOARD_DIM,BOARD_DIM]} raw hit-count grids,
        in ABSOLUTE board coordinates (not yet ego-rotated for player 2).

        Computed for BOTH physical players regardless of whose turn it is.
        `own_hand`/`opp_hidden` give exact vs. bounded coin-availability for
        whichever side is `active` vs. its opponent (see generate_observation).

        Known simplification: Ensign's grant_move can reposition a friendly
        Berserker before its chain would be evaluated (extending its reach by
        up to 1 hex in a narrow set of directions); not modeled here — the
        added complexity isn't justified by that marginal reach gain.
        """
        opponent = 3 - active

        def coin_gate(side, coin_id):
            return (own_hand[coin_id] >= 1) if side == active else (opp_hidden[coin_id] >= 1)

        grids = {(side, kind): np.zeros((BOARD_DIM, BOARD_DIM), dtype=np.float32)
                 for side in (1, 2) for kind in THREAT_KINDS}

        for side in (1, 2):
            marshalls = [u for u in self.board.units if u.player_id == side
                         and UNIT_BY_ID[u.id].tactic == 'grant_attack']
            for u in self.board.units:
                if u.player_id != side:
                    continue
                info = UNIT_BY_ID[u.id]
                own_ok = coin_gate(side, u.id)
                grant_ok = bool(marshalls) and info.can_normal_attack and any(
                    coin_gate(side, m.id) and self._hex_distances(m.loc, 2).get(u.loc, 0) > 0
                    for m in marshalls
                )
                if not (own_ok or grant_ok):
                    continue
                for cell, kind, hits in self._threat_contributions(u):
                    grids[(side, kind)][cell] += hits
        return grids

    def _continuation_actions(self):
        """Legal continuation ids for the current pending sub-turn (absolute frame)."""
        p = self.state.pending
        active = self.active_player
        r, q = p.unit_loc
        ids = []

        if p.kind == 'move_then_attack:move':
            for d, (dr, dq) in enumerate(self.board.offsets):
                if (r + dr, q + dq) in self.board.get_free_adjacent_cells(r, q):
                    ids.append(self.encode_action(d, r, q))  # reuse move verbs 0-5
        elif p.kind == 'move_then_attack:attack':
            mover = self.board.get_unit_at(r, q)
            for d, (dr, dq) in enumerate(self.board.offsets):
                enemy = self.board.get_unit_at(r + dr, q + dq)
                if self._can_attack(mover, enemy):
                    ids.append(self.encode_action(6 + d, r, q))  # reuse attack verbs 6-11
        elif p.kind == 'ranged_attack':
            # SELECT each legal ranged target; (tr,tq) is the TARGET cell, not the source.
            for tr, tq in self._ranged_targets((r, q), **p.data):
                ids.append(self.encode_action(SELECT_VERB, tr, tq))
        elif p.kind == 'bonus_move':
            # Swordsman's free post-attack move: a single move from its cell.
            for d, (dr, dq) in enumerate(self.board.offsets):
                if (r + dr, q + dq) in self.board.get_free_adjacent_cells(r, q):
                    ids.append(self.encode_action(d, r, q))
        elif p.kind == 'extra_maneuver':
            # Berserker repeat: move / attack / control from its cell (each paid by a coin).
            mover = self.board.get_unit_at(r, q)
            for d, (dr, dq) in enumerate(self.board.offsets):
                if (r + dr, q + dq) in self.board.get_free_adjacent_cells(r, q):
                    ids.append(self.encode_action(d, r, q))
                if self._can_attack(mover, self.board.get_unit_at(r + dr, q + dq)):
                    ids.append(self.encode_action(6 + d, r, q))
            if self.board.is_valid_claim(active, (r, q)):
                ids.append(self.encode_action(CONTROL_VERB, r, q))
        elif p.kind == 'bonus_action':
            ids.extend(self._bonus_actions(p.data['coin']))
        elif p.kind == 'move_to':
            # Light Cavalry / Royal Guard: SELECT a reachable destination cell.
            for tr, tq in self._move_to_targets((r, q), p.data['max_dist'], p.data['controlled']):
                ids.append(self.encode_action(SELECT_VERB, tr, tq))
        elif p.kind == 'line_charge':
            # Lancer: SELECT an in-line enemy (the move destination is implied).
            for tr, tq in self._line_charge_targets((r, q), p.data['max_dist']):
                ids.append(self.encode_action(SELECT_VERB, tr, tq))
        elif p.kind == 'grant_attack:select':
            for tr, tq in self._grant_attack_targets(p.data['origin'], p.data['range']):
                ids.append(self.encode_action(SELECT_VERB, tr, tq))
        elif p.kind == 'grant_move:select':
            for tr, tq in self._grant_move_targets(p.data['origin'], p.data['range']):
                ids.append(self.encode_action(SELECT_VERB, tr, tq))
        elif p.kind == 'grant_attack:strike':
            # The chosen unit makes a normal adjacent attack.
            mover = self.board.get_unit_at(r, q)
            for d, (dr, dq) in enumerate(self.board.offsets):
                if self._can_attack(mover, self.board.get_unit_at(r + dr, q + dq)):
                    ids.append(self.encode_action(6 + d, r, q))
        elif p.kind == 'grant_move:step':
            # The chosen unit moves one space, ending within range of the granter.
            origin, rng = p.data['origin'], p.data['range']
            within = self._hex_distances(origin, rng)
            for d, (dr, dq) in enumerate(self.board.offsets):
                dest = (r + dr, q + dq)
                if dest in self.board.get_free_adjacent_cells(r, q) and dest in within:
                    ids.append(self.encode_action(d, r, q))
        elif p.kind == 'free_maneuver':
            # Mercenary: one free move / attack / control from its cell.
            mover = self.board.get_unit_at(r, q)
            for d, (dr, dq) in enumerate(self.board.offsets):
                if (r + dr, q + dq) in self.board.get_free_adjacent_cells(r, q):
                    ids.append(self.encode_action(d, r, q))
                if mover is not None and UNIT_BY_ID[mover.id].can_normal_attack \
                        and self._can_attack(mover, self.board.get_unit_at(r + dr, q + dq)):
                    ids.append(self.encode_action(6 + d, r, q))
            if self.board.is_valid_claim(active, (r, q)):
                ids.append(self.encode_action(CONTROL_VERB, r, q))
        elif p.kind == 'footman_maneuver':
            # The Footman currently at the front of the queue makes one maneuver.
            mover = self.board.get_unit_at(r, q)
            if mover is not None:
                for d, (dr, dq) in enumerate(self.board.offsets):
                    if (r + dr, q + dq) in self.board.get_free_adjacent_cells(r, q):
                        ids.append(self.encode_action(d, r, q))
                    if UNIT_BY_ID[mover.id].can_normal_attack \
                            and self._can_attack(mover, self.board.get_unit_at(r + dr, q + dq)):
                        ids.append(self.encode_action(6 + d, r, q))
                if self.board.is_valid_claim(active, (r, q)):
                    ids.append(self.encode_action(CONTROL_VERB, r, q))

        if p.optional:
            ids.append(DECLINE_ACTION_ID)
        return ids

    def _perform_continuation(self, action_id: int) -> Action:
        """Apply one click of the owed pending sub-turn; clear/advance `pending`."""
        p = self.state.pending
        active = self.active_player

        if action_id == DECLINE_ACTION_ID and p.kind == 'footman_maneuver':
            # Declining only skips the current Footman; advance to the next in the queue.
            queue = p.data['queue'][1:]
            self.state.pending = (
                Pending('footman_maneuver', unit_loc=queue[0], optional=True, data={'queue': queue})
                if queue else None)
            return Action(reward=0.0, finishes_game=False, is_valid=True,
                          txt_result='Skipped a Footman')

        if action_id == DECLINE_ACTION_ID:
            if not p.optional:
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='This continuation is mandatory', is_valid=False)
            self.state.pending = None
            return Action(reward=0.0, finishes_game=False, is_valid=True,
                          txt_result='Declined continuation')

        if p.kind == 'bonus_action':
            # Warrior Priest: spend the freshly drawn coin on one normal action now.
            # Handled before the spatial guard because the bonus may be a face-down
            # action (pass / recruit / claim-initiative), not just a spatial one.
            if action_id not in self._bonus_actions(p.data['coin']):
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Not a legal bonus action', is_valid=False)
            action_type, action_args = self.get_action_info(action_id)
            action = self.action_dict[action_type]['act_function'](*action_args)
            action.type = action_type
            action.additional_info = action_args
            if action.is_valid:
                self.state.pending = None
            return action

        if action_id >= SPATIAL_SIZE:
            return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                          txt_result='Not a legal continuation', is_valid=False)
        verb, r, q = self.decode_action(action_id)

        # Directional continuations (move/attack) act FROM the pending unit's cell, so
        # (r,q) must equal unit_loc. SELECT continuations point AT a target cell, so
        # that invariant does not apply and each kind validates its own target below.
        if p.kind == 'move_then_attack:move':
            if (r, q) != p.unit_loc:
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Continuation must act on the pending unit', is_valid=False)
            if not (0 <= verb <= 5):
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Must move first', is_valid=False)
            offset = self.board.offsets[verb]
            end = (r + offset[0], q + offset[1])
            if end not in self.board.get_free_adjacent_cells(r, q):
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Target cell not free', is_valid=False)
            unit = self.board.get_unit_at(r, q)
            unit.move(loc=end)
            self.exploration_map_dict[active][end] += 1
            # Then attack — optional so a move into a position with no enemy adjacent
            # is not a softlock (matches "attack, if able").
            self.state.pending = Pending('move_then_attack:attack', unit_loc=end, optional=True)
            return Action(reward=MOVE_NEG_REWARD_PER_TURN, finishes_game=False,
                          txt_result='Moved; choose an attack', is_valid=True)

        if p.kind == 'move_then_attack:attack':
            if (r, q) != p.unit_loc:
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Continuation must act on the pending unit', is_valid=False)
            if not (6 <= verb <= 11):
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Expected an attack', is_valid=False)
            offset = self.board.offsets[verb - 6]
            target = (r + offset[0], q + offset[1])
            mover = self.board.get_unit_at(r, q)
            enemy = self.board.get_unit_at(*target)
            if not self._can_attack(mover, enemy):
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='No legal enemy at target cell', is_valid=False)
            self._resolve_attack(mover, enemy)
            self.state.pending = None
            return Action(reward=ATTACK_REWARD, finishes_game=False,
                          txt_result='Moved and attacked', is_valid=True)

        if p.kind == 'ranged_attack':
            # SELECT a ranged target; (r,q) is the target cell, not the unit's cell.
            if verb != SELECT_VERB:
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Expected a target selection', is_valid=False)
            if (r, q) not in self._ranged_targets(p.unit_loc, **p.data):
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Not a legal ranged target', is_valid=False)
            # Ranged attacker is 2 cells away, so a Pikeman counter never applies here.
            self._resolve_attack(self.board.get_unit_at(*p.unit_loc), self.board.get_unit_at(r, q))
            self.state.pending = None
            return Action(reward=ATTACK_REWARD, finishes_game=False,
                          txt_result='Archer attacked', is_valid=True)

        if p.kind == 'bonus_move':
            # Swordsman's free post-attack move (no coin cost).
            if (r, q) != p.unit_loc:
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Continuation must act on the pending unit', is_valid=False)
            if not (0 <= verb <= 5):
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Expected a move', is_valid=False)
            offset = self.board.offsets[verb]
            end = (r + offset[0], q + offset[1])
            if end not in self.board.get_free_adjacent_cells(r, q):
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Target cell not free', is_valid=False)
            self.board.get_unit_at(r, q).move(loc=end)
            self.exploration_map_dict[active][end] += 1
            self.state.pending = None
            return Action(reward=MOVE_NEG_REWARD_PER_TURN, finishes_game=False,
                          txt_result='Bonus move', is_valid=True)

        if p.kind == 'extra_maneuver':
            # Berserker: each extra maneuver is paid by removing one of its own coins.
            if (r, q) != p.unit_loc:
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Continuation must act on the pending unit', is_valid=False)
            unit = self.board.get_unit_at(r, q)
            if unit is None or unit.stack < 2:
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='No coin to pay for an extra maneuver', is_valid=False)
            if 0 <= verb <= 5:  # move
                offset = self.board.offsets[verb]
                end = (r + offset[0], q + offset[1])
                if end not in self.board.get_free_adjacent_cells(r, q):
                    return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                                  txt_result='Target cell not free', is_valid=False)
                self._damage_unit(unit)  # pay from the stack (stack >= 2, so it survives)
                unit.move(loc=end)
                self.exploration_map_dict[active][end] += 1
                reward, new_loc = MOVE_NEG_REWARD_PER_TURN, end
            elif 6 <= verb <= 11:  # attack
                offset = self.board.offsets[verb - 6]
                enemy = self.board.get_unit_at(r + offset[0], q + offset[1])
                if not self._can_attack(unit, enemy):
                    return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                                  txt_result='No legal enemy at target cell', is_valid=False)
                self._damage_unit(unit)  # pay first
                self._resolve_attack(unit, enemy)  # may kill the Berserker via a counter
                reward, new_loc = ATTACK_REWARD, (r, q)
            elif verb == CONTROL_VERB:  # control
                if not self.board.is_valid_claim(active, (r, q)):
                    return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                                  txt_result='Invalid claim', is_valid=False)
                self._damage_unit(unit)  # pay
                self.board.change_base_control(player_id=active, base_loc=(r, q))
                if len(self.board.get_controlled_bases(active)) >= self.winning_base_count:
                    self.state.pending = None
                    return Action(reward=WIN_REWARD, finishes_game=True, is_valid=True,
                                  txt_result=f'Player {active} won')
                reward, new_loc = CLAIM_BASE_REWARD, (r, q)
            else:
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Expected a maneuver', is_valid=False)
            # Re-open the chain if the Berserker survived and can still pay; else end it.
            survivor = self.board.get_unit_at(*new_loc)
            if survivor is not None and survivor is unit and survivor.stack >= 2:
                self.state.pending = Pending('extra_maneuver', unit_loc=new_loc, optional=True)
            else:
                self.state.pending = None
            return Action(reward=reward, finishes_game=False, is_valid=True,
                          txt_result='Berserker extra maneuver')

        if p.kind == 'move_to':
            if verb != SELECT_VERB:
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Expected a destination selection', is_valid=False)
            if (r, q) not in self._move_to_targets(p.unit_loc, p.data['max_dist'], p.data['controlled']):
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Not a legal destination', is_valid=False)
            unit = self.board.get_unit_at(*p.unit_loc)
            unit.move(loc=(r, q))
            self.exploration_map_dict[active][(r, q)] += 1
            self.state.pending = None
            return Action(reward=MOVE_NEG_REWARD_PER_TURN, finishes_game=False,
                          txt_result='Moved', is_valid=True)

        if p.kind == 'line_charge':
            if verb != SELECT_VERB:
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Expected a target selection', is_valid=False)
            targets = self._line_charge_targets(p.unit_loc, p.data['max_dist'])
            if (r, q) not in targets:
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Not a legal charge target', is_valid=False)
            unit = self.board.get_unit_at(*p.unit_loc)
            unit.move(loc=targets[(r, q)])
            self.exploration_map_dict[active][targets[(r, q)]] += 1
            self._resolve_attack(unit, self.board.get_unit_at(r, q))
            self.state.pending = None
            return Action(reward=ATTACK_REWARD, finishes_game=False,
                          txt_result='Lancer charge', is_valid=True)

        if p.kind == 'grant_attack:select':
            if verb != SELECT_VERB or (r, q) not in self._grant_attack_targets(
                    p.data['origin'], p.data['range']):
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Not a grantable unit', is_valid=False)
            self.state.pending = Pending('grant_attack:strike', unit_loc=(r, q), optional=False)
            return Action(reward=0.0, finishes_game=False, is_valid=True,
                          txt_result='Chosen unit will attack')

        if p.kind == 'grant_attack:strike':
            if (r, q) != p.unit_loc:
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Strike must come from the chosen unit', is_valid=False)
            if not (6 <= verb <= 11):
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Expected an attack', is_valid=False)
            action, new_loc = self._resolve_free_maneuver((r, q), verb)
            if action.is_valid:
                self.state.pending = None
                # The granted unit's own on-attack attribute still triggers (FAQ:
                # Swordsman / Berserker / Warrior Priest fire off a Marshall-granted attack).
                self._fire_maneuver_triggers(self.board.get_unit_at(*new_loc), 'attack')
            return action

        if p.kind == 'grant_move:select':
            if verb != SELECT_VERB or (r, q) not in self._grant_move_targets(
                    p.data['origin'], p.data['range']):
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Not a grantable unit', is_valid=False)
            self.state.pending = Pending('grant_move:step', unit_loc=(r, q), optional=False,
                                         data=dict(p.data))
            return Action(reward=0.0, finishes_game=False, is_valid=True,
                          txt_result='Chosen unit will move')

        if p.kind == 'grant_move:step':
            if (r, q) != p.unit_loc or not (0 <= verb <= 5):
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Expected a move from the chosen unit', is_valid=False)
            offset = self.board.offsets[verb]
            dest = (r + offset[0], q + offset[1])
            within = self._hex_distances(p.data['origin'], p.data['range'])
            if dest not in self.board.get_free_adjacent_cells(r, q) or dest not in within:
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Destination not free or out of range', is_valid=False)
            self.board.get_unit_at(r, q).move(loc=dest)
            self.exploration_map_dict[active][dest] += 1
            self.state.pending = None
            # The granted unit's own attribute still triggers (FAQ: a Berserker may
            # continue with stack-paid maneuvers after an Ensign-granted move).
            self._fire_maneuver_triggers(self.board.get_unit_at(*dest), 'move')
            return Action(reward=MOVE_NEG_REWARD_PER_TURN, finishes_game=False,
                          txt_result='Granted move', is_valid=True)

        if p.kind == 'free_maneuver':
            if (r, q) != p.unit_loc:
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Continuation must act on the pending unit', is_valid=False)
            action, _ = self._resolve_free_maneuver((r, q), verb)
            if action.is_valid:
                self.state.pending = None
            return action

        if p.kind == 'footman_maneuver':
            if (r, q) != p.unit_loc:
                return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                              txt_result='Maneuver must act on the queued Footman', is_valid=False)
            action, _ = self._resolve_free_maneuver((r, q), verb)
            if not action.is_valid:
                return action
            if action.finishes_game:
                self.state.pending = None
                return action
            queue = p.data['queue'][1:]
            self.state.pending = (
                Pending('footman_maneuver', unit_loc=queue[0], optional=True, data={'queue': queue})
                if queue else None)
            return action

        # Unknown pending kind — clear it to avoid a stuck turn.
        self.state.pending = None
        return Action(reward=INVALID_ACTION_REWARD, finishes_game=False,
                      txt_result=f'Unknown pending kind {p.kind!r}', is_valid=False)

    def get_active_player_units(self):
        return [u for u in self.board.units if u.player_id == self.active_player]


# Precomputed full remap table — vectorises the P2 ego-centric mask translation.
# Shape (ACTION_SPACE_SIZE,); face-down actions are identity, spatial actions are
# rotated 180°. Built once at import time (~400 µs), reused every observation call.
_REMAP_TABLE = np.array([WarChestEnv.remap_action(a) for a in range(ACTION_SPACE_SIZE)], dtype=np.int64)

# Coordinate planes (row_coord, col_coord): static per-cell values, so both the
# raw and P2-rotated variants are built once at import time. Rotating a raw
# index grid 180° automatically yields the correctly-flipped ego-centric values
# (cell (r,q) of the rotated array holds the raw value at (s-r, s-q)), the same
# trick already used for raw_board/expl in generate_observation.
_COORD_NORM = BOARD_DIM - 1  # 6
_ROW_COORD_RAW = (np.arange(BOARD_DIM, dtype=np.float32)[:, None] / _COORD_NORM).repeat(BOARD_DIM, axis=1)
_COL_COORD_RAW = (np.arange(BOARD_DIM, dtype=np.float32)[None, :] / _COORD_NORM).repeat(BOARD_DIM, axis=0)
_ROW_COORD_ROT = np.rot90(_ROW_COORD_RAW, 2).copy()
_COL_COORD_ROT = np.rot90(_COL_COORD_RAW, 2).copy()

# ---------------------------------------------------------------------------
# Vectorised obs_global helpers.
# All arrays are built once at import time and reused every generate_observation call.
# _counter_to_deck_vec / _counter_to_unit_vec iterate only over non-zero Counter
# entries (typically 3–5), replacing O(17) or O(16) per-element Python loops.
# ---------------------------------------------------------------------------
_DECK_LIST = list(DECK)                                          # stable DECK iteration order
_DECK_COIN_TO_IDX = {c: i for i, c in enumerate(_DECK_LIST)}    # coin → position in DECK vec
_TOTAL_COINS_VEC = np.array([TOTAL_COINS[c] for c in _DECK_LIST], dtype=np.float32)
_UNIT_COIN_TO_IDX = {c: i for i, c in enumerate(UNIT_COINS)}    # coin → position in unit vec
_TOTAL_COINS_UNIT_VEC = np.array([TOTAL_COINS[c] for c in UNIT_COINS], dtype=np.float32)
_SUPPLY_CAP_VEC = np.array([SUPPLY_CAP[c] for c in RECRUIT_TYPES], dtype=np.float32)
_UNIT_IN_DECK = np.array([_DECK_COIN_TO_IDX[c] for c in UNIT_COINS], dtype=np.int64)


def _counter_to_deck_vec(counter) -> np.ndarray:
    """Counter → float32[N_COIN_TYPES] in _DECK_LIST order; iterates only non-zero entries."""
    v = np.zeros(N_COIN_TYPES, dtype=np.float32)
    for coin, cnt in counter.items():
        i = _DECK_COIN_TO_IDX.get(coin)
        if i is not None:
            v[i] = cnt
    return v


def _counter_to_unit_vec(counter) -> np.ndarray:
    """Counter → float32[NUM_UNIT_TYPES] in UNIT_COINS order; iterates only non-zero entries."""
    v = np.zeros(NUM_UNIT_TYPES, dtype=np.float32)
    for coin, cnt in counter.items():
        i = _UNIT_COIN_TO_IDX.get(coin)
        if i is not None:
            v[i] = cnt
    return v
