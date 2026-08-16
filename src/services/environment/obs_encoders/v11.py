"""Observation encoder v11 — v10 + draw-share features (docs/history.md 2026-07-25).

Owns everything version-specific about the OBS_VERSION 11 observation: the board-
plane layout, the global-feature layout, the normalizers, the ego (P2) 180°
rotation, and the derivation of the threat / base-reach feature planes plus the
material-at-risk / E_opp_hand / base-reach scalars.

The game engine (WarChestEnv) exposes only stable rules-queries
(`unit_threat_footprint`, `attack_enabler_coins`, `unit_base_reach_cells`, and the
maneuver/reach geometry primitives) plus raw state. This module decides how to
*weight* (the coin-availability model), *aggregate*, *normalize*, *lay out* and
*rotate* those quantities into tensors — the axes that vary between obs versions.
A future v11 (e.g. threat weighted by P(coin in the opponent's hand)) is a change
here, not in the engine.

`encode(view)` is a pure function of a GameStateView (satisfied by WarChestEnv):
it never mutates the env. That is what lets N agents of different obs versions
encode the same authoritative game state independently — the round-robin gauntlet.
"""
from collections import Counter

import numpy as np
import gymnasium as gym

from ..roster import UNIT_IDS, NUM_UNIT_TYPES, TOTAL_COINS, SUPPLY_CAP, MAX_TOTAL, ROYAL_ID
from ..game_state import DECK, HAND_SIZE, UNITS_PER_PLAYER
from ..cell_ids import (
    INVALID_CELL_ID, EMPTY_CELL_ID, UNCONTROLLED_BASE_CELL_ID,
    CONTROLLED_BASE_PLAYER_1_CELL_ID, CONTROLLED_BASE_PLAYER_2_CELL_ID,
)
# Action-space + geometry constants are stable across obs versions and owned by the
# engine; import them (no cycle: warchest_env never imports this module at module top).
from ..warchest_env import (
    WarChestEnv, BOARD_DIM, N_COIN_TYPES, ACTION_SPACE_SIZE,
    RECRUIT_TYPES,
)

# --------------------------------------------------------------------------- #
# Plane layout (see the original block that lived in warchest_env).
# --------------------------------------------------------------------------- #
N_BASE_PLANES = 6
OWN_UNIT_PLANE_BASE = N_BASE_PLANES               # 6
OPP_UNIT_PLANE_BASE = N_BASE_PLANES + NUM_UNIT_TYPES  # 22

# Threat planes: graded hit-count a side could land on each cell this turn, split
# by delivery mechanism so the CNN needn't re-derive tactic geometry.
#   melee  — adjacent attacks (incl. Marshall-granted, Berserker distance-1)
#   ranged — Archer/Crossbowman ranged_attack (no movement)
#   charge — move-then-strike landing at hex-distance >= 2 (Lancer/Cavalry/Berserker)
THREAT_KINDS = ('melee', 'ranged', 'charge')
N_THREAT_KINDS = len(THREAT_KINDS)
N_THREAT_PLANES = 2 * N_THREAT_KINDS  # own_* + enemy_* per kind = 6
OWN_THREAT_PLANE_BASE = OPP_UNIT_PLANE_BASE + NUM_UNIT_TYPES     # 38
ENEMY_THREAT_PLANE_BASE = OWN_THREAT_PLANE_BASE + N_THREAT_KINDS  # 41
THREAT_NORM = MAX_TOTAL  # 5 — same normaliser as unit stacks; clipped to 1.0

# Coordinate planes: ego-centric row/col index, static.
N_COORD_PLANES = 2
COORD_PLANE_BASE = ENEMY_THREAT_PLANE_BASE + N_THREAT_KINDS  # 44
ROW_COORD_PLANE = COORD_PLANE_BASE       # 44
COL_COORD_PLANE = COORD_PLANE_BASE + 1   # 45

# Base-control reach planes: objective-analogue of the threat planes. A 0/1 grid
# of base cells a side could move onto (then claim) this turn, gated by coin
# availability (own hand / opponent hidden pool, worst-case, exactly as threats).
N_BASE_REACH_PLANES = 2
BASE_REACH_PLANE_BASE = COORD_PLANE_BASE + N_COORD_PLANES  # 46
OWN_BASE_REACH_PLANE = BASE_REACH_PLANE_BASE       # 46
ENEMY_BASE_REACH_PLANE = BASE_REACH_PLANE_BASE + 1  # 47

BOARD_CHANNELS = BASE_REACH_PLANE_BASE + N_BASE_REACH_PLANES  # 6 + 32 + 6 + 2 + 2 = 48

# Unit coin types (deployable); the royal coin has no board unit.
UNIT_COINS = UNIT_IDS
STACK_NORM = MAX_TOTAL  # 5 — max coins of one type on one stack
# Max coins a player can hold across hand+bag (4 units × total + 1 royal); bag-size norm.
OWNED_TOTAL = UNITS_PER_PLAYER * MAX_TOTAL + 1  # 21

# Global feature layout (ego-centric). Per-type vectors run over the full coin
# universe (DECK, len C=17) or the unit types (U=16); absent types are zero.
#   [0] round fraction  [1] my bases  [2] opp bases  [3] my initiative
#   own (known): hand[C] bag[C] discard[C] supply[U] bag_size[1] owned[C]
#                p_soon[C] p_mean[C]  (draw-share features, docs/IDEAS.md #R2)
#   opponent (public): on_board[U] faceup[C] supply[U] hidden_pool[C] owned[C]
#                      opp_hand_size[1]  init_transferred[1]
#   material-at-risk: own_at_risk[1] opp_at_risk[1]
#   E_opp_hand[C]; base-control reach scalars: bases_i_can_claim[1]
#     my_bases_under_flip_threat[1] win_proximity_alarm[1]
#   + pending-context one-hot (PENDING_CTX_DIM)
PENDING_KINDS = (
    'move_then_attack:move',   # Cavalry: the move stage
    'move_then_attack:attack',  # Cavalry: the (mandatory) attack stage
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
# 10 coin-vectors + 3 unit-vectors + 12 standalone scalars + pending one-hot.
# (v11 adds p_soon[C] + p_mean[C] over v10's 8 coin-vectors.)
GLOBAL_DIM = 10 * N_COIN_TYPES + 3 * NUM_UNIT_TYPES + 12 + PENDING_CTX_DIM  # 245
OBS_VERSION = 11

# Where the per-type blocks sit inside the flat global vector, for a consumer that
# wants to read them AS per-type vectors rather than as 245 anonymous slots — the
# unit-type embedding of docs/IDEAS.md A1 contracts each block against a shared
# table instead of giving every type its own weight column. Purely descriptive: the
# encoded observation is unchanged, which is why A1 needs no OBS_VERSION bump.
# The offsets follow the concatenation order in `encode` and are pinned against a
# real encode by `tests/test_unit_embedding.py` — keep them in step with it.
#   0      [round, my_bases, opp_bases, initiative]
#   4      hand[C]      21 bag[C]      38 discard[C]
#   55     supply[U]    71 [bag_size]  72 owned[C]
#   89     p_soon[C]   106 p_mean[C]  123 opp_on_board[U]
#   139    opp_faceup[C]              156 opp_supply[U]
#   172    opp_hidden[C]              189 opp_owned[C]
#   206    [opp_hand_size, init_transferred]
#   208    [own_at_risk, opp_at_risk]
#   210    E_opp_hand[C]
#   227    [bases_i_can_claim, my_bases_under_flip, win_alarm]
#   230    pending one-hot
DECK_BLOCK_OFFSETS = (4, 21, 38, 72, 89, 106, 139, 172, 189, 210)
UNIT_BLOCK_OFFSETS = (55, 123, 156)

# Privileged critic-only features: opponent's true hidden split, per coin (C).
#   [0:C] opp hand   [C:2C] opp bag   [2C:3C] opp face-down discard
PRIV_DIM = 3 * N_COIN_TYPES  # 51

# --------------------------------------------------------------------------- #
# Precomputed arrays (built once at import).
# --------------------------------------------------------------------------- #
# Full remap table — vectorises the P2 ego-centric mask translation. Face-down
# actions are identity; spatial actions rotate 180°.
_REMAP_TABLE = np.array([WarChestEnv.remap_action(a) for a in range(ACTION_SPACE_SIZE)],
                        dtype=np.int64)

# Coordinate planes: static per-cell row/col index; raw + P2-rotated variants.
_COORD_NORM = BOARD_DIM - 1  # 6
_ROW_COORD_RAW = (np.arange(BOARD_DIM, dtype=np.float32)[:, None] / _COORD_NORM).repeat(BOARD_DIM, axis=1)
_COL_COORD_RAW = (np.arange(BOARD_DIM, dtype=np.float32)[None, :] / _COORD_NORM).repeat(BOARD_DIM, axis=0)
_ROW_COORD_ROT = np.rot90(_ROW_COORD_RAW, 2).copy()
_COL_COORD_ROT = np.rot90(_COL_COORD_RAW, 2).copy()

_DECK_LIST = list(DECK)                                        # stable DECK iteration order
_DECK_COIN_TO_IDX = {c: i for i, c in enumerate(_DECK_LIST)}   # coin → position in DECK vec
_TOTAL_COINS_VEC = np.array([TOTAL_COINS[c] for c in _DECK_LIST], dtype=np.float32)
_UNIT_COIN_TO_IDX = {c: i for i, c in enumerate(UNIT_COINS)}   # coin → position in unit vec
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


class ObsEncoderV11:
    """OBS_VERSION 11 encoder. v10 + own-side draw-share features (docs/IDEAS.md #R2).

    Adds two per-type [C] vectors to the own-side global block:
      - p_soon[t]: expected share of the *next* hand that is type t (imminent draw
        share; hypergeometric mean, one reshuffle if the bag empties mid-draw).
      - p_mean[t]: steady-state share of type t in the recirculating pool
        (bag+hand+discard) — the concentration/dilution the coin economy moves.
    The gap p_soon-p_mean reads as "loaded now vs. stuck behind a reshuffle"; high
    p_mean concentration is the combo/tempo signal bolstering-away-other-units buys.
    Stateless; a single instance is reused per env.
    """

    version = OBS_VERSION
    board_channels = BOARD_CHANNELS
    global_dim = GLOBAL_DIM
    priv_dim = PRIV_DIM
    # Which board-plane channels hold own/opponent unit-stack counts, for readouts
    # (Critic's A2 gather-pool) that need per-cell occupancy rather than raw features.
    own_unit_channels = slice(OWN_UNIT_PLANE_BASE, OWN_UNIT_PLANE_BASE + NUM_UNIT_TYPES)
    opp_unit_channels = slice(OPP_UNIT_PLANE_BASE, OPP_UNIT_PLANE_BASE + NUM_UNIT_TYPES)
    # Per-type structure of the flat global vector (docs/IDEAS.md A1). `deck_*` blocks
    # run over DECK (units + the royal coin), `unit_*` blocks over the unit types only;
    # `deck_unit_positions` picks the unit slots out of a DECK block in unit-plane order
    # and `deck_royal_position` is the one slot left over.
    deck_block_offsets = DECK_BLOCK_OFFSETS
    unit_block_offsets = UNIT_BLOCK_OFFSETS
    deck_block_len = N_COIN_TYPES
    unit_block_len = NUM_UNIT_TYPES
    deck_unit_positions = tuple(int(i) for i in _UNIT_IN_DECK)
    deck_royal_position = _DECK_COIN_TO_IDX[ROYAL_ID]

    def observation_space(self):
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

    # ------------------------------------------------------------------ #
    # Feature-plane derivation. The coin-availability model + aggregation
    # live here (the versioned part); per-unit footprints / reach cells are
    # engine rules-queries on the view.
    # ------------------------------------------------------------------ #
    def threat_grids(self, view, active, own_hand, opp_hidden):
        """{(player_id, kind): np.ndarray[BOARD_DIM,BOARD_DIM]} raw hit-count grids,
        ABSOLUTE coords (not yet ego-rotated). Worst-case availability model:
        a coin counts as available if the side could hold it."""
        def coin_available(side, coin_id):
            return (own_hand[coin_id] >= 1) if side == active else (opp_hidden[coin_id] >= 1)

        grids = {(side, kind): np.zeros((BOARD_DIM, BOARD_DIM), dtype=np.float32)
                 for side in (1, 2) for kind in THREAT_KINDS}

        for side in (1, 2):
            for u in view.board.units:
                if u.player_id != side:
                    continue
                footprint = view.unit_threat_footprint(u)
                if not footprint:
                    continue
                if not any(coin_available(side, c) for c in view.attack_enabler_coins(u)):
                    continue
                for cell, kind, hits in footprint:
                    grids[(side, kind)][cell] += hits
        return grids

    def base_reach_grids(self, view, active, own_hand, opp_hidden):
        """{side: np.ndarray[BOARD_DIM,BOARD_DIM]} 0/1 claimable-base reach grids,
        ABSOLUTE coords. Coin-availability gated exactly as threat_grids."""
        def coin_available(side, coin_id):
            return (own_hand[coin_id] >= 1) if side == active else (opp_hidden[coin_id] >= 1)

        grids = {side: np.zeros((BOARD_DIM, BOARD_DIM), dtype=np.float32) for side in (1, 2)}
        for side in (1, 2):
            for u in view.board.units:
                if u.player_id != side or not coin_available(side, u.id):
                    continue
                for cell in view.unit_base_reach_cells(u):
                    grids[side][cell] = 1.0
        return grids

    def encode(self, view):
        active = view.active_player
        opponent = 3 - active
        s = view.board.board_size - 1  # 6

        raw_board = view.board.board
        expl = view.exploration_map_dict[active]
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
        for u in view.board.units:
            r, q = u.loc
            if active == 2:
                r, q = s - r, s - q
            owner_base = OWN_UNIT_PLANE_BASE if u.player_id == active else OPP_UNIT_PLANE_BASE
            board_enc[owner_base + (u.id - 1), r, q] = u.stack / STACK_NORM
        # Coordinate planes: static, ego-centric row/col index.
        board_enc[ROW_COORD_PLANE] = _ROW_COORD_ROT if active == 2 else _ROW_COORD_RAW
        board_enc[COL_COORD_PLANE] = _COL_COORD_ROT if active == 2 else _COL_COORD_RAW
        # Global features [GLOBAL_DIM] — ego-centric coin-counting.
        my_bases = len(view.board.get_controlled_bases(active))
        opp_bases = len(view.board.get_controlled_bases(opponent))

        own_hand = view.state.hands[active]
        own_bag = view.state.bags[active]
        own_discard = view.state.discard_faceup[active] + view.state.discard_facedown[active]
        own_supply = view.state.supply[active]

        # On-board counts are stack heights (committed coins), one unit per type.
        opp_on_board = Counter()
        for u in view.board.units:
            if u.player_id == opponent:
                opp_on_board[u.id] += u.stack
        opp_faceup = view.state.discard_faceup[opponent]
        opp_supply = view.state.supply[opponent]

        def in_play(pid):
            """Coins still in the cycle = owned-by-composition minus boxed."""
            o = view.state.owned(pid)
            b = view.state.boxed[pid]
            return Counter({c: o[c] - b[c] for c in DECK})

        own_owned = in_play(active)
        opp_owned = in_play(opponent)

        # Threat planes: opponent coin-availability is bounded (hand+bag+facedown
        # discard, unknown split); own availability is exact (own_hand).
        opp_hidden = Counter({
            c: opp_owned[c] - opp_on_board[c] - opp_faceup[c] - opp_supply[c] for c in UNIT_IDS
        })
        threat_grids = self.threat_grids(view, active, own_hand, opp_hidden)
        for i, kind in enumerate(THREAT_KINDS):
            own_grid = threat_grids[(active, kind)]
            enemy_grid = threat_grids[(opponent, kind)]
            if active == 2:
                own_grid = np.rot90(own_grid, 2)
                enemy_grid = np.rot90(enemy_grid, 2)
            board_enc[OWN_THREAT_PLANE_BASE + i] = np.clip(own_grid / THREAT_NORM, 0.0, 1.0)
            board_enc[ENEMY_THREAT_PLANE_BASE + i] = np.clip(enemy_grid / THREAT_NORM, 0.0, 1.0)

        # Material-at-risk (globals): min(hits, stack) summed per side, using RAW
        # threat grids (pre-clip); grids and u.loc are both absolute coords.
        enemy_hits = sum(threat_grids[(opponent, k)] for k in THREAT_KINDS)
        own_hits = sum(threat_grids[(active, k)] for k in THREAT_KINDS)
        own_at_risk = opp_at_risk = 0.0
        for u in view.board.units:
            if u.player_id == active:
                own_at_risk += min(enemy_hits[u.loc], u.stack)
            else:
                opp_at_risk += min(own_hits[u.loc], u.stack)

        # Base-control reach planes + scalars (objective-analogue of the threats).
        base_reach = self.base_reach_grids(view, active, own_hand, opp_hidden)
        own_reach, enemy_reach = base_reach[active], base_reach[opponent]
        bases_i_can_claim = float(own_reach.sum())
        my_bases_under_flip = sum(
            enemy_reach[loc] for loc in view.board.get_controlled_bases(active))
        # Opponent one base from winning AND able to take a base this turn unless answered.
        win_alarm = float(opp_bases == view.winning_base_count - 1 and enemy_reach.sum() > 0)
        if active == 2:
            own_reach = np.rot90(own_reach, 2)
            enemy_reach = np.rot90(enemy_reach, 2)
        board_enc[OWN_BASE_REACH_PLANE] = own_reach
        board_enc[ENEMY_BASE_REACH_PLANE] = enemy_reach

        # Convert all counters to dense numpy vectors; remaining ops are vectorised.
        hand_v    = _counter_to_deck_vec(own_hand)
        bag_v     = _counter_to_deck_vec(own_bag)
        discard_v = _counter_to_deck_vec(own_discard)
        supply_v  = _counter_to_unit_vec(own_supply)
        owned_v   = _counter_to_deck_vec(own_owned)

        # Draw-share features (docs/IDEAS.md #R2), own-side, per type; already in [0,1].
        # p_soon: expected share of the next hand that is type t (hypergeometric mean;
        # one reshuffle from the discard if the bag empties mid-draw).
        bag_size = bag_v.sum()
        if bag_size >= HAND_SIZE:
            e_soon = HAND_SIZE * bag_v / bag_size
        else:
            disc_total = discard_v.sum()
            rest = HAND_SIZE - bag_size
            e_soon = bag_v + (rest * discard_v / disc_total if disc_total > 0
                              else np.zeros(N_COIN_TYPES, dtype=np.float32))
        p_soon = e_soon / HAND_SIZE
        # p_mean: steady-state share of type t in the recirculating pool (bag+hand+discard).
        recirc = bag_v + hand_v + discard_v
        recirc_total = recirc.sum()
        p_mean = (recirc / recirc_total if recirc_total > 0
                  else np.zeros(N_COIN_TYPES, dtype=np.float32))

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

        # E_opp_hand: expected copies of each type in the opponent's hand right now,
        # = hidden_pool * opp_hand_size / hidden_total (hypergeometric mean).
        opp_hand_size = sum(view.state.hands[opponent].values())
        hidden_nonneg = np.clip(hidden_v, 0.0, None)
        hidden_total = hidden_nonneg.sum()
        e_opp_hand = hidden_nonneg * (opp_hand_size / hidden_total) if hidden_total > 0 \
            else np.zeros(N_COIN_TYPES, dtype=np.float32)

        global_feats = np.concatenate([
            np.array([
                min(view.state.round_number / view.max_rounds, 1.0),
                my_bases / view.winning_base_count,
                opp_bases / view.winning_base_count,
                float(view.state.initiative_owner == active),
            ], dtype=np.float32),
            hand_v / _TOTAL_COINS_VEC,
            bag_v / _TOTAL_COINS_VEC,
            discard_v / _TOTAL_COINS_VEC,
            supply_v / _SUPPLY_CAP_VEC,
            np.array([sum(own_bag.values()) / OWNED_TOTAL], dtype=np.float32),
            owned_v / _TOTAL_COINS_VEC,
            # own-side draw-share features (already normalised shares in [0,1])
            p_soon,
            p_mean,
            onboard_v / _TOTAL_COINS_UNIT_VEC,
            faceup_v / _TOTAL_COINS_VEC,
            opp_sup_v / _SUPPLY_CAP_VEC,
            hidden_v / _TOTAL_COINS_VEC,
            opp_own_v / _TOTAL_COINS_VEC,
            np.array([
                opp_hand_size / HAND_SIZE,
                float(view.state.initiative_transferred_this_round),
            ], dtype=np.float32),
            # material-at-risk
            np.array([
                min(own_at_risk / OWNED_TOTAL, 1.0),
                min(opp_at_risk / OWNED_TOTAL, 1.0),
            ], dtype=np.float32),
            # expected opponent hand, per type
            e_opp_hand / _TOTAL_COINS_VEC,
            # base-control reach scalars
            np.array([
                min(bases_i_can_claim / view.winning_base_count, 1.0),
                min(my_bases_under_flip / view.winning_base_count, 1.0),
                win_alarm,
            ], dtype=np.float32),
        ])

        # Pending-context one-hot: which mid-tactic continuation (if any).
        ctx = np.zeros(PENDING_CTX_DIM, dtype=np.float32)
        if view.state.pending is None:
            ctx[0] = 1.0
        else:
            ctx[1 + PENDING_KIND_IDX[view.state.pending.kind]] = 1.0
        global_feats = np.concatenate([global_feats, ctx])
        # Valid action mask [ACTION_SPACE_SIZE]
        valid_ids = view.get_possible_actions()
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

    def encode_privileged(self, view):
        """Opponent's true hidden coin split — critic-only (never given to the policy).

        Ego-centric: the opponent is 3 - active. Per coin: hand, bag, face-down
        discard counts, normalized by initial owned.
        """
        opp = 3 - view.active_player
        hand = view.state.hands[opp]
        bag = view.state.bags[opp]
        fd = view.state.discard_facedown[opp]
        feats = (
            [hand[c] / TOTAL_COINS[c] for c in DECK]
            + [bag[c] / TOTAL_COINS[c] for c in DECK]
            + [fd[c] / TOTAL_COINS[c] for c in DECK]
        )
        return np.array(feats, dtype=np.float32)
