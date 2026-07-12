"""Pure click-resolution logic for interactive human play (see interactive_renderer.py).

Kept free of matplotlib so the action-bucketing/decoding can be exercised directly
against real `env.get_possible_actions()` output without driving a GUI.

Two-phase click model:
  1. Anchor selection — click a board cell (a friendly unit) or a hand coin.
     `compute_anchors` buckets every legal id under the cell/coin a first click on
     it would need. During a `pending` continuation this usually collapses to a
     single anchor (the renderer auto-selects it, skipping phase 1) — except
     Warrior Priest's `bonus_action`, which can offer ids anchored to several of
     the player's units, same as a normal turn.
  2. Kind resolution — `group_by_kind` splits an anchor's ids by verb. A kind
     resolves immediately (bolster/claim_base/tactic-initiate/pass/claim_initiative/
     decline: always a single id) or needs a second click among `targets_for_group`
     (move/attack/select/deploy: a destination cell; recruit: a take-coin).
"""
from .board import Board
from .warchest_env import (
    WarChestEnv, SPATIAL_SIZE, DEPLOY_VERBS, BOLSTER_VERB, TACTIC_VERB, SELECT_VERB,
    CONTROL_VERB, MOVE_ACTION, ATTACK_ACTION, CLAIM_BASE_ACTION, DEPLOY_ACTION,
    BOLSTER_ACTION, TACTIC_ACTION, CLAIM_INITIATIVE_ACTION, PASS_ACTION, RECRUIT_ACTION,
    DECLINE_ACTION,
)

KIND_MOVE = 'move'
KIND_ATTACK = 'attack'
KIND_CLAIM_BASE = 'claim_base'
KIND_BOLSTER = 'bolster'
KIND_TACTIC = 'tactic'
KIND_SELECT = 'select'
KIND_DEPLOY = 'deploy'
KIND_PASS = 'pass'
KIND_CLAIM_INITIATIVE = 'claim_initiative'
KIND_RECRUIT = 'recruit'
KIND_DECLINE = 'decline'

KIND_LABELS = {
    KIND_MOVE: 'Move', KIND_ATTACK: 'Attack', KIND_CLAIM_BASE: 'Claim Base',
    KIND_BOLSTER: 'Bolster', KIND_TACTIC: 'Tactic', KIND_SELECT: 'Select',
    KIND_DEPLOY: 'Deploy', KIND_PASS: 'Pass', KIND_CLAIM_INITIATIVE: 'Claim Initiative',
    KIND_RECRUIT: 'Recruit', KIND_DECLINE: 'Decline',
}

# Kinds where a single anchor+kind combination is always exactly one action id —
# no second click needed, commit as soon as the kind is chosen (or auto-chosen).
IMMEDIATE_KINDS = {KIND_CLAIM_BASE, KIND_BOLSTER, KIND_TACTIC, KIND_PASS,
                  KIND_CLAIM_INITIATIVE, KIND_DECLINE}
# Kinds needing a follow-up board-cell click.
CELL_TARGET_KINDS = {KIND_MOVE, KIND_ATTACK, KIND_SELECT, KIND_DEPLOY}

_FACEDOWN_KIND_BY_TYPE = {
    CLAIM_INITIATIVE_ACTION: KIND_CLAIM_INITIATIVE,
    PASS_ACTION: KIND_PASS,
    RECRUIT_ACTION: KIND_RECRUIT,
    DECLINE_ACTION: KIND_DECLINE,
}


def decode_id(action_id):
    """Decode any legal action id into (kind, args).

    Handles SELECT_VERB ids directly via `WarChestEnv.decode_action`, since
    `WarChestEnv.get_action_info` predates the pending-continuation SELECT verb
    and raises on it.
    """
    if action_id >= SPATIAL_SIZE:
        action_type, args = WarChestEnv.decode_facedown(action_id)
        return _FACEDOWN_KIND_BY_TYPE[action_type], args
    verb, r, q = WarChestEnv.decode_action(action_id)
    if 0 <= verb <= 5:
        return KIND_MOVE, (verb, r, q)
    if 6 <= verb <= 11:
        return KIND_ATTACK, (verb, r, q)
    if verb == CONTROL_VERB:
        return KIND_CLAIM_BASE, (r, q)
    if verb == BOLSTER_VERB:
        return KIND_BOLSTER, (r, q)
    if verb == TACTIC_VERB:
        return KIND_TACTIC, (r, q)
    if verb == SELECT_VERB:
        return KIND_SELECT, (r, q)
    if verb in DEPLOY_VERBS:
        return KIND_DEPLOY, (DEPLOY_VERBS[verb], r, q)
    raise ValueError(f'undecodable verb {verb} in action_id {action_id}')


def compute_anchors(env, legal_ids):
    """Bucket legal ids by the cell/coin a first click on them would need.

    Returns {('cell', (r,q)): [ids]} | {('coin', coin_id): [ids]} | {('decline', None): [ids]}.
    """
    anchors = {}
    pending = env.state.pending
    for aid in legal_ids:
        kind, args = decode_id(aid)
        if kind in (KIND_MOVE, KIND_ATTACK, KIND_CLAIM_BASE, KIND_BOLSTER, KIND_TACTIC):
            key = ('cell', (args[-2], args[-1]))
        elif kind == KIND_SELECT:
            # The continuation's acting unit, not the target — that's the 2nd click.
            key = ('cell', tuple(pending.unit_loc))
        elif kind == KIND_DEPLOY:
            key = ('coin', args[0])
        elif kind in (KIND_PASS, KIND_CLAIM_INITIATIVE):
            key = ('coin', args[0])
        elif kind == KIND_RECRUIT:
            key = ('coin', args[0])  # the paid coin
        elif kind == KIND_DECLINE:
            key = ('decline', None)
        else:
            continue
        anchors.setdefault(key, []).append(aid)
    return anchors


def group_by_kind(ids):
    groups = {}
    for aid in ids:
        kind, _ = decode_id(aid)
        groups.setdefault(kind, []).append(aid)
    return groups


def targets_for_group(kind, ids):
    """Map each id's follow-up target to the id: a (r,q) cell for move/attack/
    select/deploy, or a coin id for recruit (choosing what to take)."""
    targets = {}
    for aid in ids:
        k, args = decode_id(aid)
        if k == KIND_MOVE:
            verb, r, q = args
            dr, dq = Board.offsets[verb]
            targets[(r + dr, q + dq)] = aid
        elif k == KIND_ATTACK:
            verb, r, q = args
            dr, dq = Board.offsets[verb - 6]
            targets[(r + dr, q + dq)] = aid
        elif k == KIND_SELECT:
            r, q = args
            targets[(r, q)] = aid
        elif k == KIND_DEPLOY:
            _, r, q = args
            targets[(r, q)] = aid
        elif k == KIND_RECRUIT:
            pay, take = args
            targets[take] = aid
    return targets


def nearest_cell(x, y, cells, hex_radius=0.5):
    """Closest (r,q) in `cells` to data-coords (x,y), or None if none is within
    one hex radius (a miss-click off the board)."""
    best, best_d = None, None
    for (r, q) in cells:
        cx, cy = WarChestEnv.convert_hex_grid_to_cartesian(r, q, hex_radius=hex_radius)
        d = (cx - x) ** 2 + (cy - y) ** 2
        if best_d is None or d < best_d:
            best, best_d = (r, q), d
    if best is None or best_d > hex_radius ** 2:
        return None
    return best


def nearest_key(x, y, positions):
    """Closest key in `positions` ({key: (x, y)}) to data-coords (x, y)."""
    best, best_d = None, None
    for key, (cx, cy) in positions.items():
        d = (cx - x) ** 2 + (cy - y) ** 2
        if best_d is None or d < best_d:
            best, best_d = key, d
    return best
