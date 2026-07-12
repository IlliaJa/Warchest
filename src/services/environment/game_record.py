"""Time-proof game (de)serialization: plain-JSON game states + a semantic event log.

Deliberately independent of policy/critic/obs-encoder identity: a `GameState` is
pure data (board grid + unit/coin counters), so a saved record stays loadable and
re-scorable by any future net, as long as the *rules* (GameState's own shape)
haven't changed. Not pickle: pickle bakes in class/module paths, which would break
on an unrelated rename even though nothing about the recorded game changed.

The event log is one entry per draw/reshuffle/action, in play order, written for
an engine with full information (self-play, human-vs-model). Every event carries
a `hidden` flag marking whether that information would be visible to an opponent
under the real rules (face-down coins, hidden draws) — a future online-play
recorder can reuse this exact schema, simply omitting/nulling whatever it doesn't
actually observe, and setting `info_level` to `'observed'` instead of `'full'`.
"""
import json
from collections import Counter

import numpy as np

from .board import Board
from .game_state import GameState, Pending
from .roster import COIN_BY_ID
from .units import UNIT_CLASS_BY_ID

FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# GameState <-> plain dict
# ---------------------------------------------------------------------------

def game_state_to_dict(state: GameState) -> dict:
    board = state.board
    units = [
        {'id': u.id, 'player_id': u.player_id, 'loc': list(u.loc), 'stack': u.stack}
        for u in board.units
    ]
    pending = None
    if state.pending is not None:
        p = state.pending
        pending = {'kind': p.kind, 'unit_loc': list(p.unit_loc), 'optional': p.optional, 'data': p.data}

    def _counters(field):
        return {str(pid): dict(counter) for pid, counter in field.items()}

    return {
        'board_grid': board.board.tolist(),
        'units': units,
        'active_player': state.active_player,
        'action_count': state.action_count,
        'compositions': {str(pid): list(types) for pid, types in state.compositions.items()},
        'bags': _counters(state.bags),
        'hands': _counters(state.hands),
        'discard_faceup': _counters(state.discard_faceup),
        'discard_facedown': _counters(state.discard_facedown),
        'supply': _counters(state.supply),
        'boxed': _counters(state.boxed),
        'initiative_owner': state.initiative_owner,
        'initiative_transferred_this_round': state.initiative_transferred_this_round,
        'pending': pending,
        'round_number': state.round_number,
        'last_action_type': state.last_action_type,
        'last_coin': state.last_coin,
        'last_coin_player': state.last_coin_player,
        'last_recruited_coin': state.last_recruited_coin,
    }


def game_state_from_dict(d: dict) -> GameState:
    board = Board()
    board.board = np.array(d['board_grid'], dtype=int)
    for ud in d['units']:
        unit = UNIT_CLASS_BY_ID[ud['id']](player_id=ud['player_id'], board=board)
        unit.loc = tuple(ud['loc'])
        unit.stack = ud['stack']
        board.units.append(unit)

    pending = None
    if d['pending'] is not None:
        p = d['pending']
        pending = Pending(kind=p['kind'], unit_loc=tuple(p['unit_loc']),
                          optional=p['optional'], data=p['data'])

    def _counters(field):
        return {int(pid): Counter({int(c): n for c, n in counts.items()})
                for pid, counts in field.items()}

    return GameState(
        board=board,
        active_player=d['active_player'],
        action_count=d['action_count'],
        compositions={int(pid): tuple(types) for pid, types in d['compositions'].items()},
        bags=_counters(d['bags']),
        hands=_counters(d['hands']),
        discard_faceup=_counters(d['discard_faceup']),
        discard_facedown=_counters(d['discard_facedown']),
        supply=_counters(d['supply']),
        boxed=_counters(d['boxed']),
        initiative_owner=d['initiative_owner'],
        initiative_transferred_this_round=d['initiative_transferred_this_round'],
        pending=pending,
        round_number=d['round_number'],
        last_action_type=d['last_action_type'],
        last_coin=d['last_coin'],
        last_coin_player=d['last_coin_player'],
        last_recruited_coin=d['last_recruited_coin'],
    )


def _coin_name(coin_id):
    return COIN_BY_ID[coin_id].name if coin_id is not None else None


# ---------------------------------------------------------------------------
# Event builders — called from WarChestEnv hooks (_draw_hand, _reshuffle,
# _apply_action) only when `event_log` is being tracked (save_game_history=True).
# ---------------------------------------------------------------------------

def build_draw_event(player: int, round_number: int, coins) -> dict:
    names = [_coin_name(c) for c in coins]
    text = f"P{player} drew {', '.join(names)}" if names else f'P{player} drew nothing (bag empty)'
    return {
        'ply_kind': 'draw', 'round': round_number, 'player': player,
        'coin_ids': list(coins), 'coins': names, 'count': len(coins),
        'hidden': True, 'text': text,
    }


def build_reshuffle_event(player: int, round_number: int) -> dict:
    return {
        'ply_kind': 'reshuffle', 'round': round_number, 'player': player,
        'hidden': False, 'text': f"P{player}'s discard reshuffled into the bag",
    }


def _decode_continuation_click(env, action_id: int) -> dict:
    """Generic (kind-agnostic) description of a tactic-continuation click.

    Doesn't special-case every `_continuation_actions` kind (there are a dozen+);
    `pending_kind` + the decoded cell(s) are enough for a human or later tooling
    to work out exactly what happened even where the text is generic.
    """
    from . import warchest_env as we

    if action_id >= we.SPATIAL_SIZE:
        return {'summary': 'facedown continuation'}
    verb, r, q = we.WarChestEnv.decode_action(action_id)
    if 0 <= verb <= 5:
        dr, dq = Board.offsets[verb]
        return {'from': [r, q], 'to': [r + dr, q + dq], 'summary': f'move to ({r + dr},{q + dq})'}
    if 6 <= verb <= 11:
        dr, dq = Board.offsets[verb - 6]
        return {'from': [r, q], 'target': [r + dr, q + dq], 'summary': f'attack ({r + dr},{q + dq})'}
    if verb == we.CONTROL_VERB:
        return {'cell': [r, q], 'summary': f'control ({r},{q})'}
    if verb == we.SELECT_VERB:
        return {'target': [r, q], 'summary': f'select ({r},{q})'}
    return {'cell': [r, q], 'summary': f'verb={verb} at ({r},{q})'}


def build_action_event(env, action, *, cont_kind=None, cont_unit=None, pre_target=None,
                       state_dict=None) -> dict:
    """Build one semantic event dict for a just-applied, valid `Action`.

    `pre_target`: (unit_id, player_id) of the defender at an attack's target cell,
    captured by the caller before the attack resolved (it may have been eliminated
    by the time this runs). `cont_unit`/`cont_kind`: the pending continuation's
    acting unit/kind, captured by the caller the same way.
    """
    from . import warchest_env as we

    state = env.state
    board = env.board
    a_type = action.type
    args = action.additional_info
    player = action.player_id

    event = {
        'ply_kind': 'action', 'round': state.round_number, 'player': player,
        'action_id': action.id, 'action_type': a_type, 'valid': action.is_valid,
        'reward': action.reward, 'finishes_game': action.finishes_game,
        'result_text': action.txt_result,
    }
    if state_dict is not None:
        event['state'] = state_dict

    if a_type == we.MOVE_ACTION:
        verb, r, q = args
        dr, dq = board.offsets[verb]
        coin = state.last_coin
        event.update(
            coin_id=coin, coin=_coin_name(coin), frm=[r, q], to=[r + dr, q + dq], hidden=False,
            text=f'P{player} moved {_coin_name(coin)} from ({r},{q}) to ({r + dr},{q + dq})',
        )
    elif a_type == we.ATTACK_ACTION:
        verb, r, q = args
        dr, dq = board.offsets[verb - 6]
        tr, tq = r + dr, q + dq
        coin = state.last_coin
        defender_name = _coin_name(pre_target[0]) if pre_target else None
        eliminated = board.get_unit_at(tr, tq) is None
        event.update(
            coin_id=coin, coin=_coin_name(coin), frm=[r, q], target=[tr, tq], hidden=False,
            defender=defender_name, defender_eliminated=eliminated,
            text=(f"P{player}'s {_coin_name(coin)} attacked {defender_name or 'target'} "
                  f'at ({tr},{tq})' + (' — eliminated' if eliminated else '')),
        )
    elif a_type == we.CLAIM_BASE_ACTION:
        verb, r, q = args
        coin = state.last_coin
        event.update(
            coin_id=coin, coin=_coin_name(coin), cell=[r, q], hidden=False,
            text=f'P{player} claimed base ({r},{q}) with {_coin_name(coin)}',
        )
    elif a_type == we.DEPLOY_ACTION:
        coin, r, q = args
        event.update(
            coin_id=coin, coin=_coin_name(coin), to=[r, q], hidden=False,
            text=f'P{player} deployed {_coin_name(coin)} to ({r},{q})',
        )
    elif a_type == we.BOLSTER_ACTION:
        r, q = args
        coin = state.last_coin
        event.update(
            coin_id=coin, coin=_coin_name(coin), cell=[r, q], hidden=False,
            text=f'P{player} bolstered {_coin_name(coin)} at ({r},{q})',
        )
    elif a_type == we.TACTIC_ACTION and cont_kind is None:
        r, q = args
        unit = board.get_unit_at(r, q)
        coin = state.last_coin
        unit_name = _coin_name(unit.id) if unit is not None else None
        event.update(
            coin_id=coin, coin=_coin_name(coin), unit=unit_name, cell=[r, q], hidden=False,
            text=f"P{player} triggered {unit_name}'s tactic at ({r},{q}), paying {_coin_name(coin)}",
        )
    elif a_type == we.TACTIC_ACTION:  # cont_kind is not None: a pending continuation click
        unit_name = _coin_name(cont_unit.id) if cont_unit is not None else None
        detail = _decode_continuation_click(env, action.id)
        event.update(
            pending_kind=cont_kind, unit=unit_name, hidden=False, **detail,
            text=f"P{player}'s {unit_name} continuation ({cont_kind}): {detail.get('summary', '')}",
        )
    elif a_type == we.CLAIM_INITIATIVE_ACTION:
        coin = state.last_coin
        event.update(
            coin_id=coin, coin=_coin_name(coin), hidden=True,
            text=f'P{player} claimed initiative, paying {_coin_name(coin)} face-down',
        )
    elif a_type == we.PASS_ACTION:
        coin = state.last_coin
        event.update(
            coin_id=coin, coin=_coin_name(coin), hidden=True,
            text=f'P{player} passed, discarding {_coin_name(coin)} face-down',
        )
    elif a_type == we.RECRUIT_ACTION:
        pay, take = args
        event.update(
            take_id=take, take=_coin_name(take), pay_id=pay, pay=_coin_name(pay),
            take_hidden=False, pay_hidden=True, hidden=True,
            text=f'P{player} recruited {_coin_name(take)}, paying {_coin_name(pay)} face-down',
        )
    return event


# ---------------------------------------------------------------------------
# GameRecord container: assembled after a game ends, from env.event_log
# ---------------------------------------------------------------------------

def determine_result(action, truncated: bool) -> dict:
    """Winner/reason from the terminating `Action`, mirroring the reward semantics
    in `warchest_env.py` exactly (WIN_REWARD for the conquering player, LOSS_REWARD
    for a forfeiting player_id — i.e. the *other* player wins the forfeit case).
    """
    if truncated:
        return {'winner': None, 'reason': 'truncated_max_rounds'}
    if action.finishes_game:
        if action.reward > 0:
            return {'winner': action.player_id, 'reason': 'bases_controlled'}
        if action.reward < 0:
            return {'winner': 3 - action.player_id, 'reason': 'forfeit_no_actions'}
    return {'winner': None, 'reason': 'unknown'}


def build_game_record(env, players: dict, result: dict) -> dict:
    """Assemble a `GameRecord` dict from a finished game.

    `env` must have been constructed with `save_game_history=True`. `players`:
    e.g. `{1: 'human', 2: 'policy:warchest_ppo_20260707-0026'}`.
    """
    if env.history is None or env.event_log is None:
        raise ValueError('env must be constructed with save_game_history=True to build a record')
    return {
        'format': FORMAT_VERSION,
        'info_level': 'full',
        'players': {str(pid): name for pid, name in players.items()},
        'initial_state': game_state_to_dict(env.history[0]),
        'events': env.event_log,
        'result': result,
    }


def _json_default(obj):
    """Defensive numpy-scalar/array conversion at the serialization boundary.

    Board coordinates trace back to `np.where` in places, so `numpy.int64`
    (and similarly `numpy.floating`) can end up in an action id, a reward, or a
    coin id well before it reaches here — rather than chase every call site
    that might hand one to us, normalize at the one place it actually matters.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def save_game_record(record: dict, path: str) -> None:
    """Write `record` with one compact line per event.

    Fully pretty-printing (`indent=2`) would explode every event's embedded
    board `state` into dozens of lines each — across a ~100+-event game that
    buries the human-readable part (`text`, `action_type`, `coin`...) instead of
    making it scannable. One line per event is still valid JSON, but lets a
    human read the game turn by turn; every other top-level field (including the
    one-off `initial_state`) stays a single compact line too.
    """
    keys = list(record.keys())
    lines = ['{']
    for i, key in enumerate(keys):
        comma = '' if i == len(keys) - 1 else ','
        if key == 'events':
            event_lines = ',\n'.join(
                f'    {json.dumps(ev, default=_json_default)}' for ev in record[key]
            )
            lines.append('  "events": [')
            lines.append(event_lines)
            lines.append(f'  ]{comma}')
        else:
            lines.append(f'  {json.dumps(key)}: {json.dumps(record[key], default=_json_default)}{comma}')
    lines.append('}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def load_game_record(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def record_to_history(record: dict):
    """Rebuild a `list[GameState]` matching `env.history`'s shape (index 0 = initial
    state, index i = state after the i-th recorded action), so the existing
    `GameRenderer` can replay a saved record exactly like a live game's history.
    """
    states = [game_state_from_dict(record['initial_state'])]
    for ev in record['events']:
        if ev.get('ply_kind') == 'action' and ev.get('state') is not None:
            states.append(game_state_from_dict(ev['state']))
    return states
