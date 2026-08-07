"""Scenario tests for BolsterBot — each asserts one clause of the archetype brief.

Run: `python -m pytest tests/test_bolster_bot.py -q` (or `python tests/test_bolster_bot.py`).
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.services.environment.warchest_env import (
    WarChestEnv, SPATIAL_SIZE, DEPLOY_VERBS, CONTROL_VERB, BOLSTER_VERB, UNIT_CLASS_BY_COIN,
)
from src.services.bots.bolster_bot import (
    BolsterBot, KEY_UNIT_IDS, DEPLOY_VERB_BY_UNIT, START_BASES,
)
from src.services.gauntlet import lookahead_agent
from src.services.environment.roster import ROYAL_ID

BERSERKER, PRIEST, PIKEMAN, CAVALRY, MARSHALL = 8, 16, 10, 3, 12
SWORDSMAN, KNIGHT, LIGHT_CAV, LANCER = 1, 2, 4, 5
BOT = 2  # BolsterBot plays player 2 in these tests

BOT_COMP = [BERSERKER, PRIEST, MARSHALL, CAVALRY]
OPP_COMP = [SWORDSMAN, KNIGHT, LIGHT_CAV, LANCER]


def make_env(bot_comp=BOT_COMP, opp_comp=OPP_COMP):
    env = WarChestEnv(save_game_history=False)
    np.random.seed(0)
    env.reset(options={'force_units': {BOT: list(bot_comp), 3 - BOT: list(opp_comp)}})
    env.state.active_player = BOT
    env.state.pending = None
    # clear any units the draw/opening might imply (none deploy automatically, but be safe)
    env.board.units = [u for u in env.board.units if False]
    return env


def place(env, pid, uid, loc, stack=1):
    u = UNIT_CLASS_BY_COIN[uid](player_id=pid, board=env.board)
    u.place_on_board(loc)
    u.stack = stack
    env.board.units.append(u)
    return u


def set_hand(env, hand):
    env.state.hands[BOT] = Counter(hand)


def decode(a):
    if a >= SPATIAL_SIZE:
        return ('facedown',) + WarChestEnv.decode_facedown(a)
    return ('spatial',) + WarChestEnv.decode_action(a)


# --------------------------------------------------------------------------- #
# Point 2 — deploy a drawn PAIR of a filler (the Marshall bug)
# --------------------------------------------------------------------------- #
def test_deploys_filler_pair():
    env = make_env()
    set_hand(env, {MARSHALL: 2})              # 2 Marshall coins, no key coin
    a = BolsterBot().act(env)
    verb = WarChestEnv.decode_action(a)[0] if a < SPATIAL_SIZE else None
    assert verb == DEPLOY_VERB_BY_UNIT[MARSHALL], f'expected Marshall deploy, got {decode(a)}'


# Point 2 — a SINGLE filler with a recruit available must NOT be deployed
def test_single_filler_recruits_not_deploys():
    env = make_env()
    place(env, BOT, PRIEST, (2, 5))           # a key unit occupies one home → only 1 base free
    set_hand(env, {CAVALRY: 1})               # one lone filler coin
    a = BolsterBot().act(env)
    assert a >= SPATIAL_SIZE, f'a single filler must not be deployed, got {decode(a)}'
    kind, args = WarChestEnv.decode_facedown(a)
    assert kind == 'recruit' and args[1] in KEY_UNIT_IDS, f'expected key recruit, got {decode(a)}'


# Point 2 — 3 different coins, both bases free → deploy the single most-important filler
def test_single_filler_deploys_when_both_bases_free():
    env = make_env()
    set_hand(env, {CAVALRY: 1, MARSHALL: 1})       # both in comp, both 2nd coins in bag
    env.state.bags[BOT] = Counter({CAVALRY: 1, MARSHALL: 1})
    a = BolsterBot().act(env)
    assert a < SPATIAL_SIZE and WarChestEnv.decode_action(a)[0] == DEPLOY_VERB_BY_UNIT[CAVALRY], \
        f'expected Cavalry (most important, 2nd coin in bag) deploy, got {decode(a)}'


# Deploy the filler whose SECOND coin is still in the bag (drawable next round)
def test_prefers_filler_with_second_coin_in_bag():
    env = make_env()
    set_hand(env, {CAVALRY: 1, MARSHALL: 1})
    env.state.bags[BOT] = Counter({MARSHALL: 1})   # only Marshall's 2nd coin is in the bag
    a = BolsterBot().act(env)
    assert a < SPATIAL_SIZE and WarChestEnv.decode_action(a)[0] == DEPLOY_VERB_BY_UNIT[MARSHALL], \
        f'Cavalry is more important but has no 2nd coin in bag → expected Marshall, got {decode(a)}'


# Once both key units are deployed, a filler may deploy even with only one base free
def test_filler_deploys_when_keys_down_and_one_base_free():
    env = make_env()
    place(env, BOT, PRIEST, (2, 5))                # left home occupied
    place(env, BOT, BERSERKER, (4, 4))             # Berserker moved off its home → (5,6) free
    set_hand(env, {CAVALRY: 1})
    env.state.bags[BOT] = Counter({CAVALRY: 1})
    a = BolsterBot().act(env)
    assert a < SPATIAL_SIZE and WarChestEnv.decode_action(a)[0] == DEPLOY_VERB_BY_UNIT[CAVALRY], \
        f'both keys deployed + one base free → deploy filler, got {decode(a)}'


# --------------------------------------------------------------------------- #
# Point 4/5 — key-unit attack priority (KILL first)
# --------------------------------------------------------------------------- #
def test_berserker_attacks_over_claim():
    env = make_env()
    place(env, BOT, BERSERKER, (2, 2), stack=2)   # on a claimable neutral base ...
    place(env, 1, SWORDSMAN, (3, 3), stack=1)     # ... and adjacent to a killable enemy
    set_hand(env, {BERSERKER: 1})
    a = BolsterBot().act(env)
    verb = WarChestEnv.decode_action(a)[0] if a < SPATIAL_SIZE else None
    assert verb is not None and 6 <= verb <= 11, f'Berserker must attack, not {decode(a)}'


# Berserker is combat-only: it must never claim/capture a base, even with no enemy
# nearby and a legal claim available (falls through to bolstering off its home instead).
def test_berserker_never_claims_base():
    env = make_env()
    place(env, BOT, BERSERKER, (2, 2), stack=1)   # claimable neutral base, no adjacent enemy
    set_hand(env, {BERSERKER: 1})
    a = BolsterBot().act(env)
    assert a < SPATIAL_SIZE, f'expected a spatial action, got {decode(a)}'
    verb, r, q = WarChestEnv.decode_action(a)
    assert verb != CONTROL_VERB, f'Berserker must never claim/capture a base, got {decode(a)}'
    assert (verb, (r, q)) == (BOLSTER_VERB, (2, 2)), \
        f'expected Berserker to bolster instead of claim, got {decode(a)}'


# --------------------------------------------------------------------------- #
# Point 4/5/6 — key units deploy on their flank homes
# --------------------------------------------------------------------------- #
def test_deploys_berserker_on_right_home():
    env = make_env()
    set_hand(env, {BERSERKER: 1})
    a = BolsterBot().act(env)
    assert a < SPATIAL_SIZE
    verb, r, q = WarChestEnv.decode_action(a)
    assert verb == DEPLOY_VERB_BY_UNIT[BERSERKER] and (r, q) == (5, 6), \
        f'Berserker should deploy on its right home (5,6), got {decode(a)}'


def test_deploys_priest_on_left_home():
    env = make_env()
    set_hand(env, {PRIEST: 1})
    a = BolsterBot().act(env)
    assert a < SPATIAL_SIZE
    verb, r, q = WarChestEnv.decode_action(a)
    assert verb == DEPLOY_VERB_BY_UNIT[PRIEST] and (r, q) == (2, 5), \
        f'Priest should deploy on its left home (2,5), got {decode(a)}'


# --------------------------------------------------------------------------- #
# Point 3 / exception — never bolster a key unit on a home base
# --------------------------------------------------------------------------- #
def test_no_bolster_on_home_base():
    env = make_env()
    place(env, BOT, PRIEST, (2, 5), stack=1)   # Priest sitting on its home base, stack 1
    set_hand(env, {PRIEST: 1})
    a = BolsterBot().act(env)
    if a < SPATIAL_SIZE:
        verb, r, q = WarChestEnv.decode_action(a)
        assert not (verb == BOLSTER_VERB and (r, q) in set(START_BASES[BOT])), \
            f'must not bolster on a home base, got {decode(a)}'


# --------------------------------------------------------------------------- #
# Royal-coin economy top-up: recruit a filler paying the Royal coin once coins in
# circulation (hand + bag, all types) drop to <=4.
# --------------------------------------------------------------------------- #
def test_recruit_filler_with_royal_when_coins_low():
    comp = [SWORDSMAN, KNIGHT, LIGHT_CAV, LANCER]      # no key units in the draft
    env = make_env(bot_comp=comp, opp_comp=BOT_COMP)   # BOT_COMP doesn't overlap with `comp`
    set_hand(env, {ROYAL_ID: 1})
    env.state.bags[BOT] = Counter()                    # hand + bag total = 1 <= 4
    a = BolsterBot().act(env)
    assert a >= SPATIAL_SIZE, f'expected a recruit (face-down), got {decode(a)}'
    kind, args = WarChestEnv.decode_facedown(a)
    assert kind == 'recruit' and args == (ROYAL_ID, LIGHT_CAV), \
        f'expected recruit LIGHT_CAV paying ROYAL, got {decode(a)}'


def test_no_royal_recruit_when_coins_plentiful():
    comp = [SWORDSMAN, KNIGHT, LIGHT_CAV, LANCER]
    env = make_env(bot_comp=comp, opp_comp=BOT_COMP)
    set_hand(env, {ROYAL_ID: 1})                       # bag left at its full starting count
    a = BolsterBot().act(env)
    if a >= SPATIAL_SIZE:
        kind, args = WarChestEnv.decode_facedown(a)
        assert not (kind == 'recruit' and args[0] == ROYAL_ID), \
            f'must not spend Royal on a recruit while coins in circulation are plentiful, got {decode(a)}'


# --------------------------------------------------------------------------- #
# Point 6 — integration: fillers never maneuver, never get recruited (except the
# Royal-funded top-up); keys never bolster on a home base; Berserker never claims;
# over full games.
# --------------------------------------------------------------------------- #
def test_integration_constraints():
    bot = BolsterBot()
    opp = lookahead_agent('la', time_budget=0.03)
    filler_maneuvers = filler_recruits = onbase = over_bolster = berserker_claims = 0
    for seed in range(300, 316):
        env = WarChestEnv(save_game_history=False)
        np.random.seed(seed)
        env.reset(options={'force_units': {BOT: list(KEY_UNIT_IDS)}})
        agents = {BOT: bot, 3 - BOT: opp}
        for _ in range(2000):
            pid = env.active_player
            if pid == BOT:
                a = bot.act(env)
                if a >= SPATIAL_SIZE:
                    kind, args = WarChestEnv.decode_facedown(a)
                    # a Royal-funded filler top-up is the one sanctioned exception
                    if kind == 'recruit' and args[1] not in KEY_UNIT_IDS and args[0] != ROYAL_ID:
                        filler_recruits += 1
                else:
                    v, r, q = WarChestEnv.decode_action(a)
                    u = env.board.get_unit_at(r, q)
                    if v not in DEPLOY_VERBS and v != BOLSTER_VERB and u is not None \
                            and u.player_id == BOT and u.id not in KEY_UNIT_IDS:
                        filler_maneuvers += 1
                    if v == BOLSTER_VERB and u is not None and u.id not in KEY_UNIT_IDS and u.stack >= 2:
                        over_bolster += 1
                    if v == BOLSTER_VERB and (r, q) in set(START_BASES[BOT]) \
                            and u is not None and u.id in KEY_UNIT_IDS:
                        onbase += 1
                    if v == CONTROL_VERB and u is not None and u.id == BERSERKER:
                        berserker_claims += 1
            else:
                a = agents[pid].act(env)
            _, _, term, trunc, info = env.step(a)
            if not info['action'].is_valid:
                _, _, term, trunc, info = env.make_random_step()
            if term or trunc:
                break
    assert filler_maneuvers == 0, f'fillers maneuvered {filler_maneuvers}x'
    assert filler_recruits == 0, f'fillers recruited {filler_recruits}x'
    assert over_bolster == 0, f'fillers bolstered past 2 {over_bolster}x'
    assert onbase == 0, f'key units bolstered on a home base {onbase}x'
    assert berserker_claims == 0, f'Berserker claimed/captured a base {berserker_claims}x'


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f'PASS  {fn.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'FAIL  {fn.__name__}: {e}')
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
