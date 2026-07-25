"""SimGreedyBot: the 1-ply simulation greedy must stay legal, never mutate the
live env, and — unlike the obs-only GreedyBot — actually use the whole game
(tactics, and by extension recruit/bolster/initiative it now scores rather than
ignores).
"""
import numpy as np

from src.services.environment.warchest_env import WarChestEnv
from src.services.bots.greedy_sim_bot import SimGreedyBot
from _helpers import archer_scenario, blank_env, place, SWORDSMAN


def test_does_not_mutate_the_live_env():
    np.random.seed(0)
    env = WarChestEnv()
    env.reset()
    bot = SimGreedyBot()
    active_before = env.active_player
    round_before = env.state.round_number
    units_before = [(u.id, u.loc, u.player_id, u.stack) for u in env.board.units]
    legal_before = env.get_possible_actions()

    action = bot.act(env)

    assert action in legal_before
    assert env.active_player == active_before
    assert env.state.round_number == round_before
    assert [(u.id, u.loc, u.player_id, u.stack) for u in env.board.units] == units_before


def test_plays_a_full_legal_episode():
    np.random.seed(1)
    env = WarChestEnv()
    env.reset()
    bot = SimGreedyBot()
    for _ in range(4000):
        legal = env.get_possible_actions()
        action = bot.act(env)
        assert action in legal
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    else:
        raise AssertionError('episode did not terminate within the step budget')


def test_initiates_a_tactic_to_make_a_kill():
    """An Archer can only attack via its tactic. With an enemy at range 2 and a
    stack of 1 (a kill), the bot must choose the tactic — the obs-only GreedyBot
    could never initiate one. Then drive the pending sub-turn and confirm the kill.
    """
    np.random.seed(2)
    env, archer, far = archer_scenario()
    # Give P2 a real hand so the 2-ply opponent reply is a genuine move rather than
    # an empty-hand redraw that cycles the turn straight back to P1.
    from collections import Counter
    env.state.hands[2] = Counter({1: 1})
    bot = SimGreedyBot()

    first = bot.act(env)
    assert bot._classify(first) == 'tactic'

    # Drive the whole P1 sub-turn (tactic initiate -> select target) to completion.
    for _ in range(6):
        action = bot.act(env)
        env.step(action)
        if env.active_player != 1:
            break
    assert far.loc not in [u.loc for u in env.board.units if u.player_id == 2], \
        'the range-2 enemy should have been killed by the Archer tactic'


def test_takes_the_immediately_winning_move():
    """Standing one base short of the win, with a unit sitting on a claimable base,
    the bot must take the control that wins rather than anything else.
    """
    np.random.seed(3)
    env = blank_env(active=1, initiative=1)
    from collections import Counter
    env.state.compositions = {1: (SWORDSMAN,), 2: (SWORDSMAN,)}
    env.state.hands[1] = Counter({SWORDSMAN: 1})
    # A real P2 hand, so passing doesn't just redraw the turn back to P1 (which
    # would let "pass now, win next" tie the immediate win).
    env.state.hands[2] = Counter({SWORDSMAN: 1})
    # give P1 winning_base_count - 1 bases already
    neutral = [(0, 1), (2, 2), (5, 3), (1, 3), (4, 4), (6, 5)]
    have = len(env.board.get_controlled_bases(1))
    for loc in neutral[:env.winning_base_count - 1 - have]:
        env.board.change_base_control(1, loc)
    assert len(env.board.get_controlled_bases(1)) == env.winning_base_count - 1
    # a swordsman parked on the next claimable (still-neutral) base
    target = neutral[env.winning_base_count - 1 - have]
    place(env, SWORDSMAN, 1, target, stack=1)

    action = SimGreedyBot().act(env)
    assert SimGreedyBot._classify(action) == 'control'
    _, _, terminated, _, _ = env.step(action)
    assert terminated, 'claiming the 6th base should end the game as a win'
