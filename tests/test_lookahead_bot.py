"""LookaheadBot: must return legal actions, never mutate the live env while
searching, and play a full episode through cleanly — mirrors test_bots.py's
scope for RandomBot/GreedyBot, but via the env-taking `act(env)` interface
(docs/lookahead_bot_plan.md) instead of `Bot.act(obs)`.
"""
from src.services.environment.warchest_env import WarChestEnv
from src.services.bots.lookahead_bot import LookaheadBot


def _tiny_bot(**kwargs):
    # Small time budget + branching cap so the test suite stays fast.
    return LookaheadBot(time_budget=0.05, max_branching=4, **kwargs)


def test_lookahead_bot_does_not_mutate_the_live_env():
    env = WarChestEnv()
    env.reset()
    bot = _tiny_bot()

    active_before = env.active_player
    legal_before = env.get_possible_actions()
    round_before = env.state.round_number
    units_before = [(u.id, u.loc, u.player_id) for u in env.board.units]

    action = bot.act(env)

    assert action in legal_before
    assert env.active_player == active_before
    assert env.state.round_number == round_before
    assert [(u.id, u.loc, u.player_id) for u in env.board.units] == units_before


def test_lookahead_bot_plays_a_full_legal_episode():
    env = WarChestEnv()
    env.reset()
    bot = _tiny_bot()
    for _ in range(30):
        legal = env.get_possible_actions()
        action = bot.act(env)
        assert action in legal
        _, _, terminated, truncated, info = env.step(action)
        assert info['action'].is_valid
        if terminated or truncated:
            env.reset()


def test_lookahead_bot_fair_mode_plays_a_full_legal_episode():
    """see_opponent_hand=False re-splits the opponent's hand+bag — check that path too."""
    env = WarChestEnv()
    env.reset()
    bot = _tiny_bot(see_opponent_hand=False)
    for _ in range(30):
        legal = env.get_possible_actions()
        action = bot.act(env)
        assert action in legal
        _, _, terminated, truncated, info = env.step(action)
        assert info['action'].is_valid
        if terminated or truncated:
            env.reset()
