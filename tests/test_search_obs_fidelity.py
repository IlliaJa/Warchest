"""A search bot must evaluate positions on the same observation the nets were trained on.

`GameState` carries everything about a position except one thing: `exploration_map_dict`
(per-player per-cell visit counts) lives on `WarChestEnv`, is created in `set_init_state`
and incremented on every move. `set_state` therefore cannot restore it, and `_apply`
increments the *sim* env's copy for every line a search simulates — so board plane 5, which
is exactly that map, arrived at the nets as a stale accumulating count belonging to no
position at all. Measured before the fix: substituting the sim env's plane 5 for the real
one changes the policy's top-1 move on 8.3 % of states (mean TV 0.123), against the 13.1 %
top-1 change that is the entire value of a 1.0 s search over its own prior (docs/IDEAS.md
R.10 M3, R.10.14).

`LookaheadBot._sync_exploration_map` is the fix and these tests are its contract. They use
the plain `LookaheadBot` (no checkpoints, no search) since the property is about the
observation, not about any net.
"""
import numpy as np

from src.services.bots.lookahead_bot import LookaheadBot, _clone_state
from src.services.environment.obs_encoders import latest_encoder
from src.services.environment.warchest_env import WarChestEnv

EXPLORATION_PLANE = 5


def _env_with_history(n_plies=20, seed=5):
    """A live env some way into a game, so its exploration map is not all zeros."""
    env = WarChestEnv(save_game_history=False)
    np.random.seed(seed)
    env.reset()
    for _ in range(n_plies):
        legal = env.get_possible_actions()
        _, _, term, trunc, _ = env.step(int(np.random.choice(legal)))
        if term or trunc:
            break
    return env


def test_the_live_env_actually_accumulates_an_exploration_map():
    # Guards the premise: if this plane were always zero the rest of the file proves nothing.
    env = _env_with_history()
    plane = env.generate_observation()['board'][EXPLORATION_PLANE]
    assert plane.max() > 0.0


def test_sim_env_plane_matches_the_live_env_after_prepare_root():
    env = _env_with_history()
    encoder = latest_encoder()
    bot = LookaheadBot(time_budget=0.0)
    state, _ = bot._prepare_root(env, env.active_player)

    bot._sim_env.set_state(state)
    bot._sync_exploration_map()
    np.testing.assert_array_equal(
        encoder.encode(bot._sim_env)['board'][EXPLORATION_PLANE],
        env.generate_observation()['board'][EXPLORATION_PLANE],
    )


def test_the_whole_board_matches_not_just_the_plane_that_was_broken():
    env = _env_with_history()
    encoder = latest_encoder()
    bot = LookaheadBot(time_budget=0.0)
    state, _ = bot._prepare_root(env, env.active_player)

    bot._sim_env.set_state(state)
    bot._sync_exploration_map()
    np.testing.assert_array_equal(encoder.encode(bot._sim_env)['board'],
                                  env.generate_observation()['board'])


def test_simulated_moves_do_not_leak_into_later_encodes():
    """The failure mode that made this arbitrary rather than merely stale: `_apply`
    increments the sim env's own map, once per simulated ply, for the whole life of the bot.
    """
    env = _env_with_history()
    encoder = latest_encoder()
    bot = LookaheadBot(time_budget=0.0)
    state, queues = bot._prepare_root(env, env.active_player)

    for _ in range(12):
        legal = bot._legal_from(state)
        if not legal:
            break
        child = _clone_state(state)
        child_queues = {1: list(queues[1]), 2: list(queues[2])}
        bot._apply(child, child_queues, int(np.random.choice(legal)))

    bot._sim_env.set_state(state)
    bot._sync_exploration_map()
    np.testing.assert_array_equal(
        encoder.encode(bot._sim_env)['board'][EXPLORATION_PLANE],
        env.generate_observation()['board'][EXPLORATION_PLANE],
    )


def test_without_the_sync_the_plane_is_wrong_which_is_why_the_sync_exists():
    # The control: the bug is real and the fix is what removes it, rather than the plane
    # being incidentally equal for some other reason.
    env = _env_with_history()
    encoder = latest_encoder()
    bot = LookaheadBot(time_budget=0.0)
    state, _ = bot._prepare_root(env, env.active_player)

    bot._sim_env.set_state(state)  # no _sync_exploration_map()
    assert not np.array_equal(
        encoder.encode(bot._sim_env)['board'][EXPLORATION_PLANE],
        env.generate_observation()['board'][EXPLORATION_PLANE],
    )


def test_sync_is_a_no_op_before_any_root_was_prepared():
    # `LookaheadCriticBot._calibrate_value_scale` encodes states outside a search; the sync
    # must not raise there, it simply has nothing to install.
    bot = LookaheadBot(time_budget=0.0)
    assert bot._root_exploration is None
    bot._sync_exploration_map()


def test_each_root_re_reads_the_map_instead_of_caching_the_first_one():
    # Driven by an explicit mutation rather than by more random plies: the map only grows on
    # *move* actions, so a fixed number of random plies is not a reliable way to change it.
    env = _env_with_history()
    encoder = latest_encoder()
    bot = LookaheadBot(time_budget=0.0)
    state, _ = bot._prepare_root(env, env.active_player)
    bot._sim_env.set_state(state)
    bot._sync_exploration_map()
    early = encoder.encode(bot._sim_env)['board'][EXPLORATION_PLANE].copy()

    env.exploration_map_dict[env.active_player][(3, 3)] += 7
    state, _ = bot._prepare_root(env, env.active_player)
    bot._sim_env.set_state(state)
    bot._sync_exploration_map()
    later = encoder.encode(bot._sim_env)['board'][EXPLORATION_PLANE]

    np.testing.assert_array_equal(later, env.generate_observation()['board'][EXPLORATION_PLANE])
    assert not np.array_equal(early, later)


def test_the_snapshot_is_a_copy_so_a_search_cannot_write_back_into_the_live_env():
    env = _env_with_history()
    bot = LookaheadBot(time_budget=0.0)
    state, queues = bot._prepare_root(env, env.active_player)
    before = {p: env.exploration_map_dict[p].copy() for p in (1, 2)}

    bot._sim_env.set_state(state)
    bot._sync_exploration_map()
    legal = bot._legal_from(state)
    for _ in range(8):
        child = _clone_state(state)
        bot._apply(child, {1: list(queues[1]), 2: list(queues[2])}, int(np.random.choice(legal)))

    for p in (1, 2):
        np.testing.assert_array_equal(env.exploration_map_dict[p], before[p])
