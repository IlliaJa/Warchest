"""`RandomEvalBot` — the B1 randomised-coefficient family (docs/IDEAS.md B1).

Three properties this file is here to pin, in order of how much damage a regression
would do:

  1. **The default θ changes nothing.** `RandomEvalBot(theta=LEGACY_THETA)` must make the
     same decisions as a plain `SimGreedyBot`. The θ parametrisation touches
     `HeuristicEvaluator.evaluate`, which is the leaf of every search bot in the repo and
     the distribution `LookaheadCriticBot`'s value-scale calibration is moment-matched
     against — a silent shift there would void measurements taken with bots that never
     heard of θ. Asserted on *action values under a shared determinization*, not over a
     whole game: these bots re-sample the future draw order from the global RNG on every
     `act()`, so even two plain `SimGreedyBot`s drift apart. See the test's own docstring.
  2. **A different θ is a different bot.** The whole proposal is worthless if the search
     washes the coefficients out; a test that only checked (1) would pass on a no-op.
  3. **θ never touches the global RNG.** `gauntlet.play_game` re-seeds the global RNG per
     game to pin the draft, and the schedule is antithetic — the two games of a pair
     replay one draft with the seats swapped. A bot that drew θ from that stream would
     desynchronise the pair and silently destroy the variance reduction it exists for.
"""
import numpy as np
import pytest

from src.services.environment.warchest_env import WarChestEnv
from src.services.bots.evaluation import (
    THETA_KEYS, THETA_RANGES, LEGACY_THETA, RICH_THETA, HeuristicEvaluator,
    normalize_theta, sample_theta, theta_tag, format_theta,
)
from src.services.bots.greedy_sim_bot import SimGreedyBot
from src.services.bots.lookahead_bot import LookaheadBot, _clone_state
from src.services.bots.random_eval_bot import RandomEvalBot, RandomEvalLookaheadBot


def _fresh_env(seed):
    env = WarChestEnv(save_game_history=False)
    np.random.seed(seed)
    env.reset()
    return env


def _play_actions(bot, seed, n_steps=60):
    """Drive `bot` against itself for `n_steps` plies; return the action sequence."""
    env = _fresh_env(seed)
    actions = []
    for _ in range(n_steps):
        action = bot.act(env)
        actions.append(action)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return actions


# --------------------------------------------------------------------------- #
# 1. the default θ is the status quo
# --------------------------------------------------------------------------- #
def test_default_theta_changes_no_action_value_across_many_states():
    """The equivalence guard: θ=`LEGACY_THETA` must not change a single action's value.

    Compared on **one shared determinization** per probe, not by calling `act()` twice.
    That detail is load-bearing and has nothing to do with θ: `LookaheadBot._prepare_root`
    samples a fresh determinization of the future draw order from the *global* RNG on every
    `act()` call (documented design — see its module docstring), and the draw is sensitive
    enough that re-pinning the seed around each call is not sufficient — two bots differing
    only in construction order get different queues. Two *plain* `SimGreedyBot`s already
    diverge on the same seeded game for exactly this reason.

    So the comparison hands both bots the same root state and the same queues and checks
    every legal action's value. That is strictly stronger than comparing the argmax — it
    would catch a θ effect too small to flip a decision — and it is exactly deterministic.
    """
    np.random.seed(1234)
    plain = SimGreedyBot()
    np.random.seed(1234)
    themed = RandomEvalBot(theta=LEGACY_THETA)

    compared = 0
    for i in range(25):
        env = _fresh_env(200 + i)
        for _ in range(i % 9):          # spread the probes over the opening, not just ply 0
            env.make_random_step()
        root_player = env.active_player
        legal = env.get_possible_actions()
        if len(legal) <= 1:
            continue
        np.random.seed(99)
        root_state, queues = plain._prepare_root(env, root_player)
        for action in legal:
            expected = plain._value_after_my_turn(
                _clone_state(root_state), {1: list(queues[1]), 2: list(queues[2])},
                action, root_player)
            actual = themed._value_after_my_turn(
                _clone_state(root_state), {1: list(queues[1]), 2: list(queues[2])},
                action, root_player)
            assert actual == expected, (
                f'theta=LEGACY changed action {action}\'s value at probe {i}: '
                f'{actual!r} vs {expected!r}')
            compared += 1
    assert compared > 100, f'only {compared} action values compared — probe set too thin'


def test_legacy_theta_leaf_is_bit_identical_to_the_pre_theta_evaluator():
    env = _fresh_env(11)
    p = env.active_player
    default = HeuristicEvaluator(shaping_anneal=1.0)
    explicit = HeuristicEvaluator(shaping_anneal=1.0, theta=LEGACY_THETA)
    # Bit-identical, not approx: LookaheadCriticBot calibrates on this exact distribution.
    assert explicit.evaluate(env, p) == default.evaluate(env, p)


def test_rich_theta_reproduces_enable_new_terms():
    env = _fresh_env(12)
    p = env.active_player
    rich = HeuristicEvaluator(enable_new_terms=True)
    via_theta = HeuristicEvaluator(theta=RICH_THETA)
    assert via_theta.evaluate(env, p) == rich.evaluate(env, p)


def test_theta_and_enable_new_terms_together_are_a_contradiction():
    with pytest.raises(ValueError, match='not both'):
        HeuristicEvaluator(enable_new_terms=True, theta={'economy': 3.0})


def test_rich_eval_kwarg_is_rejected():
    """θ subsumes `rich_eval`; accepting both would give two disagreeing sources of truth."""
    with pytest.raises(TypeError, match='rich_eval'):
        RandomEvalBot(rich_eval=True)


# --------------------------------------------------------------------------- #
# 2. a different θ is a different bot
# --------------------------------------------------------------------------- #
def test_each_term_scales_linearly_and_independently():
    """Doubling one θ entry doubles exactly that term's contribution, leaving the rest.

    The evaluator folds θ into precomputed coefficients on the hot path, so this is the
    guard that the folding kept the terms separable.
    """
    env = _fresh_env(13)
    p = env.active_player
    for key in THETA_KEYS:
        base = HeuristicEvaluator(theta=RICH_THETA).evaluate(env, p)
        doubled = HeuristicEvaluator(theta={**RICH_THETA, key: 2.0}).evaluate(env, p)
        term = HeuristicEvaluator(
            theta={k: (1.0 if k == key else 0.0) for k in THETA_KEYS}).evaluate(env, p)
        assert doubled == pytest.approx(base + term), f'{key} did not scale linearly'


def test_zeroing_a_term_removes_it_entirely():
    env = _fresh_env(14)
    p = env.active_player
    full = HeuristicEvaluator(theta=RICH_THETA).evaluate(env, p)
    without = HeuristicEvaluator(theta={**RICH_THETA, 'economy': 0.0}).evaluate(env, p)
    economy = HeuristicEvaluator(theta=RICH_THETA)._economy(env, p, 3 - p)
    assert without == pytest.approx(full - economy)


def test_a_recruit_hungry_theta_recruits_more_than_the_default():
    """The economy term exists to make `recruit` register as a gain; a big weight on it
    must actually change what the bot commits to, or the family is a no-op.
    """
    default = RandomEvalBot(theta=LEGACY_THETA)
    hungry = RandomEvalBot(theta={**LEGACY_THETA, 'economy': 10.0})
    for seed in (3, 4, 5):
        _play_actions(default, seed=seed, n_steps=40)
        _play_actions(hungry, seed=seed, n_steps=40)
    default_rate = default.usage['recruit'] / max(sum(default.usage.values()), 1)
    hungry_rate = hungry.usage['recruit'] / max(sum(hungry.usage.values()), 1)
    assert hungry_rate > default_rate, (
        f'economy=10 recruited {hungry_rate:.2f} of the time vs the default '
        f'{default_rate:.2f} — theta is not reaching the bot\'s choices')


def test_two_sampled_thetas_diverge_within_one_game():
    """Different seeds must produce bots that actually diverge, not just differ on paper."""
    a = _play_actions(RandomEvalBot(seed=0), seed=21)
    b = _play_actions(RandomEvalBot(seed=1), seed=21)
    assert a != b


# --------------------------------------------------------------------------- #
# 3. sampling hygiene
# --------------------------------------------------------------------------- #
def test_theta_sampling_never_touches_the_global_rng():
    """See the module docstring: the antithetic draft pairing depends on this.

    Scoped to `new_episode()`, which is what fires *inside* the schedule (once per game,
    after `play_game` has seeded the global RNG and drawn the draft). Construction is
    deliberately not covered: it builds a `WarChestEnv`, which does consume global
    randomness, and it happens once up front — before any game seed is set.
    """
    bot = RandomEvalBot(seed=99, resample_each_episode=True)
    np.random.seed(1234)
    # Keys AND position: an ordinary draw only advances the position — the 624-word key
    # array is untouched until the generator regenerates, so comparing keys alone would
    # miss exactly the leak this test exists to catch.
    keys_before, pos_before = np.random.get_state()[1].copy(), np.random.get_state()[2]
    for _ in range(20):
        bot.new_episode()
    assert len(bot.theta_history) == 21  # it really did resample 20 times
    assert np.array_equal(np.random.get_state()[1], keys_before)
    assert np.random.get_state()[2] == pos_before


def test_same_seed_same_theta_different_seed_different_theta():
    assert RandomEvalBot(seed=5).theta == RandomEvalBot(seed=5).theta
    assert RandomEvalBot(seed=5).theta != RandomEvalBot(seed=6).theta


def test_sampled_theta_respects_its_ranges():
    rng = np.random.default_rng(0)
    for _ in range(500):
        theta = sample_theta(rng)
        assert set(theta) == set(THETA_KEYS)
        for key, value in theta.items():
            lo, hi = THETA_RANGES[key]
            assert value == 0.0 or lo <= value <= hi, f'{key}={value} outside [{lo}, {hi}]'
    # `base` is never zeroed — a bot that does not want bases is RandomBot, which the pool
    # already has (see THETA_ZERO_PROB).
    assert all(sample_theta(np.random.default_rng(s))['base'] > 0 for s in range(200))


def test_the_family_reaches_both_zero_and_the_top_of_each_range():
    """A family whose draws all sit near 1.0 would be a parameter sweep, not archetypes.

    "Large" is relative to each key's own range — `durability` is capped at 1.0 by
    measurement (see THETA_RANGES) while `economy` reaches 12, so an absolute threshold
    would either be vacuous for one or unreachable for the other.
    """
    rng = np.random.default_rng(3)
    draws = [sample_theta(rng) for _ in range(300)]
    for key in ('durability', 'economy', 'tempo', 'progress'):
        hi = THETA_RANGES[key][1]
        assert any(d[key] == 0.0 for d in draws), f'{key} is never switched off'
        assert any(d[key] >= 0.7 * hi for d in draws), f'{key} never nears its own ceiling'


def test_new_episode_resamples_only_when_asked():
    pinned = RandomEvalBot(seed=2, resample_each_episode=False)
    first = dict(pinned.theta)
    for _ in range(5):
        pinned.new_episode()
    assert pinned.theta == first
    assert len(pinned.theta_history) == 1

    rolling = RandomEvalBot(seed=2, resample_each_episode=True)
    start = dict(rolling.theta)
    rolling.new_episode()
    assert rolling.theta != start
    assert len(rolling.theta_history) == 2


# --------------------------------------------------------------------------- #
# θ validation + labelling
# --------------------------------------------------------------------------- #
def test_partial_theta_fills_from_the_legacy_default():
    assert normalize_theta({'economy': 4.0}) == {**LEGACY_THETA, 'economy': 4.0}


def test_unknown_theta_key_raises_rather_than_being_ignored():
    with pytest.raises(ValueError, match='unknown theta keys'):
        normalize_theta({'econ': 4.0})


def test_negative_theta_is_rejected():
    with pytest.raises(ValueError, match='must be >= 0'):
        normalize_theta({'material': -1.0})


def test_theta_tag_names_the_most_stretched_term_not_the_biggest_number():
    """`tempo` reaches 20x and `base` only 2x, so ranking raw multipliers would tag almost
    every draw `ini`. The tag is range-relative.
    """
    # base at its 2.0 ceiling (stretch 1.0) beats a 2.1x tempo (a quarter of tempo's own
    # log-range), even though 2.1 is the larger raw multiplier.
    assert theta_tag({'base': 2.0, 'tempo': 2.1}) == 'bas'
    # ... and loses to a tempo at *its* ceiling. Both at 1.0 stretch would tie, so `base`
    # sits mid-range here to make the comparison strict.
    assert theta_tag({'base': 1.0, 'tempo': 20.0}) == 'ini'
    # A zeroed term never wins the tag.
    assert theta_tag({k: 0.0 for k in THETA_KEYS if k != 'base'}) == 'bas'


def test_format_theta_lists_every_key():
    text = format_theta(sample_theta(np.random.default_rng(1)))
    assert text.count('=') == len(THETA_KEYS)


# --------------------------------------------------------------------------- #
# integration: the bot is still a legal, non-mutating bot
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('seed', [0, 3, 7])
def test_a_sampled_theta_plays_a_full_legal_episode(seed):
    """No θ in the family may produce an illegal move or a hang — an opponent-pool entrant
    that crashes a rollout is worse than no entrant.
    """
    bot = RandomEvalBot(seed=seed)
    env = _fresh_env(100 + seed)
    for _ in range(4000):
        legal = env.get_possible_actions()
        action = bot.act(env)
        assert action in legal
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    else:
        raise AssertionError('episode did not terminate within the step budget')


def test_does_not_mutate_the_live_env():
    bot = RandomEvalBot(seed=1)
    env = _fresh_env(5)
    units_before = [(u.id, u.loc, u.player_id, u.stack) for u in env.board.units]
    active_before = env.active_player
    bot.act(env)
    assert env.active_player == active_before
    assert [(u.id, u.loc, u.player_id, u.stack) for u in env.board.units] == units_before


# --------------------------------------------------------------------------- #
# Training-pool wiring (docs/IDEAS.md B1)
# --------------------------------------------------------------------------- #
def test_opponent_pool_resamples_theta_every_episode():
    """The pool hands out one shared instance, so per-episode variety exists only if
    `sample()` fires the hook. Without it the whole `random_eval` slice would be a single
    fixed playstyle — strictly worse than the greedy it replaces.
    """
    from src.services.opponent_pool import OpponentPool

    pool = OpponentPool(p_random=0.0, p_greedy=0.0, p_pool=0.0, p_random_eval=1.0)
    thetas = []
    for _ in range(6):
        bot, label = pool.sample(None, 'cpu')
        assert label == 'random_eval'
        thetas.append(tuple(bot.theta[k] for k in THETA_KEYS))
    assert len(set(thetas)) == len(thetas), 'theta did not change between episodes'


def test_random_eval_weight_is_off_by_default_and_excluded_from_sampling():
    """A zero weight must keep the bot out of the type list entirely — it is built lazily,
    and merely listing it would construct one in every rollout worker for nothing.
    """
    from src.services.opponent_pool import OpponentPool

    pool = OpponentPool(p_random=1.0, p_greedy=0.0, p_pool=0.0)
    assert pool.weights['p_random_eval'] == 0.0
    for _ in range(20):
        _, label = pool.sample(None, 'cpu')
        assert label == 'random'
    assert pool._random_eval_bot is None


def test_random_eval_is_routed_as_an_env_reading_bot():
    """`RandomEvalBot.act` takes the live env, not the ego-centric obs. Landing in the
    wrong branch of `_opponent_env_action` would pass it a dict and crash mid-rollout.
    """
    from src.services.environment.rollout_core import (
        _SEARCH_OPP_TYPES, OPP_GROUP_IDX, OPP_ONEHOT_SLOT, OPP_TYPE_IDX, opp_group_id,
    )

    assert 'random_eval' in _SEARCH_OPP_TYPES
    # Its own advantage-centring group (it is a distinct strength tier), but the `greedy`
    # critic one-hot slot, which is pinned at 3 wide by existing checkpoints.
    assert opp_group_id('random_eval') == OPP_GROUP_IDX['random_eval']
    assert OPP_ONEHOT_SLOT['random_eval'] == OPP_TYPE_IDX['greedy']


# --------------------------------------------------------------------------- #
# The same θ on a deeper base bot (RandomEvalLookaheadBot)
# --------------------------------------------------------------------------- #
def test_both_bases_draw_the_same_theta_from_the_same_seed():
    """The two family members are only comparable if one seed means one θ on both. If the
    sampling drifted, a `--base greedy` vs `--base lookahead` measurement would be
    comparing different playstyles, not different search depths.
    """
    greedy = RandomEvalBot(seed=11)
    deep = RandomEvalLookaheadBot(seed=11, time_budget=0.01)
    assert greedy.theta == deep.theta
    assert greedy.name != deep.name  # ... but they must not collide in a gauntlet field


def test_lookahead_base_default_theta_still_plays_like_a_plain_lookahead_bot():
    """Same guard as the SimGreedy one: threading θ through `LookaheadBot` must not perturb
    the default. Compared on the leaf value rather than an action sequence — LookaheadBot
    is wall-clock budgeted, so its action choices are not reproducible run to run.
    """
    env = _fresh_env(19)
    p = env.active_player
    plain = LookaheadBot(time_budget=0.01)
    themed = RandomEvalLookaheadBot(theta=LEGACY_THETA, time_budget=0.01)
    plain._sim_env.set_state(env.state)
    themed._sim_env.set_state(env.state)
    assert themed._evaluator.evaluate(themed._sim_env, p) == \
        plain._evaluator.evaluate(plain._sim_env, p)


def test_lookahead_base_records_a_verb_profile():
    """`usage` moved up to LookaheadBot so both depths report the same behaviour profile —
    without it `eval_theta_family.py --base lookahead` would read an empty counter and
    report every arm as identical.
    """
    bot = RandomEvalLookaheadBot(seed=2, time_budget=0.01)
    env = _fresh_env(23)
    for _ in range(12):
        action = bot.act(env)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    assert sum(bot.usage.values()) > 0
    assert set(bot.usage) <= {'move', 'attack', 'control', 'bolster', 'deploy', 'recruit',
                              'tactic', 'claim_initiative', 'pass', 'decline', 'select'}


def test_lookahead_base_plays_a_full_legal_episode():
    bot = RandomEvalLookaheadBot(seed=5, time_budget=0.01)
    env = _fresh_env(29)
    for _ in range(4000):
        legal = env.get_possible_actions()
        action = bot.act(env)
        assert action in legal
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    else:
        raise AssertionError('episode did not terminate within the step budget')


def test_lookahead_theta_spec_round_trips_through_the_gauntlet_builder():
    import pickle
    import torch
    from src.services.gauntlet import build_agent

    spec = {'kind': 'random_eval_lookahead', 'name': 'la0eco',
            'kwargs': {'theta': {'economy': 5.0}, 'seed': 1, 'time_budget': 0.02}}
    agent = build_agent(pickle.loads(pickle.dumps(spec)), device=torch.device('cpu'))
    assert agent.name == 'la0eco'
    assert agent.theta['economy'] == 5.0
    assert agent.time_budget == 0.02
