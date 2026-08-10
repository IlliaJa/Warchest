"""`PolicyThetaBot` — policy proposes, θ-weighted simulation checks (docs/IDEAS.md B1).

What these tests exist to protect, in order of how much a regression would cost:

  1. **The candidate set really is the policy's top-k.** That pruning is where all of this
     bot's strength comes from — the measured 0.53-0.78 against `lookahead_critic` is not
     the heuristic's doing (the same heuristic on an unpruned `SimGreedyBot` scores ~0.0
     against that opponent). A bug that silently widened or ignored the prior would leave a
     bot that still plays legally, still looks fine in a smoke test, and quietly loses.
  2. **`policy_weight` spans the two regimes it is documented to span** — 0 means θ
     decides among the candidates, large means the policy's own first choice wins. This is
     the dial that trades strength against behavioural variety, and the measured trade is
     sharp (see `FAMILY_POLICY_WEIGHT`), so it has to actually be connected.
  3. **A resampling bot draws only from the verified family.** The training pool cannot
     re-measure its opponent; the raw θ prior contains members that lose outright (0.08 at
     the low end), so drawing from it in the pool would silently hand the learner free wins.
"""
import numpy as np
import pytest
import torch

from src.services.environment.warchest_env import WarChestEnv
from src.services.bots.evaluation import THETA_KEYS, LEGACY_THETA, normalize_theta
from src.services.bots.policy_theta_bot import (
    PolicyThetaBot, POLICY_THETA_FAMILY, POLICY_THETA_FAMILY_MEMBERS,
    FAMILY_POLICY_WEIGHT, FAMILY_TOP_K,
)

POLICY = 'data/warchest_ppo_20260808-0607.pth'
pytestmark = pytest.mark.skipif(
    not __import__('os').path.exists(POLICY),
    reason=f'{POLICY} not present (checkpoints are not in the repo)')


def _fresh_env(seed, plies=0):
    env = WarChestEnv(save_game_history=False)
    np.random.seed(seed)
    env.reset()
    for _ in range(plies):
        env.make_random_step()
    return env


def _bot(**kwargs):
    kwargs.setdefault('policy_path', POLICY)
    return PolicyThetaBot(**kwargs)


# --------------------------------------------------------------------------- #
# 1. the policy really is the proposer
# --------------------------------------------------------------------------- #
def test_chosen_move_is_always_inside_the_policys_top_k():
    """The pruning is the strength. Checked against the bot's own prior computation so the
    test cannot drift from the implementation's ego-frame remapping.
    """
    bot = _bot(top_k=4, seed=0)
    for i in range(12):
        env = _fresh_env(400 + i, plies=i % 7)
        legal = env.get_possible_actions()
        if len(legal) <= 1:
            continue
        logp = bot._priors(env, env.active_player)
        ranked = sorted(legal, key=lambda a: logp[bot._ego_index(a, env.active_player)],
                        reverse=True)
        assert bot.act(env) in ranked[:4]


def test_a_large_policy_weight_reproduces_the_policys_own_first_choice():
    """At a big enough prior weight nothing the simulation sees can overrule it — the
    documented "degenerates to the raw policy" end of the dial.
    """
    bot = _bot(policy_weight=1e6, top_k=8, seed=0)
    for i in range(10):
        env = _fresh_env(500 + i, plies=i % 5)
        legal = env.get_possible_actions()
        if len(legal) <= 1:
            continue
        logp = bot._priors(env, env.active_player)
        best = max(legal, key=lambda a: logp[bot._ego_index(a, env.active_player)])
        assert bot.act(env) == best


def test_policy_weight_zero_lets_theta_pick_a_different_move():
    """The other end: with no prior term, two θ must disagree inside the same candidate
    set. If they never did, the family would be cosmetic.

    Probed on **developed** positions on purpose. Measured disagreement rates for the two
    most distant family members (0/60 over plies 0-10, 2/60 over 10-40, 10/59 over 30-90):
    in the opening every unit is still in supply, so the heuristic terms θ re-weights —
    material at risk, stack durability, distance to a capturable base — have nothing to
    read yet and every member ranks the candidates identically. The family separates where
    there is a position to have an opinion about.
    """
    a = _bot(policy_weight=0.0, top_k=FAMILY_TOP_K, theta=POLICY_THETA_FAMILY[0], seed=0)
    b = _bot(policy_weight=0.0, top_k=FAMILY_TOP_K, theta=POLICY_THETA_FAMILY[5], seed=0)
    differing = compared = 0
    for i, plies in enumerate(range(30, 90, 6)):
        for s in range(6):
            env = _fresh_env(900 + s * 17 + i, plies=plies)
            if len(env.get_possible_actions()) <= 1:
                continue
            # The determinization is sampled from the global RNG per act() call, so it is
            # re-pinned before each arm — otherwise this compares draw luck, not θ.
            np.random.seed(31)
            x = a.act(env)
            np.random.seed(31)
            differing += (b.act(env) != x)
            compared += 1
    assert compared >= 30, 'too few usable probes to conclude anything'
    assert differing > 0, 'no θ in the family ever changes the chosen move'


# --------------------------------------------------------------------------- #
# 2. the shipped family
# --------------------------------------------------------------------------- #
def test_every_family_member_spells_out_all_eight_keys():
    """Guards the bug this test was written for: the brawler shipped once with `risk`
    omitted, and `normalize_theta` fills a missing *legacy* key from `LEGACY_THETA` — i.e.
    1.0, not 0.0. The bot that carried these win rates had `risk` = 0.0, so the omission
    silently substituted a different bot for a verified one. Writing every key out, zeros
    included, makes that class of mistake impossible rather than merely unlikely.
    """
    for member in POLICY_THETA_FAMILY_MEMBERS:
        assert set(member.theta) == set(THETA_KEYS), (
            f'member {member.role!r} omits {sorted(set(THETA_KEYS) - set(member.theta))} — '
            f'missing legacy keys silently default to 1.0, not 0.0')


def test_family_roles_are_unique_and_column_header_safe():
    """The gauntlet truncates column headers to 6 chars and names entrants `pt<i><role>`,
    so roles must be 3 chars and distinct or two members become one column.
    """
    roles = [m.role for m in POLICY_THETA_FAMILY_MEMBERS]
    assert len(set(roles)) == len(roles)
    assert all(len(r) == 3 for r in roles)


def test_recorded_win_rates_are_all_above_the_bar():
    """Every member is supposed to beat `lookahead_critic`. If a future edit adds a member
    below 0.5, the family stops being what its documentation claims.
    """
    for member in POLICY_THETA_FAMILY_MEMBERS:
        assert member.wr > 0.5, f'{member.role} recorded at {member.wr}'


def test_every_shipped_family_member_is_a_valid_theta():
    for theta in POLICY_THETA_FAMILY:
        full = normalize_theta(theta)
        assert set(full) == set(THETA_KEYS)
        assert all(v >= 0.0 for v in full.values())


def test_the_family_contains_the_default_theta_as_its_control():
    """A family with no incumbent in it has nothing to compare against."""
    assert any(normalize_theta(t) == normalize_theta(LEGACY_THETA)
               for t in POLICY_THETA_FAMILY)


def test_the_family_is_not_just_one_theta_repeated():
    keys = {tuple(normalize_theta(t)[k] for k in THETA_KEYS) for t in POLICY_THETA_FAMILY}
    assert len(keys) == len(POLICY_THETA_FAMILY)


def test_resampling_draws_only_from_the_verified_family():
    """See the module docstring — the raw prior contains θ that lose outright."""
    bot = _bot(seed=3, resample_each_episode=True)
    allowed = {tuple(normalize_theta(t)[k] for k in THETA_KEYS) for t in POLICY_THETA_FAMILY}
    seen = set()
    for _ in range(40):
        bot.new_episode()
        drawn = tuple(bot.theta[k] for k in THETA_KEYS)
        assert drawn in allowed, 'drew a θ from outside the verified family'
        seen.add(drawn)
    assert len(seen) > 1, 'resampling never actually changed θ'


def test_family_defaults_match_the_measured_configuration():
    """The win rates in `POLICY_THETA_FAMILY`'s comments were measured at these two
    settings; a default that drifts away from them makes those numbers a lie.
    """
    bot = _bot()
    assert bot.policy_weight == FAMILY_POLICY_WEIGHT
    assert bot.top_k == FAMILY_TOP_K


# --------------------------------------------------------------------------- #
# 3. it is still a well-behaved bot
# --------------------------------------------------------------------------- #
def test_plays_a_full_legal_episode():
    bot = _bot(seed=1)
    env = _fresh_env(700)
    for _ in range(4000):
        legal = env.get_possible_actions()
        action = bot.act(env)
        assert action in legal
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    else:
        raise AssertionError('episode did not terminate within the step budget')
    assert sum(bot.usage.values()) > 0


def test_does_not_mutate_the_live_env():
    bot = _bot(seed=2)
    env = _fresh_env(701)
    before = [(u.id, u.loc, u.player_id, u.stack) for u in env.board.units]
    active = env.active_player
    bot.act(env)
    assert env.active_player == active
    assert [(u.id, u.loc, u.player_id, u.stack) for u in env.board.units] == before


def test_gauntlet_spec_round_trips():
    import pickle
    from src.services.gauntlet import build_agent

    spec = {'kind': 'policy_theta', 'name': 'pt0',
            'kwargs': {'policy_path': POLICY, 'theta': POLICY_THETA_FAMILY[1],
                       'top_k': 8, 'policy_weight': 0.0}}
    agent = build_agent(pickle.loads(pickle.dumps(spec)), device=torch.device('cpu'))
    assert agent.name == 'pt0'
    assert agent.top_k == 8
    assert agent.theta == normalize_theta(POLICY_THETA_FAMILY[1])


def test_opponent_pool_routes_it_as_an_env_reading_bot():
    """`act` takes the live env, not the ego obs — the wrong branch of
    `_opponent_env_action` would hand it a dict and crash mid-rollout.
    """
    from src.services.environment.rollout_core import (
        _SEARCH_OPP_TYPES, OPP_GROUP_IDX, OPP_ONEHOT_SLOT, OPP_TYPE_IDX,
    )

    assert 'policy_theta' in _SEARCH_OPP_TYPES
    assert 'policy_theta' in OPP_GROUP_IDX
    # Policy-derived, so it conditions on the `pool` slot rather than `greedy`.
    assert OPP_ONEHOT_SLOT['policy_theta'] == OPP_TYPE_IDX['pool']
