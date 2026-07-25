"""HeuristicEvaluator: the legacy path must stay the exact base/material/pos/risk
formula (LookaheadCriticBot's calibration depends on it), and each new term must
move the score in the right direction for the feature it represents.
"""
import numpy as np
import pytest

from src.services.environment.warchest_env import WarChestEnv
from src.services.environment.rollout_core import SHAPING_C, C_MAT
from src.services.bots.evaluation import HeuristicEvaluator
from _helpers import blank_env, place, SWORDSMAN, KNIGHT


def _fresh_env(seed=0):
    env = WarChestEnv(save_game_history=False)
    np.random.seed(seed)
    env.reset()
    return env


def test_legacy_evaluate_is_exactly_base_mat_pos_risk():
    """enable_new_terms=False reproduces the old `_leaf_potential`: base PBRS +
    material PBRS + positional + risk, and nothing else.
    """
    env = _fresh_env(1)
    ev = HeuristicEvaluator(shaping_anneal=1.0, enable_new_terms=False)
    p = env.active_player
    opp = 3 - p
    base = (SHAPING_C
            * (len(env.board.get_controlled_bases(p)) - len(env.board.get_controlled_bases(opp)))
            * env.winning_base_count)
    mat = C_MAT * (env.boxed_total(opp) - env.boxed_total(p))
    pos = ev.POS_COEFF * (ev._nearest_dist(env, opp) - ev._nearest_dist(env, p))
    risk = ev.RISK_COEFF * ev._material_at_risk(env, p, opp)
    assert ev.evaluate(env, p) == pytest.approx(base + mat + pos + risk)


def test_new_terms_are_purely_additive_over_the_legacy_score():
    """Turning new terms on adds exactly the four new-term functions — it never
    perturbs the legacy base/material/pos/risk contribution.
    """
    env = _fresh_env(2)
    p = env.active_player
    opp = 3 - p
    legacy = HeuristicEvaluator(enable_new_terms=False)
    rich = HeuristicEvaluator(enable_new_terms=True)
    extra = (rich._durability(env, p, opp) + rich._economy(env, p, opp)
             + rich._tempo(env, p) + rich._progress(env, p, opp))
    assert rich.evaluate(env, p) == pytest.approx(legacy.evaluate(env, p) + extra)


def test_shaping_anneal_scales_only_material_not_base():
    """shaping_anneal multiplies the material term (as in rollout_core) but leaves
    the base PBRS untouched — a regression guard on the delegation.
    """
    env = blank_env(active=1)
    place(env, SWORDSMAN, 1, (3, 3), stack=1)
    place(env, KNIGHT, 2, (3, 4), stack=3)  # gives a nonzero material picture
    env.board.change_base_control(2, (0, 1))  # opp gets a base -> nonzero base_diff
    full = HeuristicEvaluator(shaping_anneal=1.0, enable_new_terms=False).evaluate(env, 1)
    half = HeuristicEvaluator(shaping_anneal=0.5, enable_new_terms=False).evaluate(env, 1)
    # base term is identical; only the (here zero-boxed) material term is annealed,
    # so with no boxed coins the two scores match — proves base isn't annealed.
    assert full == pytest.approx(half)


def test_durability_rewards_own_bolstered_stacks():
    env = blank_env(active=1)
    place(env, SWORDSMAN, 1, (3, 3), stack=3)   # 2 bolstered coins
    place(env, KNIGHT, 2, (3, 4), stack=1)      # opp not bolstered
    ev = HeuristicEvaluator(enable_new_terms=True)
    assert ev._durability(env, 1, 2) == pytest.approx(2 * ev.DUR_COEFF)
    # symmetric: opponent's perspective is the negation
    assert ev._durability(env, 2, 1) == pytest.approx(-2 * ev.DUR_COEFF)


def test_durability_is_capped_per_unit():
    env = blank_env(active=1)
    place(env, SWORDSMAN, 1, (3, 3), stack=10)  # absurd stack, must be capped
    ev = HeuristicEvaluator(enable_new_terms=True)
    assert ev._durability(env, 1, 2) == pytest.approx(ev._MAX_BOLSTER * ev.DUR_COEFF)


def test_economy_rewards_draining_own_supply():
    env = blank_env(active=1)
    env.state.supply[1][SWORDSMAN] = 1   # low own supply
    env.state.supply[2][SWORDSMAN] = 3   # high opp supply
    ev = HeuristicEvaluator(enable_new_terms=True)
    assert ev._economy(env, 1, 2) == pytest.approx((3 - 1) * ev.ECON_COEFF)


def test_tempo_rewards_holding_initiative():
    env = blank_env(active=1, initiative=1)
    ev = HeuristicEvaluator(enable_new_terms=True)
    assert ev._tempo(env, 1) == pytest.approx(ev.INIT_COEFF)
    assert ev._tempo(env, 2) == pytest.approx(-ev.INIT_COEFF)


def test_progress_bonus_only_at_one_base_from_winning():
    env = blank_env(active=1)
    ev = HeuristicEvaluator(enable_new_terms=True)
    p = 1
    opp = 2
    # give p exactly winning_base_count - 1 bases
    neutral = [(0, 1), (2, 2), (5, 3), (1, 3), (4, 4), (6, 5)]
    have = len(env.board.get_controlled_bases(p))
    need = env.winning_base_count - 1 - have
    for loc in neutral[:need]:
        env.board.change_base_control(p, loc)
    assert len(env.board.get_controlled_bases(p)) == env.winning_base_count - 1
    assert ev._progress(env, p, opp) == pytest.approx(ev.PROG_COEFF)
    # one more base -> no longer "one short", bonus disappears
    env.board.change_base_control(p, neutral[need])
    assert ev._progress(env, p, opp) == pytest.approx(0.0)
