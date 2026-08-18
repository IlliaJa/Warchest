"""The base-differential PBRS term rides the shaping anneal (docs/IDEAS.md R.0.3).

It used to be the one dense term deliberately held at full weight, which at the 0.1
anneal floor priced one base at 33 boxed coins and paid more per episode (0.205) than
the win/loss terminal did (0.125). `play_episode` now scales it by its own
`base_shaping_anneal`, and `ppo.py` feeds that the same multiplier as holding/material.

The property under test is arithmetic, not behavioural: the multiplier is applied to a
reward computed from states the trajectory already visited, so it cannot change which
actions are taken. Two runs from identical seeds must therefore visit the same states
and differ *only* by a factor on `r_shaping`.
"""
import random

import numpy as np
import pytest
import torch

from src.services.bots import RandomBot
from src.services.environment.rollout_core import play_episode, SHAPING_C
from src.services.environment.warchest_env import WarChestEnv
from src.services.policy.policy import Policy


def run_episode(base_anneal, *, shaping_anneal=0.1, seed=17):
    """One episode vs RandomBot under a fixed seed, returning its score decomposition."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = WarChestEnv()
    policy = Policy(torch.device('cpu'))
    return play_episode(
        env, policy, RandomBot(), main_pid=1, opp_type='random',
        gamma=0.99,
        shaping_anneal=shaping_anneal,
        base_shaping_anneal=base_anneal,
        holding_reward_rate=0.0,
        max_t=200,
    )[1]


def test_base_shaping_scales_by_its_anneal_and_nothing_else_moves():
    full = run_episode(1.0)
    tenth = run_episode(0.1)

    # Same seed => same trajectory: the multiplier is applied after the fact.
    assert full['turns'] == tenth['turns']
    assert full['outcome'] == tenth['outcome']
    for part in ('r_attack', 'r_holding', 'r_material', 'r_terminal', 'r_tempo', 'r_other'):
        assert full[part] == pytest.approx(tenth[part]), part

    # A shaping-free episode would make the assertion vacuous.
    assert abs(full['r_shaping']) > 1e-6
    assert tenth['r_shaping'] == pytest.approx(0.1 * full['r_shaping'])


def test_score_absorbs_the_scaled_term():
    """The anneal reaches `main_score`, not just the logged decomposition."""
    full = run_episode(1.0)
    tenth = run_episode(0.1)
    assert full['main_score'] - tenth['main_score'] == pytest.approx(0.9 * full['r_shaping'])


def test_default_is_the_pre_anneal_behaviour():
    """`base_shaping_anneal` defaults to 1.0 so callers unaware of the split are unchanged."""
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    env = WarChestEnv()
    policy = Policy(torch.device('cpu'))
    default = play_episode(
        env, policy, RandomBot(), main_pid=1, opp_type='random',
        gamma=0.99, shaping_anneal=0.1, holding_reward_rate=0.0, max_t=200,
    )[1]
    assert default['r_shaping'] == pytest.approx(run_episode(1.0)['r_shaping'])


def test_base_to_material_ratio_is_now_flat_across_the_anneal():
    """Why the change was made: annealing both terms together freezes their ratio.

    With only material annealed, one base was worth 3.3 boxed coins at batch 1 and 33 at
    the floor — the anneal itself was widening the gap R.0.3 named.
    """
    from src.services.environment.rollout_core import C_MAT
    at_start, at_floor = 1.0, 0.1
    old_ratio_start = SHAPING_C / (at_start * C_MAT)
    old_ratio_floor = SHAPING_C / (at_floor * C_MAT)
    assert old_ratio_floor == pytest.approx(10 * old_ratio_start)

    new_ratio_start = (at_start * SHAPING_C) / (at_start * C_MAT)
    new_ratio_floor = (at_floor * SHAPING_C) / (at_floor * C_MAT)
    assert new_ratio_floor == pytest.approx(new_ratio_start)
