"""Parallel round-robin gauntlet: task-list parity + spec-based agent rebuilding."""
import pickle

import numpy as np
import pytest
import torch

from src.services.gauntlet import (
    round_robin, build_agent, greedy_agent, random_agent,
)
from src.services.gauntlet_parallel import round_robin_parallel
from src.services.bots.lookahead_bot import LookaheadBot
from src.services.policy.policy import Policy
from src.services.policy.checkpoint import save_policy_checkpoint
from src.services.environment.obs_encoders import latest_encoder, LATEST_VERSION


# --------------------------------------------------------------------------- #
# Correctness: parallel dispatch must reproduce the sequential result matrix
# --------------------------------------------------------------------------- #
def test_round_robin_parallel_matches_sequential():
    specs = [{'kind': 'random', 'name': 'random'}, {'kind': 'greedy', 'name': 'greedy'}]
    agents = [random_agent('random'), greedy_agent('greedy')]

    sequential = round_robin(agents, k_games=6, seed=0)
    parallel = round_robin_parallel(specs, ['random', 'greedy'], k_games=6, seed=0, n_workers=2)

    assert np.array_equal(sequential['wins'], parallel['wins'])
    assert np.array_equal(sequential['games'], parallel['games'])
    assert sequential['ratings'] == parallel['ratings']
    assert sequential['intransitive_fraction'] == parallel['intransitive_fraction']


def test_round_robin_parallel_more_workers_than_tasks():
    # n_workers larger than the task count: extra workers must idle and shut down cleanly.
    specs = [{'kind': 'random', 'name': 'random'}, {'kind': 'greedy', 'name': 'greedy'}]
    out = round_robin_parallel(specs, ['random', 'greedy'], k_games=2, seed=0, n_workers=8)
    assert out['games'][0, 1] == 2 and out['games'][1, 0] == 2


# --------------------------------------------------------------------------- #
# build_agent: every spec kind reconstructs a working agent
# --------------------------------------------------------------------------- #
def test_build_agent_greedy_and_random():
    g = build_agent({'kind': 'greedy', 'name': 'greedy'}, device='cpu')
    r = build_agent({'kind': 'random', 'name': 'random'}, device='cpu')
    assert g.name == 'greedy' and r.name == 'random'


def test_build_agent_lookahead():
    a = build_agent({'kind': 'lookahead', 'name': 'lookahead',
                      'kwargs': {'time_budget': 0.01, 'max_branching': 4}}, device='cpu')
    assert a.name == 'lookahead'


def test_build_agent_policy_from_checkpoint(tmp_path):
    enc = latest_encoder()
    policy = Policy(device=torch.device('cpu'), hidden_dim=16, obs_encoder=enc)
    ckpt_path = tmp_path / 'ckpt.pth'
    save_policy_checkpoint(policy, str(ckpt_path), obs_version=LATEST_VERSION, hidden_dim=16)

    agent = build_agent({'kind': 'policy', 'path': str(ckpt_path)}, device='cpu')
    assert agent is not None
    from src.services.gauntlet import play_game
    res = play_game(agent, random_agent(), seed=0)
    assert res in (0, 1, 2)


# --------------------------------------------------------------------------- #
# Why the spec layer exists at all: live lookahead bots are unpicklable
# --------------------------------------------------------------------------- #
def test_lookahead_bot_is_not_picklable():
    # `_sim_env._draw_one` is monkeypatched to a bound method whose `__name__` doesn't
    # match the attribute it's stored under (`_determinized_draw_one` vs `_draw_one`);
    # pickle's bound-method reduction resolves by `getattr(obj, func.__name__)` at
    # *unpickling* time, so `dumps` succeeds but the round trip fails on `loads`.
    bot = LookaheadBot(name='lookahead', time_budget=0.01)
    with pytest.raises((AttributeError, TypeError, pickle.PicklingError)):
        pickle.loads(pickle.dumps(bot))
