"""Golden-output regression guard for observation encoding.

Records a deterministic sequence of encoded observations (board / global /
valid_action_mask / privileged) into a committed fixture, then asserts the live
encoder reproduces it byte-for-byte. This is the safety net for the obs-encoder
extraction refactor (docs/next_steps.md Step 1): any silent drift in a
normalizer, plane layout, or the P2 ego-rotation trips this test.

The rollout drives the env directly in the ABSOLUTE action frame
(get_possible_actions -> step), so it exercises generate_observation /
get_privileged_features without going through any policy or remap. Seeding once
up front makes the whole game (owner draw, draft, every action) deterministic.

Regenerate the fixture intentionally with:
    WARCHEST_REGEN_GOLDEN=1 env -u PYTHONPATH .venv/bin/python -m pytest \
        tests/test_obs_golden.py
"""
import os

import numpy as np
import pytest

from src.services.environment.warchest_env import WarChestEnv
from src.services.environment.obs_encoders import get_encoder

_SEED = 12345
_STEPS = 300


def _fixture(version):
    return os.path.join(os.path.dirname(__file__), 'fixtures', f'golden_obs_v{version}.npz')


def _deterministic_rollout(version, steps=_STEPS, seed=_SEED):
    """Collect encoded observations over a fixed random-but-seeded rollout.

    Returns stacked arrays keyed like the obs dict, plus privileged and the
    acting player id, one row per step (resetting on terminal/truncation).
    """
    env = WarChestEnv(obs_encoder=get_encoder(version))
    np.random.seed(seed)
    env.reset()

    boards, globals_, masks, privs, actives = [], [], [], [], []
    for _ in range(steps):
        obs = env.generate_observation()
        boards.append(obs['board'])
        globals_.append(obs['global'])
        masks.append(obs['valid_action_mask'])
        privs.append(env.get_privileged_features())
        actives.append(env.active_player)

        action = int(np.random.choice(env.get_possible_actions()))
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()

    return {
        'board': np.asarray(boards, dtype=np.float32),
        'global': np.asarray(globals_, dtype=np.float32),
        'mask': np.asarray(masks),
        'privileged': np.asarray(privs, dtype=np.float32),
        'active': np.asarray(actives, dtype=np.int64),
    }


@pytest.mark.parametrize('version', [10, 11])
def test_obs_encoding_matches_golden(version):
    data = _deterministic_rollout(version)
    fixture = _fixture(version)

    if os.environ.get('WARCHEST_REGEN_GOLDEN') or not os.path.exists(fixture):
        os.makedirs(os.path.dirname(fixture), exist_ok=True)
        np.savez_compressed(fixture, **data)
        pytest.skip(f'golden baseline (re)generated at {fixture}; rerun to compare')

    gold = np.load(fixture)
    # Exact equality: the encoder is deterministic, so a refactor that preserves
    # behavior must reproduce identical arrays (float op order included).
    np.testing.assert_array_equal(data['active'], gold['active'])
    np.testing.assert_array_equal(data['mask'], gold['mask'])
    np.testing.assert_array_equal(data['board'], gold['board'])
    np.testing.assert_array_equal(data['global'], gold['global'])
    np.testing.assert_array_equal(data['privileged'], gold['privileged'])
