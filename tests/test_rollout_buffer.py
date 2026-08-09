"""Advantage normalisation: the per-opponent centring that replaces the critic's one-hot.

Why this exists (docs/next_iteration.md §3.5, §5 row 6). The critic predicts one value
function across the whole opponent pool, but the returns are wildly opponent-dependent —
measured win rates 1.000 / 0.825 / 0.525 against random / greedy / self. A critic that cannot
tell the opponents apart therefore under-predicts against the weak ones and over-predicts
against the strong ones, so `A = G - V` carries a per-opponent *offset*: every action taken
against `random` looks good and every action against a pool snapshot looks bad, whatever the
action actually was. `critic_v1`/`v2` paid for that with a 3-wide opponent one-hot input;
`critic_v3` drops the input and removes the offset here instead, which is the only reason
dropping it is safe. These two halves must stay matched, so both are pinned:
`test_critic_arch.py` covers the head, this file covers the advantage.

The properties that matter, and what breaks if each regresses:
  1. per-group means come out (the offset is gone from the gradient);
  2. groups are NOT rescaled to unit variance individually (that would upweight the
     near-deterministic vs-random group's noise to match the group carrying real signal);
  3. the batch still has unit std (so the policy step size does not silently change);
  4. a tiny group falls back to the batch mean (a mean from a handful of correlated steps is
     noise, and subtracting it injects bias rather than removing it);
  5. `adv_norm='global'` still reproduces the pre-2026-08-09 behaviour exactly.
"""
import numpy as np
import pytest
import torch

from src.services.environment.rollout_core import OPP_GROUP_IDX, opp_group_id
from src.utils.rollout_buffer import RolloutBuffer

DEV = torch.device('cpu')


def _buffer(group_rewards, *, ep_len=8):
    """A buffer of one episode per (group, reward) pair, with V(s) fixed at 0.

    With values identically zero, GAE reduces to a discounted sum of that episode's rewards,
    so each group's advantages are a known constant scaled by the discount — which makes the
    per-group mean an exactly predictable quantity rather than something to eyeball.
    """
    buf = RolloutBuffer()
    for gid, reward in group_rewards:
        for _ in range(ep_len):
            buf.add_step(obs=None, action=0, log_prob=torch.tensor(0.0), reward=reward,
                         opp_onehot=np.zeros(3, dtype=np.float32),
                         privileged=np.zeros(2, dtype=np.float32), opp_id=gid)
        buf.end_episode()
    buf._opp_ids_arr = np.array(buf._opp_ids, dtype=np.int64)
    buf.set_values([0.0] * len(buf._rewards))
    return buf


def _episodes(group, reward, n):
    return [(group, reward)] * n


def test_per_opponent_centring_removes_the_between_group_offset():
    # Two opponents, one paying 10x the other: a pure identity offset, no action quality.
    buf = _buffer(_episodes(0, 1.0, 12) + _episodes(2, 0.1, 12))
    buf.compute_gae(gamma=0.99, lam=0.97, device=DEV, adv_norm='per_opponent')

    ids = buf._opp_ids_arr
    adv = buf.advantages.numpy()
    # Each group's post-normalisation mean is ~0: the offset is gone from the gradient.
    for gid in (0, 2):
        assert abs(adv[ids == gid].mean()) < 1e-5

    # And the offsets that were removed are reported, with the right sign and ordering.
    assert set(buf.adv_group_offsets) == {0, 2}
    assert buf.adv_group_offsets[0] > buf.adv_group_offsets[2]


def test_global_normalisation_leaves_the_offset_in_place():
    """The baseline arm must still be biased — otherwise the A/B compares nothing."""
    buf = _buffer(_episodes(0, 1.0, 12) + _episodes(2, 0.1, 12))
    buf.compute_gae(gamma=0.99, lam=0.97, device=DEV, adv_norm='global')

    ids = buf._opp_ids_arr
    adv = buf.advantages.numpy()
    assert adv[ids == 0].mean() > 0.5   # everything vs the weak opponent looks good
    assert adv[ids == 2].mean() < -0.5  # everything vs the strong one looks bad
    assert buf.adv_group_offsets == {}


def test_groups_keep_their_relative_spread_and_are_not_individually_rescaled():
    """Per-group z-scoring would equalise these; per-group centring must not."""
    # One step per episode, so each advantage IS its reward and the group spreads are exactly
    # what this sets them to (with longer episodes the GAE decay across steps dominates and
    # both groups end up equally wide, which measures nothing).
    # Group 0: near-constant returns, like always beating `random`. Group 2: real variance.
    narrow = [(0, 1.0 + 1e-4 * i) for i in range(96)]
    wide = [(2, r) for r in np.linspace(-2.0, 2.0, 96)]
    buf = _buffer(narrow + wide, ep_len=1)
    buf.compute_gae(gamma=0.99, lam=0.97, device=DEV, adv_norm='per_opponent')

    ids = buf._opp_ids_arr
    adv = buf.advantages.numpy()
    narrow_std = adv[ids == 0].std()
    wide_std = adv[ids == 2].std()
    assert wide_std > 10 * narrow_std, (
        f'the low-variance group was amplified to match the high-variance one '
        f'(narrow={narrow_std:.4g}, wide={wide_std:.4g}) — that is per-group z-scoring, '
        f'which hands the near-deterministic opponent as much gradient weight as the one '
        f'carrying the real signal'
    )


def test_batch_still_has_unit_std_so_the_step_size_does_not_change():
    buf = _buffer(_episodes(0, 1.0, 12) + _episodes(2, 0.1, 12))
    buf.compute_gae(gamma=0.99, lam=0.97, device=DEV, adv_norm='per_opponent')
    # Scaling happens AFTER centring; scaling on the pre-centring std would land well
    # below 1.0 here, because removing the between-group spread shrinks the std a lot.
    # Sample std (correction=1), matching the `.std()` the normalisation itself divides by.
    assert buf.advantages.std().item() == pytest.approx(1.0, abs=1e-4)


def test_a_group_too_small_to_estimate_falls_back_to_the_batch_mean():
    # One episode of 8 steps is far below MIN_GROUP_SAMPLES; the big group is above it.
    buf = _buffer(_episodes(0, 1.0, 12) + _episodes(4, 5.0, 1))
    assert RolloutBuffer.MIN_GROUP_SAMPLES > 8
    buf.compute_gae(gamma=0.99, lam=0.97, device=DEV, adv_norm='per_opponent')

    assert 4 not in buf.adv_group_offsets, 'a mean from 8 correlated steps is noise, not a mean'
    assert 0 in buf.adv_group_offsets
    # Left uncentred, the outlier group keeps its (large) deviation rather than being
    # flattened to zero mean by its own unreliable estimate.
    ids = buf._opp_ids_arr
    assert abs(buf.advantages.numpy()[ids == 4].mean()) > 1.0


def test_missing_opp_ids_degrade_to_plain_global_centring():
    """Buffers built by paths that never set opp_ids must not crash or mis-slice."""
    buf = _buffer(_episodes(0, 1.0, 12))
    buf._opp_ids_arr = None
    buf.compute_gae(gamma=0.99, lam=0.97, device=DEV, adv_norm='per_opponent')
    assert buf.adv_group_offsets == {}
    assert abs(float(buf.advantages.mean())) < 1e-5


def test_unknown_adv_norm_is_rejected():
    buf = _buffer(_episodes(0, 1.0, 12))
    with pytest.raises(ValueError, match='adv_norm'):
        buf.compute_gae(gamma=0.99, lam=0.97, device=DEV, adv_norm='per_episode')


# --------------------------------------------------------------------------- #
# The group labelling itself
# --------------------------------------------------------------------------- #

def test_search_opponents_get_their_own_groups_unlike_the_critic_onehot():
    """The whole point of a separate id: `OPP_ONEHOT_SLOT` collapses these onto `pool`,
    and finetune is 75 % pool / 25 % lookahead_critic — two opponents of different
    strength whose offsets must not be averaged together."""
    assert opp_group_id('lookahead_critic') != opp_group_id('pool')
    assert opp_group_id('puct') != opp_group_id('pool')
    assert len({opp_group_id(t) for t in OPP_GROUP_IDX}) == len(OPP_GROUP_IDX)


def test_an_unregistered_opponent_gets_a_fallback_group_instead_of_crashing():
    """Adding a bot to the pool must not kill a running job over a missing dict entry."""
    gid = opp_group_id('some_future_exploiter')
    assert gid not in OPP_GROUP_IDX.values()
    assert gid == opp_group_id('another_future_exploiter')  # shared bucket, warned once


def test_batch_mean_is_zero_even_when_a_group_falls_back():
    """A non-zero mean advantage is a uniform push on every sampled action, so the
    fallback path must not leave one behind."""
    buf = _buffer(_episodes(0, 1.0, 12) + _episodes(4, 5.0, 1))
    buf.compute_gae(gamma=0.99, lam=0.97, device=DEV, adv_norm='per_opponent')
    assert abs(float(buf.advantages.mean())) < 1e-5
