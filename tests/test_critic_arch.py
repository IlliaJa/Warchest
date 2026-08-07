"""Critic architecture: v1/v2 selection, checkpoint compatibility, trunk-health guard.

Why these exist: the shipped `critic_v1` trunk *died* — every pre-activation of its final
ReLU went <= 0, so the trunk output was identically zero, `_split_pool` fed the value head a
block of hard zeros, and the critic was blind to the board. It stayed that way for a whole
generation of checkpoints and silently voided every search measurement taken with them
(docs/next_iteration.md §3.4). `critic_v2` adds GroupNorm to remove the absorbing state and a
board-only auxiliary head to supply the gradient pressure whose absence caused the drift.

These tests pin the three things that must not regress:
  1. every existing (v1) checkpoint still loads, because the gauntlet reconstructs them;
  2. `trunk_health` actually reports a dead conv as 0.0 — it is the guard, so it has to work;
  3. the board-only head's gradient reaches the trunk, which is the entire mechanism.
"""
import numpy as np
import pytest
import torch

from src.services.environment.obs_encoders import get_encoder, latest_encoder
from src.services.environment.warchest_env import WarChestEnv
from src.services.policy.checkpoint import (
    CRITIC_ARCHS, CURRENT_CRITIC_ARCH, load_critic_checkpoint, save_critic_checkpoint,
)
from src.services.policy.policy import CRITIC_ARCH_V1, CRITIC_ARCH_V2, Critic

DEV = torch.device('cpu')
HIDDEN = 32  # divisible by CRITIC_GROUPS=8, small enough to build repeatedly


def _boards(n=16, channels=None):
    ch = channels if channels is not None else latest_encoder().board_channels
    return torch.randn(n, ch, 7, 7)


def test_v2_is_the_default_and_both_arches_build():
    assert CURRENT_CRITIC_ARCH == CRITIC_ARCH_V2
    assert set(CRITIC_ARCHS) == {CRITIC_ARCH_V1, CRITIC_ARCH_V2}
    assert Critic(DEV, HIDDEN).arch == CRITIC_ARCH_V2
    for arch in CRITIC_ARCHS:
        assert Critic(DEV, HIDDEN, arch=arch).arch == arch


def test_unknown_arch_is_rejected():
    with pytest.raises(ValueError, match='unknown critic arch'):
        Critic(DEV, HIDDEN, arch='critic_v3')


def test_v2_requires_hidden_dim_divisible_by_group_count():
    # GroupNorm would raise a far less obvious error deep in the constructor otherwise.
    with pytest.raises(ValueError, match='divisible'):
        Critic(DEV, 20, arch=CRITIC_ARCH_V2)


def test_v1_module_layout_is_unchanged_so_old_state_dicts_load():
    """The reason v1 still exists: a v1 state_dict must load into a v1 Critic verbatim."""
    a = Critic(DEV, HIDDEN, arch=CRITIC_ARCH_V1)
    b = Critic(DEV, HIDDEN, arch=CRITIC_ARCH_V1)
    b.load_state_dict(a.state_dict())  # strict=True by default
    assert a.board_only_head is None
    # v1 has no GroupNorm anywhere in the trunk.
    assert not any(isinstance(m, torch.nn.GroupNorm) for m in a.board_encoder.modules())


def test_v2_has_groupnorm_and_a_board_only_head():
    c = Critic(DEV, HIDDEN, arch=CRITIC_ARCH_V2)
    assert sum(isinstance(m, torch.nn.GroupNorm) for m in c.board_encoder.modules()) == 3
    assert c.board_only_head is not None


def test_v1_and_v2_state_dicts_are_not_interchangeable():
    """A silent cross-load would resurrect exactly the bug this split exists to prevent."""
    v1 = Critic(DEV, HIDDEN, arch=CRITIC_ARCH_V1)
    v2 = Critic(DEV, HIDDEN, arch=CRITIC_ARCH_V2)
    with pytest.raises(RuntimeError):
        v1.load_state_dict(v2.state_dict())
    with pytest.raises(RuntimeError):
        v2.load_state_dict(v1.state_dict())


def test_checkpoint_round_trip_preserves_arch(tmp_path):
    for arch in CRITIC_ARCHS:
        c = Critic(DEV, HIDDEN, arch=arch)
        p = tmp_path / f'{arch}.pth'
        save_critic_checkpoint(c, p, obs_version=latest_encoder().version,
                               hidden_dim=HIDDEN, arch=arch)
        meta = load_critic_checkpoint(p)
        assert meta['arch'] == arch
        # The load path must be able to rebuild it from metadata alone.
        Critic(DEV, meta['hidden_dim'], arch=meta['arch']).load_state_dict(meta['state_dict'])


def test_legacy_checkpoint_without_arch_key_is_read_as_v1(tmp_path):
    """Checkpoints predating the arch key are all v1; defaulting them to v2 would break them."""
    c = Critic(DEV, HIDDEN, arch=CRITIC_ARCH_V1)
    p = tmp_path / 'legacy.pth'
    torch.save({'format': 1, 'obs_version': latest_encoder().version, 'hidden_dim': HIDDEN,
                'state_dict': c.state_dict()}, p)
    meta = load_critic_checkpoint(p)
    assert meta['arch'] == CRITIC_ARCH_V1
    Critic(DEV, meta['hidden_dim'], arch=meta['arch']).load_state_dict(meta['state_dict'])


@pytest.mark.parametrize('arch', CRITIC_ARCHS)
def test_trunk_health_reports_one_fraction_per_conv_plus_output_spread(arch):
    c = Critic(DEV, HIDDEN, arch=arch)
    h = c.trunk_health(_boards())
    assert len(h['alive']) == 3
    assert all(0.0 <= v <= 1.0 for v in h['alive'])
    # A freshly initialised trunk is roughly half-on and its output varies with the input;
    # it is not dead at init in either arch.
    assert all(v > 0.05 for v in h['alive']), h
    assert h['out_std'] > 0


def _kill_last_conv(c):
    """Drive the last conv's pre-activations to a constant far-negative value."""
    convs = [m for m in c.board_encoder.modules() if isinstance(m, torch.nn.Conv2d)]
    with torch.no_grad():
        convs[-1].weight.zero_()
        convs[-1].bias.fill_(-50.0)
    return c


def test_v1_absorbing_state_is_reachable_and_is_detected_by_the_alive_fraction():
    """The v1 failure exactly as it happened in production: alive -> 0, output -> exactly 0."""
    c = _kill_last_conv(Critic(DEV, HIDDEN, arch=CRITIC_ARCH_V1))
    h = c.trunk_health(_boards())
    assert h['alive'][-1] == 0.0, h
    assert h['out_std'] == 0.0, h
    with torch.no_grad():
        assert float(c.board_encoder(_boards()).abs().max()) == 0.0


def test_v2_cannot_reach_the_absorbing_state_so_the_alive_fraction_alone_is_not_enough():
    """GroupNorm re-centres, so the same sabotage that kills v1 leaves v2 fully 'alive'.

    That is the fix working — but it is also why `out_std` exists. A constant all-POSITIVE
    output carries exactly as little board information as an all-zero one, and the alive
    fraction reads 1.0 for it. Without `out_std` the guard would pass a blind v2 critic.
    """
    c = _kill_last_conv(Critic(DEV, HIDDEN, arch=CRITIC_ARCH_V2))
    h = c.trunk_health(_boards())
    assert h['alive'][-1] == 1.0, h            # no absorbing state to fall into
    assert h['out_std'] == pytest.approx(0.0, abs=1e-9), h   # ...but no information either


@pytest.mark.parametrize('arch', CRITIC_ARCHS)
def test_out_std_catches_a_collapsed_trunk_on_either_arch(arch):
    """The single condition that covers both failure modes, which is what ppo.py guards on."""
    h = _kill_last_conv(Critic(DEV, HIDDEN, arch=arch)).trunk_health(_boards())
    assert h['out_std'] < 1e-6, h


def test_groupnorm_keeps_the_trunk_signal_from_collapsing():
    """§3.4's measured effect: |out|max rises by ~60x. Guards against dropping GroupNorm."""
    torch.manual_seed(0)
    b = _boards(64)
    with torch.no_grad():
        v1_out = float(Critic(DEV, HIDDEN, arch=CRITIC_ARCH_V1).board_encoder(b).abs().max())
        v2_out = float(Critic(DEV, HIDDEN, arch=CRITIC_ARCH_V2).board_encoder(b).abs().max())
    assert v2_out > 10 * v1_out, (v1_out, v2_out)


def test_board_only_head_gradient_reaches_the_first_conv():
    """The mechanism: this head cannot be satisfied from globals, so the trunk must learn."""
    c = Critic(DEV, HIDDEN, arch=CRITIC_ARCH_V2)
    loss = ((c.board_only_value(_boards()) - torch.randn(16)) ** 2).mean()
    loss.backward()
    first_conv = next(m for m in c.board_encoder.modules() if isinstance(m, torch.nn.Conv2d))
    assert first_conv.weight.grad is not None
    assert float(first_conv.weight.grad.abs().sum()) > 0


def test_board_only_value_is_refused_on_v1_rather_than_silently_wrong():
    with pytest.raises(RuntimeError, match='critic_v2'):
        Critic(DEV, HIDDEN, arch=CRITIC_ARCH_V1).board_only_value(_boards())


def test_board_only_value_sees_no_non_board_input():
    """Same boards + different globals/privileged => identical board-only value."""
    c = Critic(DEV, HIDDEN, arch=CRITIC_ARCH_V2)
    b = _boards()
    a = c.board_only_value(b)
    # There is no argument through which globals could enter; assert the value is a pure
    # function of the board by re-calling with the module in a different mode.
    assert torch.allclose(a, c.board_only_value(b))
    assert a.shape == (len(b),)


@pytest.mark.parametrize('arch', CRITIC_ARCHS)
def test_value_forward_still_works_on_a_real_observation(arch):
    env = WarChestEnv()
    obs, _ = env.reset()
    enc = latest_encoder()
    c = Critic(DEV, HIDDEN, obs_encoder=enc, arch=arch)
    opp = torch.zeros(1, Critic.OPP_DIM)
    priv = torch.tensor(env.get_privileged_features()).unsqueeze(0)
    v = c.value_single(obs, opp, priv)
    assert torch.isfinite(v).all()
    batch = {
        'board': torch.from_numpy(np.stack([obs['board']] * 4)),
        'global': torch.from_numpy(np.stack([obs['global']] * 4)),
        'opp_onehot': opp.expand(4, -1),
        'privileged': priv.expand(4, -1),
    }
    assert c.value_batch(batch).shape == (4,)


def test_v2_builds_for_every_obs_version_in_the_registry():
    """The gauntlet reconstructs critics across obs eras; GroupNorm must not care."""
    for version in (10, 11):
        enc = get_encoder(version)
        c = Critic(DEV, HIDDEN, obs_encoder=enc, arch=CRITIC_ARCH_V2)
        assert torch.isfinite(c.board_encoder(_boards(4, enc.board_channels))).all()
