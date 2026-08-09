"""The eval-phase reference opponent: a frozen checkpoint the run is scored against.

`wr_vs_greedy_eval` / `wr_vs_random_eval` both saturate long before the policy stops
improving, so a run cannot tell from them whether generation N+1 beat generation N.
The reference match answers that directly by playing the current policy against a saved
checkpoint every eval.

The property that makes the number mean anything is *when* the checkpoint is chosen: once
intermediate checkpoints are written to `data/` during a run, resolving "the newest policy"
at eval time would re-baseline the run onto its own recent output and pin the score near
0.5 by construction. Pinned here:
  1. `latest_policy_checkpoint` picks by mtime (intermediate names need not sort);
  2. the trainer resolves and loads it ONCE, at construction, and `_maybe_eval` never
     touches the filesystem again — this is the whole point of the change;
  3. an unloadable checkpoint degrades to a warning, since losing a measurement is not a
     reason to lose a training run.
"""
import os

import torch

from src.app.ppo import (
    ReferenceOpponent, latest_policy_checkpoint, POLICY_CKPT_GLOB,
)
from src.services.environment.obs_encoders import get_encoder, latest_encoder
from src.services.environment.warchest_env import WarChestEnv
from src.services.policy.checkpoint import save_policy_checkpoint, save_critic_checkpoint
from src.services.policy.policy import Policy, Critic


def _save_policy(path, hidden_dim=16):
    enc = latest_encoder()
    policy = Policy(device='cpu', hidden_dim=hidden_dim, obs_encoder=enc)
    save_policy_checkpoint(policy, path, obs_version=enc.version, hidden_dim=hidden_dim)
    return policy


def test_latest_policy_checkpoint_picks_newest_by_mtime(tmp_path):
    # Deliberately reversed name order vs. save order: the run-final checkpoints happen to
    # sort by their timestamp suffix, but an intermediate one saved mid-run need not.
    old = tmp_path / 'warchest_ppo_zzz.pth'
    new = tmp_path / 'warchest_ppo_aaa.pth'
    _save_policy(str(old))
    _save_policy(str(new))
    os.utime(old, (1_000, 1_000))
    os.utime(new, (2_000, 2_000))

    assert latest_policy_checkpoint(str(tmp_path / 'warchest_ppo_*.pth')) == str(new)


def test_latest_policy_checkpoint_is_none_when_empty(tmp_path):
    assert latest_policy_checkpoint(str(tmp_path / 'warchest_ppo_*.pth')) is None


def test_reference_opponent_uses_env_obs_when_versions_match():
    """No encoder => the env's already-computed obs is used as-is.

    Re-encoding it would be pure waste in a loop that runs `eval_episodes` games, so the
    trainer only attaches an encoder on a version mismatch.
    """
    env = WarChestEnv(save_game_history=False, debug_mode=False)
    obs, _ = env.reset()
    enc = latest_encoder()
    policy = Policy(device='cpu', hidden_dim=16, obs_encoder=enc)
    ref = ReferenceOpponent(policy, path='data/warchest_ppo_fake.pth',
                            obs_version=enc.version, encoder=None, env=None)

    assert ref.name == 'warchest_ppo_fake'
    action, _, _ = ref.act(obs)
    assert obs['valid_action_mask'][action]


def test_reference_opponent_reencodes_on_version_mismatch():
    """An older-obs checkpoint is sized to its own encoder, so the env obs is not a valid
    input for it — the reference re-derives one, exactly as the gauntlet's PolicyAgent does.
    """
    env = WarChestEnv(save_game_history=False, debug_mode=False)
    obs, _ = env.reset()
    own_enc = get_encoder(latest_encoder().version)
    policy = Policy(device='cpu', hidden_dim=16, obs_encoder=own_enc)
    calls = []
    real_encode = own_enc.encode
    own_enc.encode = lambda view: (calls.append(view), real_encode(view))[1]

    ref = ReferenceOpponent(policy, path='p.pth', obs_version=own_enc.version,
                            encoder=own_enc, env=env)
    action, _, _ = ref.act(obs)

    assert calls == [env], 'the reference must re-encode the env, not reuse the passed obs'
    assert obs['valid_action_mask'][action]


def _trainer(reference_path, **hp_over):
    """A minimal PPOTrainer. Kept tiny (hidden_dim 16, 1 episode) — these tests are about
    checkpoint plumbing, not learning."""
    from src.app.ppo import PPOTrainer

    env = WarChestEnv(save_game_history=False, debug_mode=False)
    hp = {
        'n_batches': 1, 'collect_episodes': 1, 'max_t': 40, 'gamma': 0.99, 'lam': 0.9,
        'ppo_epochs': 1, 'ppo_eps': 0.2, 'entropy_coeff': 0.02,
        'holding_reward_rate': env.default_holding_reward_rate(), 'minibatch_size': 64,
        'n_workers': 1, 'lr_actor': 3e-4, 'lr_critic': 3e-4,
        'hidden_dim': 16, 'critic_hidden_dim': 16, 'print_every': 1,
        'eval_every': 1, 'eval_episodes': 1,
        'p_random_initial': 1.0, 'p_greedy_initial': 0.0, 'p_pool_initial': 0.0,
        'p_lookahead_critic_initial': 0.0,
        'p_random_finetune': 1.0, 'p_greedy_finetune': 0.0, 'p_pool_finetune': 0.0,
        'p_lookahead_critic_finetune': 0.0,
        'lookahead_critic_time_budget': 0.1, 'wr_greedy_finetune_threshold': 2.0,
        'critic_arch': 'critic_v3', 'adv_norm': 'per_opponent', 'trunk_health_every': 0,
        'reference_policy_path': reference_path,
        **hp_over,
    }

    def policy_constructor():
        return Policy(device='cpu', hidden_dim=hp['hidden_dim'])

    policy = policy_constructor()
    critic = Critic(device='cpu', hidden_dim=hp['critic_hidden_dim'], arch=hp['critic_arch'])
    return PPOTrainer(env, policy, critic,
                      torch.optim.Adam(policy.parameters(), lr=3e-4),
                      torch.optim.Adam(critic.parameters(), lr=3e-4),
                      policy_constructor, hp, 'cpu')


def test_reference_is_loaded_once_and_eval_never_reresolves(tmp_path, monkeypatch):
    """The regression this change exists to prevent: a run that saves intermediate
    checkpoints must not start evaluating itself against its own output."""
    import src.app.ppo as ppo_mod

    path = tmp_path / 'warchest_ppo_ref.pth'
    _save_policy(str(path))
    trainer = _trainer(str(path))
    assert trainer._eval_reference is not None
    loaded = trainer._eval_reference.policy

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError('eval must not re-resolve the reference checkpoint')

    monkeypatch.setattr(ppo_mod, 'latest_policy_checkpoint', _must_not_be_called)
    monkeypatch.setattr(ppo_mod, 'load_policy_checkpoint', _must_not_be_called)

    trainer._maybe_eval(1)
    assert trainer._eval_reference.policy is loaded


def test_no_reference_when_path_is_none():
    trainer = _trainer(None)
    assert trainer._eval_reference is None
    trainer._maybe_eval(1)  # must not raise


def test_unloadable_reference_degrades_to_warning(tmp_path):
    """A critic checkpoint pointed at --reference-policy is the realistic mistake, and it
    must cost the measurement, not the run."""
    bad = tmp_path / 'warchest_critic_x.pth'
    critic = Critic(device='cpu', hidden_dim=16, arch='critic_v3')
    save_critic_checkpoint(critic, str(bad), obs_version=latest_encoder().version,
                           hidden_dim=16, arch='critic_v3')

    trainer = _trainer(str(bad))
    assert trainer._eval_reference is None
    trainer._maybe_eval(1)


def test_reference_glob_matches_the_saved_checkpoint_name():
    """`POLICY_CKPT_GLOB` has to match what the trainer's own save path produces, or the
    default reference silently never resolves."""
    import fnmatch
    assert fnmatch.fnmatch('data/warchest_ppo_20260808-0607.pth', POLICY_CKPT_GLOB)
