"""Policy/critic checkpoint (de)serialization with obs-version + architecture metadata.

A bare `state_dict` is meaningless without knowing which obs encoder and network
architecture produced it (docs/next_steps.md Step 1 "the compatibility blocker").
Checkpoints are therefore saved as a metadata envelope so any era's policy (or
critic) can be reconstructed and dropped into the round-robin gauntlet.

Envelope (torch.save of a dict):
    {'format': 1, 'arch': str, 'obs_version': int, 'hidden_dim': int,
     'state_dict': {param: cpu tensor}}

Legacy checkpoints (a bare state_dict saved before this existed) are still
loadable — they are assumed to be the current arch at the latest obs version,
which is what they were in practice when saved.

The critic was historically never persisted (training-only, discarded after each
run) — `save_critic_checkpoint`/`load_critic_checkpoint` mirror the policy pair
exactly, saved as a separate file alongside the policy checkpoint rather than a
new key on the existing envelope, so `load_policy_checkpoint` and every existing
caller (gauntlet.py) are untouched.

Critic envelopes additionally carry optional `return_mean`/`return_std` floats
— the `ReturnNormalizer` EMA (ppo.py) in effect when the checkpoint was saved,
letting a consumer outside the training loop (`LookaheadCriticBot`) recover the
critic's real reward-scale value exactly instead of approximating it. Absent
(`None`) on any checkpoint saved before this pair of fields existed.
"""
import torch

from ..environment.obs_encoders import LATEST_VERSION

# Architecture id for the current factored-head Policy. Bump (and register a
# loader) when the network class changes; the gauntlet keys reconstruction on it.
CURRENT_ARCH = 'policy_factored_v1'
# Architecture id for the current Critic (docs/rewards.md's widened critic-only trunk).
CURRENT_CRITIC_ARCH = 'critic_v1'


def save_policy_checkpoint(policy, path, *, obs_version, hidden_dim, arch=CURRENT_ARCH):
    """Save `policy`'s weights plus the metadata needed to rebuild it later."""
    payload = {
        'format': 1,
        'arch': arch,
        'obs_version': obs_version,
        'hidden_dim': hidden_dim,
        'state_dict': {k: v.detach().cpu() for k, v in policy.state_dict().items()},
    }
    torch.save(payload, path)


def load_policy_checkpoint(path, map_location='cpu', *, default_hidden_dim=64):
    """Return {'state_dict', 'obs_version', 'arch', 'hidden_dim'} for any checkpoint.

    Accepts both the metadata envelope and a legacy bare state_dict.
    """
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(obj, dict) and 'state_dict' in obj and 'obs_version' in obj:
        arch = obj.get('arch', CURRENT_ARCH)
        if arch == CURRENT_CRITIC_ARCH:
            raise ValueError(
                f"{path!r} is a CRITIC checkpoint (arch={arch!r}), not a policy checkpoint "
                f"— it was likely copied from a data/warchest_critic_*.pth file by mistake. "
                f"Use the matching data/warchest_ppo_*.pth from the same training run instead."
            )
        return {
            'state_dict': obj['state_dict'],
            'obs_version': obj['obs_version'],
            'arch': arch,
            'hidden_dim': obj.get('hidden_dim', default_hidden_dim),
        }
    # Legacy: a bare state_dict (OrderedDict of param -> tensor) with no metadata.
    return {
        'state_dict': obj,
        'obs_version': LATEST_VERSION,
        'arch': CURRENT_ARCH,
        'hidden_dim': default_hidden_dim,
    }


def save_critic_checkpoint(critic, path, *, obs_version, hidden_dim, arch=CURRENT_CRITIC_ARCH,
                            return_mean=None, return_std=None):
    """Save `critic`'s weights plus the metadata needed to rebuild it later.

    `return_mean`/`return_std`: the `ReturnNormalizer` EMA in effect at save time
    (ppo.py) — the critic is trained to predict *normalised* returns
    (`(return - mean) / std`), and that normalisation is undone
    (`value * std + mean`) everywhere the critic's output is treated as a real
    value during training (rollout bootstrapping, GAE). Without these, nothing
    records what that undoing needs to be: a consumer outside the training loop
    (`LookaheadCriticBot`) has no way to recover the network's real-reward
    scale and has to approximate it (see `_calibrate_value_scale`'s docstring).
    Optional and omitted by default (`None`) so existing call sites don't need
    to change; a checkpoint saved without them just has no `return_mean`/
    `return_std` keys, same as every checkpoint saved before this pair existed.
    """
    payload = {
        'format': 1,
        'arch': arch,
        'obs_version': obs_version,
        'hidden_dim': hidden_dim,
        'state_dict': {k: v.detach().cpu() for k, v in critic.state_dict().items()},
    }
    if return_mean is not None and return_std is not None:
        payload['return_mean'] = float(return_mean)
        payload['return_std'] = float(return_std)
    torch.save(payload, path)


def load_critic_checkpoint(path, map_location='cpu', *, default_hidden_dim=64):
    """Return {'state_dict', 'obs_version', 'arch', 'hidden_dim', 'return_mean',
    'return_std'} for any critic checkpoint saved by `save_critic_checkpoint`.
    No legacy format exists for the critic (it was never persisted before this
    pair was added). `return_mean`/`return_std` are `None` for checkpoints
    saved before that pair of fields existed — callers must handle that case,
    there's no way to recover them after the fact (see `save_critic_checkpoint`).
    """
    obj = torch.load(path, map_location=map_location, weights_only=False)
    arch = obj.get('arch', CURRENT_CRITIC_ARCH)
    if arch == CURRENT_ARCH:
        # The #1 real-world cause: someone copied a data/warchest_ppo_*.pth (policy)
        # over the critic path by mistake. Catching it here, keyed on the saved
        # `arch` metadata, turns a cryptic "Missing/Unexpected key(s)" state_dict
        # error into an actionable one.
        raise ValueError(
            f"{path!r} is a POLICY checkpoint (arch={arch!r}), not a critic checkpoint "
            f"— it was likely copied from a data/warchest_ppo_*.pth file by mistake. "
            f"Copy the matching data/warchest_critic_*.pth from the same training run "
            f"to this path instead."
        )
    return {
        'state_dict': obj['state_dict'],
        'obs_version': obj['obs_version'],
        'arch': arch,
        'hidden_dim': obj.get('hidden_dim', default_hidden_dim),
        'return_mean': obj.get('return_mean'),
        'return_std': obj.get('return_std'),
    }
