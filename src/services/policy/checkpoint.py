"""Policy checkpoint (de)serialization with obs-version + architecture metadata.

A bare `state_dict` is meaningless without knowing which obs encoder and network
architecture produced it (docs/next_steps.md Step 1 "the compatibility blocker").
Checkpoints are therefore saved as a metadata envelope so any era's policy can be
reconstructed and dropped into the round-robin gauntlet.

Envelope (torch.save of a dict):
    {'format': 1, 'arch': str, 'obs_version': int, 'hidden_dim': int,
     'state_dict': {param: cpu tensor}}

Legacy checkpoints (a bare state_dict saved before this existed) are still
loadable — they are assumed to be the current arch at the latest obs version,
which is what they were in practice when saved.
"""
import torch

from ..environment.obs_encoders import LATEST_VERSION

# Architecture id for the current factored-head Policy. Bump (and register a
# loader) when the network class changes; the gauntlet keys reconstruction on it.
CURRENT_ARCH = 'policy_factored_v1'


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
        return {
            'state_dict': obj['state_dict'],
            'obs_version': obj['obs_version'],
            'arch': obj.get('arch', CURRENT_ARCH),
            'hidden_dim': obj.get('hidden_dim', default_hidden_dim),
        }
    # Legacy: a bare state_dict (OrderedDict of param -> tensor) with no metadata.
    return {
        'state_dict': obj,
        'obs_version': LATEST_VERSION,
        'arch': CURRENT_ARCH,
        'hidden_dim': default_hidden_dim,
    }
