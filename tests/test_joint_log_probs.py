"""Verify scatter-based _joint_log_probs is numerically identical to the loop-based original."""
import torch
import torch.nn.functional as F

from src.services.environment.warchest_env import ACTION_SPACE_SIZE, N_FACTORED_VERBS
from src.services.policy.policy import Policy


def _joint_log_probs_loop(flat_logits, verb_logits, mask, verb_index, group_mat):
    """Original loop-based reference — kept verbatim from the pre-scatter version."""
    NEG = -1e9
    B = flat_logits.shape[0]
    ml = flat_logits.masked_fill(~mask, NEG)
    g = verb_index.unsqueeze(0).expand(B, -1)
    gmax = torch.stack(
        [ml[:, group_mat[v]].max(dim=1).values for v in range(N_FACTORED_VERBS)],
        dim=1,
    )
    shifted = ml - gmax.gather(1, g)
    exp_shifted = shifted.exp() * mask.float()
    gsum = torch.stack(
        [exp_shifted[:, group_mat[v]].sum(dim=1) for v in range(N_FACTORED_VERBS)],
        dim=1,
    )
    within_logp = (shifted - gsum.clamp_min(1e-12).log().gather(1, g)).masked_fill(~mask, NEG)
    verb_mask = gsum > 0
    verb_logp = F.log_softmax(verb_logits.masked_fill(~verb_mask, NEG), dim=1)
    joint = verb_logp.gather(1, g) + within_logp
    return joint.masked_fill(~mask, NEG)


def test_joint_log_probs():
    torch.manual_seed(42)
    policy = Policy(device='cpu', hidden_dim=64)

    cases = [
        # (batch_size, n_legal, description)
        (1, 30, 'single obs, sparse mask'),
        (1, 200, 'single obs, dense mask'),
        (8, 50, 'batch, sparse mask'),
        (8, 500, 'batch, dense mask'),
        (1, 1, 'single legal action'),
        (1, ACTION_SPACE_SIZE, 'all actions legal'),
    ]

    for B, n_legal, desc in cases:
        mask = torch.zeros(B, ACTION_SPACE_SIZE, dtype=torch.bool)
        for b in range(B):
            ids = torch.randperm(ACTION_SPACE_SIZE)[:n_legal]
            mask[b, ids] = True

        flat_logits = torch.randn(B, ACTION_SPACE_SIZE)
        verb_logits = torch.randn(B, N_FACTORED_VERBS)

        ref = _joint_log_probs_loop(flat_logits, verb_logits, mask,
                                    policy._verb_index, policy._group_mat)
        new = policy._joint_log_probs(flat_logits, verb_logits, mask)

        # Compare only legal positions; NEG positions are -1e9 in both and
        # may differ by a tiny float rounding amount.
        any_legal = mask.any(dim=0)
        legal_diff = (ref[:, any_legal] - new[:, any_legal]).abs().max().item()
        assert legal_diff < 1e-4, f'{desc}: legal diff {legal_diff:.2e}'
        print(f'PASS  {desc:<40s}  legal_diff={legal_diff:.2e}')

    print('\nAll cases passed.')


def test_empty_verb_group():
    """Verb groups with no legal actions must not produce NaN or inf."""
    torch.manual_seed(7)
    policy = Policy(device='cpu', hidden_dim=64)

    # Only allow actions from a single verb type (V_MOVE = 0)
    from src.services.environment.warchest_env import VERB_OF_ACTION, V_MOVE
    import numpy as np
    move_ids = torch.from_numpy(np.where(VERB_OF_ACTION == V_MOVE)[0]).long()
    mask = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool)
    mask[0, move_ids[:3]] = True  # just 3 legal move actions

    flat_logits = torch.randn(1, ACTION_SPACE_SIZE)
    verb_logits = torch.randn(1, N_FACTORED_VERBS)

    ref = _joint_log_probs_loop(flat_logits, verb_logits, mask,
                                policy._verb_index, policy._group_mat)
    new = policy._joint_log_probs(flat_logits, verb_logits, mask)

    assert not torch.isnan(new).any(), 'NaN in output'
    assert not torch.isinf(new[mask]).any(), 'Inf in legal positions'

    legal_diff = (ref[mask] - new[mask]).abs().max().item()
    assert legal_diff < 1e-4, f'empty-verb-group diff {legal_diff:.2e}'
    print(f'PASS  empty verb group  legal_diff={legal_diff:.2e}')


if __name__ == '__main__':
    test_joint_log_probs()
    test_empty_verb_group()
