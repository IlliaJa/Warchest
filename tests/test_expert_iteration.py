"""`_sharpen_target` (src/services/expert_iteration.py) — the CE-target sharpening
exponent added after `data/exit/round0.npz` (2026-08-18) showed the raw visit
distribution is less decisive than the policy already distilling toward it (mean
entropy 0.720 nats vs a pre-distill policy entropy of 0.469), and unsharpened
distillation dragged the policy's own entropy up to 0.875 in one round (docs/IDEAS.md
R.10.9). See that section for the full measurement.
"""
import torch

from src.services.expert_iteration import _sharpen_target


def _target(*probs):
    return torch.tensor([probs], dtype=torch.float32)


def test_visit_temp_1_is_a_no_op():
    t = _target(0.5, 0.3, 0.2, 0.0)
    torch.testing.assert_close(_sharpen_target(t, 1.0), t)


def test_visit_temp_below_1_sharpens_and_stays_normalised():
    t = _target(0.5, 0.3, 0.2, 0.0)
    sharp = _sharpen_target(t, 0.5)
    torch.testing.assert_close(sharp.sum(dim=1), torch.tensor([1.0]))

    def entropy(p):
        p = p.clamp_min(1e-12)
        return float(-(p * p.log()).sum())

    assert entropy(sharp) < entropy(t)


def test_visit_temp_preserves_zeros_and_argmax():
    t = _target(0.5, 0.3, 0.2, 0.0)
    sharp = _sharpen_target(t, 0.3)
    assert sharp[0, 3].item() == 0.0
    assert sharp.argmax(dim=1).item() == t.argmax(dim=1).item()


def test_visit_temp_above_1_flattens():
    t = _target(0.7, 0.2, 0.1)
    flat = _sharpen_target(t, 2.0)

    def entropy(p):
        p = p.clamp_min(1e-12)
        return float(-(p * p.log()).sum())

    assert entropy(flat) > entropy(t)
