"""`_sharpen_target` and `ReplayWindow` (src/services/expert_iteration.py).

`_sharpen_target` is the CE-target sharpening exponent added after
`data/exit/round0.npz` (2026-08-18) showed the raw visit distribution is less decisive
than the policy already distilling toward it (mean entropy 0.720 nats vs a pre-distill
policy entropy of 0.469), and unsharpened distillation dragged the policy's own entropy
up to 0.875 in one round (docs/IDEAS.md R.10.9). See that section for the full
measurement.

`ReplayWindow` is the sliding window over rounds' self-play data added to close
docs/IDEAS.md R.10.5a / R.10.8 item 2: each round used to distil on its own dataset
only, ~25k narrow, single-network samples with nothing from earlier rounds to anchor
it.
"""
import numpy as np
import torch

from src.services.expert_iteration import ReplayWindow, SelfPlayDataset, _sharpen_target


def _fake_dataset(n):
    ds = SelfPlayDataset()
    ds.boards = np.zeros((n, 1), dtype=np.float32)
    ds.globals = np.zeros((n, 1), dtype=np.float32)
    ds.masks = np.ones((n, 1), dtype=bool)
    ds.visit_targets = np.ones((n, 1), dtype=np.float32)
    ds.opp_onehots = np.zeros((n, 1), dtype=np.float32)
    ds.privileged = np.zeros((n, 1), dtype=np.float32)
    ds.z = np.zeros(n, dtype=np.float32)
    return ds


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


def test_replay_window_single_round_is_a_no_op():
    w = ReplayWindow(max_rounds=3)
    ds = _fake_dataset(5)
    w.push(ds)
    assert len(w) == 1
    assert w.concat() is ds


def test_replay_window_concatenates_within_the_window():
    w = ReplayWindow(max_rounds=2)
    w.push(_fake_dataset(3))
    w.push(_fake_dataset(4))
    assert len(w) == 2
    assert len(w.concat()) == 7


def test_replay_window_drops_the_oldest_round_past_its_size():
    w = ReplayWindow(max_rounds=2)
    w.push(_fake_dataset(3))
    w.push(_fake_dataset(4))
    w.push(_fake_dataset(5))
    assert len(w) == 2
    assert len(w.concat()) == 9  # the first round's 3 samples were dropped


def test_replay_window_clamps_max_rounds_to_at_least_one():
    w = ReplayWindow(max_rounds=0)
    assert w.max_rounds == 1
