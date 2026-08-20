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

from src.services.expert_iteration import (
    ReplayWindow, SelfPlayDataset, _kl_to_reference, _sharpen_target, _split_by_game,
)


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


def test_kl_to_reference_is_zero_when_identical():
    joint = torch.tensor([[0.5, 0.3, 0.2]]).log()
    kl = _kl_to_reference(joint, joint)
    assert abs(float(kl)) < 1e-6


def test_kl_to_reference_is_positive_for_different_distributions():
    ref = torch.tensor([[0.5, 0.3, 0.2]]).log()
    new = torch.tensor([[0.2, 0.3, 0.5]]).log()
    kl = _kl_to_reference(ref, new)
    assert float(kl) > 0


def test_kl_to_reference_ignores_illegal_actions():
    ref = torch.tensor([[0.6, 0.4, 1e-30]]).log()
    ref[0, 2] = -1e9
    new_diff = ref.clone()
    new_diff[0, 2] = -50.0  # wildly different from ref's -1e9 there; exp(-1e9) == 0, so it shouldn't matter
    kl = _kl_to_reference(ref, new_diff)
    assert abs(float(kl)) < 1e-6


# --------------------------------------------------------------------------- #
# game_ids + the by-game held-out split (docs/IDEAS.md R.10.5c / R.10.8 item 3)
# --------------------------------------------------------------------------- #
def _labelled_dataset(*game_sizes):
    """A stacked dataset built through `add`/`label_last`, i.e. with real game ids."""
    ds = SelfPlayDataset()
    for n in game_sizes:
        for _ in range(n):
            ds.add(board=np.zeros((1,), dtype=np.float32),
                   global_feats=np.zeros((1,), dtype=np.float32),
                   mask=np.ones((1,), dtype=bool),
                   visit_target=np.ones((1,), dtype=np.float32),
                   opp_onehot=np.zeros((1,), dtype=np.float32),
                   privileged=np.zeros((1,), dtype=np.float32),
                   mover=1)
        ds.label_last(n, winner=1)
    return ds.stack()


def test_label_last_tags_each_game_with_its_own_id():
    ds = _labelled_dataset(3, 2)
    assert ds.game_ids.tolist() == [0, 0, 0, 1, 1]


def test_concat_rebases_game_ids_so_workers_do_not_collide():
    merged = SelfPlayDataset.concat([_labelled_dataset(2, 1), _labelled_dataset(3)])
    # Two games in the first part, one in the second — three distinct games, not two.
    assert merged.game_ids.tolist() == [0, 0, 1, 2, 2, 2]


def test_concat_tolerates_a_part_without_game_ids():
    legacy = _fake_dataset(4)  # arrays set directly, game_ids left None
    merged = SelfPlayDataset.concat([legacy, _labelled_dataset(2)])
    assert len(merged.game_ids) == 6
    assert merged.game_ids.tolist()[-2:] == [4, 4]  # one game, after 4 legacy ids


def test_save_load_round_trips_game_ids(tmp_path):
    path = str(tmp_path / 'ds.npz')
    _labelled_dataset(2, 3).save(path)
    assert SelfPlayDataset.load(path).game_ids.tolist() == [0, 0, 1, 1, 1]


def test_load_falls_back_to_per_sample_ids_for_a_pre_game_ids_dataset(tmp_path):
    path = str(tmp_path / 'legacy.npz')
    ds = _labelled_dataset(3)
    np.savez_compressed(path, boards=ds.boards, globals=ds.globals, masks=ds.masks,
                        visit_targets=ds.visit_targets, opp_onehots=ds.opp_onehots,
                        privileged=ds.privileged, z=ds.z)
    assert SelfPlayDataset.load(path).game_ids.tolist() == [0, 1, 2]


def test_split_by_game_never_puts_one_game_on_both_sides():
    game_ids = np.repeat(np.arange(10), 8)  # 10 games x 8 samples
    for _ in range(20):
        train_idx, val_idx = _split_by_game(game_ids, 0.2)
        assert not (set(game_ids[train_idx]) & set(game_ids[val_idx]))
        assert len(train_idx) + len(val_idx) == len(game_ids)
        # Whole games only, so the split lands on multiples of the game length.
        assert len(val_idx) % 8 == 0


def test_split_by_game_holds_out_roughly_val_frac():
    game_ids = np.repeat(np.arange(20), 5)
    _, val_idx = _split_by_game(game_ids, 0.25)
    assert len(val_idx) == 25  # 5 of 20 games


def test_split_by_game_never_holds_out_every_game():
    game_ids = np.repeat(np.arange(2), 4)
    train_idx, val_idx = _split_by_game(game_ids, 0.9)
    assert len(train_idx) == 4 and len(val_idx) == 4


def test_split_by_game_falls_back_to_samples_below_two_games():
    game_ids = np.zeros(10, dtype=np.int64)
    train_idx, val_idx = _split_by_game(game_ids, 0.2)
    assert len(val_idx) == 2 and len(train_idx) == 8


def test_split_by_game_with_no_val_frac_trains_on_everything():
    train_idx, val_idx = _split_by_game(np.repeat(np.arange(4), 3), 0.0)
    assert len(train_idx) == 12 and len(val_idx) == 0


# --------------------------------------------------------------------------- #
# Apprentice-driven state distribution (docs/IDEAS.md R.10.12)
# --------------------------------------------------------------------------- #
class _ScriptedBot:
    """Stands in for `PuctBot` without loading a checkpoint or running a search.

    Its 'search' spreads visits over the last two legal actions while its 'policy prior'
    is a point mass on the first, so the two distributions are distinguishable in the
    recorded data: a prior-sampled ply is one where the played move came from a one-entry
    distribution, while the target row always has the search's two entries.
    """

    def act(self, env):
        legal = env.get_possible_actions()
        self.last_stats = {
            'visit_counts': {legal[-1]: 0.75, legal[0]: 0.25},
            'policy_priors': {legal[0]: 1.0},
            'policy_argmax': legal[0],
        }
        return legal[-1]


def _play_scripted(apprentice_frac, max_turns=10):
    from src.services.environment.warchest_env import WarChestEnv
    from src.services.expert_iteration import play_selfplay_game

    env = WarChestEnv(save_game_history=False)
    ds = SelfPlayDataset()
    stats = play_selfplay_game(_ScriptedBot(), env, ds, temperature=0.0, temp_moves=0,
                               max_turns=max_turns, apprentice_frac=apprentice_frac)
    return ds.stack(), stats


def test_apprentice_frac_zero_keeps_the_search_as_the_actor():
    ds, stats = _play_scripted(0.0)
    assert stats['n_samples'] > 0
    assert stats['apprentice_sum'] == 0


def test_apprentice_frac_one_hands_every_ply_to_the_raw_policy():
    ds, stats = _play_scripted(1.0)
    assert stats['n_samples'] > 0
    assert stats['apprentice_sum'] == stats['n_samples']


def test_apprentice_play_still_records_the_search_target_not_the_prior():
    # The whole point: the states come from the student, the labels from the teacher. The
    # prior is a point mass, the search target has two entries — so two non-zeros per row
    # proves the recorded target is the search's even on prior-played plies.
    ds, stats = _play_scripted(1.0)
    nonzero = (ds.visit_targets > 0).sum(axis=1)
    assert stats['apprentice_sum'] == stats['n_samples']
    assert set(nonzero.tolist()) <= {1, 2}  # 1 only where the position had a single legal move
    assert (nonzero == 2).sum() > 0


def test_apprentice_plies_are_labelled_with_the_game_outcome_like_any_other():
    ds, stats = _play_scripted(1.0)
    assert len(ds.z) == stats['n_samples']
    assert set(np.unique(ds.z).tolist()) <= {-1.0, 1.0}
    assert ds.game_ids.tolist() == [0] * stats['n_samples']


# --------------------------------------------------------------------------- #
# Early stopping / best-epoch restore in `distill` (R.10.5c, R.10.8 item 3)
# --------------------------------------------------------------------------- #
class _StubPolicy(torch.nn.Module):
    """The one method `distill` calls on a policy, over a linear head."""

    def __init__(self, n_actions, n_global):
        super().__init__()
        self.lin = torch.nn.Linear(n_global, n_actions)

    def joint_log_probs_batch(self, batch):
        logits = self.lin(batch['global']).masked_fill(~batch['mask'], -1e9)
        return torch.log_softmax(logits, dim=1)


class _StubCritic(torch.nn.Module):
    def __init__(self, n_global):
        super().__init__()
        self.lin = torch.nn.Linear(n_global, 1)

    def value_batch(self, batch):
        return self.lin(batch['global'])


def _stub_training_data(n_games=8, per_game=8, n_actions=4, n_global=3, seed=0):
    """Games whose targets are pure noise, so nothing generalises and the held-out loss
    turns upward almost immediately — the regime that makes early stopping observable.
    """
    rng = np.random.default_rng(seed)
    ds = SelfPlayDataset()
    for _ in range(n_games):
        for _ in range(per_game):
            tgt = np.zeros(n_actions, dtype=np.float32)
            tgt[rng.integers(n_actions)] = 1.0
            ds.add(board=np.zeros((1,), dtype=np.float32),
                   global_feats=rng.normal(size=n_global).astype(np.float32),
                   mask=np.ones(n_actions, dtype=bool),
                   visit_target=tgt,
                   opp_onehot=np.zeros(3, dtype=np.float32),
                   privileged=np.zeros(1, dtype=np.float32),
                   mover=1)
        ds.label_last(per_game, winner=1)
    return ds.stack()


def _distill_stub(**kw):
    from src.services.expert_iteration import distill

    torch.manual_seed(0)
    np.random.seed(0)
    ds = _stub_training_data()
    return distill(ds, _StubPolicy(4, 3), _StubCritic(3), minibatch_size=16,
                   lr_policy=0.2, lr_critic=0.2, val_frac=0.25, log_every=0, **kw)


def test_distill_reports_a_held_out_split_taken_by_game():
    res = _distill_stub(epochs=1)
    assert res['n_val_games'] == 2  # 25 % of 8 games
    assert res['n_val'] == 16  # whole games, 8 samples each


def test_distill_stops_early_when_the_held_out_loss_stops_improving():
    res = _distill_stub(epochs=10, patience=1)
    assert res['epochs_run'] < 10
    assert 1 <= res['best_epoch'] <= res['epochs_run']


def test_distill_returns_the_best_epoch_not_the_last():
    res = _distill_stub(epochs=10, patience=1)
    best_seen = min(h['val_ce'] for h in res['history'])
    assert res['val']['ce'] <= best_seen + 1e-6
    assert res['val']['ce'] < res['history'][-1]['val_ce']  # the last epoch was worse


def test_distill_without_early_stopping_runs_every_epoch():
    res = _distill_stub(epochs=3, early_stop=False)
    assert res['epochs_run'] == 3
    assert res['best_epoch'] is None
    assert 'val_ce' not in res['history'][0]


# --------------------------------------------------------------------------- #
# Disagreement weighting (the 5.6 % of samples that carry a correction)
# --------------------------------------------------------------------------- #
def test_minibatches_carry_the_sample_index_so_weights_can_be_attached():
    ds = _labelled_dataset(4)
    seen = []
    for batch in ds.iter_minibatches(2, 'cpu', shuffle=True):
        seen.extend(batch['index'].tolist())
    assert sorted(seen) == [0, 1, 2, 3]


def test_disagreement_mask_flags_exactly_the_samples_the_policy_gets_wrong():
    from src.services.expert_iteration import _disagreement_mask

    torch.manual_seed(0)
    ds = _stub_training_data(n_games=3, per_game=4)
    policy = _StubPolicy(4, 3)
    idx = np.arange(len(ds.z))
    flags = _disagreement_mask(ds, policy, idx, 'cpu')

    expected = []
    with torch.inference_mode():
        for i in idx:
            batch = {'board': torch.from_numpy(ds.boards[i:i + 1]),
                     'global': torch.from_numpy(ds.globals[i:i + 1]),
                     'mask': torch.from_numpy(ds.masks[i:i + 1])}
            am = int(policy.joint_log_probs_batch(batch).argmax(dim=1).item())
            expected.append(am != int(ds.visit_targets[i].argmax()))
    assert flags.tolist() == expected


def test_disagree_weight_one_is_bit_identical_to_no_weighting():
    def run(weight):
        torch.manual_seed(0)
        np.random.seed(0)
        ds = _stub_training_data()
        pol = _StubPolicy(4, 3)
        from src.services.expert_iteration import distill
        distill(ds, pol, _StubCritic(3), epochs=2, minibatch_size=16, lr_policy=0.1,
                val_frac=0.25, log_every=0, train_critic=False, visit_temp=1.0,
                disagree_weight=weight)
        return torch.cat([p.flatten() for p in pol.parameters()])

    torch.testing.assert_close(run(1.0), run(1.0))
    assert not torch.allclose(run(1.0), run(6.0))


def test_disagree_weight_moves_the_policy_further_on_the_samples_it_up_weights():
    from src.services.expert_iteration import _disagreement_mask, distill

    def run(weight):
        torch.manual_seed(0)
        np.random.seed(0)
        ds = _stub_training_data()
        pol = _StubPolicy(4, 3)
        idx = np.arange(len(ds.z))
        before = _disagreement_mask(ds, pol, idx, 'cpu')
        distill(ds, pol, _StubCritic(3), epochs=6, minibatch_size=16, lr_policy=0.1,
                val_frac=0.25, log_every=0, train_critic=False, visit_temp=1.0,
                early_stop=False, disagree_weight=weight)
        after = _disagreement_mask(ds, pol, idx, 'cpu')
        # How many of the originally-disagreeing samples the policy now matches.
        return int((before & ~after).sum())

    assert run(8.0) > run(1.0)
