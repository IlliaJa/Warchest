"""The third opponent phase and the `puct_live` opponent it exists for (2026-08-20).

`lookahead_critic` and a *frozen* puct are both opponents the policy eventually overtakes —
the last run ended at `wr_lookahead` 0.71 and ~0.60 against frozen puct — and an opponent you
beat contributes near-noise to the gradient once advantages are centred per opponent
(docs/IDEAS.md B8) while still costing full rollout time. `finetune2` hands their share to a
search whose priors are the *current* policy's newest snapshot, which cannot be overtaken.

Two things these tests pin, both of which are easy to get wrong:
  * the promotion trigger reads a rolling mean over ~100 training episodes, so it must hold
    over several evals and must never fall back (a flip-flopping opponent distribution is
    worse than either phase);
  * `puct_live` must actually re-point at the newest snapshot, and must degrade to the
    frozen behaviour when there are none yet.
"""
from collections import deque
from types import SimpleNamespace

import numpy as np

from src.services.environment.rollout_core import (
    OPP_GROUP_IDX, OPP_ONEHOT_SLOT, OPP_TYPE_IDX, _SEARCH_OPP_TYPES, opp_group_id,
)
from src.services.opponent_pool import OpponentPool


# --------------------------------------------------------------------------- #
# Registration: B8's caveat is that a new opponent kind silently lands in the
# warned fallback advantage bucket, or worse, widens the critic's one-hot.
# --------------------------------------------------------------------------- #
def test_puct_live_shares_pucts_critic_onehot_slot():
    # Widening OPP_TYPE_IDX would break every v1/v2 critic checkpoint, so the new kind must
    # reuse an existing slot — the same one the frozen search bots use.
    assert 'puct_live' not in OPP_TYPE_IDX
    assert OPP_ONEHOT_SLOT['puct_live'] == OPP_ONEHOT_SLOT['puct'] == OPP_TYPE_IDX['pool']


def test_puct_live_has_its_own_advantage_group():
    assert OPP_GROUP_IDX['puct_live'] != OPP_GROUP_IDX['puct']
    assert opp_group_id('puct_live') == OPP_GROUP_IDX['puct_live']


def test_puct_live_is_an_env_taking_search_opponent():
    assert 'puct_live' in _SEARCH_OPP_TYPES


# --------------------------------------------------------------------------- #
# The pool
# --------------------------------------------------------------------------- #
def _pool(**kw):
    base = dict(p_random=0.0, p_greedy=0.0, p_pool=0.0)
    base.update(kw)
    return OpponentPool(max_size=4, snapshot_every=1, **base)


def test_weights_round_trip_through_set_weights():
    pool = _pool()
    pool.set_weights(p_random=0.0, p_greedy=0.0, p_pool=0.5, p_puct_live=0.5)
    assert pool.weights['p_puct_live'] == 0.5
    assert pool.weights['p_pool'] == 0.5


def test_a_kind_with_no_weight_is_never_sampled():
    pool = _pool(p_pool=1.0, p_puct_live=0.0)
    pool.append_snapshot({'w': np.zeros(1)})
    stub = SimpleNamespace(load_state_dict=lambda *_: None, eval=lambda: None)
    stub.to = lambda *_: stub
    kinds = {pool.sample(lambda: stub, 'cpu')[1] for _ in range(30)}
    assert kinds == {'pool'}


def test_live_puct_reloads_only_when_a_newer_snapshot_arrived():
    """The reload is keyed on the pool's monotonic append counter, so an episode does not
    pay a `load_state_dict` per sample — the same trick the `pool` opponent uses.
    """
    pool = _pool(p_puct_live=1.0)
    loads = []
    pool._puct_live_bot = SimpleNamespace(
        _policy=SimpleNamespace(load_state_dict=lambda sd: loads.append(sd),
                               eval=lambda: None))

    pool.append_snapshot({'gen': 1})
    pool._get_puct_live_bot()
    pool._get_puct_live_bot()
    assert loads == [{'gen': 1}]  # second call is a no-op

    pool.append_snapshot({'gen': 2})
    pool._get_puct_live_bot()
    assert loads == [{'gen': 1}, {'gen': 2}]


def test_live_puct_falls_back_to_the_checkpoint_before_any_snapshot_exists():
    pool = _pool(p_puct_live=1.0)
    loads = []
    pool._puct_live_bot = SimpleNamespace(
        _policy=SimpleNamespace(load_state_dict=lambda sd: loads.append(sd),
                               eval=lambda: None))
    pool._get_puct_live_bot()
    assert loads == []  # nothing to load: it plays the weights it was built with


def test_the_two_puct_variants_are_configured_independently():
    pool = _pool(puct_time_budget=0.1, puct_max_simulations=None,
                 puct_live_time_budget=1.0, puct_live_max_simulations=100)
    assert (pool._puct_time_budget, pool._puct_max_simulations) == (0.1, None)
    assert (pool._puct_live_time_budget, pool._puct_live_max_simulations) == (1.0, 100)


def test_both_variants_default_to_the_measured_strong_configuration():
    # blind + forced playouts measured 0.733 against the raw policy where the cheating,
    # quota-less default measured 0.567 (docs/IDEAS.md R.10.13/R.10.14).
    pool = _pool()
    assert pool._puct_blind is True
    assert pool._puct_forced_playouts_k == 2.0


# --------------------------------------------------------------------------- #
# The phase machine, exercised on a stub trainer (no nets, no rollouts)
# --------------------------------------------------------------------------- #
class _StubPool:
    def __init__(self):
        self.applied = []

    def set_weights(self, **kw):
        self.applied.append(kw)


def _trainer(wr_lookahead, *, confirm=3, threshold=0.60, min_episodes=50, min_batch=0):
    from src.app.ppo import PPOTrainer

    t = SimpleNamespace(
        _pool=_StubPool(),
        _wr_finetune_threshold=0.75,
        _wr_finetune2_threshold=threshold,
        _finetune2_confirm_evals=confirm,
        _finetune2_min_batch=min_batch,
        _finetune2_min_episodes=min_episodes,
        _in_finetune2=False,
        _finetune2_streak=0,
        _wr_vs_lookahead=deque(wr_lookahead, maxlen=100),
        _opp_weights_initial={'tag': 'initial'},
        _opp_weights_finetune={'tag': 'finetune'},
        _opp_weights_finetune2={'tag': 'finetune2'},
    )
    t._apply_opponent_phase = PPOTrainer._apply_opponent_phase.__get__(t, SimpleNamespace)
    return t


def test_below_the_greedy_threshold_the_phase_is_initial():
    t = _trainer([1] * 100)
    assert t._apply_opponent_phase(0.5, batch_num=100) == 'initial'
    assert t._pool.applied[-1] == {'tag': 'initial'}


def test_one_crossing_is_not_enough_to_promote():
    t = _trainer([1] * 80)  # wr_lookahead = 1.0, well over the threshold
    assert t._apply_opponent_phase(0.9, batch_num=100) == 'finetune'
    assert t._apply_opponent_phase(0.9, batch_num=100) == 'finetune'
    assert t._apply_opponent_phase(0.9, batch_num=100) == 'finetune2'


def test_a_dip_resets_the_streak():
    t = _trainer([1] * 80)
    t._apply_opponent_phase(0.9, batch_num=100)
    t._apply_opponent_phase(0.9, batch_num=100)
    t._wr_vs_lookahead = deque([0] * 80, maxlen=100)  # win rate collapses
    assert t._apply_opponent_phase(0.9, batch_num=100) == 'finetune'
    t._wr_vs_lookahead = deque([1] * 80, maxlen=100)
    assert t._apply_opponent_phase(0.9, batch_num=100) == 'finetune'  # streak restarted


def test_promotion_is_one_way():
    t = _trainer([1] * 80, confirm=1)
    assert t._apply_opponent_phase(0.9, batch_num=100) == 'finetune2'
    t._wr_vs_lookahead = deque([0] * 80, maxlen=100)
    # Neither a collapsed win rate nor a collapsed greedy eval takes it back: flipping the
    # opponent distribution on a noisy rolling mean is worse than either phase.
    assert t._apply_opponent_phase(0.9, batch_num=100) == 'finetune2'
    assert t._apply_opponent_phase(0.1, batch_num=100) == 'finetune2'


def test_too_few_recorded_episodes_blocks_promotion():
    t = _trainer([1] * 10, confirm=1, min_episodes=50)
    for _ in range(5):
        assert t._apply_opponent_phase(0.9, batch_num=100) == 'finetune'


def test_min_batch_holds_finetune_open_on_a_warm_start():
    t = _trainer([1] * 80, confirm=1, min_batch=500)
    assert t._apply_opponent_phase(0.9, batch_num=100) == 'finetune'
    assert t._apply_opponent_phase(0.9, batch_num=500) == 'finetune2'


def test_a_win_rate_under_the_threshold_never_promotes():
    t = _trainer([1] * 30 + [0] * 70, confirm=1)  # 0.30 against lookahead_critic
    for _ in range(4):
        assert t._apply_opponent_phase(0.9, batch_num=100) == 'finetune'
    assert all(w == {'tag': 'finetune'} for w in t._pool.applied)


# --------------------------------------------------------------------------- #
# Warm start (--init-policy / --init-critic, docs/IDEAS.md #19)
# --------------------------------------------------------------------------- #
def test_return_normalizer_restore_puts_the_critic_back_on_its_own_scale():
    from src.app.ppo import ReturnNormalizer

    norm = ReturnNormalizer()
    norm.restore(0.35, 0.62)
    assert norm.mean == 0.35 and norm.std == 0.62
    # denormalize(normalize(x)) == x, i.e. a warm-started critic's output lands where it was
    # trained to land instead of being read through a fresh (0, 1).
    assert abs(norm.denormalize(norm.normalize(1.25)) - 1.25) < 1e-9


def test_restore_is_blended_into_not_overwritten_by_the_first_batch():
    import torch

    from src.app.ppo import ReturnNormalizer

    norm = ReturnNormalizer(alpha=0.1)
    norm.restore(1.0, 1.0)
    norm.update(torch.tensor([0.0, 0.0, 0.0, 0.0]))
    # A fresh normaliser would jump straight to the batch's (0, ~0); a restored one moves
    # 10 % of the way, which is what "initialised" has to mean for the restore to be worth
    # anything at all.
    assert 0.85 < norm.mean < 0.95


def test_restore_floors_a_degenerate_std():
    from src.app.ppo import ReturnNormalizer

    norm = ReturnNormalizer()
    norm.restore(0.0, 0.0)
    assert norm.std > 0  # never divides by zero on a checkpoint with a collapsed scale


def test_adopt_checkpoint_shape_overrides_the_cli_and_says_so(caplog):
    from src.app.ppo import _adopt_checkpoint_shape
    from src.services.environment.obs_encoders import LATEST_VERSION

    hp = {'policy_arch': 'policy_v1', 'hidden_dim': 64}
    meta = {'arch': 'policy_factored_v2', 'hidden_dim': 128, 'obs_version': LATEST_VERSION}
    with caplog.at_level('INFO'):
        _adopt_checkpoint_shape(hp, meta, 'policy_arch', 'hidden_dim', 'ckpt.pth')
    assert hp == {'policy_arch': 'policy_factored_v2', 'hidden_dim': 128}
    assert 'warm start' in caplog.text


def test_adopt_checkpoint_shape_refuses_an_obs_version_mismatch():
    from src.app.ppo import _adopt_checkpoint_shape
    from src.services.environment.obs_encoders import LATEST_VERSION

    meta = {'arch': 'policy_v1', 'hidden_dim': 64, 'obs_version': LATEST_VERSION - 1}
    try:
        _adopt_checkpoint_shape({'policy_arch': 'policy_v1', 'hidden_dim': 64}, meta,
                                'policy_arch', 'hidden_dim', 'old.pth')
    except SystemExit as exc:
        assert 'OBS_VERSION' in str(exc)
    else:
        raise AssertionError('expected a SystemExit naming the obs version')
