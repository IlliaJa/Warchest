"""Warm-start fixes (2026-08-21, docs/IDEAS.md R.10.17).

`--init-policy` restores weights; it does not move the *schedules*, and every schedule in
`PPOTrainer` is a function of fractional progress through the run. The 2026-08-21 sparring
run is the measurement: warm-started from a checkpoint that ended its own run at
`ent_frac` 0.19, it was trained at 6.6x that checkpoint's final entropy coefficient and
7.9x its final LR, with the holding/material/base shaping anneal back at 1.0. `ent_frac`
went 0.21 -> 0.49 in 40 batches, and over 680 games it scored **0.218 +- 0.016 against the
checkpoint it started from**, landing on the cold run's own curve at the same batch index
(`ent_frac` 0.475 vs 0.473, `wr_vs_lookahead_critic` 0.256 vs 0.267 at batches 241-344).

Three things are pinned here:
  * `--schedule-start-frac` moves where the run starts in that schedule, and 0.0 must
    reproduce every pre-change run exactly;
  * `--start-in-finetune2` enters the sparring phase at batch 1, because its win-rate gate
    is unreachable — it wants `wr_vs_lookahead_critic` >= 0.60 and the checkpoint that ended
    the 2026-08-17 run at 0.71 reads 0.477 now that the exploration-plane fix has stopped
    handicapping the search;
  * a rolling win rate with no games behind it reports WR_NO_GAMES, not 0.0. That
    distinction is what made `wr_vs_puct_live_train` read as a flat total loss through a run
    in which `puct_live` never played a single episode.
"""
import inspect
import re
from collections import deque
from types import SimpleNamespace

from pytest import approx

from src.app.ppo import (
    PPOTrainer, SHAPING_ANNEAL_FINAL, SHAPING_ANNEAL_HALF_FRAC, SHAPING_ANNEAL_INIT,
    WR_NO_GAMES,
)


# --------------------------------------------------------------------------- #
# The schedule offset
# --------------------------------------------------------------------------- #
class _StubOptimizer:
    def __init__(self):
        self.param_groups = [{'lr': 0.0}]


def _trainer(*, n_batches=1500, start_frac=0.0):
    t = SimpleNamespace(
        _n_batches=n_batches,
        _schedule_start_frac=start_frac,
        _entropy_coeff=None,
        _entropy_coeff_init=0.025,
        _entropy_coeff_final=0.003,
        _verb_entropy_coeff=None,
        _verb_entropy_coeff_init=0.02,
        _verb_entropy_coeff_final=0.01,
        _lr_actor_init=3e-4,
        _lr_critic_init=3e-4,
        _lr_final_frac=0.1,
        _shaping_anneal=None,
        _actor_optimizer=_StubOptimizer(),
        _critic_optimizer=_StubOptimizer(),
    )
    for name in ('_schedule_frac', '_update_schedules', '_compute_shaping_anneal'):
        setattr(t, name, getattr(PPOTrainer, name).__get__(t, SimpleNamespace))
    return t


def test_a_fresh_run_is_unchanged_by_the_new_knob():
    """frac 0.0 must reproduce every pre-2026-08-21 run: init values on batch 1, final
    values on the last batch, shaping at the floor from the halfway point."""
    t = _trainer(start_frac=0.0)

    t._update_schedules(1)
    assert t._entropy_coeff == 0.025
    assert t._verb_entropy_coeff == 0.02
    assert t._actor_optimizer.param_groups[0]['lr'] == 3e-4
    assert t._shaping_anneal == SHAPING_ANNEAL_INIT

    t._update_schedules(1500)
    assert abs(t._entropy_coeff - 0.003) < 1e-12
    assert abs(t._verb_entropy_coeff - 0.01) < 1e-12
    assert abs(t._actor_optimizer.param_groups[0]['lr'] - 3e-5) < 1e-12
    assert t._shaping_anneal == approx(SHAPING_ANNEAL_FINAL)

    # The shaping anneal still reaches its floor just past SHAPING_ANNEAL_HALF_FRAC of the
    # run, as it did before the refactor. It now divides by (n_batches - 1) rather than
    # n_batches, so the anneal is 0.07 % faster than the pre-change one — under a tenth of
    # one batch, and the entropy/LR schedules, which had the (n_batches - 1) denominator all
    # along, are bit-identical.
    half = int(1500 * SHAPING_ANNEAL_HALF_FRAC)
    assert t._compute_shaping_anneal(half + 1) == approx(SHAPING_ANNEAL_FINAL)
    assert t._compute_shaping_anneal(half) > SHAPING_ANNEAL_FINAL
    assert t._compute_shaping_anneal(half - 300) > 0.25


def test_frac_one_holds_the_final_values_for_the_whole_run():
    """The setting for continuing from an end-of-run checkpoint: nothing to re-anneal."""
    t = _trainer(start_frac=1.0)
    for batch in (1, 2, 150, 300):
        t._update_schedules(batch)
        assert abs(t._entropy_coeff - 0.003) < 1e-12
        assert abs(t._verb_entropy_coeff - 0.01) < 1e-12
        assert abs(t._actor_optimizer.param_groups[0]['lr'] - 3e-5) < 1e-12
        assert t._shaping_anneal == approx(SHAPING_ANNEAL_FINAL)


def test_a_partial_offset_stretches_the_remainder_over_the_run():
    """frac 0.5 starts halfway and still ends at the final values, so the knob can never
    leave a run stranded mid-anneal — which is the failure mode it exists to remove."""
    t = _trainer(start_frac=0.5)
    assert t._schedule_frac(1) == 0.5
    assert t._schedule_frac(1500) == 1.0

    t._update_schedules(1)
    mid = t._entropy_coeff
    assert 0.003 < mid < 0.025
    t._update_schedules(1500)
    assert abs(t._entropy_coeff - 0.003) < 1e-12
    # Shaping is already at the floor: its anneal completes at half of the schedule.
    assert t._compute_shaping_anneal(1) == approx(SHAPING_ANNEAL_FINAL)


def test_the_schedule_fraction_is_monotone_and_clamped():
    t = _trainer(start_frac=0.3)
    fracs = [t._schedule_frac(b) for b in range(1, 1502)]
    assert fracs == sorted(fracs)
    assert fracs[0] == 0.3
    assert max(fracs) == 1.0  # batch_num past n_batches must not overshoot


def test_the_schedule_stubs_read_the_same_attributes_the_real_trainer_sets():
    """Same guard as tests/test_finetune2_phase.py: a SimpleNamespace stub accepts any
    attribute name, so an attribute typo would pass here and crash the first real batch."""
    known = set(re.findall(r'self\.(_\w+)\s*=', inspect.getsource(PPOTrainer.__init__)))
    known |= {n for n in dir(PPOTrainer) if n.startswith('_')}  # method calls, not state
    for method in (PPOTrainer._update_schedules, PPOTrainer._schedule_frac,
                   PPOTrainer._compute_shaping_anneal):
        read = {n for n in method.__code__.co_names if n.startswith('_')}
        assert not read - known, (
            f'{method.__name__} never initialised: {sorted(read - known)}')


# --------------------------------------------------------------------------- #
# Entering finetune2 directly
# --------------------------------------------------------------------------- #
class _StubPool:
    def __init__(self):
        self.applied = []

    def set_weights(self, **kw):
        self.applied.append(kw)


def _phase_trainer(*, in_finetune2):
    t = SimpleNamespace(
        _pool=_StubPool(),
        _wr_finetune_threshold=0.75,
        _wr_finetune2_threshold=0.60,
        _finetune2_confirm_evals=3,
        _finetune2_min_batch=0,
        _finetune2_min_episodes=50,
        _in_finetune2=in_finetune2,
        _finetune2_streak=0,
        _wr_vs_lookahead_critic=deque(maxlen=100),
        _opp_weights_initial={'tag': 'initial'},
        _opp_weights_finetune={'tag': 'finetune'},
        _opp_weights_finetune2={'tag': 'finetune2'},
    )
    t._apply_opponent_phase = PPOTrainer._apply_opponent_phase.__get__(t, SimpleNamespace)
    return t


def test_start_in_finetune2_needs_neither_the_win_rate_nor_the_greedy_gate():
    # No lookahead_critic episodes recorded at all, and a greedy eval far under the
    # finetune threshold: the un-gated phase must still be finetune2 from the first eval.
    t = _phase_trainer(in_finetune2=True)
    assert t._apply_opponent_phase(0.1, batch_num=1) == 'finetune2'
    assert t._pool.applied[-1] == {'tag': 'finetune2'}


def test_without_the_flag_an_empty_deque_still_blocks_promotion():
    t = _phase_trainer(in_finetune2=False)
    assert t._apply_opponent_phase(0.9, batch_num=1) == 'finetune'


# --------------------------------------------------------------------------- #
# "No games played" is not a win rate of zero
# --------------------------------------------------------------------------- #
def test_an_unplayed_opponent_reports_the_sentinel_not_zero():
    assert PPOTrainer._rolling_wr(deque(maxlen=100)) == WR_NO_GAMES
    assert WR_NO_GAMES < 0.0, 'the sentinel must be outside the win-rate range'


def test_a_played_opponent_reports_its_mean():
    assert PPOTrainer._rolling_wr(deque([1, 1, 0, 0], maxlen=100)) == 0.5
    assert PPOTrainer._rolling_wr(deque([0, 0], maxlen=100)) == 0.0


# --------------------------------------------------------------------------- #
# The eval opponents
# --------------------------------------------------------------------------- #
def test_the_eval_phase_no_longer_plays_random():
    """`wr_vs_random_eval` read exactly 1.000 for all 1500 batches of the 2026-08-17 run
    and all ~350 of the next one, at a third of the eval budget. Pinned by name because the
    game is cheap to reintroduce by accident.
    """
    src = inspect.getsource(PPOTrainer._maybe_eval)
    body = src.split('"""')[2]  # skip the docstring, which explains the removal
    assert 'RandomBot' not in body
    assert "'random'" not in body, 'the Elo tracker is fed no `random` games any more'
    assert 'wr_random' not in body
    # And the import it needed is gone from the module, so a reintroduction cannot be silent.
    import src.app.ppo as ppo_module
    assert not hasattr(ppo_module, 'RandomBot')
