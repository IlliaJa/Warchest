"""Draft pairing (opt-in, measured NOT to help) + explicit draft lists (docs/IDEAS.md L5).

The draft really does decide games: with the same deterministic bot on both sides and the
two compositions swapped between seats, the same composition won both games **63.3 %** of
the time (190/300, se 2.8 pp). So replaying each draft once per colour looks like free
variance reduction, and `build_task_list(paired=True)` implements it.

**It is off by default, because it was measured and the mechanism does not engage.** The
reduction needs the two games of a pair to be negatively correlated; measured on 150 pairs
each, `greedy_fast` vs `greedy_sim` gives `r = -0.003 +/- 0.082` and two policy checkpoints
give `r = -0.005 +/- 0.082`. The 63.3 % was measured with one deterministic bot playing
*itself*; real entrants differ, and policy agents sample their actions, so an identical
opening diverges on the first ply. Two direct variance checks (n=120) landed at ratio 1.29
and 0.77 — ~1.4 sigma in opposite directions, i.e. noise.

These tests therefore pin the *schedule* (it does what it says, and the default reproduces
every previously recorded gauntlet number bit-for-bit), not a variance claim. The one
load-bearing env property is `test_same_seed_reproduces_the_draft`: without it, `paired=True`
would silently be a no-op.

A forced-draft archetype bot could never have used the swap anyway — its composition follows
the *agent*, not the seat, so it is the treatment rather than a nuisance draw. Its control is
common random numbers across arms, which is `eval_bolster.build_draft_list`; the second half
of this file pins that, and that half is on by default because it is a robustness fix rather
than a variance claim.
"""
import numpy as np
import pytest

from src.app.eval_bolster import build_draft_list
from src.services.bots.bolster_bot import KEY_UNIT_IDS
from src.services.environment.game_state import UNITS_PER_PLAYER
from src.services.environment.warchest_env import WarChestEnv
from src.services.gauntlet import build_task_list


# --------------------------------------------------------------------------- #
# The env property the pairing depends on
# --------------------------------------------------------------------------- #

def _draft_at(seed):
    np.random.seed(seed)
    env = WarChestEnv(save_game_history=False)
    env.reset()
    return dict(env.state.compositions)


def test_same_seed_reproduces_the_draft():
    """If this ever stops holding, paired tasks silently become unpaired."""
    assert _draft_at(1234) == _draft_at(1234)


def test_different_seeds_give_different_drafts():
    """Guards against the opposite failure: a schedule that repeats one draft forever."""
    drafts = {tuple(sorted(_draft_at(s)[1])) for s in range(12)}
    assert len(drafts) > 1


def test_compositions_are_four_each_and_disjoint():
    comps = _draft_at(7)
    assert len(comps[1]) == len(comps[2]) == UNITS_PER_PLAYER
    assert not set(comps[1]) & set(comps[2])


# --------------------------------------------------------------------------- #
# The paired schedule
# --------------------------------------------------------------------------- #

def _pairs(tasks):
    """Group a single matchup's tasks into consecutive (even, odd) game pairs."""
    return list(zip(tasks[0::2], tasks[1::2]))


def test_paired_tasks_replay_each_draft_once_per_colour():
    tasks = build_task_list(2, k_games=6, seed=0, paired=True)
    assert len(tasks) == 6
    for first, second in _pairs(tasks):
        assert first[2] == second[2], 'the two games of a pair must share a seed'
        assert first[3] != second[3], 'and must be played with opposite colours'


def test_paired_schedule_uses_each_draft_exactly_twice():
    tasks = build_task_list(3, k_games=4, seed=0, paired=True)
    seeds = [t[2] for t in tasks]
    counts = {s: seeds.count(s) for s in set(seeds)}
    assert set(counts.values()) == {2}


def test_seeds_never_collide_across_matchups():
    """A draft shared between two different matchups would correlate their results."""
    tasks = build_task_list(4, k_games=4, seed=0, paired=True)
    by_match = {}
    for i, j, seed, _ in tasks:
        by_match.setdefault(seed, set()).add((i, j))
    assert all(len(m) == 1 for m in by_match.values())


def test_odd_k_games_does_not_leak_the_trailing_seed_into_the_next_matchup():
    """The unpartnered last game still consumes a seed; reusing it would pair two
    different matchups on one draft."""
    tasks = build_task_list(3, k_games=3, seed=0, paired=True)
    by_match = {}
    for i, j, seed, _ in tasks:
        by_match.setdefault(seed, set()).add((i, j))
    assert all(len(m) == 1 for m in by_match.values())


def test_the_default_is_unpaired_so_recorded_numbers_stay_reproducible():
    """Paired was measured not to help (module docstring), and switching the default
    would silently change every number ever recorded from this gauntlet."""
    assert build_task_list(3, k_games=4, seed=5) == build_task_list(
        3, k_games=4, seed=5, paired=False)
    tasks = build_task_list(3, k_games=4, seed=5, paired=False)
    seeds = [t[2] for t in tasks]
    assert seeds == list(range(5, 5 + len(tasks))), 'old schedule: a fresh seed every game'
    assert [t[3] for t in tasks] == [g % 2 == 0 for _ in range(3) for g in range(4)]


def test_paired_and_unpaired_schedule_the_same_games_and_colours():
    """Only the seeds differ; the matchup/colour layout must be untouched."""
    a = build_task_list(4, k_games=4, seed=0, paired=True)
    b = build_task_list(4, k_games=4, seed=0, paired=False)
    assert [(i, j, c) for i, j, _, c in a] == [(i, j, c) for i, j, _, c in b]


# --------------------------------------------------------------------------- #
# Forced-draft bots: common random numbers instead
# --------------------------------------------------------------------------- #

def test_draft_list_is_deterministic_in_its_seed():
    """The whole point: two arms sharing a seed face identical drafts."""
    assert build_draft_list(8, KEY_UNIT_IDS, seed=3) == build_draft_list(8, KEY_UNIT_IDS, seed=3)
    assert build_draft_list(8, KEY_UNIT_IDS, seed=3) != build_draft_list(8, KEY_UNIT_IDS, seed=4)


def test_draft_list_pins_the_key_units_to_the_bot_and_keeps_sides_disjoint():
    for bot_comp, opp_comp in build_draft_list(20, KEY_UNIT_IDS, seed=0):
        assert set(KEY_UNIT_IDS) <= set(bot_comp), 'the archetype must always be draftable'
        assert len(bot_comp) == len(opp_comp) == UNITS_PER_PLAYER
        assert not set(bot_comp) & set(opp_comp)
        assert len(set(bot_comp)) == UNITS_PER_PLAYER, 'no duplicate types within a side'


def test_draft_list_varies_the_opponent_across_games():
    """CRN fixes *which* opponents are faced; it must not collapse to one opponent."""
    opps = {o for _, o in build_draft_list(20, KEY_UNIT_IDS, seed=0)}
    assert len(opps) > 1


def test_draft_list_rejects_more_key_units_than_a_player_drafts():
    with pytest.raises(SystemExit):
        build_draft_list(4, list(range(1, UNITS_PER_PLAYER + 2)), seed=0)


def test_pinned_drafts_actually_reach_the_env():
    """`force_units` for both sides must fully determine the draft — if the env still
    redrew either side, the CRN guarantee would be silent fiction."""
    (bot_comp, opp_comp), = build_draft_list(1, KEY_UNIT_IDS, seed=0)
    seen = []
    for seed in (11, 22):  # different RNG state, same pinned draft
        np.random.seed(seed)
        env = WarChestEnv(save_game_history=False)
        env.reset(options={'force_units': {1: list(bot_comp), 2: list(opp_comp)}})
        seen.append(dict(env.state.compositions))
    assert seen[0] == seen[1]
    assert set(seen[0][1]) == set(bot_comp)
    assert set(seen[0][2]) == set(opp_comp)
