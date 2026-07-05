"""Verify that the vectorised obs_global is numerically identical to the reference loop version."""
import sys
sys.path.insert(0, '.')

import numpy as np
from collections import Counter

from src.services.environment.warchest_env import (
    WarChestEnv,
    DECK, UNIT_COINS, RECRUIT_TYPES, N_COIN_TYPES, NUM_UNIT_TYPES,
    TOTAL_COINS, SUPPLY_CAP, HAND_SIZE,
)
# Obs-layout constants and the vectorised global-feature helpers live in the v10 encoder.
from src.services.environment.obs_encoders.v10 import (
    OWNED_TOTAL, GLOBAL_DIM, THREAT_KINDS, PENDING_CTX_DIM, PENDING_KIND_IDX,
    _DECK_LIST, _DECK_COIN_TO_IDX, _TOTAL_COINS_VEC,
    _UNIT_COIN_TO_IDX, _TOTAL_COINS_UNIT_VEC, _SUPPLY_CAP_VEC, _UNIT_IN_DECK,
    _counter_to_deck_vec, _counter_to_unit_vec,
)


# ---------------------------------------------------------------------------
# Reference: verbatim copy of the original Python-loop global feature computation.
# Used as the expected value; not run during training.
# ---------------------------------------------------------------------------

def _ref_global_feats(env) -> np.ndarray:
    """Original loop-based global feature vector (exact copy of pre-vectorisation code)."""
    active = env.active_player
    opponent = 3 - active

    own_hand = env.state.hands[active]
    own_bag = env.state.bags[active]
    own_discard = env.state.discard_faceup[active] + env.state.discard_facedown[active]
    own_supply = env.state.supply[active]

    opp_on_board = Counter()
    for u in env.board.units:
        if u.player_id == opponent:
            opp_on_board[u.id] += u.stack
    opp_faceup = env.state.discard_faceup[opponent]
    opp_supply = env.state.supply[opponent]

    def in_play(pid):
        o = env.state.owned(pid)
        b = env.state.boxed[pid]
        return Counter({c: o[c] - b[c] for c in DECK})

    own_owned = in_play(active)
    opp_owned = in_play(opponent)
    opp_hidden = {
        c: opp_owned[c] - opp_on_board[c] - opp_faceup[c] - opp_supply[c] for c in DECK
    }

    # OBS_VERSION 10 blocks — recompute via the same public helpers (unit-tested
    # separately) plus the documented formulas, independent of the concat code.
    threat = env._threat_grids(active, own_hand, opp_hidden)
    enemy_hits = sum(threat[(opponent, k)] for k in THREAT_KINDS)
    own_hits = sum(threat[(active, k)] for k in THREAT_KINDS)
    own_at_risk = sum(min(enemy_hits[u.loc], u.stack)
                      for u in env.board.units if u.player_id == active)
    opp_at_risk = sum(min(own_hits[u.loc], u.stack)
                      for u in env.board.units if u.player_id == opponent)

    opp_hand_size = sum(env.state.hands[opponent].values())
    hidden_nonneg = [max(opp_hidden[c], 0.0) for c in DECK]
    hidden_total = sum(hidden_nonneg)
    if hidden_total > 0:
        e_opp = [h * opp_hand_size / hidden_total for h in hidden_nonneg]
    else:
        e_opp = [0.0] * len(DECK)

    base = env._base_reach_grids(active, own_hand, opp_hidden)
    own_reach, enemy_reach = base[active], base[opponent]
    bases_i_can_claim = float(own_reach.sum())
    my_bases_under_flip = sum(enemy_reach[loc]
                              for loc in env.board.get_controlled_bases(active))
    win_alarm = float(
        len(env.board.get_controlled_bases(opponent)) == env.winning_base_count - 1
        and enemy_reach.sum() > 0)

    def norm(counter):
        return [counter[c] / TOTAL_COINS[c] for c in DECK]

    def norm_units(counter):
        return [counter[c] / TOTAL_COINS[c] for c in UNIT_COINS]

    def norm_supply(counter):
        return [counter[c] / SUPPLY_CAP[c] for c in RECRUIT_TYPES]

    feats = np.array(
        [
            min(env.state.round_number / env.max_rounds, 1.0),
            len(env.board.get_controlled_bases(active)) / env.winning_base_count,
            len(env.board.get_controlled_bases(opponent)) / env.winning_base_count,
            float(env.state.initiative_owner == active),
        ]
        + norm(own_hand)
        + norm(own_bag)
        + norm(own_discard)
        + norm_supply(own_supply)
        + [sum(own_bag.values()) / OWNED_TOTAL]
        + norm(own_owned)
        + norm_units(opp_on_board)
        + norm(opp_faceup)
        + norm_supply(opp_supply)
        + [opp_hidden[c] / TOTAL_COINS[c] for c in DECK]
        + norm(opp_owned)
        + [opp_hand_size / HAND_SIZE]
        + [float(env.state.initiative_transferred_this_round)]
        + [min(own_at_risk / OWNED_TOTAL, 1.0), min(opp_at_risk / OWNED_TOTAL, 1.0)]
        + [e_opp[i] / TOTAL_COINS[DECK[i]] for i in range(len(DECK))]
        + [min(bases_i_can_claim / env.winning_base_count, 1.0),
           min(my_bases_under_flip / env.winning_base_count, 1.0),
           win_alarm],
        dtype=np.float32,
    )

    # Append the pending-context one-hot (same logic as generate_observation).
    ctx = np.zeros(PENDING_CTX_DIM, dtype=np.float32)
    if env.state.pending is None:
        ctx[0] = 1.0
    else:
        ctx[1 + PENDING_KIND_IDX[env.state.pending.kind]] = 1.0
    return np.concatenate([feats, ctx])


# ---------------------------------------------------------------------------
# Unit tests for the helper functions
# ---------------------------------------------------------------------------

def test_counter_to_deck_vec_basic():
    """_counter_to_deck_vec round-trips: vec[i] == counter[_DECK_LIST[i]]."""
    c = Counter({_DECK_LIST[0]: 3, _DECK_LIST[5]: 1, _DECK_LIST[-1]: 2})
    v = _counter_to_deck_vec(c)
    assert v.shape == (N_COIN_TYPES,)
    assert v.dtype == np.float32
    assert v[0] == 3.0
    assert v[5] == 1.0
    assert v[-1] == 2.0
    assert v.sum() == 6.0, 'unexpected non-zero entries'
    print('PASS  _counter_to_deck_vec basic')


def test_counter_to_unit_vec_basic():
    """_counter_to_unit_vec round-trips: vec[i] == counter[UNIT_COINS[i]]."""
    c = Counter({UNIT_COINS[0]: 2, UNIT_COINS[3]: 4})
    v = _counter_to_unit_vec(c)
    assert v.shape == (NUM_UNIT_TYPES,)
    assert v.dtype == np.float32
    assert v[0] == 2.0
    assert v[3] == 4.0
    assert v.sum() == 6.0
    print('PASS  _counter_to_unit_vec basic')


def test_counter_to_deck_vec_empty():
    v = _counter_to_deck_vec(Counter())
    assert (v == 0).all()
    print('PASS  _counter_to_deck_vec empty')


def test_unit_in_deck_positions():
    """_UNIT_IN_DECK maps unit coins to their correct positions in _DECK_LIST."""
    for i, coin in enumerate(UNIT_COINS):
        assert _DECK_LIST[_UNIT_IN_DECK[i]] == coin, (
            f'UNIT_COINS[{i}]={coin} not at _UNIT_IN_DECK[{i}]={_UNIT_IN_DECK[i]}'
        )
    print('PASS  _UNIT_IN_DECK positions')


# ---------------------------------------------------------------------------
# End-to-end equivalence tests
# ---------------------------------------------------------------------------

def test_obs_global_equivalence_random_game():
    """Run a full random game; compare vectorised obs['global'] with reference at every step."""
    np.random.seed(42)
    env = WarChestEnv()
    obs, _ = env.reset()
    max_steps = 300
    mismatches = []

    for step in range(max_steps):
        ref = _ref_global_feats(env)
        got = obs['global']

        assert got.shape == (GLOBAL_DIM,), f'step {step}: shape {got.shape} != ({GLOBAL_DIM},)'
        if not np.array_equal(got, ref):
            max_diff = np.abs(got - ref).max()
            if max_diff > 1e-6:
                mismatches.append((step, max_diff, np.where(np.abs(got - ref) > 1e-6)[0].tolist()))

        valid = np.where(obs['valid_action_mask'])[0]
        action = int(np.random.choice(valid))
        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            obs, _ = env.reset()

    assert not mismatches, (
        f'{len(mismatches)} mismatches found:\n'
        + '\n'.join(f'  step={s} max_diff={d:.2e} positions={p[:5]}' for s, d, p in mismatches[:5])
    )
    print(f'PASS  random game ({max_steps} steps, no mismatches)')


def test_obs_global_shape_and_range():
    """Sanity: global obs has the right shape and plausible value range."""
    np.random.seed(7)
    env = WarChestEnv()
    obs, _ = env.reset()
    for _ in range(50):
        assert obs['global'].shape == (GLOBAL_DIM,)
        # All values in [-0.1, 1.1] (small slack for opp_hidden which could be slightly
        # negative if boxed coins mismatch due to floating-point; game logic prevents this
        # in practice, so any violation here signals a real bug).
        assert obs['global'].min() >= -0.11, f'unexpected negative: {obs["global"].min():.3f}'
        assert obs['global'].max() <= 1.11, f'unexpected >1: {obs["global"].max():.3f}'
        valid = np.where(obs['valid_action_mask'])[0]
        obs, _, done, truncated, _ = env.step(int(np.random.choice(valid)))
        if done or truncated:
            obs, _ = env.reset()
    print('PASS  shape and range (50 steps)')


def test_obs_global_p1_and_p2():
    """Explicitly verify both players' observations are handled correctly."""
    np.random.seed(99)
    env = WarChestEnv()
    obs, _ = env.reset()
    p1_checked = p2_checked = False
    for _ in range(100):
        active = env.active_player
        ref = _ref_global_feats(env)
        got = obs['global']
        # atol matches the random-game equivalence test: E_opp_hand's divide-by-pool-sum
        # is order-sensitive in float32 (the reference computes it in float64 then casts),
        # so tolerate ~1e-6; this test's job is verifying P1/P2 rotation, not bit-exactness.
        np.testing.assert_allclose(got, ref, atol=1e-6, err_msg=f'P{active} mismatch')
        if active == 1:
            p1_checked = True
        else:
            p2_checked = True
        valid = np.where(obs['valid_action_mask'])[0]
        obs, _, done, truncated, _ = env.step(int(np.random.choice(valid)))
        if done or truncated:
            obs, _ = env.reset()
    assert p1_checked and p2_checked, 'Did not exercise both players'
    print('PASS  P1 and P2 observations both verified')


if __name__ == '__main__':
    test_counter_to_deck_vec_basic()
    test_counter_to_unit_vec_basic()
    test_counter_to_deck_vec_empty()
    test_unit_in_deck_positions()
    test_obs_global_equivalence_random_game()
    test_obs_global_shape_and_range()
    test_obs_global_p1_and_p2()
    print('\nAll tests passed.')
