"""Action-space and observation contract: sizing, the verb partition, encode/decode,
the P2 180° remap, the SELECT spatial verb, and observation shapes. This is the
schema that breaks first on any phase bump, so it is grouped on its own.
"""
import numpy as np

from src.services.environment.warchest_env import (
    WarChestEnv, ACTION_SPACE_SIZE, SPATIAL_SIZE, FACEDOWN_SIZE, GLOBAL_DIM,
    BOARD_CHANNELS, PRIV_DIM, N_VERBS, N_COIN_TYPES, BOARD_DIM,
    DEPLOY_VERBS, DEPLOY_VERB_BASE, BOLSTER_VERB, CONTROL_VERB, TACTIC_VERB,
    SELECT_VERB, DECLINE_ACTION_ID, RECRUIT_TYPES, VERB_OF_ACTION, UNIT_COINS,
    N_FACTORED_VERBS, OBS_VERSION, PENDING_CTX_DIM,
    V_MOVE, V_ATTACK, V_CONTROL, V_DEPLOY, V_BOLSTER, V_CLAIM, V_PASS, V_RECRUIT,
    V_TACTIC, V_DECLINE, V_SELECT,
    CLAIM_INITIATIVE_ACTION, PASS_ACTION, RECRUIT_ACTION, DECLINE_ACTION,
)
from src.services.environment import roster


# --------------------------------------------------------------------------- #
# Sizing
# --------------------------------------------------------------------------- #

def test_action_space_sizes():
    assert N_COIN_TYPES == 17
    assert len(DEPLOY_VERBS) == 16
    # Phase 4 appended a TACTIC verb then a SELECT verb past the 16 deploy verbs.
    assert SELECT_VERB == 31
    assert N_VERBS == DEPLOY_VERB_BASE + 16 + 2  # 32
    assert SPATIAL_SIZE == N_VERBS * BOARD_DIM * BOARD_DIM  # 1568
    # face-down = claim(C) + pass(C) + recruit(take 16 × pay C) + decline(1, Phase 4)
    assert FACEDOWN_SIZE == 2 * N_COIN_TYPES + 16 * N_COIN_TYPES + 1
    assert ACTION_SPACE_SIZE == SPATIAL_SIZE + FACEDOWN_SIZE  # 1875
    assert DECLINE_ACTION_ID == ACTION_SPACE_SIZE - 1  # the very last id
    assert tuple(DEPLOY_VERBS.values()) == UNIT_COINS
    assert OBS_VERSION == 10


def test_global_dim_includes_pending_context():
    assert PENDING_CTX_DIM == 15  # 'no pending' + 14 tactic/attribute continuation kinds
    # OBS_VERSION 10: +E_opp_hand (8th coin-vector) + 5 scalars (2 material-at-risk,
    # 3 base-control reach) on top of the OBS_VERSION 9 layout.
    assert GLOBAL_DIM == 8 * N_COIN_TYPES + 3 * roster.NUM_UNIT_TYPES + 12 + PENDING_CTX_DIM


# --------------------------------------------------------------------------- #
# Verb partition for the factored head
# --------------------------------------------------------------------------- #

def test_verb_of_action_partition():
    assert N_FACTORED_VERBS == 11
    assert VERB_OF_ACTION.shape == (ACTION_SPACE_SIZE,)
    assert set(np.unique(VERB_OF_ACTION)).issubset(set(range(N_FACTORED_VERBS)))
    # spot-check representative ids per verb
    assert VERB_OF_ACTION[WarChestEnv.encode_action(0, 3, 3)] == V_MOVE
    assert VERB_OF_ACTION[WarChestEnv.encode_action(7, 3, 3)] == V_ATTACK
    assert VERB_OF_ACTION[WarChestEnv.encode_action(CONTROL_VERB, 3, 3)] == V_CONTROL
    assert VERB_OF_ACTION[WarChestEnv.encode_action(BOLSTER_VERB, 3, 3)] == V_BOLSTER
    assert VERB_OF_ACTION[WarChestEnv.encode_action(DEPLOY_VERB_BASE, 3, 3)] == V_DEPLOY
    assert VERB_OF_ACTION[WarChestEnv.encode_action(TACTIC_VERB, 3, 3)] == V_TACTIC
    assert VERB_OF_ACTION[WarChestEnv.encode_action(SELECT_VERB, 3, 3)] == V_SELECT
    assert VERB_OF_ACTION[SPATIAL_SIZE] == V_CLAIM
    assert VERB_OF_ACTION[SPATIAL_SIZE + N_COIN_TYPES] == V_PASS
    assert VERB_OF_ACTION[SPATIAL_SIZE + 2 * N_COIN_TYPES] == V_RECRUIT
    assert VERB_OF_ACTION[DECLINE_ACTION_ID] == V_DECLINE


# --------------------------------------------------------------------------- #
# Encode / decode / remap
# --------------------------------------------------------------------------- #

def test_encode_decode_roundtrips():
    for verb in (0, 5, 6, 11, CONTROL_VERB, BOLSTER_VERB, DEPLOY_VERB_BASE, N_VERBS - 1):
        for (r, q) in ((0, 0), (3, 4), (6, 6)):
            a = WarChestEnv.encode_action(verb, r, q)
            assert WarChestEnv.decode_action(a) == (verb, r, q)
    # face-down round-trips
    for coin in (UNIT_COINS[0], roster.ROYAL_ID):
        a = WarChestEnv.encode_facedown(0, coin)  # claim
        assert WarChestEnv.decode_facedown(a) == (CLAIM_INITIATIVE_ACTION, (coin,))
        a = WarChestEnv.encode_facedown(1, coin)  # pass
        assert WarChestEnv.decode_facedown(a) == (PASS_ACTION, (coin,))
    take, pay = RECRUIT_TYPES[3], roster.ROYAL_ID
    a = WarChestEnv.encode_recruit(take, pay)
    assert WarChestEnv.decode_facedown(a) == (RECRUIT_ACTION, (pay, take))


def test_remap_action_self_inverse_and_facedown_identity():
    for verb in (0, 3, 8, CONTROL_VERB, BOLSTER_VERB, DEPLOY_VERB_BASE + 5):
        a = WarChestEnv.encode_action(verb, 1, 2)
        assert WarChestEnv.remap_action(WarChestEnv.remap_action(a)) == a
    fd = SPATIAL_SIZE + 5
    assert WarChestEnv.remap_action(fd) == fd  # face-down ids are non-spatial


def test_tactic_and_decline_encode_decode_and_remap():
    a = WarChestEnv.encode_action(TACTIC_VERB, 1, 2)
    assert WarChestEnv.decode_action(a) == (TACTIC_VERB, 1, 2)
    # tactic is spatial: rotates 180° for P2 but its verb is unchanged; self-inverse
    assert WarChestEnv.remap_action(WarChestEnv.remap_action(a)) == a
    rv, rr, rq = WarChestEnv.decode_action(WarChestEnv.remap_action(a))
    assert rv == TACTIC_VERB and (rr, rq) == (5, 4)
    # decline is non-spatial: maps to itself and decodes to the decline action
    assert WarChestEnv.remap_action(DECLINE_ACTION_ID) == DECLINE_ACTION_ID
    assert WarChestEnv.decode_facedown(DECLINE_ACTION_ID) == (DECLINE_ACTION, ())


def test_select_is_spatial_and_remaps_like_a_cell():
    a = WarChestEnv.encode_action(SELECT_VERB, 1, 2)
    assert WarChestEnv.decode_action(a) == (SELECT_VERB, 1, 2)
    # spatial: rotates 180° for P2, verb unchanged (not a direction); self-inverse.
    assert WarChestEnv.remap_action(WarChestEnv.remap_action(a)) == a
    rv, rr, rq = WarChestEnv.decode_action(WarChestEnv.remap_action(a))
    assert rv == SELECT_VERB and (rr, rq) == (5, 4)


# --------------------------------------------------------------------------- #
# Observation shapes
# --------------------------------------------------------------------------- #

def test_observation_shapes():
    env = WarChestEnv()
    obs, _ = env.reset()
    assert obs['board'].shape == (BOARD_CHANNELS, BOARD_DIM, BOARD_DIM)
    assert obs['global'].shape == (GLOBAL_DIM,)
    assert obs['valid_action_mask'].shape == (ACTION_SPACE_SIZE,)
    assert env.get_privileged_features().shape == (PRIV_DIM,)
