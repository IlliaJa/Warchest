"""Phase 3 — 16-unit vanilla roster + per-game 4-of-16 disjoint drafting.

Covers the new schema (roster totals, drafting, full-roster action/obs sizing,
per-type encodings) and re-covers the still-valid economy/round invariants under
the new generation (coin conservation across deploy/bolster/attack/recruit/
draw/reshuffle, coin-gating, round controller).
"""
from collections import Counter

import numpy as np
import torch

from src.services.environment.warchest_env import (
    WarChestEnv, ACTION_SPACE_SIZE, SPATIAL_SIZE, FACEDOWN_SIZE, GLOBAL_DIM,
    BOARD_CHANNELS, PRIV_DIM, N_VERBS, N_COIN_TYPES, BOARD_DIM,
    DEPLOY_VERBS, DEPLOY_VERB_BASE, BOLSTER_VERB, CONTROL_VERB, RECRUIT_TYPES,
    COIN_TO_IDX, ROYAL_COIN_IDX, VERB_OF_ACTION, UNIT_COINS, N_FACTORED_VERBS,
    OWN_UNIT_PLANE_BASE, OPP_UNIT_PLANE_BASE,
    V_MOVE, V_ATTACK, V_CONTROL, V_DEPLOY, V_BOLSTER, V_CLAIM, V_PASS, V_RECRUIT,
    DEPLOY_ACTION, RECRUIT_ACTION, MOVE_ACTION, ATTACK_ACTION,
    CLAIM_INITIATIVE_ACTION, PASS_ACTION,
)
from src.services.environment.game_state import (
    DECK, build_bag, build_supply, UNITS_PER_PLAYER, HAND_SIZE,
)
from src.services.environment import roster
from src.services.bots.greedy_bot import GreedyBot
from src.services.bots.random_bot import RandomBot


# --------------------------------------------------------------------------- #
# Roster
# --------------------------------------------------------------------------- #

def test_roster_has_16_units_plus_royal():
    assert roster.NUM_UNIT_TYPES == 16
    assert [u.id for u in roster.UNIT_TYPES] == list(range(1, 17))
    assert roster.ROYAL_ID == 17
    assert roster.TOTAL_COINS[roster.ROYAL_ID] == 1


def test_roster_totals_and_supply_split():
    # Every unit owns 4 or 5 coins; bag keeps 2, supply gets the rest (>= 2).
    for u in roster.UNIT_TYPES:
        assert u.total_coins in (4, 5)
        assert roster.SUPPLY_CAP[u.id] == u.total_coins - roster.BAG_PER_UNIT
        assert roster.SUPPLY_CAP[u.id] >= 2
    # A couple of known cards from docs/UNITS.md.
    assert roster.TOTAL_COINS[1] == 5  # Swordsman x5
    assert roster.TOTAL_COINS[2] == 4  # Knight x4


# --------------------------------------------------------------------------- #
# Action-space sizing
# --------------------------------------------------------------------------- #

def test_action_space_sizes():
    assert N_COIN_TYPES == 17
    assert len(DEPLOY_VERBS) == 16
    # Phase 4 appended a TACTIC verb past the 16 deploy verbs, so N_VERBS grew by 1.
    assert N_VERBS == DEPLOY_VERB_BASE + 16 + 1  # 31
    assert SPATIAL_SIZE == N_VERBS * BOARD_DIM * BOARD_DIM  # 1519
    # face-down = claim(C) + pass(C) + recruit(take 16 × pay C) + decline(1, Phase 4)
    assert FACEDOWN_SIZE == 2 * N_COIN_TYPES + 16 * N_COIN_TYPES + 1
    assert ACTION_SPACE_SIZE == SPATIAL_SIZE + FACEDOWN_SIZE
    assert tuple(DEPLOY_VERBS.values()) == UNIT_COINS


def test_verb_of_action_partition():
    assert VERB_OF_ACTION.shape == (ACTION_SPACE_SIZE,)
    assert set(np.unique(VERB_OF_ACTION)).issubset(set(range(N_FACTORED_VERBS)))
    # spot-check representative ids per verb
    assert VERB_OF_ACTION[WarChestEnv.encode_action(0, 3, 3)] == V_MOVE
    assert VERB_OF_ACTION[WarChestEnv.encode_action(7, 3, 3)] == V_ATTACK
    assert VERB_OF_ACTION[WarChestEnv.encode_action(CONTROL_VERB, 3, 3)] == V_CONTROL
    assert VERB_OF_ACTION[WarChestEnv.encode_action(BOLSTER_VERB, 3, 3)] == V_BOLSTER
    assert VERB_OF_ACTION[WarChestEnv.encode_action(DEPLOY_VERB_BASE, 3, 3)] == V_DEPLOY
    assert VERB_OF_ACTION[SPATIAL_SIZE] == V_CLAIM
    assert VERB_OF_ACTION[SPATIAL_SIZE + N_COIN_TYPES] == V_PASS
    assert VERB_OF_ACTION[SPATIAL_SIZE + 2 * N_COIN_TYPES] == V_RECRUIT


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


# --------------------------------------------------------------------------- #
# Drafting
# --------------------------------------------------------------------------- #

def test_draft_is_disjoint_and_sized():
    for seed in range(20):
        np.random.seed(seed)
        env = WarChestEnv()
        env.reset()
        c1, c2 = env.state.compositions[1], env.state.compositions[2]
        assert len(c1) == UNITS_PER_PLAYER and len(c2) == UNITS_PER_PLAYER
        assert len(set(c1)) == UNITS_PER_PLAYER  # distinct within a player
        assert set(c1).isdisjoint(c2)            # players never share a unit
        assert set(c1) | set(c2) <= set(UNIT_COINS)


def test_initial_bag_and_supply_match_composition():
    np.random.seed(1)
    env = WarChestEnv()
    env.reset()
    for pid in (1, 2):
        comp = env.state.compositions[pid]
        # At reset the hand is drawn from the bag, nothing discarded yet, so
        # bag + hand must reconstruct the starting bag.
        recon = env.state.bags[pid] + env.state.hands[pid]
        assert recon == build_bag(comp)
        assert env.state.supply[pid] == build_supply(comp)
        # royal coin is present and bag-only (1 copy total)
        assert recon[roster.ROYAL_ID] == 1


# --------------------------------------------------------------------------- #
# Legality / masks
# --------------------------------------------------------------------------- #

def _drive(env, n, seed=0):
    np.random.seed(seed)
    for _ in range(n):
        _, _, t, tr, _ = env.make_random_step()
        if t or tr:
            return True
    return False


def test_possible_actions_never_reference_absent_types():
    for seed in range(10):
        env = WarChestEnv()
        env.reset()
        _drive(env, 8, seed=seed)
        active = env.active_player
        comp = set(env.state.compositions[active])
        for a in env.get_possible_actions():
            kind, args = env.get_action_info(a)
            if kind == DEPLOY_ACTION:
                coin = args[0]
                assert coin in comp  # can only deploy a drafted type
            if kind == RECRUIT_ACTION:
                pay, take = args
                assert take in comp  # can only recruit a drafted supply type


def test_move_and_attack_are_coin_gated():
    for seed in range(10):
        env = WarChestEnv()
        env.reset()
        _drive(env, 12, seed=seed)
        hand = env.state.hands[env.active_player]
        for a in env.get_possible_actions():
            kind, args = env.get_action_info(a)
            if kind in (MOVE_ACTION, ATTACK_ACTION):
                _, r, q = args
                unit = env.board.get_unit_at(r, q)
                assert unit is not None and unit.player_id == env.active_player
                assert unit.id in hand  # must hold the matching coin


# --------------------------------------------------------------------------- #
# Mechanics: deploy / recruit / per-type planes
# --------------------------------------------------------------------------- #

def _find_action(env, kind):
    for a in env.get_possible_actions():
        k, args = env.get_action_info(a)
        if k == kind:
            return a, args
    return None, None


def test_deploy_places_unit_and_sets_its_plane():
    for seed in range(40):
        env = WarChestEnv()
        env.reset()
        a, args = _find_action(env, DEPLOY_ACTION)
        if a is None:
            continue
        coin, r, q = args
        active = env.active_player
        env.step(a)
        unit = env.board.get_unit_at(r, q)
        assert unit is not None and unit.id == coin and unit.player_id == active
        obs = env.generate_observation()  # from the NEW active player's view
        # The unit now belongs to the opponent of whoever is active next; its
        # plane (own or opp) must carry a nonzero stack value somewhere.
        own = obs['board'][OWN_UNIT_PLANE_BASE:OWN_UNIT_PLANE_BASE + 16]
        opp = obs['board'][OPP_UNIT_PLANE_BASE:OPP_UNIT_PLANE_BASE + 16]
        assert (own.sum() + opp.sum()) > 0
        return
    raise AssertionError('no deploy action found across resets')


def test_recruit_moves_supply_to_faceup_and_pays_facedown():
    for seed in range(60):
        np.random.seed(seed)
        env = WarChestEnv()
        env.reset()
        a, args = _find_action(env, RECRUIT_ACTION)
        if a is None:
            _drive(env, 3, seed=seed)
            a, args = _find_action(env, RECRUIT_ACTION)
            if a is None:
                continue
        pay, take = args
        active = env.active_player
        sup_before = env.state.supply[active][take]
        fu_before = env.state.discard_faceup[active][take]
        fd_before = env.state.discard_facedown[active][pay]
        env.step(a)
        assert env.state.supply[active][take] == sup_before - 1
        assert env.state.discard_faceup[active][take] == fu_before + 1
        assert env.state.discard_facedown[active][pay] == fd_before + 1
        return
    raise AssertionError('no recruit action found across resets')


# --------------------------------------------------------------------------- #
# Economy invariant + full game
# --------------------------------------------------------------------------- #

def _zone_plus_board(env, pid):
    c = Counter()
    s = env.state
    for z in (s.hands[pid], s.bags[pid], s.discard_faceup[pid],
              s.discard_facedown[pid], s.supply[pid], s.boxed[pid]):
        c += z
    for u in env.board.units:
        if u.player_id == pid:
            c[u.id] += u.stack
    return c


def test_coin_conservation_across_a_full_game():
    """Coins only move between zones (attack -> box); per-type totals are fixed."""
    for seed in range(8):
        np.random.seed(seed)
        env = WarChestEnv()
        env.reset()
        owned = {pid: env.state.owned(pid) for pid in (1, 2)}
        for _ in range(400):
            _, _, t, tr, info = env.make_random_step()
            assert info['action'].is_valid
            for pid in (1, 2):
                live = _zone_plus_board(env, pid)
                for c in DECK:
                    assert live[c] == owned[pid][c], (seed, pid, c, live[c], owned[pid][c])
            if t or tr:
                break


def test_full_random_game_terminates_with_valid_actions():
    np.random.seed(3)
    env = WarChestEnv()
    env.reset()
    done = False
    for _ in range(600):
        _, _, t, tr, info = env.make_random_step()
        assert info['action'].is_valid
        if t or tr:
            done = True
            break
    assert done


def test_round_controller_advances_rounds():
    np.random.seed(2)
    env = WarChestEnv()
    env.reset()
    start_round = env.state.round_number
    for _ in range(120):
        _, _, t, tr, _ = env.make_random_step()
        if t or tr:
            break
    assert env.state.round_number > start_round


# --------------------------------------------------------------------------- #
# Obs shapes + bots + nets
# --------------------------------------------------------------------------- #

def test_observation_shapes():
    env = WarChestEnv()
    obs, _ = env.reset()
    assert obs['board'].shape == (BOARD_CHANNELS, BOARD_DIM, BOARD_DIM)
    assert obs['global'].shape == (GLOBAL_DIM,)
    assert obs['valid_action_mask'].shape == (ACTION_SPACE_SIZE,)
    assert env.get_privileged_features().shape == (PRIV_DIM,)


def test_both_bots_return_legal_actions():
    env = WarChestEnv()
    obs, _ = env.reset()
    for bot in (RandomBot(), GreedyBot()):
        for _ in range(20):
            a, _, _ = bot.act(obs)
            assert obs['valid_action_mask'][a] == 1
            env_a = WarChestEnv.remap_action(a) if env.active_player == 2 else a
            obs, _, t, tr, info = env.step(env_a)
            assert info['action'].is_valid
            if t or tr:
                obs, _ = env.reset()


def test_policy_and_critic_forward_under_new_schema():
    from src.services.policy.policy import Policy, Critic
    dev = torch.device('cpu')
    env = WarChestEnv()
    obs, _ = env.reset()
    pol, cri = Policy(device=dev), Critic(device=dev)
    probs = pol.forward(obs).squeeze(0).detach()
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    legal = torch.tensor(obs['valid_action_mask'].astype(bool))
    assert float(probs[~legal].sum()) < 1e-6
    opp = torch.zeros(1, Critic.OPP_DIM)
    priv = torch.tensor(env.get_privileged_features()).unsqueeze(0)
    v = cri.value_single(obs, opp, priv)
    assert torch.isfinite(v).all()
