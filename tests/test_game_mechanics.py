"""Game mechanics: setup/drafting, the round/turn controller, coin-economy actions
(deploy / bolster / recruit / attack), coin-gating legality, and the coin-conservation
invariant across full random games.
"""
import numpy as np

from src.services.environment.warchest_env import (
    WarChestEnv, UNIT_COINS,
    DEPLOY_ACTION, RECRUIT_ACTION, MOVE_ACTION, ATTACK_ACTION,
)
from src.services.environment.obs_encoders.v10 import (
    OWN_UNIT_PLANE_BASE, OPP_UNIT_PLANE_BASE,
)
from src.services.environment.game_state import (
    DECK, build_bag, build_supply, UNITS_PER_PLAYER,
)
from src.services.environment import roster
from _helpers import drive, find_action, zone_plus_board


# --------------------------------------------------------------------------- #
# Setup / drafting
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
# Legality / coin-gating
# --------------------------------------------------------------------------- #

def test_possible_actions_never_reference_absent_types():
    for seed in range(10):
        env = WarChestEnv()
        env.reset()
        drive(env, 8, seed=seed)
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
        drive(env, 12, seed=seed)
        hand = env.state.hands[env.active_player]
        for a in env.get_possible_actions():
            kind, args = env.get_action_info(a)
            if kind in (MOVE_ACTION, ATTACK_ACTION):
                _, r, q = args
                unit = env.board.get_unit_at(r, q)
                assert unit is not None and unit.player_id == env.active_player
                assert unit.id in hand  # must hold the matching coin


# --------------------------------------------------------------------------- #
# Economy actions: deploy / recruit / per-type planes
# --------------------------------------------------------------------------- #

def test_deploy_places_unit_and_sets_its_plane():
    for seed in range(40):
        env = WarChestEnv()
        env.reset()
        a, args = find_action(env, DEPLOY_ACTION)
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
        a, args = find_action(env, RECRUIT_ACTION)
        if a is None:
            drive(env, 3, seed=seed)
            a, args = find_action(env, RECRUIT_ACTION)
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
# Conservation invariant + full-game / round controller
# --------------------------------------------------------------------------- #

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
                live = zone_plus_board(env, pid)
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
