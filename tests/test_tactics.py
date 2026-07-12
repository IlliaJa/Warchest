"""Tactics: the pending sub-turn state machine plus each unit's tactic end-to-end.

  Cavalry:  TACTIC@unit  →  move-dir (mandatory)  →  attack-dir (mandatory)
  Archer:   TACTIC@unit  →  SELECT target (mandatory ranged attack, 2 away)

Covers the reusable mechanism (a multi-step tactic resolves as masked continuation
clicks without inflating the action space), the pending-context one-hot, the Archer
"no normal attack" restriction, P2-via-remap, and coin conservation with tactics in
play. New units add cases here as Phase 4 grows.
"""
from collections import Counter

import numpy as np

from src.services.environment.warchest_env import (
    WarChestEnv, SPATIAL_SIZE, TACTIC_VERB, SELECT_VERB, DECLINE_ACTION_ID,
)
from src.services.environment.obs_encoders.v10 import PENDING_CTX_DIM, PENDING_KIND_IDX
from src.services.environment.game_state import DECK
from _helpers import (
    cavalry_scenario, archer_scenario, zone_plus_board, blank_env, place,
    CAV, ARCHER, LIGHT_CAV, LANCER, CROSSBOW, FOOTMAN, ENSIGN, MARSHALL, ROYAL_GUARD, ROYAL,
    BERSERKER,
    A, B, C, MOVE_DIR_A_TO_B, ATK_DIR_B_TO_C,
    AR, FAR, ATK_DIR_AR_TO_ADJ,
)


# --------------------------------------------------------------------------- #
# Cavalry — move_then_attack (directional continuations)
# --------------------------------------------------------------------------- #

def test_cavalry_tactic_full_move_then_attack():
    env, cav, enemy = cavalry_scenario()
    tac = WarChestEnv.encode_action(TACTIC_VERB, *A)
    assert tac in env.get_possible_actions()

    # 1) Initiate: coin paid face-up, pending opened, turn does NOT pass.
    env.step(tac)
    assert env.state.hands[1][CAV] == 0
    assert env.state.discard_faceup[1][CAV] == 1
    assert env.active_player == 1
    assert env.state.pending is not None and env.state.pending.kind == 'move_then_attack:move'

    # 2) Move step: only move-dirs from A, no decline (mandatory).
    cont = env.get_possible_actions()
    assert DECLINE_ACTION_ID not in cont
    assert all(0 <= WarChestEnv.decode_action(a)[0] <= 5 for a in cont)
    move = WarChestEnv.encode_action(MOVE_DIR_A_TO_B, *A)
    assert move in cont
    env.step(move)
    assert env.board.get_unit_at(*B) is cav and env.board.get_unit_at(*A) is None
    assert env.active_player == 1
    assert env.state.pending.kind == 'move_then_attack:attack' and env.state.pending.unit_loc == B

    # 3) Attack step: mandatory (no decline), and the adjacent enemy is targetable.
    cont = env.get_possible_actions()
    assert DECLINE_ACTION_ID not in cont
    atk = WarChestEnv.encode_action(6 + ATK_DIR_B_TO_C, *B)
    assert atk in cont
    env.step(atk)
    # enemy (stack 1) removed → boxed; pending cleared → turn passed on.
    assert env.board.get_unit_at(*C) is None
    assert env.state.boxed[2][1] == 1
    assert env.state.pending is None


def test_cavalry_attack_step_is_mandatory():
    """'Move and then attack' — both halves are mandatory, so the attack cannot be
    declined once the step lands next to an enemy."""
    env, cav, enemy = cavalry_scenario()
    env.step(WarChestEnv.encode_action(TACTIC_VERB, *A))
    env.step(WarChestEnv.encode_action(MOVE_DIR_A_TO_B, *A))
    assert env.state.pending.kind == 'move_then_attack:attack'
    assert DECLINE_ACTION_ID not in env.get_possible_actions()
    _, _, _, _, info = env.step(DECLINE_ACTION_ID)
    assert not info['action'].is_valid          # cannot skip the mandatory attack
    assert env.state.pending.kind == 'move_then_attack:attack'  # state unchanged
    assert env.board.get_unit_at(*C) is enemy


def test_cavalry_tactic_unavailable_without_a_completable_attack():
    """With no move that ends adjacent to an enemy, the tactic cannot start — the
    Cavalry falls back to a normal move (it never degrades into a bare move-tactic)."""
    env, cav, enemy = cavalry_scenario()
    env.board.remove_unit(enemy)                 # nothing left to attack anywhere
    assert env.board.get_free_adjacent_cells(*A) # the Cavalry can still move normally
    assert not env._tactic_startable(cav)
    assert WarChestEnv.encode_action(TACTIC_VERB, *A) not in env.get_possible_actions()


def test_cavalry_move_step_is_mandatory():
    env, cav, enemy = cavalry_scenario()
    env.step(WarChestEnv.encode_action(TACTIC_VERB, *A))
    assert DECLINE_ACTION_ID not in env.get_possible_actions()
    _, _, _, _, info = env.step(DECLINE_ACTION_ID)
    assert not info['action'].is_valid          # cannot skip the mandatory move
    assert env.state.pending.kind == 'move_then_attack:move'  # state unchanged


def test_tactic_blocked_when_no_coin_in_hand():
    env, cav, enemy = cavalry_scenario()
    env.state.hands[1] = Counter()  # drop the Cavalry coin
    assert WarChestEnv.encode_action(TACTIC_VERB, *A) not in env.get_possible_actions()


def test_pending_context_onehot_tracks_the_subturn():
    env, cav, enemy = cavalry_scenario()
    ctx = env.generate_observation()['global'][-PENDING_CTX_DIM:]
    assert ctx[0] == 1.0 and ctx[1:].sum() == 0.0     # normal play

    env.step(WarChestEnv.encode_action(TACTIC_VERB, *A))
    ctx = env.generate_observation()['global'][-PENDING_CTX_DIM:]
    assert ctx[0] == 0.0
    assert ctx[1 + PENDING_KIND_IDX['move_then_attack:move']] == 1.0

    env.step(WarChestEnv.encode_action(MOVE_DIR_A_TO_B, *A))
    ctx = env.generate_observation()['global'][-PENDING_CTX_DIM:]
    assert ctx[1 + PENDING_KIND_IDX['move_then_attack:attack']] == 1.0


def test_turn_does_not_pass_mid_tactic():
    env, cav, enemy = cavalry_scenario()
    env.step(WarChestEnv.encode_action(TACTIC_VERB, *A))
    assert env.active_player == 1
    env.step(WarChestEnv.encode_action(MOVE_DIR_A_TO_B, *A))
    assert env.active_player == 1  # still mid-tactic
    # mask only ever offers continuations while pending is set
    for a in env.get_possible_actions():
        assert a == DECLINE_ACTION_ID or a < SPATIAL_SIZE


def test_p2_cavalry_tactic_via_remap():
    """The whole flow must also work in P2's rotated frame (as the trainer drives it)."""
    env, cav, enemy = cavalry_scenario()
    # Flip ownership: cavalry is P2's, enemy is P1's; P2 to act.
    cav.player_id = 2
    enemy.player_id = 1
    env.state.active_player = 2
    env.state.compositions = {1: (1,), 2: (CAV,)}
    env.state.hands = {1: Counter(), 2: Counter({CAV: 1})}

    obs = env.generate_observation()
    mask = obs['valid_action_mask']
    # The tactic id appears in the mask at the *rotated* cell.
    tac_abs = WarChestEnv.encode_action(TACTIC_VERB, *A)
    tac_ego = WarChestEnv.remap_action(tac_abs)
    assert mask[tac_ego] == 1
    env.step(WarChestEnv.remap_action(tac_ego))  # trainer remaps ego→absolute before step
    assert env.state.pending.kind == 'move_then_attack:move'

    move_abs = WarChestEnv.encode_action(MOVE_DIR_A_TO_B, *A)
    assert env.generate_observation()['valid_action_mask'][WarChestEnv.remap_action(move_abs)] == 1
    env.step(move_abs)
    assert env.board.get_unit_at(*B) is cav
    atk_abs = WarChestEnv.encode_action(6 + ATK_DIR_B_TO_C, *B)
    env.step(atk_abs)
    assert env.board.get_unit_at(*C) is None  # enemy removed


# --------------------------------------------------------------------------- #
# Archer — ranged_attack (the SELECT primitive)
# --------------------------------------------------------------------------- #

def test_archer_ranged_tactic_full_select():
    env, archer, far = archer_scenario()
    tac = WarChestEnv.encode_action(TACTIC_VERB, *AR)
    assert tac in env.get_possible_actions()

    # 1) Initiate: coin paid face-up, ranged_attack pending opened, turn does NOT pass.
    env.step(tac)
    assert env.state.hands[1][ARCHER] == 0
    assert env.state.discard_faceup[1][ARCHER] == 1
    assert env.active_player == 1
    assert env.state.pending is not None and env.state.pending.kind == 'ranged_attack'

    # 2) Select step: every continuation is a SELECT at a legal target cell; the
    #    distance-2 enemy is offered, the step is mandatory (no decline).
    cont = env.get_possible_actions()
    assert DECLINE_ACTION_ID not in cont
    assert all(WarChestEnv.decode_action(a)[0] == SELECT_VERB for a in cont)
    sel = WarChestEnv.encode_action(SELECT_VERB, *FAR)
    assert sel in cont

    # 3) Resolve: target loses its (only) coin → boxed, unit removed, pending cleared.
    env.step(sel)
    assert env.board.get_unit_at(*FAR) is None
    assert env.state.boxed[2][1] == 1
    assert env.state.pending is None


def test_archer_cannot_normal_attack_but_can_use_tactic():
    env, archer, far = archer_scenario(adjacent_enemy=True)
    actions = env.get_possible_actions()
    # No normal (adjacent) attack on the adjacent enemy — the restriction holds.
    normal_atk = WarChestEnv.encode_action(6 + ATK_DIR_AR_TO_ADJ, *AR)
    assert normal_atk not in actions
    assert not any(6 <= WarChestEnv.decode_action(a)[0] <= 11
                   for a in actions if a < SPATIAL_SIZE)
    # But the ranged tactic against the distance-2 enemy is available.
    assert WarChestEnv.encode_action(TACTIC_VERB, *AR) in actions


def test_archer_tactic_requires_a_target_in_range():
    env, archer, far = archer_scenario()
    env.board.remove_unit(far)  # no enemy at distance 2 anymore
    assert WarChestEnv.encode_action(TACTIC_VERB, *AR) not in env.get_possible_actions()


def test_archer_select_step_is_mandatory():
    env, archer, far = archer_scenario()
    env.step(WarChestEnv.encode_action(TACTIC_VERB, *AR))
    assert DECLINE_ACTION_ID not in env.get_possible_actions()
    _, _, _, _, info = env.step(DECLINE_ACTION_ID)
    assert not info['action'].is_valid               # cannot skip the mandatory attack
    assert env.state.pending.kind == 'ranged_attack'  # state unchanged
    assert env.board.get_unit_at(*FAR) is far


def test_archer_pending_context_onehot():
    env, archer, far = archer_scenario()
    env.step(WarChestEnv.encode_action(TACTIC_VERB, *AR))
    ctx = env.generate_observation()['global'][-PENDING_CTX_DIM:]
    assert ctx[0] == 0.0
    assert ctx[1 + PENDING_KIND_IDX['ranged_attack']] == 1.0


def test_p2_archer_tactic_via_remap():
    """The ranged SELECT flow must also work in P2's rotated frame."""
    env, archer, far = archer_scenario()
    archer.player_id = 2
    far.player_id = 1
    env.state.active_player = 2
    env.state.compositions = {1: (1, 2), 2: (ARCHER,)}
    env.state.hands = {1: Counter(), 2: Counter({ARCHER: 1})}

    tac_abs = WarChestEnv.encode_action(TACTIC_VERB, *AR)
    assert env.generate_observation()['valid_action_mask'][WarChestEnv.remap_action(tac_abs)] == 1
    env.step(tac_abs)
    assert env.state.pending.kind == 'ranged_attack'

    sel_abs = WarChestEnv.encode_action(SELECT_VERB, *FAR)
    assert env.generate_observation()['valid_action_mask'][WarChestEnv.remap_action(sel_abs)] == 1
    env.step(sel_abs)
    assert env.board.get_unit_at(*FAR) is None  # enemy removed in the absolute frame


# --------------------------------------------------------------------------- #
# Pending machine invariants under random play
# --------------------------------------------------------------------------- #

def test_coin_conservation_holds_with_tactics_in_play():
    """Tactics only move coins between zones (init→face-up, attack→box)."""
    saw_pending = False
    for seed in range(12):
        np.random.seed(seed)
        env = WarChestEnv()
        env.reset()
        owned = {pid: env.state.owned(pid) for pid in (1, 2)}
        for _ in range(400):
            _, _, t, tr, info = env.make_random_step()
            assert info['action'].is_valid
            saw_pending = saw_pending or env.state.pending is not None
            for pid in (1, 2):
                live = zone_plus_board(env, pid)
                for c in DECK:
                    assert live[c] == owned[pid][c], (seed, pid, c)
            if t or tr:
                break
    # The slice is only meaningful if random play actually reaches the tactic path.
    assert saw_pending


def test_mid_pending_actions_are_always_available():
    """While a tactic is pending the active player always has a legal continuation."""
    for seed in range(20):
        np.random.seed(seed)
        env = WarChestEnv()
        env.reset()
        for _ in range(300):
            if env.state.pending is not None:
                assert env.get_possible_actions(), 'pending sub-turn must not softlock'
            _, _, t, tr, _ = env.make_random_step()
            if t or tr:
                break


# --------------------------------------------------------------------------- #
# Cluster 1/2/4 tactics (SELECT-driven destinations / targets / grants)
# --------------------------------------------------------------------------- #

def test_light_cavalry_moves_up_to_two():
    env = blank_env(active=1)
    env.state.compositions = {1: (LIGHT_CAV,), 2: ()}
    env.state.hands[1] = Counter({LIGHT_CAV: 1})
    env.state.bags = {1: Counter({LIGHT_CAV: 1}), 2: Counter()}
    lc = place(env, LIGHT_CAV, 1, (3, 3), stack=1)

    env.step(WarChestEnv.encode_action(TACTIC_VERB, 3, 3))
    assert env.state.pending.kind == 'move_to'
    far = WarChestEnv.encode_action(SELECT_VERB, 3, 5)   # two empty steps away
    assert far in env.get_possible_actions()
    env.step(far)
    assert env.board.get_unit_at(3, 5) is lc and env.state.pending is None


def test_lancer_charges_in_a_straight_line():
    env = blank_env(active=1)
    env.state.compositions = {1: (LANCER,), 2: (CAV,)}
    env.state.hands[1] = Counter({LANCER: 1})
    env.state.bags = {1: Counter({LANCER: 1}), 2: Counter({CAV: 1})}
    lancer = place(env, LANCER, 1, (3, 3), stack=1)
    place(env, CAV, 2, (3, 5), stack=1)                  # 2 away in a line, (3,4) empty

    # Lancer has no normal attack and the enemy is not adjacent, so only the tactic shows.
    env.step(WarChestEnv.encode_action(TACTIC_VERB, 3, 3))
    assert env.state.pending.kind == 'line_charge'
    sel = WarChestEnv.encode_action(SELECT_VERB, 3, 5)
    assert sel in env.get_possible_actions()
    env.step(sel)
    assert env.board.get_unit_at(3, 4) is lancer         # ended adjacent to the target
    assert env.board.get_unit_at(3, 5) is None           # target struck
    assert env.state.pending is None


def test_crossbowman_straight_line_blocked_by_intervening_unit():
    env = blank_env(active=1)
    env.state.compositions = {1: (CROSSBOW,), 2: (CAV,)}
    env.state.hands[1] = Counter({CROSSBOW: 1})
    env.state.bags = {1: Counter({CROSSBOW: 1}), 2: Counter({CAV: 1})}
    place(env, CROSSBOW, 1, (3, 3), stack=1)
    place(env, CAV, 2, (3, 5), stack=1)        # 2 away straight
    place(env, CAV, 2, (3, 4), stack=1)        # blocks the line

    # The line is blocked, so no ranged target → the tactic is not offered.
    assert WarChestEnv.encode_action(TACTIC_VERB, 3, 3) not in env.get_possible_actions()


def test_marshall_grants_a_normal_attack():
    env = blank_env(active=1)
    env.state.compositions = {1: (MARSHALL, CAV), 2: (LIGHT_CAV,)}
    env.state.hands[1] = Counter({MARSHALL: 1})
    env.state.bags = {1: Counter({MARSHALL: 1}), 2: Counter({LIGHT_CAV: 1})}
    place(env, MARSHALL, 1, (3, 3), stack=1)
    place(env, CAV, 1, (3, 4), stack=1)        # friendly within range
    place(env, LIGHT_CAV, 2, (3, 5), stack=1)  # adjacent to the friendly unit

    env.step(WarChestEnv.encode_action(TACTIC_VERB, 3, 3))
    assert env.state.pending.kind == 'grant_attack:select'
    env.step(WarChestEnv.encode_action(SELECT_VERB, 3, 4))    # choose the ally
    assert env.state.pending.kind == 'grant_attack:strike' and env.state.pending.unit_loc == (3, 4)
    env.step(WarChestEnv.encode_action(6 + 2, 3, 4))          # ally attacks (3,5)
    assert env.board.get_unit_at(3, 5) is None and env.state.pending is None


def test_ensign_grants_a_move():
    env = blank_env(active=1)
    env.state.compositions = {1: (ENSIGN, CAV), 2: ()}
    env.state.hands[1] = Counter({ENSIGN: 1})
    env.state.bags = {1: Counter({ENSIGN: 1}), 2: Counter()}
    place(env, ENSIGN, 1, (3, 3), stack=1)
    ally = place(env, CAV, 1, (3, 4), stack=1)

    env.step(WarChestEnv.encode_action(TACTIC_VERB, 3, 3))
    assert env.state.pending.kind == 'grant_move:select'
    env.step(WarChestEnv.encode_action(SELECT_VERB, 3, 4))    # choose the ally
    assert env.state.pending.kind == 'grant_move:step'
    env.step(WarChestEnv.encode_action(2, 3, 4))              # ally moves to (3,5), still ≤2 from ensign
    assert env.board.get_unit_at(3, 5) is ally and env.state.pending is None


def test_footman_tactic_maneuvers_each_footman():
    env = blank_env(active=1)
    env.state.compositions = {1: (FOOTMAN,), 2: ()}
    env.state.hands[1] = Counter({FOOTMAN: 1})
    env.state.bags = {1: Counter({FOOTMAN: 1}), 2: Counter()}
    f1 = place(env, FOOTMAN, 1, (3, 3), stack=1)
    f2 = place(env, FOOTMAN, 1, (0, 1), stack=1)             # a second Footman (its attribute)

    env.step(WarChestEnv.encode_action(TACTIC_VERB, 3, 3))
    assert env.state.pending.kind == 'footman_maneuver' and env.state.pending.unit_loc == (3, 3)
    env.step(WarChestEnv.encode_action(2, 3, 3))            # first Footman moves to (3,4)
    assert f1.loc == (3, 4) and env.state.pending.unit_loc == (0, 1)
    env.step(WarChestEnv.encode_action(2, 0, 1))           # second Footman moves to (0,2)
    assert f2.loc == (0, 2) and env.state.pending is None


def test_royal_guard_moves_to_a_controlled_location_paying_royal():
    env = blank_env(active=1)
    env.state.compositions = {1: (ROYAL_GUARD,), 2: ()}
    env.state.hands[1] = Counter({ROYAL: 1})               # paid by the Royal coin, not the RG coin
    env.state.bags = {1: Counter({ROYAL: 1}), 2: Counter()}
    rg = place(env, ROYAL_GUARD, 1, (3, 3), stack=1)
    env.board.change_base_control(1, (3, 5))               # a controlled, empty destination 2 away

    assert WarChestEnv.encode_action(TACTIC_VERB, 3, 3) in env.get_possible_actions()
    env.step(WarChestEnv.encode_action(TACTIC_VERB, 3, 3))
    assert env.state.pending.kind == 'move_to'
    assert env.state.hands[1][ROYAL] == 0                  # the Royal coin was spent
    env.step(WarChestEnv.encode_action(SELECT_VERB, 3, 5))
    assert env.board.get_unit_at(3, 5) is rg and env.state.pending is None


def test_light_cavalry_cannot_move_over_an_adjacent_unit():
    env = blank_env(active=1)
    env.state.compositions = {1: (LIGHT_CAV,), 2: (CAV,)}
    env.state.hands[1] = Counter({LIGHT_CAV: 1})
    env.state.bags = {1: Counter({LIGHT_CAV: 1}), 2: Counter({CAV: 1})}
    place(env, LIGHT_CAV, 1, (3, 3), stack=1)
    place(env, CAV, 2, (3, 4), stack=1)            # blocks the straight path to (3,5)

    env.step(WarChestEnv.encode_action(TACTIC_VERB, 3, 3))
    cont = env.get_possible_actions()
    # (3,5) is only reachable through the blocked (3,4) → not offered.
    assert WarChestEnv.encode_action(SELECT_VERB, 3, 5) not in cont
    # but a cell reached around the obstacle (via (4,4)) still is.
    assert WarChestEnv.encode_action(SELECT_VERB, 4, 5) in cont


def test_lancer_cannot_make_a_melee_attack():
    env = blank_env(active=1)
    env.state.compositions = {1: (LANCER,), 2: (CAV,)}
    env.state.hands[1] = Counter({LANCER: 1})
    env.state.bags = {1: Counter({LANCER: 1}), 2: Counter({CAV: 1})}
    place(env, LANCER, 1, (3, 3), stack=1)
    place(env, CAV, 2, (3, 4), stack=1)            # adjacent enemy

    actions = env.get_possible_actions()
    # No normal (adjacent) attack verb is ever offered for the Lancer.
    assert not any(6 <= WarChestEnv.decode_action(a)[0] <= 11
                   for a in actions if a < SPATIAL_SIZE)
    # And with the enemy adjacent (not 2-away in a clear line), the charge tactic is unavailable too.
    assert WarChestEnv.encode_action(TACTIC_VERB, 3, 3) not in actions


def test_berserker_continues_after_a_marshall_granted_attack():
    env = blank_env(active=1)
    env.state.compositions = {1: (MARSHALL, BERSERKER), 2: (CAV,)}
    env.state.hands[1] = Counter({MARSHALL: 1})
    env.state.bags = {1: Counter({MARSHALL: 1}), 2: Counter({CAV: 1})}
    place(env, MARSHALL, 1, (3, 3), stack=1)
    ber = place(env, BERSERKER, 1, (3, 4), stack=3)   # bolstered Berserker, within range
    place(env, CAV, 2, (3, 5), stack=1)               # adjacent to the Berserker

    env.step(WarChestEnv.encode_action(TACTIC_VERB, 3, 3))       # Marshall tactic
    env.step(WarChestEnv.encode_action(SELECT_VERB, 3, 4))       # choose the Berserker
    assert env.state.pending.kind == 'grant_attack:strike'
    env.step(WarChestEnv.encode_action(6 + 2, 3, 4))            # Berserker attacks (3,5)
    assert env.board.get_unit_at(3, 5) is None
    # Its own attribute now lets it keep maneuvering with stack coins.
    assert env.state.pending is not None and env.state.pending.kind == 'extra_maneuver'
    assert env.state.pending.unit_loc == (3, 4)
