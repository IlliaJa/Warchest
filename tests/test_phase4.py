"""Phase 4 — tactics scaffolding + the Cavalry vertical slice.

Covers the pending sub-turn state machine (the reusable mechanism that lets
multi-step tactics resolve as a sequence of masked clicks without inflating the
action space) and Cavalry's "move then attack" tactic end-to-end:

  TACTIC@unit  →  move-dir (mandatory)  →  attack-dir (optional)

plus the schema bump it introduces (TACTIC verb, DECLINE slot, pending-context
one-hot) and the still-valid coin-conservation invariant under random play.
"""
from collections import Counter

import numpy as np
import torch

from src.services.environment.warchest_env import (
    WarChestEnv, ACTION_SPACE_SIZE, SPATIAL_SIZE, FACEDOWN_SIZE, GLOBAL_DIM,
    N_VERBS, TACTIC_VERB, DECLINE_ACTION_ID, N_FACTORED_VERBS, BOARD_DIM,
    N_COIN_TYPES, OBS_VERSION, VERB_OF_ACTION, PENDING_CTX_DIM, PENDING_KIND_IDX,
    V_MOVE, V_ATTACK, V_TACTIC, V_DECLINE, TACTIC_ACTION, DECLINE_ACTION,
)
from src.services.environment import roster
from src.services.environment.units import UNIT_CLASS_BY_ID
from src.services.environment.game_state import DECK
from src.services.bots.greedy_bot import GreedyBot
from src.services.bots.random_bot import RandomBot

CAV = 3  # Cavalry coin id (roster.py)


# --------------------------------------------------------------------------- #
# Schema: sizing, verb partition, encode/decode, remap
# --------------------------------------------------------------------------- #

def test_action_space_grew_by_tactic_and_decline():
    assert TACTIC_VERB == 30                      # just past the 16 deploy verbs
    assert N_VERBS == 31
    assert SPATIAL_SIZE == N_VERBS * BOARD_DIM * BOARD_DIM  # 1519
    # face-down = claim(C) + pass(C) + recruit(16*C) + decline(1)
    assert FACEDOWN_SIZE == 2 * N_COIN_TYPES + 16 * N_COIN_TYPES + 1  # 307
    assert ACTION_SPACE_SIZE == SPATIAL_SIZE + FACEDOWN_SIZE          # 1826
    assert DECLINE_ACTION_ID == ACTION_SPACE_SIZE - 1                 # the very last id
    assert OBS_VERSION == 5


def test_global_dim_includes_pending_context():
    assert PENDING_CTX_DIM == 3  # none + 2 cavalry kinds
    assert GLOBAL_DIM == 7 * N_COIN_TYPES + 3 * roster.NUM_UNIT_TYPES + 7 + PENDING_CTX_DIM


def test_verb_partition_has_tactic_and_decline():
    assert N_FACTORED_VERBS == 10
    assert VERB_OF_ACTION.shape == (ACTION_SPACE_SIZE,)
    assert set(np.unique(VERB_OF_ACTION)).issubset(set(range(N_FACTORED_VERBS)))
    assert VERB_OF_ACTION[WarChestEnv.encode_action(TACTIC_VERB, 3, 3)] == V_TACTIC
    assert VERB_OF_ACTION[DECLINE_ACTION_ID] == V_DECLINE
    # tactic follow-ups reuse the ordinary move/attack verbs
    assert VERB_OF_ACTION[WarChestEnv.encode_action(2, 3, 3)] == V_MOVE
    assert VERB_OF_ACTION[WarChestEnv.encode_action(7, 3, 3)] == V_ATTACK


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


def test_roster_only_cavalry_has_a_tactic_so_far():
    assert roster.UNIT_BY_ID[CAV].tactic == 'cavalry'
    others = [u for u in roster.UNIT_TYPES if u.id != CAV]
    assert all(u.tactic is None for u in others)


# --------------------------------------------------------------------------- #
# Cavalry scenario: a deterministic board for the move-then-attack tactic.
#   A=(3,3) cavalry (P1)   B=(3,4) free move target   C=(2,4) enemy adjacent to B
# --------------------------------------------------------------------------- #

A, B, C = (3, 3), (3, 4), (2, 4)
MOVE_DIR_A_TO_B = 2   # offsets[2] = (0, +1):  (3,3) -> (3,4)
ATK_DIR_B_TO_C = 1    # offsets[1] = (-1, 0):  (3,4) -> (2,4)


def _cavalry_scenario():
    env = WarChestEnv()
    env.reset()
    s = env.state
    s.compositions = {1: (CAV,), 2: (1,)}
    s.active_player = 1
    s.initiative_owner = 1
    s.pending = None
    s.hands = {1: Counter({CAV: 1}), 2: Counter()}
    # Non-empty bags so the round restart after the tactic redraws a playable hand.
    s.bags = {1: Counter({CAV: 1}), 2: Counter({1: 1})}
    s.discard_faceup = {1: Counter(), 2: Counter()}
    s.discard_facedown = {1: Counter(), 2: Counter()}
    s.supply = {1: Counter(), 2: Counter()}
    s.boxed = {1: Counter(), 2: Counter()}

    env.board.units = []
    cav = UNIT_CLASS_BY_ID[CAV](player_id=1, board=env.board)
    cav.place_on_board(A)
    env.board.units.append(cav)
    enemy = UNIT_CLASS_BY_ID[1](player_id=2, board=env.board)
    enemy.place_on_board(C)
    enemy.stack = 1
    env.board.units.append(enemy)
    return env, cav, enemy


def test_cavalry_tactic_full_move_then_attack():
    env, cav, enemy = _cavalry_scenario()
    tac = WarChestEnv.encode_action(TACTIC_VERB, *A)
    assert tac in env.get_possible_actions()

    # 1) Initiate: coin paid face-up, pending opened, turn does NOT pass.
    env.step(tac)
    assert env.state.hands[1][CAV] == 0
    assert env.state.discard_faceup[1][CAV] == 1
    assert env.active_player == 1
    assert env.state.pending is not None and env.state.pending.kind == 'cavalry_move'

    # 2) Move step: only move-dirs from A, no decline (mandatory).
    cont = env.get_possible_actions()
    assert DECLINE_ACTION_ID not in cont
    assert all(0 <= WarChestEnv.decode_action(a)[0] <= 5 for a in cont)
    move = WarChestEnv.encode_action(MOVE_DIR_A_TO_B, *A)
    assert move in cont
    env.step(move)
    assert env.board.get_unit_at(*B) is cav and env.board.get_unit_at(*A) is None
    assert env.active_player == 1
    assert env.state.pending.kind == 'cavalry_attack' and env.state.pending.unit_loc == B

    # 3) Attack step: optional (decline offered), and the adjacent enemy is targetable.
    cont = env.get_possible_actions()
    assert DECLINE_ACTION_ID in cont
    atk = WarChestEnv.encode_action(6 + ATK_DIR_B_TO_C, *B)
    assert atk in cont
    env.step(atk)
    # enemy (stack 1) removed → boxed; pending cleared → turn passed on.
    assert env.board.get_unit_at(*C) is None
    assert env.state.boxed[2][1] == 1
    assert env.state.pending is None


def test_cavalry_attack_step_can_be_declined():
    env, cav, enemy = _cavalry_scenario()
    env.step(WarChestEnv.encode_action(TACTIC_VERB, *A))
    env.step(WarChestEnv.encode_action(MOVE_DIR_A_TO_B, *A))
    assert env.state.pending.kind == 'cavalry_attack'
    _, _, _, _, info = env.step(DECLINE_ACTION_ID)
    assert info['action'].is_valid
    assert env.state.pending is None
    assert env.board.get_unit_at(*C) is enemy  # untouched


def test_cavalry_move_step_is_mandatory():
    env, cav, enemy = _cavalry_scenario()
    env.step(WarChestEnv.encode_action(TACTIC_VERB, *A))
    assert DECLINE_ACTION_ID not in env.get_possible_actions()
    _, _, _, _, info = env.step(DECLINE_ACTION_ID)
    assert not info['action'].is_valid          # cannot skip the mandatory move
    assert env.state.pending.kind == 'cavalry_move'  # state unchanged


def test_tactic_blocked_when_no_coin_in_hand():
    env, cav, enemy = _cavalry_scenario()
    env.state.hands[1] = Counter()  # drop the Cavalry coin
    assert WarChestEnv.encode_action(TACTIC_VERB, *A) not in env.get_possible_actions()


def test_pending_context_onehot_tracks_the_subturn():
    env, cav, enemy = _cavalry_scenario()
    ctx = env.generate_observation()['global'][-PENDING_CTX_DIM:]
    assert ctx[0] == 1.0 and ctx[1:].sum() == 0.0     # normal play

    env.step(WarChestEnv.encode_action(TACTIC_VERB, *A))
    ctx = env.generate_observation()['global'][-PENDING_CTX_DIM:]
    assert ctx[0] == 0.0
    assert ctx[1 + PENDING_KIND_IDX['cavalry_move']] == 1.0

    env.step(WarChestEnv.encode_action(MOVE_DIR_A_TO_B, *A))
    ctx = env.generate_observation()['global'][-PENDING_CTX_DIM:]
    assert ctx[1 + PENDING_KIND_IDX['cavalry_attack']] == 1.0


def test_turn_does_not_pass_mid_tactic():
    env, cav, enemy = _cavalry_scenario()
    env.step(WarChestEnv.encode_action(TACTIC_VERB, *A))
    assert env.active_player == 1
    env.step(WarChestEnv.encode_action(MOVE_DIR_A_TO_B, *A))
    assert env.active_player == 1  # still mid-tactic
    # mask only ever offers continuations while pending is set
    for a in env.get_possible_actions():
        assert a == DECLINE_ACTION_ID or a < SPATIAL_SIZE


def test_p2_cavalry_tactic_via_remap():
    """The whole flow must also work in P2's rotated frame (as the trainer drives it)."""
    env, cav, enemy = _cavalry_scenario()
    # Flip ownership: cavalry is P2's, enemy is P1's; P2 to act.
    cav.player_id = 2
    enemy.player_id = 1
    env.state.active_player = 2
    env.state.compositions = {1: (1,), 2: (CAV,)}
    env.state.hands = {1: Counter(), 2: Counter({CAV: 1})}

    s = BOARD_DIM - 1
    def rot(loc):
        return (s - loc[0], s - loc[1])

    obs = env.generate_observation()
    mask = obs['valid_action_mask']
    # The tactic id appears in the mask at the *rotated* cell.
    tac_abs = WarChestEnv.encode_action(TACTIC_VERB, *A)
    tac_ego = WarChestEnv.remap_action(tac_abs)
    assert mask[tac_ego] == 1
    env.step(WarChestEnv.remap_action(tac_ego))  # trainer remaps ego→absolute before step
    assert env.state.pending.kind == 'cavalry_move'

    move_abs = WarChestEnv.encode_action(MOVE_DIR_A_TO_B, *A)
    assert env.generate_observation()['valid_action_mask'][WarChestEnv.remap_action(move_abs)] == 1
    env.step(move_abs)
    assert env.board.get_unit_at(*B) is cav
    atk_abs = WarChestEnv.encode_action(6 + ATK_DIR_B_TO_C, *B)
    env.step(atk_abs)
    assert env.board.get_unit_at(*C) is None  # enemy removed


# --------------------------------------------------------------------------- #
# Regression: random play + bots + nets under the new schema
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
                live = _zone_plus_board(env, pid)
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


def test_both_bots_return_legal_actions_under_phase4():
    env = WarChestEnv()
    obs, _ = env.reset()
    for bot in (RandomBot(), GreedyBot()):
        for _ in range(40):
            a, _, _ = bot.act(obs)
            assert obs['valid_action_mask'][a] == 1
            env_a = WarChestEnv.remap_action(a) if env.active_player == 2 else a
            obs, _, t, tr, info = env.step(env_a)
            assert info['action'].is_valid
            if t or tr:
                obs, _ = env.reset()


def test_policy_and_critic_forward_under_phase4_schema():
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
