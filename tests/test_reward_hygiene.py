"""Reward hygiene (docs/IDEAS.md L8): what a turn actually costs.

The tempo cost used to hang off every move-shaped result, so a turn that produced
several maneuvers paid it several times — taxing exactly the mechanics (tactics,
Berserker chains, Footman double maneuvers, the Swordsman bonus move) that exist to
buy extra maneuvers per coin. It now lands once, at the turn boundary. These tests
pin that "once", per mechanism, because the property is invisible in gameplay and
regresses silently: a future continuation branch that returns its own step penalty
would break it without breaking any behavioural test.
"""
from collections import Counter

from src.services.environment.warchest_env import (
    WarChestEnv, TACTIC_VERB, CONTROL_VERB, DECLINE_ACTION_ID,
    TURN_TEMPO_REWARD, ATTACK_REWARD, WIN_REWARD, INVALID_ACTION_REWARD,
    TYPICAL_MAIN_TURNS,
)
from _helpers import (
    blank_env, place, cavalry_scenario,
    SWORDSMAN, KNIGHT, CAV, BERSERKER, FOOTMAN,
    A, B, MOVE_DIR_A_TO_B, ATK_DIR_B_TO_C,
)

ATK2 = 6 + 2   # attack toward (3,4) from (3,3)
MV2 = 2        # move toward (3,4) from (3,3)


def turn_tempo(env, action_ids):
    """Total tempo charged while stepping `action_ids`, and the per-step breakdown."""
    charged = []
    for a in action_ids:
        _, _, _, _, info = env.step(a)
        assert info['action'].is_valid, info['action'].txt_result
        charged.append(info['action'].tempo_cost)
    return sum(charged), charged


# --------------------------------------------------------------------------- #
# One turn, one charge — whatever the turn is made of
# --------------------------------------------------------------------------- #

def test_plain_move_pays_the_tempo_cost_once():
    env = blank_env(active=1)
    env.state.compositions = {1: (KNIGHT,), 2: (CAV,)}
    env.state.hands[1] = Counter({KNIGHT: 1})
    env.state.bags = {1: Counter({KNIGHT: 1}), 2: Counter({CAV: 1})}
    place(env, KNIGHT, 1, (3, 3), stack=1)

    total, per_step = turn_tempo(env, [WarChestEnv.encode_action(MV2, 3, 3)])
    assert per_step == [TURN_TEMPO_REWARD]
    assert total == TURN_TEMPO_REWARD


def test_cavalry_tactic_pays_once_across_three_clicks():
    """Initiate + move + attack is one turn and one coin, so one tempo charge —
    landing on the click that actually ends the turn, not on the mid-tactic ones."""
    env, _, _ = cavalry_scenario()
    total, per_step = turn_tempo(env, [
        WarChestEnv.encode_action(TACTIC_VERB, *A),
        WarChestEnv.encode_action(MOVE_DIR_A_TO_B, *A),
        WarChestEnv.encode_action(6 + ATK_DIR_B_TO_C, *B),
    ])
    assert per_step == [0.0, 0.0, TURN_TEMPO_REWARD]
    assert total == TURN_TEMPO_REWARD


def test_footman_double_maneuver_pays_once_not_per_footman():
    env = blank_env(active=1)
    env.state.compositions = {1: (FOOTMAN,), 2: ()}
    env.state.hands[1] = Counter({FOOTMAN: 1})
    env.state.bags = {1: Counter({FOOTMAN: 1}), 2: Counter()}
    place(env, FOOTMAN, 1, (3, 3), stack=1)
    place(env, FOOTMAN, 1, (0, 1), stack=1)

    total, per_step = turn_tempo(env, [
        WarChestEnv.encode_action(TACTIC_VERB, 3, 3),
        WarChestEnv.encode_action(2, 3, 3),    # first Footman maneuvers
        WarChestEnv.encode_action(2, 0, 1),    # second Footman maneuvers, turn ends
    ])
    assert per_step == [0.0, 0.0, TURN_TEMPO_REWARD]
    assert total == TURN_TEMPO_REWARD


def test_berserker_chain_pays_once_regardless_of_chain_length():
    """Three maneuvers off one hand coin. The two stack-paid extras cost material,
    which material PBRS already prices — charging them tempo on top double-pays."""
    env = blank_env(active=1)
    env.state.compositions = {1: (BERSERKER,), 2: (KNIGHT,)}
    env.state.hands[1] = Counter({BERSERKER: 1})
    env.state.bags = {1: Counter({BERSERKER: 1}), 2: Counter({KNIGHT: 1})}
    place(env, BERSERKER, 1, (3, 3), stack=3)
    place(env, KNIGHT, 2, (3, 4), stack=1)

    total, per_step = turn_tempo(env, [
        WarChestEnv.encode_action(ATK2, 3, 3),   # hand-paid attack, opens the chain
        WarChestEnv.encode_action(MV2, 3, 3),    # stack-paid extra 1
        WarChestEnv.encode_action(MV2, 3, 4),    # stack-paid extra 2, chain exhausted
    ])
    assert per_step == [0.0, 0.0, TURN_TEMPO_REWARD]
    assert total == TURN_TEMPO_REWARD


def test_swordsman_bonus_move_is_free_and_declining_it_costs_the_same():
    """The free post-attack move must not cost more than declining it — that was the
    old behaviour's sharpest edge, since it made the bonus strictly worse than nothing."""
    def attack_then(second_action):
        env = blank_env(active=1)
        env.state.compositions = {1: (SWORDSMAN,), 2: (CAV,)}
        env.state.hands[1] = Counter({SWORDSMAN: 1})
        env.state.bags = {1: Counter({SWORDSMAN: 1}), 2: Counter({CAV: 1})}
        place(env, SWORDSMAN, 1, (3, 3), stack=1)
        place(env, CAV, 2, (3, 4), stack=1)
        rewards = []
        for a in [WarChestEnv.encode_action(ATK2, 3, 3), second_action]:
            _, r, _, _, info = env.step(a)
            assert info['action'].is_valid
            rewards.append(r)
        return sum(rewards)

    took_it = attack_then(WarChestEnv.encode_action(MV2, 3, 3))
    declined = attack_then(DECLINE_ACTION_ID)
    assert took_it == declined == TURN_TEMPO_REWARD


# --------------------------------------------------------------------------- #
# Where the charge must not land
# --------------------------------------------------------------------------- #

def test_winning_move_carries_no_tempo_cost():
    env = blank_env(active=1, initiative=1)
    env.state.compositions = {1: (SWORDSMAN,), 2: (SWORDSMAN,)}
    env.state.hands[1] = Counter({SWORDSMAN: 1})
    env.state.hands[2] = Counter({SWORDSMAN: 1})
    neutral = [(0, 1), (2, 2), (5, 3), (1, 3), (4, 4), (6, 5)]
    have = len(env.board.get_controlled_bases(1))
    for loc in neutral[:env.winning_base_count - 1 - have]:
        env.board.change_base_control(1, loc)
    target = neutral[env.winning_base_count - 1 - have]
    place(env, SWORDSMAN, 1, target, stack=1)

    _, reward, terminated, _, info = env.step(
        WarChestEnv.encode_action(CONTROL_VERB, *target))
    assert terminated
    assert info['action'].tempo_cost == 0.0
    assert reward == WIN_REWARD


def test_invalid_action_carries_no_tempo_cost():
    env = blank_env(active=1)
    env.state.compositions = {1: (KNIGHT,), 2: ()}
    env.state.hands[1] = Counter({KNIGHT: 1})
    place(env, KNIGHT, 1, (3, 3), stack=1)

    _, reward, _, _, info = env.step(WarChestEnv.encode_action(MV2, 0, 0))  # no unit there
    assert not info['action'].is_valid
    assert info['action'].tempo_cost == 0.0
    assert reward == INVALID_ACTION_REWARD


# --------------------------------------------------------------------------- #
# The other two hygiene items
# --------------------------------------------------------------------------- #

def test_attack_pays_no_bonus_beyond_the_tempo_cost():
    """ATTACK_REWARD is zeroed: material PBRS already pays the box-a-coin event."""
    assert ATTACK_REWARD == 0.0
    env = blank_env(active=1)
    env.state.compositions = {1: (KNIGHT,), 2: (CAV,)}
    env.state.hands[1] = Counter({KNIGHT: 1})
    env.state.bags = {1: Counter({KNIGHT: 1}), 2: Counter({CAV: 1})}
    place(env, KNIGHT, 1, (3, 3), stack=1)
    place(env, CAV, 2, (3, 4), stack=1)

    _, reward, _, _, info = env.step(WarChestEnv.encode_action(ATK2, 3, 3))
    assert info['action'].is_valid
    assert reward == TURN_TEMPO_REWARD


def test_holding_rate_is_sized_on_the_measured_turn_count():
    rate = WarChestEnv.default_holding_reward_rate()
    # Holding the largest sub-winning lead for a whole typical episode is worth the
    # intended 0.8 of a win — the property the old max_rounds-based divisor lost.
    accumulated = rate * (WarChestEnv.winning_base_count - 1) * TYPICAL_MAIN_TURNS
    assert abs(accumulated - 0.8 * WIN_REWARD) < 1e-12
    assert rate > 0
