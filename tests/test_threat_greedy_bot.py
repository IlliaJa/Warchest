"""ThreatAwareGreedyBot (docs/IDEAS.md B5): the obs-only bot that reads the threat
planes. Covers the ladder rungs that encode the item's claim — take a capture only
when it is free, run a unit that is already covered, refuse the trade the threat
planes cannot see (Pikeman's counter) — plus legality. The shared multi-source BFS
is pinned in test_greedy_bot_speed.py.
"""
from collections import Counter

import numpy as np

from src.services.environment.warchest_env import (
    WarChestEnv, BOARD_DIM, ATTACK_ACTION, MOVE_ACTION,
)
from src.services.environment.obs_encoders.v11 import (
    ENEMY_THREAT_PLANE_BASE, N_THREAT_KINDS,
)
from src.services.bots.threat_greedy_bot import ThreatAwareGreedyBot
from _helpers import blank_env, place, SWORDSMAN, KNIGHT, PIKEMAN


def _act(env, bot):
    """The bot's choice, translated back to the absolute env frame (as the gauntlet's
    HeuristicAgent does), plus the decoded action for assertions."""
    action, _, _ = bot.act(env.generate_observation())
    env_action = WarChestEnv.remap_action(action) if env.active_player == 2 else action
    return env_action, env.get_action_info(env_action)


def _duel(attacker_stack, defender_id, defender_stack, defender_composition_on_board=None):
    """P1 unit at (3,3) facing a P2 unit at (3,4), both sides holding their coin.

    `defender_composition_on_board` puts the defender's whole coin supply on the board,
    which zeroes the opponent's hidden pool — and with it every enemy threat plane, so
    a rung can be tested without the defender's own threat confounding it.
    """
    env = blank_env(active=1)
    env.state.compositions = {1: (SWORDSMAN,), 2: (defender_id,)}
    env.state.hands = {1: Counter({SWORDSMAN: 1}), 2: Counter()}
    place(env, SWORDSMAN, player=1, loc=(3, 3), stack=attacker_stack)
    place(env, defender_id, player=2, loc=(3, 4),
          stack=defender_composition_on_board or defender_stack)
    return env


def test_plays_a_full_legal_episode_from_both_seats():
    np.random.seed(0)
    env = WarChestEnv()
    obs, _ = env.reset()
    bot = ThreatAwareGreedyBot()
    for _ in range(2000):
        action, _, _ = bot.act(obs)
        assert obs['valid_action_mask'][action] == 1
        env_action = WarChestEnv.remap_action(action) if env.active_player == 2 else action
        obs, _, terminated, truncated, info = env.step(env_action)
        assert info['action'].is_valid
        if terminated or truncated:
            break
    else:
        raise AssertionError('episode did not terminate within the step budget')


def test_takes_the_free_capture():
    """Stack 2 against a lone enemy: the reply lands one hit, which does not kill the
    attacker, so the capture is free and outranks every other rung."""
    env = _duel(attacker_stack=2, defender_id=SWORDSMAN, defender_stack=1)
    action, (kind, args) = _act(env, ThreatAwareGreedyBot())
    assert kind == ATTACK_ACTION
    verb, r, q = args
    assert (r, q) == (3, 3)
    assert action in env.get_possible_actions()


def test_runs_a_unit_that_is_already_covered():
    """A Knight cannot be attacked by a stack-1 unit, so the swordsman's only answer to
    standing in its melee footprint is to leave it — and to leave it for a cell the
    Knight does not cover, not merely any legal square."""
    env = _duel(attacker_stack=1, defender_id=KNIGHT, defender_stack=1)
    obs = env.generate_observation()
    enemy_hits = obs['board'][ENEMY_THREAT_PLANE_BASE:ENEMY_THREAT_PLANE_BASE + N_THREAT_KINDS]
    assert enemy_hits[:, 3, 3].sum() > 0  # the unit really is hanging

    action, (kind, args) = _act(env, ThreatAwareGreedyBot())
    assert kind == MOVE_ACTION
    verb, r, q = args
    assert (r, q) == (3, 3)
    dest = (r + env.board.offsets[verb][0], q + env.board.offsets[verb][1])
    assert enemy_hits[:, dest[0], dest[1]].sum() == 0

    # The choice has to be informative: some legal destination is covered, so a
    # plane-blind bot could land on one.
    covered = [c for c in env.board.get_free_adjacent_cells(3, 3)
               if enemy_hits[:, c[0], c[1]].sum() > 0]
    assert covered


def test_takes_the_lethal_trade_even_when_answered():
    """Stack 1 against stack 1: the planes say the attacker's cell is covered (by the
    very unit it is about to remove), and reading them literally makes the bot retreat
    from every even trade — measured at -3.7 pp vs greedy_fast. A lethal blow is taken
    regardless."""
    env = _duel(attacker_stack=1, defender_id=SWORDSMAN, defender_stack=1)
    obs = env.generate_observation()
    enemy_hits = obs['board'][ENEMY_THREAT_PLANE_BASE:ENEMY_THREAT_PLANE_BASE + N_THREAT_KINDS]
    assert enemy_hits[:, 3, 3].sum() > 0  # the attacker is nominally hanging

    action, (kind, args) = _act(env, ThreatAwareGreedyBot())
    assert kind == ATTACK_ACTION
    assert args[1:] == (3, 3)


def test_does_not_walk_off_a_base_it_holds():
    """A parked unit is a lock (docs/IDEAS.md B3: move-blocking makes an occupied base
    unstealable), so the march rung moves anything else first."""
    env = blank_env(active=1)
    env.state.compositions = {1: (SWORDSMAN,), 2: ()}
    env.state.hands = {1: Counter({SWORDSMAN: 2}), 2: Counter()}
    base = env.board.get_controlled_bases(1)[0]
    place(env, SWORDSMAN, player=1, loc=base, stack=1)
    place(env, SWORDSMAN, player=1, loc=(3, 3), stack=1)

    action, (kind, args) = _act(env, ThreatAwareGreedyBot())
    assert kind == MOVE_ACTION
    assert args[1:] != base


def test_refuses_the_trade_the_planes_cannot_see():
    """Pikeman's counter removes a coin from an adjacent attacker even when the Pikeman
    dies, so a lone attacker trades itself for nothing. Threat planes model attacks, not
    counters, which is why this is a separate guard."""
    # The Pikeman's four coins are all on the board -> its hidden pool, and therefore
    # every enemy threat plane, is empty. Nothing but the counter rules the attack out.
    env = _duel(attacker_stack=1, defender_id=PIKEMAN, defender_stack=4,
                defender_composition_on_board=4)
    action, (kind, _) = _act(env, ThreatAwareGreedyBot())
    assert kind != ATTACK_ACTION

    # Same position with one more coin on the attacker: the counter no longer kills it,
    # and the attack rung fires.
    env = _duel(attacker_stack=2, defender_id=PIKEMAN, defender_stack=4,
                defender_composition_on_board=4)
    action, (kind, args) = _act(env, ThreatAwareGreedyBot())
    assert kind == ATTACK_ACTION
    assert args[1:] == (3, 3)
