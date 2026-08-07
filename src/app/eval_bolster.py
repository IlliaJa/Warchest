"""Evaluate BolsterBot vs a chosen opponent with a *forced* Berserker/Priest draft.

The bolster archetype is only defined when the bot holds a Berserker and/or a Warrior
Priest, and the env drafts randomly, so this harness pins those units into the bot's
composition via `reset(options={'force_units': ...})` — something the bot can't do for
itself (the draft happens before it acts). Plays `--games` games with balanced colors
(bot as P1 half, P2 half) and reports the bot's win rate plus a verb-usage breakdown, so
"does it actually bolster / use the Berserker chain?" is visible alongside the win rate.

    python src/app/eval_bolster.py --games 60 --opponent lookahead
    python src/app/eval_bolster.py --games 60 --opponent lookahead --key-units 8      # berserker-only
    python src/app/eval_bolster.py --games 60 --opponent greedy_sim
"""
import argparse
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np

from src.services.environment.warchest_env import WarChestEnv
from src.services.bots.bolster_bot import BolsterBot, KEY_UNIT_IDS
import glob
import torch

from src.services.gauntlet import (
    lookahead_agent, greedy_sim_agent, greedy_fast_agent, random_agent, checkpoint_agent,
)


def _build_opponent(name, *, time_budget, policy_path=None):
    if name == 'lookahead':
        return lookahead_agent('lookahead', time_budget=time_budget)
    if name == 'greedy_sim':
        return greedy_sim_agent('greedy_sim')
    if name == 'greedy_fast':
        return greedy_fast_agent('greedy_fast')
    if name == 'random':
        return random_agent('random')
    if name == 'policy':
        path = policy_path or max(glob.glob('data/warchest_ppo_*.pth'))
        agent = checkpoint_agent(path, torch.device('cpu'))
        if agent is None:
            raise SystemExit(f'could not load policy checkpoint {path!r}')
        return agent
    raise SystemExit(f'unknown opponent {name!r}')


def play_game(bot, opp, bot_pid, force_units, *, seed, max_turns=2000):
    """One game. `bot` plays as `bot_pid`; its composition is forced to `force_units`.
    Returns (result, turns) where result is 'win'/'loss'/'draw' from the bot's view.
    """
    env = WarChestEnv(save_game_history=False)
    np.random.seed(seed)
    env.reset(options={'force_units': {bot_pid: list(force_units)}})
    agents = {bot_pid: bot, 3 - bot_pid: opp}

    for t in range(max_turns):
        pid = env.active_player
        action = agents[pid].act(env)
        _, _, terminated, truncated, info = env.step(action)
        if not info['action'].is_valid:
            _, _, terminated, truncated, info = env.make_random_step()
        if terminated:
            winner = pid
            return ('win' if winner == bot_pid else 'loss'), t + 1
        if truncated:
            return 'draw', t + 1
    return 'draw', max_turns


def main():
    ap = argparse.ArgumentParser(description='Evaluate BolsterBot vs an opponent (forced draft).')
    ap.add_argument('--games', type=int, default=60)
    ap.add_argument('--opponent', default='lookahead',
                    choices=['lookahead', 'greedy_sim', 'greedy_fast', 'random', 'policy'])
    ap.add_argument('--policy-path', default=None, help='Checkpoint for --opponent policy (default: newest).')
    ap.add_argument('--time-budget', type=float, default=0.1, help='Opponent (lookahead) per-move budget.')
    ap.add_argument('--bot-time-budget', type=float, default=0.1, help='BolsterBot per-move search budget.')
    ap.add_argument('--key-units', type=int, nargs='+', default=list(KEY_UNIT_IDS),
                    help='Unit ids forced into the bot draft (default: Berserker 8 + Warrior Priest 16).')
    ap.add_argument('--build-target', type=int, default=4, help='Berserker stack the controller builds to.')
    ap.add_argument('--max-branching', type=int, default=8)
    ap.add_argument('--pure', action='store_true',
                    help='Disable the archetype controller (pure LookaheadBot + forced draft baseline).')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    bot = BolsterBot(build_target=args.build_target, time_budget=args.bot_time_budget,
                     max_branching=args.max_branching, archetype=not args.pure)
    opp = _build_opponent(args.opponent, time_budget=args.time_budget, policy_path=args.policy_path)

    results = Counter()
    turns = []
    t0 = time.perf_counter()
    for g in range(args.games):
        bot_pid = 1 if g % 2 == 0 else 2   # alternate colors
        res, n = play_game(bot, opp, bot_pid, args.key_units, seed=args.seed + g)
        results[res] += 1
        turns.append(n)
        if (g + 1) % 10 == 0:
            wr = results['win'] / (g + 1)
            print(f'  [{g + 1}/{args.games}] WR={wr:.3f}  ({dict(results)})')

    n = args.games
    wr = results['win'] / n
    print(f'\nBolsterBot(key={args.key_units}) vs {args.opponent}: '
          f'WR={wr:.3f}  W={results["win"]} L={results["loss"]} D={results["draw"]}  '
          f'(n={n}, {time.perf_counter() - t0:.1f}s, turns avg={np.mean(turns):.0f})')
    total = sum(bot.usage.values()) or 1
    usage = {k: f'{v / total:.2f}' for k, v in bot.usage.most_common()}
    print(f'bot verb usage: {usage}')


if __name__ == '__main__':
    main()
