"""Play out a BolsterBot vs LookaheadBot game and render it with replay controls.

Forces BolsterBot's draft to contain the key units (Berserker + Warrior Priest by
default) — same forcing eval_bolster.py uses — then plays the game and opens the
same replay renderer demo.py uses (env.render_game).

    python src/app/demo_bolster.py
    python src/app/demo_bolster.py --bolster-player 2 --time-budget 0.3
    python src/app/demo_bolster.py --key-units 8              # berserker-only
    python src/app/demo_bolster.py --seed 7
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np

from src.services.environment.warchest_env import WarChestEnv
from src.services.gauntlet import bolster_agent, lookahead_agent
from src.services.bots.bolster_bot import KEY_UNIT_IDS


def main():
    ap = argparse.ArgumentParser(description='Replay BolsterBot vs LookaheadBot.')
    ap.add_argument('--bolster-player', type=int, default=1, choices=[1, 2],
                    help='Which side BolsterBot plays (default: 1).')
    ap.add_argument('--build-target', type=int, default=3, help='Stack height BolsterBot bolsters key units to.')
    ap.add_argument('--time-budget', type=float, default=0.3, help='LookaheadBot per-move search budget.')
    ap.add_argument('--max-branching', type=int, default=8, help='LookaheadBot branching cap.')
    ap.add_argument('--key-units', type=int, nargs='+', default=list(KEY_UNIT_IDS),
                    help='Unit ids forced into the BolsterBot draft (default: Berserker 8 + Priest 16).')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-turns', type=int, default=2000)
    args = ap.parse_args()

    bolster_pid = args.bolster_player
    look_pid = 3 - bolster_pid

    bot = bolster_agent('BolsterBot', build_target=args.build_target)
    look = lookahead_agent('LookaheadBot', time_budget=args.time_budget, max_branching=args.max_branching)
    agents = {bolster_pid: bot, look_pid: look}

    env = WarChestEnv(save_game_history=True)
    np.random.seed(args.seed)
    env.reset(options={'force_units': {bolster_pid: list(args.key_units)}})

    for _ in range(args.max_turns):
        pid = env.active_player
        action = agents[pid].act(env)
        _, _, terminated, truncated, info = env.step(action)
        if not info['action'].is_valid:
            _, _, terminated, truncated, info = env.make_random_step()
        if terminated:
            print(f'{agents[pid].name} (P{pid}) wins on turn {env.action_count}')
            break
        if truncated:
            print(f'Game truncated after {env.action_count} turns')
            break
    else:
        print(f'Game hit max_turns ({args.max_turns}) without a result')

    env.render_game(player_labels={bolster_pid: 'BolsterBot', look_pid: 'LookaheadBot'})


if __name__ == '__main__':
    main()
