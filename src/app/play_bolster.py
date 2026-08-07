"""Play Warchest yourself against BolsterBot (the Berserker/Priest bolster archetype).

Same interactive matplotlib UI as `src/app/play.py`, but the opponent is `BolsterBot`
and its draft is *forced* to contain the key units (Berserker + Warrior Priest) so you
actually face the archetype — see docs/independent_opponents.md. You are player 1 by
default; the bot is player 2 with the forced composition.

    python src/app/play_bolster.py                       # bot @0.3s/move, builds Berserker to stack 2
    python src/app/play_bolster.py --time-budget 0.5 --build-target 3
    python src/app/play_bolster.py --human-player 2       # you take the other side
    python src/app/play_bolster.py --key-units 8          # bot forced Berserker only (no Priest)

Requires a GUI (matplotlib TkAgg). Finished games are saved under data/games/.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.services.environment.interactive_renderer import PlayRenderer
from src.services.environment.warchest_env import WarChestEnv
from src.services.bots.bolster_bot import BolsterBot, KEY_UNIT_IDS
from src.services.environment.roster import UNIT_BY_ID


def main():
    ap = argparse.ArgumentParser(description='Play against BolsterBot (forced Berserker/Priest draft).')
    ap.add_argument('--build-target', type=int, default=3, help='Stack height the bot bolsters key units to.')
    ap.add_argument('--human-player', type=int, default=1, choices=[1, 2], help='Which side you play.')
    ap.add_argument('--key-units', type=int, nargs='+', default=list(KEY_UNIT_IDS),
                    help='Unit ids forced into the BOT draft (default: Berserker 8 + Warrior Priest 16).')
    ap.add_argument('--save-dir', type=str, default='data/games')
    args = ap.parse_args()

    bot_pid = 3 - args.human_player

    # Force the bot's composition, then hand the freshly-drafted env to the renderer.
    env = WarChestEnv(save_game_history=True)
    env.set_init_state(force_units={bot_pid: list(args.key_units)})

    bot = BolsterBot(build_target=args.build_target, name='BolsterBot')

    forced = ', '.join(UNIT_BY_ID[u].name for u in args.key_units)
    print(f'You are player {args.human_player}; BolsterBot is player {bot_pid}.')
    print(f'Bot forced to draft: {forced}  (scripted, instant; build_target={args.build_target})')
    print(f'Bot full composition: {[UNIT_BY_ID[u].name for u in env.state.compositions[bot_pid]]}')
    print(f'Your composition:     {[UNIT_BY_ID[u].name for u in env.state.compositions[args.human_player]]}')

    PlayRenderer(
        env, human_player=args.human_player, opponent=bot, critic=None,
        player_labels={args.human_player: 'You', bot_pid: 'BolsterBot'},
        save_dir=args.save_dir,
    )


if __name__ == '__main__':
    main()
