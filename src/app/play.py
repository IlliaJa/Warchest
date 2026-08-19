import argparse
import glob
import os

import torch

from src.services.bots.puct_bot import PuctBot
from src.services.environment.interactive_renderer import PlayRenderer
from src.services.environment.obs_encoders import get_encoder
from src.services.environment.warchest_env import WarChestEnv
from src.services.gauntlet import PolicyAgent
from src.services.policy.checkpoint import load_critic_checkpoint, load_policy_checkpoint
from src.services.policy.policy import Critic, Policy

DEFAULT_CRITIC_PATH = 'data/lookahead_critic/lookahead_critic_v6.pth'


def _find_latest_model() -> str:
    candidates = sorted(glob.glob('data/warchest_ppo_*.pth'))
    if not candidates:
        raise FileNotFoundError('No models found in data/warchest_ppo_*.pth')
    return candidates[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play Warchest yourself against a trained model.')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to the opponent policy .pth. Defaults to the latest data/warchest_ppo_*.pth.')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Fallback hidden dim for legacy policy checkpoints without arch metadata.')
    parser.add_argument('--critic-path', type=str, default=DEFAULT_CRITIC_PATH,
                        help='Path to a critic .pth for the eval overlay. Pass "" to disable it.')
    parser.add_argument('--opp-type', type=str, default='pool', choices=['random', 'greedy', 'pool'],
                        help="Critic's opponent-identity input (a training-time label; "
                             "'pool' is the closest proxy for a human opponent).")
    parser.add_argument('--save-dir', type=str, default='data/games',
                        help='Directory finished games are saved to (see game_record.py).')
    parser.add_argument('--puct', action='store_true',
                        help='Play against PuctBot (search over --model-path/--critic-path) '
                             'instead of the raw policy.')
    parser.add_argument('--puct-time-budget', type=float, default=1.0,
                        help='PuctBot per-move search budget in seconds (--puct only).')
    parser.add_argument('--puct-max-branching', type=int, default=8,
                        help='PuctBot branching cap per node (--puct only).')
    parser.add_argument('--puct-c', type=float, default=1.5,
                        help='PuctBot PUCT exploration constant (--puct only).')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model_path = args.model_path or _find_latest_model()
    print(f'Loading opponent model: {model_path}')
    ckpt = load_policy_checkpoint(model_path, map_location=device, default_hidden_dim=args.hidden_dim)
    policy_encoder = get_encoder(ckpt['obs_version'])
    print(f"arch={ckpt['arch']} obs_version={ckpt['obs_version']} hidden_dim={ckpt['hidden_dim']}")

    policy = Policy(device=device, hidden_dim=ckpt['hidden_dim'], obs_encoder=policy_encoder,
                    arch=ckpt['arch']).to(device)
    policy.load_state_dict(ckpt['state_dict'])
    policy.eval()

    if args.puct:
        if not args.critic_path:
            raise SystemExit('--puct needs a critic (--critic-path); it is the leaf value the search uses.')
        opponent = PuctBot(
            policy_path=model_path, critic_path=args.critic_path,
            c_puct=args.puct_c, max_branching=args.puct_max_branching,
            time_budget=args.puct_time_budget, device=device,
            name=f'puct({os.path.basename(model_path)})',
        )
        print(f'Opponent: PuctBot, time_budget={args.puct_time_budget}s, '
              f'max_branching={args.puct_max_branching}, c_puct={args.puct_c}')
    else:
        opponent = PolicyAgent(os.path.basename(model_path), policy, policy_encoder)

    critic, critic_encoder = None, None
    value_scale, value_shift = 1.0, 0.0
    if args.critic_path:
        try:
            cmeta = load_critic_checkpoint(args.critic_path, map_location=device)
            critic_encoder = get_encoder(cmeta['obs_version'])
            critic = Critic(device=device, hidden_dim=cmeta['hidden_dim'], obs_encoder=critic_encoder,
                            arch=cmeta['arch']).to(device)
            critic.load_state_dict(cmeta['state_dict'])
            critic.eval()
            if cmeta['return_mean'] is not None and cmeta['return_std'] is not None:
                value_scale, value_shift = cmeta['return_std'], cmeta['return_mean']
            calibrated = value_scale != 1.0 or value_shift != 0.0
            print(f"Critic loaded: {args.critic_path} "
                  f"(obs_version={cmeta['obs_version']}, calibrated={calibrated})")
        except FileNotFoundError:
            print(f'No critic checkpoint at {args.critic_path!r} — playing without the eval overlay.')

    env = WarChestEnv(save_game_history=True, obs_encoder=policy_encoder)
    PlayRenderer(
        env, human_player=1, opponent=opponent, critic=critic, critic_encoder=critic_encoder,
        opp_type=args.opp_type, value_scale=value_scale, value_shift=value_shift,
        player_labels={1: 'You', 2: opponent.name}, save_dir=args.save_dir,
    )
