import argparse
import glob

import torch

from src.services.policy.policy import Policy
from src.services.policy.checkpoint import load_policy_checkpoint
from src.services.environment.warchest_env import WarChestEnv
from src.services.environment.obs_encoders import get_encoder
from src.services.gauntlet import PolicyAgent, greedy_sim_agent, random_agent, lookahead_agent

AGENT_KINDS = ('policy', 'greedy_sim', 'random', 'lookahead')


def build_agent(kind, name, *, policy=None, encoder=None, lookahead_kwargs=None):
    """Construct a `GauntletAgent`-compatible player (`act(env)`) of the given kind."""
    if kind == 'policy':
        if policy is None:
            raise ValueError(f'{name}: a loaded policy is required (--model-path)')
        return PolicyAgent(name, policy, encoder)
    if kind == 'greedy_sim':
        return greedy_sim_agent(name, encoder)
    if kind == 'random':
        return random_agent(name, encoder)
    if kind == 'lookahead':
        return lookahead_agent(name, **(lookahead_kwargs or {}))
    raise ValueError(f'Unknown agent kind: {kind}')


def play_game(env, agent_p1, agent_p2, max_turns=2000):
    """Play one game between two `act(env)` agents and render it.

    Mirrors gauntlet.play_game's loop (illegal proposals fall back to a random
    legal move) but keeps history so the caller can render afterwards.
    """
    env.reset()
    agents = {1: agent_p1, 2: agent_p2}
    for _ in range(max_turns):
        pid = env.active_player
        action = agents[pid].act(env)
        _, _, terminated, truncated, info = env.step(action)
        if not info['action'].is_valid:
            _, _, terminated, truncated, info = env.make_random_step()
        if terminated:
            print(f'{agents[pid].name} (P{pid}) wins on turn {env.action_count}')
            return
        if truncated:
            print(f'Game truncated after {env.action_count} turns')
            return
    print(f'Game hit max_turns ({max_turns}) without a result')


def _find_latest_model() -> str:
    candidates = sorted(glob.glob('data/warchest_ppo_*.pth'))
    if not candidates:
        raise FileNotFoundError('No models found in data/warchest_ppo_*.pth')
    return candidates[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play out and render a Warchest game between two agents.')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to .pth file for the policy agent(s). Defaults to the latest data/warchest_ppo_*.pth.')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Fallback hidden dim for legacy checkpoints without arch metadata.')
    parser.add_argument('--p1', type=str, default='policy', choices=AGENT_KINDS,
                        help='Player 1 agent (default: policy).')
    parser.add_argument('--p2', type=str, default='greedy_sim', choices=AGENT_KINDS,
                        help='Player 2 agent (default: greedy_sim).')
    parser.add_argument('--lookahead-time-budget', type=float, default=0.5,
                        help='Per-move search budget in seconds, for lookahead agents.')
    parser.add_argument('--lookahead-max-branching', type=int, default=8,
                        help='Branching cap per search node, for lookahead agents.')
    parser.add_argument('--lookahead-blind', action='store_true',
                        help="Lookahead agent doesn't read the opponent's real hand (fair mode).")
    args = parser.parse_args()

    lookahead_kwargs = dict(
        time_budget=args.lookahead_time_budget,
        max_branching=args.lookahead_max_branching,
        see_opponent_hand=not args.lookahead_blind,
    )

    policy, encoder = None, None
    if 'policy' in (args.p1, args.p2):
        model_path = args.model_path or _find_latest_model()
        print(f'Loading model: {model_path}')

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)

        # Load with metadata so the net is built for the checkpoint's obs version and width
        # (falls back to CLI --hidden-dim for legacy checkpoints).
        ckpt = load_policy_checkpoint(model_path, map_location=device,
                                      default_hidden_dim=args.hidden_dim)
        encoder = get_encoder(ckpt['obs_version'])
        print(f"arch={ckpt['arch']} obs_version={ckpt['obs_version']} hidden_dim={ckpt['hidden_dim']}")

        policy = Policy(device=device, hidden_dim=ckpt['hidden_dim'], obs_encoder=encoder).to(device)
        policy.load_state_dict(ckpt['state_dict'])
        policy.eval()

    agent_p1 = build_agent(args.p1, args.p1.capitalize(), policy=policy, encoder=encoder,
                            lookahead_kwargs=lookahead_kwargs)
    agent_p2 = build_agent(args.p2, args.p2.capitalize(), policy=policy, encoder=encoder,
                            lookahead_kwargs=lookahead_kwargs)

    env = WarChestEnv(save_game_history=True, obs_encoder=encoder)
    play_game(env, agent_p1, agent_p2)
    env.render_game(player_labels={1: agent_p1.name, 2: agent_p2.name})
