"""Round-robin gauntlet CLI (docs/next_steps.md Step 1).

Plays a fixed field of agents — trained checkpoints plus the GreedyBot baseline —
all-pairs with balanced colors, then prints the win-rate matrix, a Bradley-Terry
(Elo-scaled) ranking anchored to the field, and the intransitive-triple fraction
(rock-paper-scissors detector).

Examples:
    python src/app/gauntlet.py                         # all data/*.pth + baselines
    python src/app/gauntlet.py --checkpoints a.pth b.pth --k-games 40
    python src/app/gauntlet.py --no-lookahead           # drop the LookaheadBot baseline
"""
import argparse
import glob
import os
import sys

# Make `import src...` work when run as `python src/app/gauntlet.py` from the root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch

from src.services.policy.policy import Policy
from src.services.policy.checkpoint import load_policy_checkpoint
from src.services.environment.obs_encoders import get_encoder
from src.services.gauntlet import (
    round_robin, greedy_agent, lookahead_agent, lookahead_critic_agent, PolicyAgent,
)
from src.services.bots.lookahead_critic_bot import DEFAULT_CRITIC_PATH


def _checkpoint_agent(path, device):
    """Build a PolicyAgent from a checkpoint, or None if it can't be reconstructed.

    A bare legacy checkpoint from an older architecture / obs version (different
    layer names or dims) is not loadable into the current code — resurrecting those
    is the subprocess/worktree path (docs/next_steps.md), out of scope here. We
    skip them with a warning so the gauntlet still runs on the loadable field.
    """
    try:
        meta = load_policy_checkpoint(path, map_location=device)
        encoder = get_encoder(meta['obs_version'])
        policy = Policy(device=device, hidden_dim=meta['hidden_dim'], obs_encoder=encoder).to(device)
        policy.load_state_dict(meta['state_dict'])
    except Exception as e:  # unreadable file, or incompatible arch/obs/dims
        reason = str(e).splitlines()[0] if str(e).strip() else type(e).__name__
        print(f'  ! skipping {os.path.basename(path)}: {reason}')
        return None
    policy.eval()
    name = os.path.splitext(os.path.basename(path))[0].replace('warchest_ppo_', 'ckpt_')
    return PolicyAgent(f'{name}[v{meta["obs_version"]}]', policy, encoder)


def _lookahead_critic_agent(args, device):
    """Build the LookaheadCriticBot baseline, or None if its checkpoint is missing.

    On by default (like LookaheadBot), but unlike LookaheadBot it depends on a
    critic checkpoint file that may not exist in every environment (e.g. a fresh
    checkout with no training run yet) — skip with a warning rather than crash.
    """
    if not os.path.exists(args.lookahead_critic_path):
        print(f'  ! skipping lookahead_critic: checkpoint not found at '
              f'{args.lookahead_critic_path}')
        return None
    return lookahead_critic_agent(
        'lookahead_critic',
        critic_path=args.lookahead_critic_path,
        beam_width=args.lookahead_critic_beam_width,
        max_branching=args.lookahead_critic_max_branching,
        time_budget=args.lookahead_critic_time_budget,
        see_opponent_hand=not args.lookahead_critic_blind,
        device=device,
    )


def _print_report(out):
    names = out['names']
    wr = out['win_rate']
    w = max(len(n) for n in names)

    print('\nWin-rate matrix (row vs column):')
    print(' ' * (w + 2) + '  '.join(f'{n[:6]:>6}' for n in names))
    for i, n in enumerate(names):
        cells = []
        for j in range(len(names)):
            cells.append('   -  ' if i == j else f'{wr[i, j]:6.2f}')
        print(f'{n:>{w}}  ' + '  '.join(cells))

    print('\nBradley-Terry ranking (Elo-scaled, field mean = 1000):')
    for n, r in sorted(out['ratings'].items(), key=lambda kv: -kv[1]):
        print(f'  {n:>{w}}  {r:7.1f}')

    frac = out['intransitive_fraction']
    print(f'\nIntransitive-triple fraction: {frac:.3f}'
          + ('  (cycles present — "beats predecessors" may be partly illusory)'
             if frac > 0 else '  (fully transitive field)'))


def main():
    parser = argparse.ArgumentParser(description='Warchest round-robin gauntlet.')
    parser.add_argument('--checkpoints', nargs='*', default=None,
                        help='Policy .pth paths. Defaults to data/warchest_ppo_*.pth.')
    parser.add_argument('--k-games', type=int, default=20,
                        help='Games per pair (colors balanced). Default 20.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-greedy', action='store_true', help='Drop the GreedyBot baseline.')
    parser.add_argument('--no-lookahead', action='store_true',
                        help='Drop the LookaheadBot baseline (on by default; its search '
                             'budget makes every game much slower than the other baselines).')
    parser.add_argument('--lookahead-time-budget', type=float, default=0.1,
                        help='Per-move search budget in seconds, for LookaheadBot.')
    parser.add_argument('--lookahead-max-branching', type=int, default=8,
                        help='Branching cap per search node, for LookaheadBot.')
    parser.add_argument('--lookahead-blind', action='store_true',
                        help="LookaheadBot doesn't read the opponent's real hand (fair mode).")
    parser.add_argument('--no-lookahead-critic', action='store_true',
                        help='Drop the LookaheadCriticBot baseline (on by default; skipped '
                             'automatically if its checkpoint is missing).')
    parser.add_argument('--lookahead-critic-path', default=DEFAULT_CRITIC_PATH,
                        help='Critic checkpoint path, for LookaheadCriticBot.')
    parser.add_argument('--lookahead-critic-beam-width', type=int, default=5,
                        help='Children kept per node, for LookaheadCriticBot.')
    parser.add_argument('--lookahead-critic-max-branching', type=int, default=8,
                        help='Raw legal-action cap per node before critic scoring, '
                             'for LookaheadCriticBot.')
    parser.add_argument('--lookahead-critic-time-budget', type=float, default=0.5,
                        help='Per-move search budget in seconds, for LookaheadCriticBot '
                             '(higher than LookaheadBot\'s default: the critic\'s forward '
                             'pass costs much more per node than a hand-crafted heuristic).')
    parser.add_argument('--lookahead-critic-blind', action='store_true',
                        help="LookaheadCriticBot doesn't read the opponent's real hand "
                             "(fair mode).")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    paths = args.checkpoints
    if paths is None:
        paths = sorted(glob.glob('data/warchest_ppo_*.pth'))
    agents = [a for a in (_checkpoint_agent(p, device) for p in paths) if a is not None]
    if not args.no_greedy:
        agents.append(greedy_agent('greedy'))
    if not args.no_lookahead:
        agents.append(lookahead_agent('lookahead', time_budget=args.lookahead_time_budget,
                                       max_branching=args.lookahead_max_branching,
                                       see_opponent_hand=not args.lookahead_blind))
    if not args.no_lookahead_critic:
        critic_agent = _lookahead_critic_agent(args, device)
        if critic_agent is not None:
            agents.append(critic_agent)

    if len(agents) < 2:
        raise SystemExit('Need at least 2 agents; pass --checkpoints or keep the baselines.')

    print(f'Field ({len(agents)}): ' + ', '.join(a.name for a in agents))
    print(f'Playing {args.k_games} games/pair ...')
    out = round_robin(agents, k_games=args.k_games, seed=args.seed)
    _print_report(out)


if __name__ == '__main__':
    main()
