"""Round-robin gauntlet CLI (docs/next_steps.md Step 1).

Plays a fixed field of agents — trained checkpoints plus the GreedyBot baseline —
all-pairs with balanced colors, then prints the win-rate matrix, a Bradley-Terry
(Elo-scaled) ranking anchored to the field, and the intransitive-triple fraction
(rock-paper-scissors detector).

Examples:
    python src/app/gauntlet.py                         # all data/*.pth + baselines
    python src/app/gauntlet.py --checkpoints a.pth b.pth --k-games 40
    python src/app/gauntlet.py --bots lookahead lookahead_critic policy greedy
    python src/app/gauntlet.py --bots policy greedy    # drop the lookahead baselines
"""
import argparse
import glob
import os
import sys

# Make `import src...` work when run as `python src/app/gauntlet.py` from the root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch

from src.services.gauntlet import round_robin, build_agent
from src.services.gauntlet_parallel import round_robin_parallel
from src.services.bots.lookahead_critic_bot import DEFAULT_CRITIC_PATH


def _build_specs(args):
    """Turn CLI args into agent specs (picklable, order matches the field/report).

    A spec fully describes how to (re)build one agent via `gauntlet.build_agent` —
    used directly by the sequential path (built once, in-process) and shipped as-is
    to parallel workers (each worker rebuilds the whole field once from these specs;
    see `gauntlet_parallel.py` for why live agent objects can't just be pickled).

    `args.bots` selects which participant kinds enter the field; a kind not listed
    is simply skipped.
    """
    specs = []
    if 'policy' in args.bots:
        paths = args.checkpoints
        if paths is None:
            paths = sorted(glob.glob('data/warchest_ppo_*.pth'))
        for path in paths:
            specs.append({'kind': 'policy', 'path': path})
    if 'greedy' in args.bots:
        specs.append({'kind': 'greedy', 'name': 'greedy'})
    if 'random' in args.bots:
        specs.append({'kind': 'random', 'name': 'random'})
    if 'lookahead' in args.bots:
        specs.append({'kind': 'lookahead', 'name': 'lookahead', 'kwargs': {
            'time_budget': args.lookahead_time_budget,
            'max_branching': args.lookahead_max_branching,
            'see_opponent_hand': not args.lookahead_blind,
        }})
    if 'lookahead_critic' in args.bots:
        # Depends on a critic checkpoint that may not exist in every environment (e.g. a
        # fresh checkout with no training run yet) — skip with a warning rather than crash.
        if not os.path.exists(DEFAULT_CRITIC_PATH):
            print(f'  ! skipping lookahead_critic: checkpoint not found at '
                  f'{DEFAULT_CRITIC_PATH}')
        else:
            specs.append({'kind': 'lookahead_critic', 'name': 'lookahead_critic', 'kwargs': {
                'critic_path': DEFAULT_CRITIC_PATH,
                'beam_width': args.lookahead_critic_beam_width,
                'max_branching': args.lookahead_critic_max_branching,
                'time_budget': args.lookahead_critic_time_budget,
                'see_opponent_hand': not args.lookahead_critic_blind,
            }})
    return specs


def _validate_specs(specs, device):
    """Dry-run build every spec once, dropping any that fail (missing/incompatible
    checkpoint) with a warning — so parallel workers never hit the same failure N times.
    Returns `(kept_specs, agents)`: the built agents (on `device`) double as the
    sequential path's live field, and as the source of each agent's display name
    (a 'policy' spec's name is only known after a real build, from the checkpoint path).
    """
    kept_specs, agents = [], []
    for spec in specs:
        try:
            agent = build_agent(spec, device=device)
        except Exception as e:  # 'policy' failures are already reported by build_agent
            if spec['kind'] != 'policy':
                print(f"  ! skipping {spec.get('name', spec['kind'])}: {e}")
            continue
        if spec['kind'] == 'policy':
            spec = {**spec, 'name': agent.name}
        kept_specs.append(spec)
        agents.append(agent)
    return kept_specs, agents


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
    parser.add_argument('--bots', nargs='+', default=['policy', 'greedy', 'lookahead', 'lookahead_critic'],
                        choices=['policy', 'greedy', 'random', 'lookahead', 'lookahead_critic'],
                        help='Participant kinds to include in the field. Default: '
                             'policy greedy lookahead lookahead_critic. "policy" loads '
                             'checkpoints per --checkpoints (or the data/*.pth glob); '
                             'lookahead_critic is skipped with a warning if its checkpoint '
                             'is missing.')
    parser.add_argument('--checkpoints', nargs='*', default=None,
                        help='Policy .pth paths (used when "policy" is in --bots). '
                             'Defaults to data/warchest_ppo_*.pth.')
    parser.add_argument('--k-games', type=int, default=20,
                        help='Games per pair (colors balanced). Default 20.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lookahead-time-budget', type=float, default=0.1,
                        help='Per-move search budget in seconds, for LookaheadBot.')
    parser.add_argument('--lookahead-max-branching', type=int, default=8,
                        help='Branching cap per search node, for LookaheadBot.')
    parser.add_argument('--lookahead-blind', action='store_true',
                        help="LookaheadBot doesn't read the opponent's real hand (fair mode).")
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
    parser.add_argument('--n-workers', type=int, default=min(os.cpu_count() or 4, 8),
                        help='Parallel worker processes for game play. Default: '
                             'min(cpu_count, 8). 1 (or --sequential) plays in-process on '
                             'cuda if available (matching pre-parallel behavior exactly); >1 '
                             'always evaluates on CPU in every worker (mirrors '
                             'rollout_collector.py\'s convention), since broadcasting live GPU '
                             'modules to worker processes is the fragile case that convention '
                             'avoids. LookaheadBot/LookaheadCriticBot use a wall-clock search '
                             'budget, so heavy oversubscription (n-workers well above physical '
                             'cores) silently reduces their effective search depth per game — '
                             'cap at physical core count for serious lookahead evals.')
    parser.add_argument('--sequential', action='store_true',
                        help='Shorthand for --n-workers 1.')
    args = parser.parse_args()
    if args.sequential:
        args.n_workers = 1

    detected_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Gameplay device, not just what's auto-detected: parallel workers always force CPU
    # (see --n-workers help), so `detected_device` is only actually used when sequential.
    exec_device = detected_device if args.n_workers <= 1 else torch.device('cpu')
    if args.n_workers > 1 and detected_device.type != 'cpu':
        print(f'Using device: {exec_device} (auto-detected {detected_device} is unused — '
              f'{args.n_workers} parallel workers always run on CPU)')
    else:
        print(f'Using device: {exec_device}')

    specs = _build_specs(args)
    specs, agents = _validate_specs(specs, exec_device)
    names = [a.name for a in agents]

    if len(specs) < 2:
        raise SystemExit('Need at least 2 agents; pass --bots with more kinds (and '
                          '--checkpoints if using policy).')

    print(f'Field ({len(specs)}): ' + ', '.join(names))
    print(f'Playing {args.k_games} games/pair ...')
    if args.n_workers <= 1:
        out = round_robin(agents, k_games=args.k_games, seed=args.seed)
    else:
        print(f'Using {args.n_workers} parallel worker processes (CPU-only evaluation).')
        out = round_robin_parallel(specs, names, k_games=args.k_games, seed=args.seed,
                                    n_workers=args.n_workers)
    _print_report(out)


if __name__ == '__main__':
    main()
