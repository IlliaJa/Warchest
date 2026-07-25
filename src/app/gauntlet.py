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
import logging
import os
import re
import sys

# Make `import src...` work when run as `python src/app/gauntlet.py` from the root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch

from src.services.gauntlet import round_robin, build_agent
from src.services.gauntlet_parallel import round_robin_parallel

CRITIC_GLOB = 'data/lookahead_critic/lookahead_critic_v*.pth'
POLICY_GLOB = 'data/warchest_ppo_*.pth'


def _latest_policy_path():
    """Newest `data/warchest_ppo_*.pth`, or None if none exist. The timestamped
    filenames sort chronologically as plain strings, so the lexicographic max is
    the most recent run (mirrors `_latest_critic_path`; used for policy_critic's
    move-prior policy when --policy-critic-policy is not given).
    """
    candidates = glob.glob(POLICY_GLOB)
    return max(candidates) if candidates else None


def _latest_critic_path():
    """Highest-numbered `lookahead_critic_v{N}.pth` under `data/lookahead_critic/`,
    or None if no such checkpoint exists (gauntlet always plays the newest critic,
    never a version pinned in code).
    """
    candidates = glob.glob(CRITIC_GLOB)
    if not candidates:
        return None

    def version(path):
        m = re.search(r'_v(\d+)\.pth$', os.path.basename(path))
        return int(m.group(1)) if m else -1

    return max(candidates, key=version)


def _critic_agent_name(path):
    """Short, column-header-safe display name for a lookahead_critic built from `path`.

    The report truncates matrix column headers to 6 chars, so distinct critics must
    differ within the first 6. `.../lookahead_critic_v3.pth` -> `lac_v3`; anything without
    a `_v{N}` suffix falls back to the (truncated) stem so two arbitrary paths still differ.
    """
    m = re.search(r'_v(\d+)\.pth$', os.path.basename(path))
    if m:
        return f'lac_v{m.group(1)}'
    return 'lac_' + os.path.splitext(os.path.basename(path))[0][:6]


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
    if 'greedy_fast' in args.bots:
        specs.append({'kind': 'greedy_fast', 'name': 'greedy_fast'})
    if 'random' in args.bots:
        specs.append({'kind': 'random', 'name': 'random'})
    if 'lookahead' in args.bots:
        specs.append({'kind': 'lookahead', 'name': 'lookahead', 'kwargs': {
            'time_budget': args.lookahead_time_budget,
            'max_branching': args.lookahead_max_branching,
            'see_opponent_hand': not args.lookahead_blind,
        }})
    if 'lookahead_critic' in args.bots:
        # One lookahead_critic per --lookahead-critic-checkpoints path (each named by its
        # critic version, e.g. lac_v2 / lac_v3, so several critics can be compared head to
        # head in one field). Falls back to the single newest critic when none are given.
        # Depends on a critic checkpoint that may not exist in every environment (e.g. a
        # fresh checkout with no training run yet) — skip with a warning rather than crash.
        crit_paths = args.lookahead_critic_checkpoints
        if not crit_paths:
            latest = _latest_critic_path()
            crit_paths = [latest] if latest else []
        if not crit_paths:
            print(f'  ! skipping lookahead_critic: no checkpoint matching {CRITIC_GLOB}')
        for critic_path in crit_paths:
            # A single critic keeps the plain 'lookahead_critic' name (back-compat with the
            # old single-bot default); multiple get versioned lac_v* names to stay distinct.
            name = 'lookahead_critic' if len(crit_paths) == 1 else _critic_agent_name(critic_path)
            specs.append({'kind': 'lookahead_critic', 'name': name, 'kwargs': {
                'critic_path': critic_path,
                'beam_width': args.lookahead_critic_beam_width,
                'max_branching': args.lookahead_critic_max_branching,
                'time_budget': args.lookahead_critic_time_budget,
                'see_opponent_hand': not args.lookahead_critic_blind,
            }})
    if 'policy_critic' in args.bots:
        # Needs BOTH a critic checkpoint (for scoring) and a policy checkpoint (for the
        # move prior); skip with a warning if either is missing rather than crash.
        critic_path = _latest_critic_path()
        policy_path = args.policy_critic_policy or _latest_policy_path()
        if critic_path is None:
            print(f'  ! skipping policy_critic: no critic checkpoint matching {CRITIC_GLOB}')
        elif policy_path is None:
            print(f'  ! skipping policy_critic: no policy checkpoint matching {POLICY_GLOB}')
        else:
            specs.append({'kind': 'policy_critic', 'name': 'policy_critic', 'kwargs': {
                'critic_path': critic_path,
                'policy_path': policy_path,
                'beam_width': args.lookahead_critic_beam_width,
                'max_branching': args.lookahead_critic_max_branching,
                'time_budget': args.lookahead_critic_time_budget,
                'see_opponent_hand': not args.lookahead_critic_blind,
            }})
    if 'round_critic' in args.bots:
        # Same checkpoint requirements as policy_critic (it IS a PolicyCriticBot,
        # just round-bounded); reuses the shared --lookahead-critic-* search knobs.
        critic_path = _latest_critic_path()
        policy_path = args.policy_critic_policy or _latest_policy_path()
        if critic_path is None:
            print(f'  ! skipping round_critic: no critic checkpoint matching {CRITIC_GLOB}')
        elif policy_path is None:
            print(f'  ! skipping round_critic: no policy checkpoint matching {POLICY_GLOB}')
        else:
            specs.append({'kind': 'round_critic', 'name': 'round_critic', 'kwargs': {
                'critic_path': critic_path,
                'policy_path': policy_path,
                'beam_width': args.lookahead_critic_beam_width,
                'max_branching': args.lookahead_critic_max_branching,
                # No time_budget: round_critic searches each round to its end
                # (unbounded by default; max_depth is the only backstop).
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
                path = spec.get('kwargs', {}).get('critic_path')
                where = f' (checkpoint: {path})' if path else ''
                print(f"  ! skipping {spec.get('name', spec['kind'])}{where}: {e}")
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
    # 'warchest' is the logger name every bot/ppo.py logs to (LookaheadCriticBot's
    # per-move/aggregate search stats among them — see lookahead_critic_bot.py). No
    # handler is configured anywhere else in this CLI's path, so without this,
    # nothing above WARNING ever prints. Only covers this (main) process directly —
    # the default parallel path (n_workers > 1) runs every actual game in spawned
    # worker processes that don't inherit this, so gauntlet_parallel.py's
    # `_worker_loop` configures its own copy too.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                         datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(description='Warchest round-robin gauntlet.')
    parser.add_argument('--bots', nargs='+',
                        default=['policy', 'greedy', 'lookahead', 'lookahead_critic', 'policy_critic'],
                        choices=['policy', 'greedy', 'greedy_fast', 'random', 'lookahead',
                                 'lookahead_critic', 'policy_critic', 'round_critic'],
                        help='Participant kinds to include in the field. Default: '
                             'policy greedy lookahead lookahead_critic policy_critic. "policy" '
                             'loads checkpoints per --checkpoints (or the data/*.pth glob); '
                             'lookahead_critic, policy_critic and round_critic are skipped with '
                             'a warning if their required checkpoint is missing. round_critic is '
                             'available but off by default (add it explicitly to compare).')
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
    parser.add_argument('--lookahead-critic-time-budget', type=float, default=0.1,
                        help='Per-move search budget in seconds, for LookaheadCriticBot '
                             '(higher than LookaheadBot\'s default: the critic\'s forward '
                             'pass costs much more per node than a hand-crafted heuristic).')
    parser.add_argument('--lookahead-critic-blind', action='store_true',
                        help="LookaheadCriticBot doesn't read the opponent's real hand "
                             "(fair mode).")
    parser.add_argument('--lookahead-critic-checkpoints', nargs='*', default=None,
                        help='Critic .pth paths, one lookahead_critic agent each (named by '
                             'version, e.g. lac_v2 / lac_v3) — used when "lookahead_critic" is '
                             'in --bots. Defaults to the single newest '
                             'data/lookahead_critic/lookahead_critic_v*.pth.')
    parser.add_argument('--policy-critic-policy', default=None,
                        help='Policy .pth whose actor supplies policy_critic\'s move prior. '
                             'Defaults to the newest data/warchest_ppo_*.pth. The beam width, '
                             'branching cap, time budget and blind flag are shared with the '
                             '--lookahead-critic-* options (both bots run the same search).')
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
