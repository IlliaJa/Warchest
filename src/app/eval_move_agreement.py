"""Do the cheating and blind teachers actually pick DIFFERENT moves?
(docs/search_under_uncertainty.md §8.2)

`eval_info_value.py` showed the two variants are equally *strong*. That does not clear
the expert-iteration concern in §6.4-3, because ExIt does not distill strength — it
distills the search's **visit distribution** over moves. Two searches can be equally
strong while choosing differently, and if they do, the cheating teacher's targets encode
choices the student (which never sees the opponent's hand) cannot derive from its own
observation. Equal win rates and unlearnable targets are perfectly compatible.

So this measures the thing ExIt actually consumes. Both sides of a self-play game are
driven by the *cheating* `PuctBot` — the exact configuration
`src/app/expert_iteration.py` and `src/services/selfplay_collector.py` use for data
generation — and at every one of its plies the blind variant is additionally run on the
same position. Neither search mutates the env, so both see an identical state.

Reported per position:

  top-1 agreement   how often the most-visited action (what `_select_final` returns,
                    and what the argmax of the distilled target will be) coincides
  TV distance       total-variation distance between the two normalised visit
                    distributions — the honest measure, since the ExIt policy loss is
                    fitted to the whole distribution, not just its argmax
  vs prior          how often each variant agrees with the raw policy argmax. Context:
                    if the search barely leaves its own prior, ExIt has little to teach
                    regardless of hands, which is a separate (and larger) problem

    python src/app/eval_move_agreement.py --games 12
    python src/app/eval_move_agreement.py --games 20 --time-budget 0.1 --sample-every 1
"""
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch

from src.app.gauntlet import _latest_critic_path, _latest_policy_path, CRITIC_GLOB, POLICY_GLOB
from src.services.bots.puct_bot import PuctBot
from src.services.environment.warchest_env import WarChestEnv


def _tv(p, q):
    """Total-variation distance between two {action: weight} distributions.

    Both come from `PuctBot._combine_visit_counts`, already normalised to sum 1, but the
    key sets differ whenever the two searches kept different `max_branching` children —
    so the union is taken and missing keys count as 0.
    """
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def _top1(dist):
    return max(dist, key=dist.get) if dist else None


def _play(cheat, blind, seed, sample_every, max_turns=2000):
    """One self-play game driven by `cheat` on both sides; probe `blind` alongside.

    Returns a list of per-position records. `cheat` drives because that is what ExIt
    data-gen does — probing a trajectory the blind bot helped steer would measure a
    distribution ExIt never sees.
    """
    env = WarChestEnv(save_game_history=False)
    np.random.seed(seed)
    env.reset()
    rows = []
    for ply in range(max_turns):
        legal = env.get_possible_actions()
        probe = len(legal) > 1 and ply % sample_every == 0
        action = cheat.act(env)
        if probe:
            c_vis = dict(cheat.last_stats.get('visit_counts') or {})
            c_prior = cheat.last_stats.get('policy_argmax')
            blind.act(env)
            b_vis = dict(blind.last_stats.get('visit_counts') or {})
            b_prior = blind.last_stats.get('policy_argmax')
            if c_vis and b_vis:
                rows.append({
                    'round': env.state.round_number,
                    'n_legal': len(legal),
                    'agree': _top1(c_vis) == _top1(b_vis),
                    'tv': _tv(c_vis, b_vis),
                    'cheat_vs_prior': c_prior is not None and _top1(c_vis) == c_prior,
                    'blind_vs_prior': b_prior is not None and _top1(b_vis) == b_prior,
                })
        _, _, terminated, truncated, info = env.step(action)
        if not info['action'].is_valid:
            _, _, terminated, truncated, info = env.make_random_step()
        if terminated or truncated:
            break
    return rows


def _summarise(label, rows):
    if not rows:
        print(f'  {label:<22} (no positions)')
        return
    agree = np.mean([r['agree'] for r in rows])
    tv = np.mean([r['tv'] for r in rows])
    print(f'  {label:<22} n={len(rows):<5} top-1 agreement={agree:6.1%}   mean TV={tv:.3f}')


def main():
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(
        description='Measure how much the cheating and blind PuctBot teachers disagree on moves.')
    ap.add_argument('--games', type=int, default=12)
    ap.add_argument('--sample-every', type=int, default=1,
                    help='Probe every N-th ply (1 = every ply). Default 1.')
    ap.add_argument('--time-budget', type=float, default=0.1,
                    help="Per-move search budget for both variants. Match ExIt's. Default 0.1.")
    ap.add_argument('--max-branching', type=int, default=8)
    ap.add_argument('--c-puct', type=float, default=1.5)
    ap.add_argument('--value-mode', default='shaped', choices=['shaped', 'outcome'],
                    help="ExIt data-gen uses 'outcome'; the gauntlet bot uses 'shaped'.")
    ap.add_argument('--critic-path', default=None)
    ap.add_argument('--policy-path', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    critic_path = args.critic_path or _latest_critic_path()
    policy_path = args.policy_path or _latest_policy_path()
    if critic_path is None:
        raise SystemExit(f'no critic checkpoint matching {CRITIC_GLOB}')
    if policy_path is None:
        raise SystemExit(f'no policy checkpoint matching {POLICY_GLOB}')

    def build(see):
        return PuctBot(policy_path=policy_path, critic_path=critic_path,
                       value_mode=args.value_mode, c_puct=args.c_puct,
                       max_branching=args.max_branching, time_budget=args.time_budget,
                       see_opponent_hand=see, stats_log_every=0, device=args.device,
                       name='cheat' if see else 'blind')

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    cheat, blind = build(True), build(False)
    print(f'policy: {policy_path}\ncritic: {critic_path}\n'
          f'{args.games} self-play games driven by the CHEATING teacher '
          f'(budget {args.time_budget}s, value_mode={args.value_mode})')

    rows = []
    t0 = time.perf_counter()
    for g in range(args.games):
        rows += _play(cheat, blind, args.seed + g, args.sample_every)
        print(f'  [{g + 1}/{args.games}] {len(rows)} positions, '
              f'{time.perf_counter() - t0:.0f}s', end='\r')
    print(' ' * 70, end='\r')

    if not rows:
        raise SystemExit('no positions probed')
    print(f'\nMOVE AGREEMENT — cheat vs blind on identical positions '
          f'({time.perf_counter() - t0:.0f}s)')
    _summarise('overall', rows)
    mid = np.median([r['round'] for r in rows])
    _summarise(f'early (round<={mid:.0f})', [r for r in rows if r['round'] <= mid])
    _summarise(f'late (round>{mid:.0f})', [r for r in rows if r['round'] > mid])
    # Wide positions are where a hand read has room to matter; a forced position agrees
    # trivially and would otherwise inflate the overall rate.
    _summarise('wide (>=8 legal)', [r for r in rows if r['n_legal'] >= 8])

    cp = np.mean([r['cheat_vs_prior'] for r in rows])
    bp = np.mean([r['blind_vs_prior'] for r in rows])
    print(f'\nSEARCH vs ITS OWN PRIOR (how much the tree moves off the policy at all)')
    print(f'  cheat agrees with policy argmax  {cp:6.1%}')
    print(f'  blind agrees with policy argmax  {bp:6.1%}')

    agree = np.mean([r['agree'] for r in rows])
    tv = np.mean([r['tv'] for r in rows])
    print('\nHOW TO READ')
    if agree > 0.9 and tv < 0.15:
        print(f'  Agreement {agree:.1%} / TV {tv:.3f} — the two teachers produce nearly the same')
        print('  targets, so §6.4-3 (ExIt distilling an unreachable target) is NOT a real')
        print('  problem and needs no fix. Combined with the null in §8, the hidden-hand')
        print('  track is closed.')
    else:
        print(f'  Agreement {agree:.1%} / TV {tv:.3f} — the teachers pick materially different')
        print('  moves at equal strength. That is exactly the §6.4-3 failure mode: the')
        print('  cheating teacher\'s targets carry choices the student cannot derive from its')
        print('  own observation. Re-run ExIt data-gen with see_opponent_hand=False and')
        print('  compare the distillation loss / policy-search agreement curves.')
    print(f'  Independently: if BOTH variants agree with the raw policy argmax {max(cp, bp):.0%}')
    print('  of the time, the search is barely improving on its prior and ExIt has little')
    print('  to teach regardless of hands.')


if __name__ == '__main__':
    main()
