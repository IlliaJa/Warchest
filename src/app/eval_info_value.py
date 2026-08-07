"""Measure the value of hidden information — the ceiling on every §6 proposal.

Every search bot in this repo defaults to `see_opponent_hand=True`: it reads the
opponent's real hand instead of modelling it (docs/search_under_uncertainty.md §2.2).
The fair mode (`see_opponent_hand=False`) replaces that with one uniformly random
re-split of the opponent's hand+bag — the weakest possible belief model. Every
proposal in §6 (belief filter, IS-MCTS, CFR/ReBeL) is ultimately an attempt to close
the gap between those two modes, so **the size of that gap is the ceiling on what any
of them can buy**. This measures it before any of them get built.

Two independent readings come out of one field:

  direct    cheat vs blind, head to head. WR ~ 0.50 => hidden information is worth
            nothing to this search and §6.1c-6.3 are not worth building; WR ~ 0.75
            => it is the dominant lever in the project.

  anchored  cheat vs ref and blind vs ref, played on the SAME seed block with the
            SAME colors, so the two arms share drafts and opening draws (common
            random numbers). The per-game paired difference has a far tighter
            interval than two independent win rates, and unlike the direct arm it
            cannot be distorted by a cheat-vs-blind specific matchup effect (a
            cheating searcher may be unusually good at punishing exactly the kind of
            mistake its blind twin makes).

`--pimc K` adds a third subject variant — blind, but averaging over K
determinizations — which splits the gap into the part that is pure
single-determinization *variance* (recoverable by plain PIMC, §6.1a) and the part
that is genuinely *missing information* (only a belief model recovers that, §6.1c).
If PIMC recovers most of the gap, belief modelling is not the bottleneck.

Start with `greedy_sim`: it carries no wall-clock budget and no nets, so a few
hundred games run in minutes, and its 2-ply reply model consumes the opponent's hand
directly — the flag is very much live for it. Then confirm on the bot you actually
care about.

    python src/app/eval_info_value.py --bot greedy_sim --games 200
    python src/app/eval_info_value.py --bot lookahead --games 120
    python src/app/eval_info_value.py --bot puct --games 60 --pimc 4
"""
import argparse
import logging
import math
import os
import sys
import textwrap
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch

from src.app.gauntlet import _latest_critic_path, _latest_policy_path, CRITIC_GLOB, POLICY_GLOB
from src.services.gauntlet import round_robin, build_agent
from src.services.gauntlet_parallel import round_robin_parallel

# Subject kinds this harness can build both variants of. `greedy_sim`/`lookahead` need
# no checkpoint; the rest need a critic (and the last two a policy as well).
NEEDS_CRITIC = ('lookahead_critic', 'policy_critic', 'puct')
NEEDS_POLICY = ('policy_critic', 'puct')
# Kinds that accept `n_determinizations` (LookaheadCriticBot and its subclasses).
# LookaheadBot/SimGreedyBot have no such parameter — they search once, full stop.
SUPPORTS_PIMC = NEEDS_CRITIC


# --------------------------------------------------------------------------- #
# Field construction
# --------------------------------------------------------------------------- #
def _subject_spec(args, *, name, blind, n_determinizations=1):
    """One subject variant: the bot under test, with hand visibility (and optionally
    the determinization count) as the ONLY thing that differs between variants.

    Everything else — time budget, branching, checkpoints — is shared, which is what
    makes the resulting win rate attributable to the information alone.
    """
    kind = args.bot
    kwargs = {'see_opponent_hand': not blind}

    if kind == 'lookahead':
        kwargs['time_budget'] = args.time_budget
        kwargs['max_branching'] = args.max_branching
    elif kind in NEEDS_CRITIC:
        kwargs['critic_path'] = args.critic_path
        if kind in NEEDS_POLICY:
            kwargs['policy_path'] = args.policy_path
        if kind == 'puct':
            kwargs['time_budget'] = args.time_budget
            kwargs['max_branching'] = args.max_branching
            kwargs['c_puct'] = args.puct_c
        else:
            kwargs['time_budget'] = args.time_budget
            kwargs['max_branching'] = args.max_branching
            kwargs['beam_width'] = args.beam_width
        kwargs['stats_log_every'] = 0
    if n_determinizations > 1:
        kwargs['n_determinizations'] = n_determinizations
    return {'kind': kind, 'name': name, 'kwargs': kwargs}


def _ref_spec(args):
    """The neutral anchor both subject variants are measured against.

    Must be a *different* bot from the subject: if it were the same kind with the same
    settings it would be a copy of the `cheat` arm, the anchored comparison would
    collapse into the direct one, and its 0.50 mirror-match row would carry no
    information at all.
    """
    kind = args.ref
    if kind == 'greedy_sim':
        return {'kind': 'greedy_sim', 'name': 'ref', 'kwargs': {}}
    if kind == 'greedy_fast':
        return {'kind': 'greedy_fast', 'name': 'ref'}
    if kind == 'random':
        return {'kind': 'random', 'name': 'ref'}
    if kind == 'lookahead':
        return {'kind': 'lookahead', 'name': 'ref', 'kwargs': {
            'time_budget': args.ref_time_budget,
            'max_branching': args.max_branching,
        }}
    raise SystemExit(f'unknown --ref {kind!r}')


def _build_tasks(pairs, *, k_games, seed):
    """`(i, j, game_seed, p1_is_i)` tasks where every pair draws from ONE shared seed block.

    `gauntlet.build_task_list` advances the seed across pairs, so no two pairs ever
    play the same game — right for a round-robin, wrong here. This instead gives game
    `g` of *every* pair the same seed and the same color assignment, so `cheat vs ref`
    game `g` and `blind vs ref` game `g` start from an identical draft and identical
    opening draws. Composition luck is the largest variance source in this game
    (docs/IDEAS.md #R1), and pairing removes it from the difference outright.

    Note the pairing only holds until the two arms diverge — the moment the subjects
    pick different moves the games are different games. The shared part (draft +
    opening) is still by far the biggest common factor, and the bots' own
    determinization shuffles use `random`, not `numpy`, so they never perturb the
    env's draw stream (`LookaheadBot._shuffled`).
    """
    tasks = []
    for g in range(k_games):
        for i, j in pairs:
            tasks.append((i, j, seed + g, g % 2 == 0))
    return tasks


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _subject_score(entry, subject):
    """Score in [0, 1] for agent index `subject` in one raw result entry (draw = 0.5)."""
    i, j, _seed, p1_is_i, res = entry
    if res == 0:
        return 0.5
    winner = i if ((res == 1) == p1_is_i) else j
    return 1.0 if winner == subject else 0.0


def _arm_scores(results, subject, opponent):
    """`{game_seed: subject score}` over every game between `subject` and `opponent`."""
    pair = {subject, opponent}
    return {e[2]: _subject_score(e, subject) for e in results if {e[0], e[1]} == pair}


def _wilson(score, n, z=1.96):
    """Wilson score interval for a win rate. Draws enter `score` as 0.5, which the
    interval treats as half a win — an approximation (the binomial variance is
    slightly overstated when draws are common), noted in the report when it matters.
    """
    if n == 0:
        return (float('nan'), float('nan'))
    p = score / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n)) / d
    return (center - half, center + half)


def _elo(p):
    """Win rate -> Elo difference. Clamped so a clean sweep reports a bound, not inf."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return 400.0 * math.log10(p / (1 - p))


def _paired(scores_a, scores_b):
    """Mean and standard error of the per-seed difference `a - b`, over shared seeds."""
    seeds = sorted(set(scores_a) & set(scores_b))
    if not seeds:
        return 0.0, float('nan'), 0
    d = np.array([scores_a[s] - scores_b[s] for s in seeds])
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float('nan')
    return float(d.mean()), se, len(d)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fmt_arm(label, score, n):
    lo, hi = _wilson(score, n)
    p = score / n if n else float('nan')
    return f'  {label:<26} WR={p:.3f}  [{lo:.3f}, {hi:.3f}]  ({score:.1f}/{n})'


def _verdict(p, lo, hi):
    """Turn the measured direct-arm win rate into the decision the experiment exists for."""
    if lo <= 0.5 <= hi:
        return ('NOT SIGNIFICANT — the interval still contains 0.5. Either hidden information '
                'is worth little to this search, or there are too few games; re-run with more '
                '--games before concluding anything.')
    if p < 0.55:
        return ('SMALL (<0.55) — the opponent\'s hand barely helps this search. Belief modelling '
                '(§6.1c) and the CFR/ReBeL track (§6.3) are not where the strength is; spend the '
                'effort on depth/search quality instead.')
    if p < 0.65:
        return ('MODERATE (0.55-0.65) — worth the cheap fixes (exact Bayesian belief §6.1c, the '
                'face-down leak §6.4-4, belief-averaged critic features §6.4-5), but it does not '
                'on its own justify the CFR/ReBeL track.')
    return ('LARGE (>0.65) — hidden information is a dominant lever. The blind mode is badly '
            'handicapped by its uniform re-split, belief modelling should be the next work item, '
            'and the §6.3 track has a real case.')


def _report(args, names, results, idx, elapsed):
    cheat, blind, ref = idx['cheat'], idx['blind'], idx['ref']
    print('\n' + '=' * 78)
    print(f'INFORMATION VALUE — subject: {args.bot}, anchor: {args.ref}, '
          f'{args.games} games/arm, {elapsed / 60:.1f} min')
    print('=' * 78)

    direct = _arm_scores(results, cheat, blind)
    n_direct = len(direct)
    s_direct = sum(direct.values())
    p_direct = s_direct / n_direct if n_direct else float('nan')
    lo, hi = _wilson(s_direct, n_direct)

    print('\nDIRECT — cheat vs blind, head to head')
    print(_fmt_arm('cheat vs blind', s_direct, n_direct))
    print(f'  {"":<26} = {_elo(p_direct):+.0f} Elo for seeing the hand')

    if not args.no_anchor:
        c_ref = _arm_scores(results, cheat, ref)
        b_ref = _arm_scores(results, blind, ref)
        print(f'\nANCHORED — both variants vs "{args.ref}" on the same seeds (paired)')
        print(_fmt_arm('cheat vs ref', sum(c_ref.values()), len(c_ref)))
        print(_fmt_arm('blind vs ref', sum(b_ref.values()), len(b_ref)))
        mean_d, se_d, n_d = _paired(c_ref, b_ref)
        pc = sum(c_ref.values()) / len(c_ref) if c_ref else float('nan')
        pb = sum(b_ref.values()) / len(b_ref) if b_ref else float('nan')
        print(f'  {"paired difference":<26} {mean_d:+.3f} +- {1.96 * se_d:.3f} (95%), n={n_d}')
        print(f'  {"":<26} = {_elo(pc) - _elo(pb):+.0f} Elo')
        if not (math.isnan(se_d) or se_d == 0):
            sig = 'significant' if abs(mean_d) > 1.96 * se_d else 'NOT significant'
            print(f'  {"":<26} {sig} at 95%')

        if 'pimc' in idx:
            p_ref = _arm_scores(results, idx['pimc'], ref)
            pp = sum(p_ref.values()) / len(p_ref) if p_ref else float('nan')
            print(f'\nPIMC — blind with n_determinizations={args.pimc}, same anchor and seeds')
            print(_fmt_arm(f'blind_pimc{args.pimc} vs ref', sum(p_ref.values()), len(p_ref)))
            mean_p, se_p, n_p = _paired(p_ref, b_ref)
            print(f'  {"vs blind (paired)":<26} {mean_p:+.3f} +- {1.96 * se_p:.3f} (95%), n={n_p}')
            # The variance/information split is a RATIO, so it is only defined when its
            # denominator — the cheat-vs-blind gap — is itself resolved. On an
            # unresolved gap the ratio is noise over noise: a gap of -0.030 and a PIMC
            # delta of -0.020 print as "PIMC recovered 0.67 of the gap" when both
            # numbers are indistinguishable from zero. Gate on significance rather than
            # on `gap != 0`.
            gap = pc - pb
            if math.isnan(se_d) or se_d == 0 or abs(gap) <= 1.96 * se_d:
                print(f'  {"variance/information split":<26} undefined — the cheat-vs-blind '
                      f'gap ({gap:+.3f}) is')
                print(f'  {"":<26} not resolved at 95%, so there is no gap to '
                      f'apportion.')
                print(f'  {"":<26} Raise --games until it resolves, or read this as '
                      f'"no gap".')
            else:
                frac = (pp - pb) / gap
                print(f'  {"recovered fraction of gap":<26} {frac:.2f}')
                print(f'  {"":<26} -> {max(0.0, 1 - frac):.2f} of the gap is missing '
                      f'INFORMATION (needs a belief model, §6.1c);')
                print(f'  {"":<26}    {min(1.0, max(0.0, frac)):.2f} is single-determinization '
                      f'VARIANCE (plain PIMC recovers it, §6.1a)')

    print('\nVERDICT (direct arm)')
    print(textwrap.fill(_verdict(p_direct, lo, hi), width=76,
                        initial_indent='  ', subsequent_indent='  '))
    print('=' * 78)

    print('\nPaste into docs/bots.md:')
    anchor_txt = ''
    if not args.no_anchor:
        c_ref = _arm_scores(results, cheat, ref)
        b_ref = _arm_scores(results, blind, ref)
        pc = sum(c_ref.values()) / len(c_ref) if c_ref else float('nan')
        pb = sum(b_ref.values()) / len(b_ref) if b_ref else float('nan')
        anchor_txt = f', vs {args.ref}: cheat {pc:.3f} / blind {pb:.3f} (Delta {pc - pb:+.3f})'
    print(f'  {args.bot}: cheat vs blind WR={p_direct:.3f} [{lo:.3f}, {hi:.3f}] '
          f'(n={n_direct}){anchor_txt}')


# --------------------------------------------------------------------------- #
def main():
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    ap = argparse.ArgumentParser(
        description='Measure how much the opponent\'s hidden hand is worth to a search bot.')
    ap.add_argument('--bot', default='greedy_sim',
                    choices=['greedy_sim', 'lookahead', 'lookahead_critic', 'policy_critic', 'puct'],
                    help='Subject bot. Two variants are built: cheating and blind. '
                         'Default greedy_sim (no time budget, no nets — fastest first read).')
    ap.add_argument('--ref', default='greedy_sim',
                    choices=['greedy_sim', 'greedy_fast', 'random', 'lookahead'],
                    help='Neutral anchor for the paired arm. Auto-switched to lookahead if it '
                         'would collide with --bot. Default greedy_sim.')
    ap.add_argument('--games', type=int, default=100,
                    help='Games per arm. Total games = 2x this without the anchor, 3x with it '
                         '(4x with --pimc). Default 100.')
    ap.add_argument('--no-anchor', action='store_true',
                    help='Direct arm only (cheat vs blind) — a third of the games, but no '
                         'paired cross-check.')
    ap.add_argument('--pimc', type=int, default=0,
                    help='Add a third variant: blind with this many determinizations, to split '
                         'the gap into variance vs missing information. '
                         f'Only for {", ".join(SUPPORTS_PIMC)}. 0 = off.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--time-budget', type=float, default=0.1,
                    help='Per-move search budget for the subject, in seconds (unused by '
                         'greedy_sim). Default 0.1.')
    ap.add_argument('--ref-time-budget', type=float, default=0.05,
                    help='Per-move budget for a --ref lookahead anchor. Default 0.05.')
    ap.add_argument('--max-branching', type=int, default=8)
    ap.add_argument('--beam-width', type=int, default=5,
                    help='Beam width for lookahead_critic / policy_critic.')
    ap.add_argument('--puct-c', type=float, default=1.5)
    ap.add_argument('--critic-path', default=None,
                    help=f'Critic checkpoint. Default: newest {CRITIC_GLOB}.')
    ap.add_argument('--policy-path', default=None,
                    help=f'Policy checkpoint. Default: newest {POLICY_GLOB}.')
    ap.add_argument('--n-workers', type=int, default=min(os.cpu_count() or 4, 8),
                    help='Parallel worker processes. Wall-clock-budgeted bots lose effective '
                         'search depth under oversubscription, so keep this at or below the '
                         'physical core count. Default min(cpu_count, 8).')
    ap.add_argument('--sequential', action='store_true', help='Shorthand for --n-workers 1.')
    args = ap.parse_args()
    if args.sequential:
        args.n_workers = 1

    # A same-kind anchor would just be a copy of the cheating arm (see `_ref_spec`).
    if args.ref == args.bot:
        args.ref = 'lookahead' if args.bot != 'lookahead' else 'greedy_sim'
        print(f'! --ref collided with --bot; using "{args.ref}" as the anchor instead')
    if args.pimc and args.bot not in SUPPORTS_PIMC:
        raise SystemExit(f'--pimc needs a bot that accepts n_determinizations '
                         f'({", ".join(SUPPORTS_PIMC)}); {args.bot} searches once.')
    # The variance/information split is defined against the anchor: all three variants
    # must be measured on the same neutral opponent for the recovered fraction to mean
    # anything. A direct pimc-vs-cheat match would not decompose the gap.
    if args.pimc and args.no_anchor:
        raise SystemExit('--pimc needs the anchored arm; drop --no-anchor.')
    if args.bot in NEEDS_CRITIC and args.critic_path is None:
        args.critic_path = _latest_critic_path()
        if args.critic_path is None:
            raise SystemExit(f'{args.bot} needs a critic; nothing matches {CRITIC_GLOB}')
    if args.bot in NEEDS_POLICY and args.policy_path is None:
        args.policy_path = _latest_policy_path()
        if args.policy_path is None:
            raise SystemExit(f'{args.bot} needs a policy; nothing matches {POLICY_GLOB}')

    specs = [
        _subject_spec(args, name='cheat', blind=False),
        _subject_spec(args, name='blind', blind=True),
    ]
    idx = {'cheat': 0, 'blind': 1}
    pairs = [(0, 1)]
    if not args.no_anchor:
        idx['ref'] = len(specs)
        specs.append(_ref_spec(args))
        pairs += [(idx['cheat'], idx['ref']), (idx['blind'], idx['ref'])]
        if args.pimc:
            idx['pimc'] = len(specs)
            specs.append(_subject_spec(args, name=f'blind_pimc{args.pimc}', blind=True,
                                       n_determinizations=args.pimc))
            pairs.append((idx['pimc'], idx['ref']))
    idx.setdefault('ref', idx['blind'])  # --no-anchor: the report only reads the direct arm

    names = [s['name'] for s in specs]
    tasks = _build_tasks(pairs, k_games=args.games, seed=args.seed)

    device = torch.device('cpu') if args.n_workers > 1 else torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    # Build the field once up front so a bad checkpoint fails here, not N times inside
    # freshly spawned workers.
    agents = [build_agent(spec, device=device) for spec in specs]

    print(f'Subject: {args.bot}  |  field: {", ".join(names)}  |  device: {device}')
    print(f'{len(tasks)} games total ({args.games}/arm, {len(pairs)} arms), '
          f'{args.n_workers} worker(s)')

    t0 = time.perf_counter()
    if args.n_workers <= 1:
        out = round_robin(agents, tasks=tasks)
    else:
        out = round_robin_parallel(specs, names, n_workers=args.n_workers, tasks=tasks)
    _report(args, names, out['results'], idx, time.perf_counter() - t0)


if __name__ == '__main__':
    main()
