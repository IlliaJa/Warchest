"""Does the randomised-coefficient evaluator family actually produce *different bots*?

`docs/IDEAS.md` B1 proposes sampling the 8-dim leaf-evaluator coefficient vector θ per
opponent instead of fixing it, on the claim that this yields a continuum of independent
playstyles — the archetype list `independent_opponents.md` Phase 1 wants hand-written,
generated instead. B1 also states the test, which this script runs:

    the family is working if bolster/recruit/tactic/initiative rates *span a wide range*
    across θ, and failing if every θ collapses onto the same profile.

**A range across θ is not evidence on its own**, which is the whole reason this script is
longer than a for-loop. Any 12-game sample of a stochastic bot produces a range; the
question is whether the θ-driven range is bigger than the range the *same* bot produces
across independent game blocks. So every run measures two things:

  * **treatment** — `--arms` sampled θ, each pinned for its whole block;
  * **control** — `--control-arms` blocks of the *default* θ (`LEGACY_THETA`, i.e. the
    evaluator every current bot already uses), differing only in which games they played.

and reports the spread of both. The headline number is the **spread ratio**: mean pairwise
total-variation distance between treatment verb profiles, over the same quantity for the
control. A ratio near 1 means θ changes nothing a re-seed doesn't; the family is only real
well above 1.

Two deliberate asymmetries, both conservative (they make the family look *worse*, so a
positive result is not an artefact of the design):

  * Treatment arms share common random numbers — arm i's game g and arm j's game g replay
    the same draft — so draft variance is differenced out of the treatment spread. Control
    blocks use disjoint seeds, so draft variance is left *in* the control spread.
  * A verb the bot is forced into is not counted: `SimGreedyBot.act` short-circuits when
    only one action is legal and never touches its `usage` counter. The profile is what
    the bot *chose*, not what the rules left it.

Win rate against `--opponent` is reported alongside, and summarised as an **arm health**
count. Per B1 win rate is explicitly not the gate — pool entrants are judged on coverage
and pressure, not Elo — but there is a floor below which an entrant stops being a
playstyle: a bot that passes two thirds of its turns is not "the turtle archetype", it is
a bot that hands the learning policy free wins and free advantage. Health is reported so
that trade-off is visible instead of assumed away in either direction.

`--sweep KEY` replaces the sampled arms with a ladder in one θ coordinate, everything else
at its default. That is the follow-up when the sampled run shows a dead arm: the arms of a
random draw differ in all eight coordinates at once, so they can show *that* some θ are
degenerate but never *which dial* did it.

    python src/app/eval_theta_family.py --arms 8 --games 12 --opponent greedy_sim
    python src/app/eval_theta_family.py --arms 8 --games 12 --opponent lookahead_critic
    python src/app/eval_theta_family.py --sweep durability --games 16    # one dial at a time
    python src/app/eval_theta_family.py --arms 8 --games 12 --opponent policy --dump out.json
"""
import argparse
import glob
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np

from src.services.bots.evaluation import (
    THETA_KEYS, LEGACY_THETA, sample_theta, theta_tag, format_theta,
)

# Fixed column order, so two runs' tables line up. Every value `SimGreedyBot._classify`
# can return; 'select' is a tactic continuation click rather than a turn-opening choice.
VERBS = ('move', 'attack', 'control', 'bolster', 'deploy', 'recruit',
         'tactic', 'claim_initiative', 'pass', 'decline', 'select')
# The four B1 names explicitly as the things that should move.
GATE_VERBS = ('bolster', 'recruit', 'tactic', 'claim_initiative')

# Arm-health thresholds. Not gates — an arm below them still counts toward the spread — but
# the count is reported, because an opponent-pool slice made mostly of these trades the
# coverage B1 wants for free wins and a distorted advantage group.
MIN_HEALTHY_WR = 0.10
MAX_HEALTHY_PASS = 0.25

CRITIC_GLOB = 'data/lookahead_critic/lookahead_critic_v*.pth'
POLICY_GLOB = 'data/warchest_ppo_*.pth'


# --------------------------------------------------------------------------- #
# Opponent
# --------------------------------------------------------------------------- #
def opponent_spec(name, *, time_budget, policy_path=None, critic_path=None):
    """A `gauntlet.build_agent` spec for `--opponent` — built inside each worker.

    Reuses the gauntlet's own agent factory rather than a private ladder, so an arm here
    faces exactly the bot the gauntlet calls by that name. Checkpoint paths are resolved
    in the parent and passed down, so every worker loads the same file even if a training
    run drops a newer one mid-measurement.
    """
    if name in ('greedy_sim', 'greedy_fast', 'random'):
        return {'kind': name, 'name': name}
    if name == 'lookahead':
        return {'kind': 'lookahead', 'name': 'lookahead',
                'kwargs': {'time_budget': time_budget}}
    if name == 'lookahead_critic':
        if critic_path is None:
            raise SystemExit(f'--opponent lookahead_critic needs a critic checkpoint '
                             f'(none matching {CRITIC_GLOB})')
        return {'kind': 'lookahead_critic', 'name': 'lookahead_critic',
                'kwargs': {'critic_path': critic_path, 'time_budget': time_budget,
                           'stats_log_every': 0}}
    if name == 'policy':
        if policy_path is None:
            raise SystemExit(f'--opponent policy needs a checkpoint (none matching {POLICY_GLOB})')
        return {'kind': 'policy', 'path': policy_path}
    raise SystemExit(f'unknown opponent {name!r}')


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
_W = {}


def _init_worker(root, spec, bot_kwargs, base):
    """Per-process setup: one opponent, built once, plus an empty θ->bot cache.

    Rebuilding either per game would dominate the measurement — a `LookaheadCriticBot`
    loads a checkpoint and calibrates its value scale on construction, and every bot here
    allocates its own forward-simulation `WarChestEnv`.
    """
    if root not in sys.path:
        sys.path.insert(0, root)
    import torch
    torch.set_num_threads(1)  # else N workers x intra-op threads oversubscribe cores
    from src.services.gauntlet import build_agent

    _W['opponent'] = build_agent(spec, device=torch.device('cpu'))
    _W['bot_kwargs'] = bot_kwargs
    _W['base'] = base
    _W['bots'] = {}


def _bot_for(theta):
    """Cached θ-family bot for this θ (one per worker per distinct θ).

    Which base search bot carries θ is `--base`: `greedy` is the 2-ply `SimGreedyBot`
    (~18-25 ms/move, the only one cheap enough for a rollout slice), `lookahead` is the
    alpha-beta search (~6x that, but a base bot that beats SimGreedyBot 0.79). The open
    question the second arm answers is whether θ still separates playstyles once more of
    the real env reward sits between root and leaf.
    """
    from src.services.bots.random_eval_bot import (
        RandomEvalBot, RandomEvalLookaheadBot, RandomEvalCriticBot,
    )

    key = tuple(theta[k] for k in THETA_KEYS)
    bot = _W['bots'].get(key)
    if bot is None:
        if _W['base'] == 'policy_theta':
            from src.services.bots.policy_theta_bot import PolicyThetaBot
            cls = PolicyThetaBot
        else:
            cls = {'lookahead': RandomEvalLookaheadBot, 'critic': RandomEvalCriticBot}.get(
                _W['base'], RandomEvalBot)
        bot = cls(theta=theta, **_W['bot_kwargs'])
        _W['bots'][key] = bot
    return bot


def _play(task):
    """Play one game. -> (arm, outcome, turns, usage dict, bot decisions, bot seconds).

    Returns the bot's *per-game* verb counts by clearing its counter first — the bot object
    is shared across this worker's games for the same θ.
    """
    from src.services.environment.warchest_env import WarChestEnv

    arm, theta, game_seed, bot_pid = task
    bot = _bot_for(theta)
    bot.reset_usage()
    opponent = _W['opponent']

    env = WarChestEnv(save_game_history=False)
    np.random.seed(game_seed)
    env.reset()
    for agent in (bot, opponent):
        hook = getattr(agent, 'new_episode', None)
        if hook is not None:
            hook()

    agents = {bot_pid: bot, 3 - bot_pid: opponent}
    bot_seconds = 0.0
    bot_decisions = 0
    outcome, turns = 'draw', 0
    for t in range(2000):
        pid = env.active_player
        t0 = time.perf_counter()
        action = agents[pid].act(env)
        if pid == bot_pid:
            bot_seconds += time.perf_counter() - t0
            bot_decisions += 1
        _, _, terminated, truncated, info = env.step(action)
        if not info['action'].is_valid:
            _, _, terminated, truncated, info = env.make_random_step()
        if terminated:
            outcome, turns = ('win' if pid == bot_pid else 'loss'), t + 1
            break
        if truncated:
            outcome, turns = 'draw', t + 1
            break
    else:
        turns = 2000
    return arm, outcome, turns, dict(bot.usage), bot_decisions, bot_seconds


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def profile(usage):
    """Verb counts -> a probability vector over `VERBS` (zeros if the arm never moved)."""
    total = sum(usage.get(v, 0) for v in VERBS)
    if total == 0:
        return np.zeros(len(VERBS))
    return np.array([usage.get(v, 0) / total for v in VERBS])


def mean_pairwise_tv(profiles):
    """Mean total-variation distance over all unordered pairs. 0 = identical playstyles.

    TV is the right summary here because a verb profile is a distribution and TV is its
    natural metric: it reads directly as "the fraction of moves one arm spends on verbs the
    other does not". Euclidean distance on the same vector would let one dominant verb
    (`move`, ~half of every profile) drown the four the gate is actually about.
    """
    if len(profiles) < 2:
        return float('nan')
    return float(np.mean([0.5 * np.abs(a - b).sum()
                          for a, b in itertools.combinations(profiles, 2)]))


def summarize(arms):
    """Per-arm rows -> the report's numbers. `arms` maps arm key -> aggregate dict."""
    profiles = {k: profile(v['usage']) for k, v in arms.items()}
    return {
        'profiles': profiles,
        'tv': mean_pairwise_tv(list(profiles.values())),
        'ranges': {verb: (min(p[i] for p in profiles.values()),
                          max(p[i] for p in profiles.values()))
                   for i, verb in enumerate(VERBS)},
    }


def health(arms, summary):
    """-> (unhealthy arm keys, reason string per key). See `MIN_HEALTHY_WR`."""
    pass_idx = VERBS.index('pass')
    bad = {}
    for key, a in arms.items():
        wr = a['wins'] / a['games'] if a['games'] else 0.0
        pass_rate = summary['profiles'][key][pass_idx]
        reasons = []
        if wr < MIN_HEALTHY_WR:
            reasons.append(f'WR {wr:.2f}')
        if pass_rate > MAX_HEALTHY_PASS:
            reasons.append(f'pass {pass_rate:.2f}')
        if reasons:
            bad[key] = ', '.join(reasons)
    return bad


def _print_table(title, arms, thetas, summary):
    print(f'\n{title}')
    head = f'{"arm":>10}  {"WR":>5}  {"turns":>5}  {"ms/mv":>6}  ' + '  '.join(
        f'{v[:6]:>6}' for v in VERBS)
    print(head)
    print('-' * len(head))
    for key in arms:
        a = arms[key]
        p = summary['profiles'][key]
        n = a['games']
        wr = a['wins'] / n if n else float('nan')
        ms = 1000.0 * a['bot_seconds'] / a['decisions'] if a['decisions'] else float('nan')
        print(f'{key:>10}  {wr:5.2f}  {a["turns"] / max(n, 1):5.0f}  {ms:6.1f}  '
              + '  '.join(f'{x:6.3f}' for x in p))
    lo = [summary['ranges'][v][0] for v in VERBS]
    hi = [summary['ranges'][v][1] for v in VERBS]
    print(f'{"min":>10}  {"":>5}  {"":>5}  {"":>6}  ' + '  '.join(f'{x:6.3f}' for x in lo))
    print(f'{"max":>10}  {"":>5}  {"":>5}  {"":>6}  ' + '  '.join(f'{x:6.3f}' for x in hi))
    print(f'{"range":>10}  {"":>5}  {"":>5}  {"":>6}  '
          + '  '.join(f'{h - l:6.3f}' for l, h in zip(lo, hi)))
    if thetas:
        print('\nθ per arm:')
        for key in arms:
            print(f'  {key:>10}  {format_theta(thetas[key])}')


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description='Measure the B1 θ-family behaviour spread.')
    ap.add_argument('--arms', type=int, default=8, help='Sampled θ arms (treatment).')
    ap.add_argument('--thetas', default=None,
                    help='JSON list of θ (or of search_theta.py --dump rows) to use as the '
                         'treatment arms instead of sampling. This is how a *selected* '
                         'family is verified: sampling re-draws from the prior and would '
                         'not reproduce the members that passed a strength bar.')
    ap.add_argument('--sweep', default=None, choices=list(THETA_KEYS),
                    help='Replace the sampled arms with a ladder in this single θ '
                         'coordinate (every other coordinate at its default). Answers '
                         '"what does this one dial do", which a random draw cannot.')
    ap.add_argument('--sweep-values', type=float, nargs='+',
                    default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0],
                    help='Ladder for --sweep.')
    ap.add_argument('--control-arms', type=int, default=4,
                    help='Default-θ blocks used as the noise floor. 0 disables the control '
                         '(and with it the only reason to believe the treatment spread).')
    ap.add_argument('--games', type=int, default=12, help='Games per arm, colors balanced.')
    ap.add_argument('--opponent', default='greedy_sim',
                    choices=['greedy_sim', 'greedy_fast', 'random', 'lookahead',
                             'lookahead_critic', 'policy'])
    ap.add_argument('--time-budget', type=float, default=0.1,
                    help='Per-move search budget for a search opponent.')
    ap.add_argument('--policy-path', default=None, help='Checkpoint for --opponent policy.')
    ap.add_argument('--critic-path', default=None,
                    help='Critic checkpoint for --opponent lookahead_critic.')
    ap.add_argument('--base', default='greedy',
                    choices=['greedy', 'lookahead', 'critic', 'policy_theta'],
                    help='Which search bot carries θ. "greedy" is the 2-ply SimGreedyBot '
                         '(~18-25 ms/move); "lookahead" is the alpha-beta search — ~6x the '
                         'cost, but a base bot that beats SimGreedyBot 0.79; "critic" is '
                         'the critic-guided beam search, where θ re-weights only the '
                         'hand-written half of the leaf blend.')
    ap.add_argument('--policy-weight', type=float, default=None,
                    help='Weight on the policy log-prior (--base policy_theta only).')
    ap.add_argument('--top-k', type=int, default=None,
                    help='Policy-ranked moves simulated per decision (--base policy_theta).')
    ap.add_argument('--critic-weight', type=float, default=None,
                    help='Leaf blend weight (--base critic only). Default: the shipped 0.7.')
    ap.add_argument('--branching', type=int, default=None,
                    help='Search width (--base critic only): sets both max_branching and '
                         'beam_width. Default: the shipped 5.')
    ap.add_argument('--reply-branching', type=int, default=8,
                    help="Each arm's 2nd-ply reply cap (--base greedy only). 2 is the cheap "
                         'setting B1 suggests for the training hot path.')
    ap.add_argument('--bot-time-budget', type=float, default=0.1,
                    help='Per-move search budget for each arm (--base lookahead only).')
    ap.add_argument('--seed', type=int, default=0, help='Base game seed.')
    ap.add_argument('--theta-seed', type=int, default=0,
                    help='Base seed for the θ draws; arm i uses theta_seed+i.')
    ap.add_argument('--n-workers', type=int, default=min(os.cpu_count() or 4, 8))
    ap.add_argument('--dump', default=None, help='Write the full result as JSON here.')
    args = ap.parse_args()

    critic_path = args.critic_path
    if args.opponent == 'lookahead_critic' and critic_path is None:
        found = sorted(glob.glob(CRITIC_GLOB))
        critic_path = found[-1] if found else None
    policy_path = args.policy_path
    if args.opponent == 'policy' and policy_path is None:
        found = sorted(glob.glob(POLICY_GLOB))
        policy_path = found[-1] if found else None
    spec = opponent_spec(args.opponent, time_budget=args.time_budget,
                         policy_path=policy_path, critic_path=critic_path)

    # Arm keys are display labels; θ is carried in the task so workers need no shared state.
    thetas, order = {}, []
    if args.thetas:
        with open(args.thetas) as fh:
            rows = json.load(fh)
        for i, row in enumerate(rows):
            theta = row['theta'] if isinstance(row, dict) and 'theta' in row else row
            key = f't{i}:{theta_tag(theta)}'
            thetas[key] = theta
            order.append(key)
    elif args.sweep:
        for value in args.sweep_values:
            key = f'{args.sweep[:5]}={value:g}'
            thetas[key] = {**LEGACY_THETA, args.sweep: value}
            order.append(key)
    else:
        for i in range(args.arms):
            theta = sample_theta(np.random.default_rng(args.theta_seed + i))
            key = f't{i}:{theta_tag(theta)}'
            thetas[key] = theta
            order.append(key)
    control_order = []
    for r in range(args.control_arms):
        key = f'ctl{r}'
        thetas[key] = dict(LEGACY_THETA)
        control_order.append(key)

    tasks = []
    for key in order:
        # Common random numbers across treatment arms: arm i game g and arm j game g replay
        # the same draft, so the draft cancels out of the treatment spread (see docstring).
        for g in range(args.games):
            tasks.append((key, thetas[key], args.seed + g, 1 if g % 2 == 0 else 2))
    for r, key in enumerate(control_order):
        # Disjoint seed blocks: the control is meant to *include* draft variance, so that
        # the noise floor it defines is an over-estimate rather than an under-estimate.
        base = args.seed + args.games * (r + 1) * 1000
        for g in range(args.games):
            tasks.append((key, thetas[key], base + g, 1 if g % 2 == 0 else 2))

    kind = f'sweep of {args.sweep}' if args.sweep else f'{args.arms} sampled'
    print(f'{len(thetas)} arms ({kind} + {args.control_arms} control) '
          f'x {args.games} games vs {args.opponent}, base={args.base} = {len(tasks)} games, '
          f'{args.n_workers} workers')
    for key in order:
        print(f'  {key:>10}  {format_theta(thetas[key])}')

    arms = {k: {'games': 0, 'wins': 0.0, 'turns': 0, 'usage': Counter(),
                'decisions': 0, 'bot_seconds': 0.0}
            for k in order + control_order}
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if args.base == 'policy_theta':
        bot_kwargs = {'reply_branching': args.reply_branching}
        if args.policy_weight is not None:
            bot_kwargs['policy_weight'] = args.policy_weight
        if args.top_k is not None:
            bot_kwargs['top_k'] = args.top_k
    elif args.base == 'critic':
        bot_kwargs = {'time_budget': args.bot_time_budget, 'stats_log_every': 0}
        if args.critic_weight is not None:
            bot_kwargs['critic_weight'] = args.critic_weight
        if args.branching is not None:
            bot_kwargs['max_branching'] = args.branching
            bot_kwargs['beam_width'] = args.branching
    elif args.base == 'lookahead':
        bot_kwargs = {'time_budget': args.bot_time_budget}
    else:
        bot_kwargs = {'reply_branching': args.reply_branching}
    t0 = time.perf_counter()
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.n_workers, initializer=_init_worker,
                  initargs=(root, spec, bot_kwargs, args.base)) as pool:
        for n, (key, outcome, turns, usage, decisions, seconds) in enumerate(
                pool.imap_unordered(_play, tasks), start=1):
            a = arms[key]
            a['games'] += 1
            a['wins'] += 1.0 if outcome == 'win' else (0.5 if outcome == 'draw' else 0.0)
            a['turns'] += turns
            a['usage'].update(usage)
            a['decisions'] += decisions
            a['bot_seconds'] += seconds
            if n % 20 == 0 or n == len(tasks):
                print(f'  [{n}/{len(tasks)}] {time.perf_counter() - t0:.0f}s', flush=True)

    treat = summarize({k: arms[k] for k in order})
    _print_table(f'Treatment — {kind} θ on {args.base} (vs {args.opponent}, '
                 f'{args.games} games each)', {k: arms[k] for k in order}, thetas, treat)
    ctl = None
    if control_order:
        ctl = summarize({k: arms[k] for k in control_order})
        _print_table(f'Control — {args.control_arms} blocks of the default θ',
                     {k: arms[k] for k in control_order}, None, ctl)

    print('\nSpread (mean pairwise total-variation distance between verb profiles):')
    print(f'  treatment (across θ)   {treat["tv"]:.4f}')
    if ctl:
        print(f'  control   (across seed) {ctl["tv"]:.4f}')
        ratio = treat['tv'] / ctl['tv'] if ctl['tv'] > 0 else float('inf')
        print(f'  ratio                   {ratio:.2f}x')
        verdict = ('the family generates real playstyle variety'
                   if ratio >= 2.0 else
                   'θ moves behaviour no further than a re-seed does — family NOT working')
        print(f'  -> {verdict}  (gate: ratio >= 2.0)')

    print(f'\nB1 gate verbs (range across arms; control range in brackets):')
    for verb in GATE_VERBS:
        lo, hi = treat['ranges'][verb]
        extra = ''
        if ctl:
            clo, chi = ctl['ranges'][verb]
            extra = f'   [ctl {chi - clo:.3f}]'
        print(f'  {verb:>16}  {lo:.3f} .. {hi:.3f}   range {hi - lo:.3f}{extra}')

    bad = health({k: arms[k] for k in order}, treat)
    print(f'\nArm health (WR < {MIN_HEALTHY_WR:.2f} or pass rate > {MAX_HEALTHY_PASS:.2f}): '
          f'{len(bad)}/{len(order)} unhealthy')
    for key, reason in bad.items():
        print(f'  {key:>10}  {reason}   {format_theta(thetas[key])}')
    if not bad:
        print('  (none — every arm is a playing opponent, not a punching bag)')
    if ctl:
        # Read health *relative to the control*, always. The thresholds are absolute, so a
        # strong `--opponent` flags every arm — including the unmodified default-θ bot,
        # which is the whole point of printing this: it separates "θ produced a punching
        # bag" from "SimGreedyBot loses to this opponent no matter what θ says".
        ctl_bad = health({k: arms[k] for k in control_order}, ctl)
        ctl_wr = np.mean([arms[k]['wins'] / arms[k]['games'] for k in control_order])
        treat_wr = np.mean([arms[k]['wins'] / arms[k]['games'] for k in order])
        print(f'  control (default θ, same thresholds): {len(ctl_bad)}/{len(control_order)} '
              f'unhealthy, mean WR {ctl_wr:.2f} vs treatment mean WR {treat_wr:.2f}')
        if len(ctl_bad) == len(control_order):
            print('  -> the default θ fails the same thresholds: this is the opponent being '
                  'strong, not the family being degenerate.')

    print(f'\ntotal {time.perf_counter() - t0:.0f}s')

    if args.dump:
        out = {
            'args': vars(args),
            'opponent_spec': spec,
            'arms': {k: {'games': v['games'], 'wins': v['wins'], 'turns': v['turns'],
                         'usage': dict(v['usage']), 'decisions': v['decisions'],
                         'bot_seconds': v['bot_seconds'],
                         'theta': thetas[k]}
                     for k, v in arms.items()},
            'verbs': list(VERBS),
            'treatment_tv': treat['tv'],
            'control_tv': ctl['tv'] if ctl else None,
            'unhealthy': bad,
        }
        with open(args.dump, 'w') as fh:
            json.dump(out, fh, indent=2)
        print(f'wrote {args.dump}')


if __name__ == '__main__':
    main()
