"""Search the θ space for a *strong* member of the family — docs/IDEAS.md B1/B2.

`eval_theta_family.py` answers "are these different bots". This answers "is any of them
good", which turned out to be the binding question: a random draw of 8 θ on `LookaheadBot`
scored 0.06-0.44 against `lookahead_critic` while the default θ scored 0.34, so θ moves
strength by ~40 points of win rate and nobody had ever looked for the top of that range.

Three things make this affordable enough to actually run:

  * **Common random numbers.** Candidate i's game g and candidate j's game g replay the
    same seed, so they face identical drafts and the comparison differences out the single
    largest variance source in this game (the same composition wins both games of a swapped
    pair 63 % of the time — `gauntlet.build_task_list`). Colours alternate within a
    candidate, so each also plays both seats.
  * **Successive halving.** Cheap noisy screening on many candidates, then progressively
    more games on the survivors. A flat "N games for every candidate" schedule spends most
    of its budget proving that already-hopeless θ are hopeless.
  * **Result caching by (candidate, seed).** A survivor's earlier games are never replayed
    when its game count grows between rounds, so round k costs only the *new* games.

The opponent is a `gauntlet.build_agent` spec, so this searches a best response to whatever
the field's yardstick currently is — `lookahead_critic` by default.

    python src/app/search_theta.py --opponent lookahead_critic --n-workers 12
    python src/app/search_theta.py --candidates 64 --schedule 8 16 32 64 --keep-frac 0.35
    python src/app/search_theta.py --seed-thetas best.json --candidates 48   # local search
"""
import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np

from src.services.bots.evaluation import (
    THETA_KEYS, THETA_RANGES, LEGACY_THETA, normalize_theta, sample_theta, theta_tag,
    format_theta,
)
from src.services.bots.lookahead_critic_bot import LookaheadCriticBot
from src.app.eval_theta_family import opponent_spec, CRITIC_GLOB, POLICY_GLOB

_W = {}


def _init_worker(root, spec, base, time_budget, see_opponent_hand):
    if root not in sys.path:
        sys.path.insert(0, root)
    import torch
    torch.set_num_threads(1)
    from src.services.gauntlet import build_agent

    _W['opponent'] = build_agent(spec, device=torch.device('cpu'))
    _W['base'] = base
    _W['time_budget'] = time_budget
    _W['see_opponent_hand'] = see_opponent_hand
    _W['bots'] = {}


def _bot_for(theta, max_branching, critic_weight):
    """Cached bot per (θ, width, blend) — construction allocates a whole sim env, and
    the critic base also loads a checkpoint and calibrates its value scale.
    """
    from src.services.bots.random_eval_bot import (
        RandomEvalBot, RandomEvalLookaheadBot, RandomEvalCriticBot,
    )

    key = (tuple(theta[k] for k in THETA_KEYS), max_branching, critic_weight)
    bot = _W['bots'].get(key)
    if bot is None:
        if _W['base'] == 'policy_theta':
            from src.services.bots.policy_theta_bot import PolicyThetaBot
            # The two generic search slots are reused per base (see --branchings help):
            # here width means top_k and the blend weight means policy_weight.
            bot = PolicyThetaBot(theta=theta, top_k=max_branching,
                                 policy_weight=critic_weight,
                                 see_opponent_hand=_W['see_opponent_hand'])
        elif _W['base'] == 'critic':
            # beam_width tracks max_branching: the raw cap and the survivor cap are both 5
            # by default, and searching them independently doubles the space for a knob
            # whose effect is mostly "how wide overall".
            bot = RandomEvalCriticBot(theta=theta, time_budget=_W['time_budget'],
                                      max_branching=max_branching, beam_width=max_branching,
                                      critic_weight=critic_weight, stats_log_every=0,
                                      see_opponent_hand=_W['see_opponent_hand'])
        elif _W['base'] == 'lookahead':
            bot = RandomEvalLookaheadBot(theta=theta, time_budget=_W['time_budget'],
                                         max_branching=max_branching,
                                         see_opponent_hand=_W['see_opponent_hand'])
        else:
            bot = RandomEvalBot(theta=theta, reply_branching=max_branching,
                                see_opponent_hand=_W['see_opponent_hand'])
        _W['bots'][key] = bot
    return bot


def _play(task):
    """One game. -> (candidate index, seed, score in {1.0 win, 0.5 draw, 0.0 loss})."""
    from src.services.environment.warchest_env import WarChestEnv

    idx, theta, max_branching, critic_weight, seed, bot_pid = task
    bot = _bot_for(theta, max_branching, critic_weight)
    opponent = _W['opponent']

    env = WarChestEnv(save_game_history=False)
    np.random.seed(seed)
    env.reset()
    for agent in (bot, opponent):
        hook = getattr(agent, 'new_episode', None)
        if hook is not None:
            hook()

    agents = {bot_pid: bot, 3 - bot_pid: opponent}
    for _ in range(2000):
        pid = env.active_player
        action = agents[pid].act(env)
        _, _, terminated, truncated, info = env.step(action)
        if not info['action'].is_valid:
            _, _, terminated, truncated, info = env.make_random_step()
        if terminated:
            return idx, seed, (1.0 if pid == bot_pid else 0.0)
        if truncated:
            return idx, seed, 0.5
    return idx, seed, 0.5


# --------------------------------------------------------------------------- #
# Candidate generation
# --------------------------------------------------------------------------- #
def perturb(theta, rng, *, scale=0.35):
    """Log-normal jitter around `theta`, clipped to each key's range.

    Local search around an incumbent. A zeroed coordinate stays zeroed with probability
    1/2 rather than always: "off" is a distinct regime (see THETA_ZERO_PROB), so a local
    step has to be able to both leave it and enter it, but a multiplicative jitter alone
    can never escape 0.
    """
    out = {}
    for key in THETA_KEYS:
        lo, hi = THETA_RANGES[key]
        value = theta[key]
        if value <= 0.0:
            out[key] = 0.0 if rng.random() < 0.5 else float(np.exp(rng.uniform(
                np.log(lo), np.log(hi))))
            continue
        if rng.random() < 0.10:
            out[key] = 0.0
            continue
        out[key] = float(np.clip(value * np.exp(rng.normal(0.0, scale)), lo, hi))
    return out


def build_grid(branchings, critic_weights):
    """Every (width, blend) combination at the default θ — the *interpretable* arm.

    The θ search answers "is some coefficient vector strong"; measurement says no, at
    every base tried. What remains are the two search knobs, and those are few enough to
    enumerate exhaustively, which beats sampling them: a grid gives one clean number per
    setting instead of a cloud confounded with whatever θ each candidate happened to draw.
    Candidate 0 is the stock configuration, so the baseline is always in the field.
    """
    cands = []
    for weight in critic_weights:
        for branching in branchings:
            stock = (weight == LookaheadCriticBot.CRITIC_WEIGHT and branching == 5)
            cands.append({'theta': dict(LEGACY_THETA), 'max_branching': branching,
                          'critic_weight': weight,
                          'origin': 'stock' if stock else 'grid'})
    cands.sort(key=lambda c: c['origin'] != 'stock')   # stock first, so it is candidate 0
    return cands


def build_candidates(n, rng, *, seed_thetas, branchings, critic_weights):
    """The initial pool: the default θ, any supplied incumbents, jitters of them, then
    fresh draws from the prior to fill.

    The default θ is always candidate 0 and always survives to the final round: it is the
    baseline the whole search is measured against, and dropping it mid-way would leave no
    control on the same games as the winner.
    """
    cands = [{'theta': dict(LEGACY_THETA), 'max_branching': branchings[0],
              'critic_weight': critic_weights[0], 'origin': 'default'}]
    for theta in seed_thetas:
        for branching in branchings:
            cands.append({'theta': normalize_theta(theta), 'max_branching': branching,
                          'critic_weight': critic_weights[0], 'origin': 'seed'})
    while len(cands) < n:
        if seed_thetas and rng.random() < 0.5:
            base = seed_thetas[int(rng.integers(len(seed_thetas)))]
            theta, origin = perturb(normalize_theta(base), rng), 'jitter'
        else:
            theta, origin = sample_theta(rng), 'prior'
        cands.append({'theta': theta,
                      'max_branching': int(rng.choice(branchings)),
                      'critic_weight': float(rng.choice(critic_weights)),
                      'origin': origin})
    return cands[:n]


def wilson_lower(wins, n, z=1.0):
    """Lower Wilson bound on the win rate — the ranking key.

    Ranking by raw win rate systematically promotes whichever candidate got lucky on the
    fewest games, which is exactly the failure mode successive halving is prone to. The
    lower bound penalises small n, so a candidate has to be *both* good and measured.
    """
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0))
    return (centre - margin) / denom


def main():
    ap = argparse.ArgumentParser(description='Search θ for the strongest family member.')
    ap.add_argument('--opponent', default='lookahead_critic',
                    choices=['greedy_sim', 'greedy_fast', 'random', 'lookahead',
                             'lookahead_critic', 'policy'])
    ap.add_argument('--base', default='lookahead',
                    choices=['greedy', 'lookahead', 'critic', 'policy_theta'],
                    help='Search bot carrying θ. "lookahead" is the policy-independent '
                         'alpha-beta search; "critic" is the critic-guided beam search — '
                         'stronger, but it inherits a self-play-derived critic, so it is '
                         'not an independent opponent. "greedy" measured a ceiling of '
                         '~0.08 against a trained policy.')
    ap.add_argument('--candidates', type=int, default=48)
    ap.add_argument('--grid', action='store_true',
                    help='Enumerate every (--branchings x --critic-weights) pair at the '
                         'default θ instead of sampling θ. Use this for the strength '
                         'question: θ is a behaviour dial, the search knobs are the '
                         'strength dials, and the knobs are few enough to enumerate.')
    ap.add_argument('--schedule', type=int, nargs='+', default=[8, 16, 32, 64],
                    help='Cumulative games per candidate, one entry per round.')
    ap.add_argument('--keep-frac', type=float, default=0.35,
                    help='Fraction of candidates carried into the next round.')
    ap.add_argument('--branchings', type=int, nargs='+', default=[5, 8, 12],
                    help='Search-width values to co-search with θ. Meaning per base: '
                         'lookahead/critic = max_branching, greedy = reply_branching, '
                         'policy_theta = top_k (how many policy-ranked moves get simulated).')
    ap.add_argument('--critic-weights', type=float, nargs='+', default=[0.7],
                    help='Blend weights to co-search. --base critic: the share of the '
                         'leaf taken from the critic (0.7 shipped). --base policy_theta: '
                         'policy_weight, the weight on the policy log-prior in the final '
                         'score (0 = policy-pruned SimGreedy, large = the raw policy).')
    ap.add_argument('--bot-time-budget', type=float, default=0.1,
                    help='Per-move budget for each candidate. Keep at or below the '
                         "opponent's so a win is not bought with wall-clock.")
    ap.add_argument('--opponent-time-budget', type=float, default=0.1)
    ap.add_argument('--blind', action='store_true',
                    help="Candidates don't read the opponent's real hand (fair mode).")
    ap.add_argument('--seed-thetas', default=None,
                    help='JSON list of θ dicts to seed the pool with (from a previous run '
                         "'s --dump). Turns the search into local search around them.")
    ap.add_argument('--seed', type=int, default=0, help='Base game seed (the CRN block).')
    ap.add_argument('--theta-seed', type=int, default=0)
    ap.add_argument('--n-workers', type=int, default=min(os.cpu_count() or 4, 12))
    ap.add_argument('--dump', default=None, help='Write the ranked survivors here as JSON.')
    args = ap.parse_args()

    critic_path = policy_path = None
    if args.opponent == 'lookahead_critic':
        found = sorted(glob.glob(CRITIC_GLOB))
        critic_path = found[-1] if found else None
    if args.opponent == 'policy':
        found = sorted(glob.glob(POLICY_GLOB))
        policy_path = found[-1] if found else None
    spec = opponent_spec(args.opponent, time_budget=args.opponent_time_budget,
                         policy_path=policy_path, critic_path=critic_path)

    rng = np.random.default_rng(args.theta_seed)
    seed_thetas = []
    if args.seed_thetas:
        with open(args.seed_thetas) as fh:
            seed_thetas = [row['theta'] if isinstance(row, dict) and 'theta' in row else row
                           for row in json.load(fh)]
    if args.grid:
        cands = build_grid(args.branchings, args.critic_weights)
    else:
        cands = build_candidates(args.candidates, rng, seed_thetas=seed_thetas,
                                 branchings=args.branchings,
                                 critic_weights=args.critic_weights)

    print(f'{len(cands)} candidates vs {args.opponent} '
          f'(base={args.base}, budget={args.bot_time_budget}s vs '
          f'{args.opponent_time_budget}s), schedule {args.schedule}, {args.n_workers} workers')

    scores = [dict() for _ in cands]   # candidate index -> {seed: score}
    alive = list(range(len(cands)))
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    t0 = time.perf_counter()
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.n_workers, initializer=_init_worker,
                  initargs=(root, spec, args.base, args.bot_time_budget,
                            not args.blind)) as pool:
        for rnd, n_games in enumerate(args.schedule):
            tasks = []
            for idx in alive:
                for g in range(n_games):
                    seed = args.seed + g
                    if seed in scores[idx]:
                        continue        # already played in an earlier round — reuse it
                    tasks.append((idx, cands[idx]['theta'], cands[idx]['max_branching'],
                                  cands[idx]['critic_weight'], seed,
                                  1 if g % 2 == 0 else 2))
            print(f'\nround {rnd + 1}: {len(alive)} candidates x {n_games} games '
                  f'({len(tasks)} new games)')
            for n, (idx, seed, score) in enumerate(pool.imap_unordered(_play, tasks), 1):
                scores[idx][seed] = score
                if n % 50 == 0 or n == len(tasks):
                    print(f'  [{n}/{len(tasks)}] {time.perf_counter() - t0:.0f}s', flush=True)

            ranked = sorted(alive, key=lambda i: -wilson_lower(sum(scores[i].values()),
                                                               len(scores[i])))
            for i in ranked[:8]:
                wins, n = sum(scores[i].values()), len(scores[i])
                print(f'    #{i:>3} WR {wins / n:.3f} (n={n}, lb={wilson_lower(wins, n):.3f}) '
                      f'br={cands[i]["max_branching"]:>2} cw={cands[i]["critic_weight"]:.2f} '
                      f'{cands[i]["origin"]:>7}  {format_theta(cands[i]["theta"])}')
            if rnd == len(args.schedule) - 1:
                break
            n_keep = max(2, int(round(len(alive) * args.keep_frac)))
            alive = ranked[:n_keep]
            if 0 not in alive:
                alive.append(0)   # the default-θ control always survives (see build_candidates)

    print(f'\n=== final ranking ({time.perf_counter() - t0:.0f}s) ===')
    final = sorted(alive, key=lambda i: -wilson_lower(sum(scores[i].values()),
                                                       len(scores[i])))
    out = []
    for i in final:
        wins, n = sum(scores[i].values()), len(scores[i])
        se = math.sqrt(max(wins / n * (1 - wins / n) / n, 0.0))
        tag = (' <- baseline' if cands[i]['origin'] in ('default', 'stock') else '')
        print(f'  WR {wins / n:.3f} +-{se:.3f} (n={n})  br={cands[i]["max_branching"]:>2}  '
              f'cw={cands[i]["critic_weight"]:.2f}  {theta_tag(cands[i]["theta"]):>3}  '
              f'{format_theta(cands[i]["theta"])}{tag}')
        out.append({'theta': cands[i]['theta'], 'max_branching': cands[i]['max_branching'],
                    'critic_weight': cands[i]['critic_weight'],
                    'wr': wins / n, 'n': n, 'se': se, 'origin': cands[i]['origin']})

    if args.dump:
        with open(args.dump, 'w') as fh:
            json.dump(out, fh, indent=2)
        print(f'wrote {args.dump}')


if __name__ == '__main__':
    main()
