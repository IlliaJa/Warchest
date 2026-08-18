"""What does the search actually *do* differently from the raw policy it is built on?

`next_iteration.md` §5 row 9 (run 2026-08-16, `IDEAS.md` R.0.1) measured that `PuctBot`
beats the policy that supplies its priors — 0.66 at a 0.1 s budget, 0.74 at 1.0 s. That
says search is worth ~180 Elo here; it does not say *what the extra strength is made of*,
and that matters before spending a training run distilling it:

  * if the gap is prophylaxis — the search stops hanging material and cashes free kills —
    then distilling it teaches exactly the skill the policy is missing, and ExIt is the
    right lever;
  * if the gap is base timing with the same material carelessness, the teacher is a better
    racer and distilling it will make a better racer, not a better player.

So this plays the two agents against each other with colours balanced and records, per
side, the behaviour that separates those two stories: the verb mix, how much of its own
material sits under enemy threat while it is to move (`own_at_risk`, global 208 — the
encoder already computes it and, per `IDEAS.md` B5, nothing has ever read it), how much
enemy material it boxes, and how the base differential closes.

Everything is measured from the *mover's* ego-centric observation, so the two arms are
directly comparable regardless of seat. Colours alternate per game and both arms see the
same seeds, so the draft is common random numbers across arms (`IDEAS.md` L5's surviving
half).

Usage:

    python src/app/eval_search_delta.py --games 60 --puct-time-budget 1.0
    python src/app/eval_search_delta.py --games 60 --puct-time-budget 0.1 --n-workers 12
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.services.environment.warchest_env import WarChestEnv, VERB_OF_ACTION
from src.services.environment.obs_encoders import get_encoder
from src.services.policy.checkpoint import load_policy_checkpoint, load_critic_checkpoint

VERB_NAMES = ('move', 'attack', 'control', 'deploy', 'bolster', 'claim',
              'pass', 'recruit', 'tactic', 'decline', 'select')
OWN_AT_RISK_IDX = 208
OPP_AT_RISK_IDX = 209
MY_BASES_IDX = 1
OPP_BASES_IDX = 2

# One lazily-built pair of agents per worker process: PuctBot loads two checkpoints and
# a policy forward is 0.86 ms, so rebuilding per game would dominate a short run.
_AGENTS = None


def _build_agents(policy_path, critic_path, time_budget, c_puct, max_branching, device):
    from src.services.bots.puct_bot import PuctBot
    from src.services.policy.policy import Policy
    from src.services.gauntlet import PolicyAgent

    pmeta = load_policy_checkpoint(policy_path, map_location=device)
    encoder = get_encoder(pmeta['obs_version'])
    policy = Policy(device=device, hidden_dim=pmeta['hidden_dim'], obs_encoder=encoder,
                    arch=pmeta['arch']).to(device)
    policy.load_state_dict(pmeta['state_dict'])
    policy.eval()
    raw = PolicyAgent('policy', policy, encoder)
    puct = PuctBot(policy_path=policy_path, critic_path=critic_path, c_puct=c_puct,
                   max_branching=max_branching, time_budget=time_budget, device=device,
                   stats_log_every=0)
    puct.name = 'puct'
    return {'policy': raw, 'puct': puct, 'encoder': encoder}


def _init_worker(cfg):
    global _AGENTS
    import torch
    torch.set_num_threads(1)
    _AGENTS = _build_agents(**cfg)


def _blank_side():
    return {'verbs': np.zeros(len(VERB_NAMES)), 'decisions': 0,
            'own_at_risk': 0.0, 'opp_at_risk': 0.0, 'boxed_inflicted': 0,
            'bases_final': 0, 'wins': 0, 'games': 0, 'plies': 0}


def play_one(task):
    """One game. `task = (seed, puct_seat)`. Returns per-arm behaviour sums."""
    seed, puct_seat = task
    agents = _AGENTS
    encoder = agents['encoder']
    seat = {puct_seat: 'puct', 3 - puct_seat: 'policy'}
    out = {'puct': _blank_side(), 'policy': _blank_side()}

    env = WarChestEnv(save_game_history=False)
    np.random.seed(seed)
    env.reset()
    winner = 0
    plies = 0
    for _ in range(2000):
        pid = env.active_player
        arm = seat[pid]
        rec = out[arm]
        # Ego-centric globals for whoever is to move: index 208/209 are own/opp at-risk
        # material, already normalised to [0, 1] by the encoder.
        g = encoder.encode(env)['global']
        rec['own_at_risk'] += float(g[OWN_AT_RISK_IDX])
        rec['opp_at_risk'] += float(g[OPP_AT_RISK_IDX])
        rec['decisions'] += 1

        action = agents[arm].act(env)
        _, _, terminated, truncated, info = env.step(action)
        if not info['action'].is_valid:
            _, _, terminated, truncated, info = env.make_random_step()
        else:
            rec['verbs'][VERB_OF_ACTION[action]] += 1
        plies += 1
        if terminated:
            winner = pid
            break
        if truncated:
            break

    for pid, arm in seat.items():
        other = 3 - pid
        rec = out[arm]
        rec['games'] = 1
        rec['plies'] = plies
        rec['wins'] = int(winner == pid)
        # Coins of the *opponent* permanently removed = material this arm inflicted.
        rec['boxed_inflicted'] = int(env.boxed_total(other))
        rec['bases_final'] = len(env.board.get_controlled_bases(pid))
    out['_decisive'] = int(winner != 0)
    return out


def _merge(acc, part):
    for arm in ('puct', 'policy'):
        for k, v in part[arm].items():
            acc[arm][k] = acc[arm][k] + v
    acc['_decisive'] += part['_decisive']
    return acc


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--policy', default=None, help='Policy checkpoint; default = newest data/warchest_ppo_*.pth')
    p.add_argument('--critic', default=None, help='Critic checkpoint; default = newest data/warchest_critic_*.pth')
    p.add_argument('--games', type=int, default=60, help='Total games (colours balanced).')
    p.add_argument('--puct-time-budget', type=float, default=1.0)
    p.add_argument('--puct-c', type=float, default=1.5)
    p.add_argument('--puct-max-branching', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    import glob
    policy_path = args.policy or max(glob.glob('data/warchest_ppo_*.pth'), key=os.path.getmtime)
    critic_path = args.critic or max(glob.glob('data/warchest_critic_*.pth'), key=os.path.getmtime)
    cmeta = load_critic_checkpoint(critic_path, map_location='cpu')
    print(f'policy = {policy_path}')
    print(f'critic = {critic_path}  (arch {cmeta["arch"]}, obs v{cmeta["obs_version"]})')
    print(f'{args.games} games, puct budget {args.puct_time_budget}s, {args.n_workers} workers\n')

    tasks = [(args.seed + i, 1 + (i % 2)) for i in range(args.games)]
    cfg = dict(policy_path=policy_path, critic_path=critic_path,
               time_budget=args.puct_time_budget, c_puct=args.puct_c,
               max_branching=args.puct_max_branching, device=args.device)

    acc = {'puct': _blank_side(), 'policy': _blank_side(), '_decisive': 0}
    if args.n_workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.n_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for i, part in enumerate(pool.imap_unordered(play_one, tasks), 1):
                acc = _merge(acc, part)
                if i % 10 == 0:
                    print(f'  ... {i}/{len(tasks)} games', flush=True)
    else:
        _init_worker(cfg)
        for i, t in enumerate(tasks, 1):
            acc = _merge(acc, play_one(t))
            if i % 5 == 0:
                print(f'  ... {i}/{len(tasks)} games', flush=True)

    n = acc['puct']['games']
    print(f'\n{n} games, decisive {acc["_decisive"]}/{n} '
          f'({100 * acc["_decisive"] / max(n, 1):.0f}%), '
          f'{acc["puct"]["plies"] / max(n, 1):.1f} plies/game\n')

    def wr(arm):
        w = acc[arm]['wins']
        return w / max(n, 1), np.sqrt(max(w / max(n, 1) * (1 - w / max(n, 1)), 1e-9) / max(n, 1))

    for arm in ('puct', 'policy'):
        w, se = wr(arm)
        print(f'{arm:>7}: WR {w:.3f} +/- {se:.3f}')

    print(f'\n{"metric":<26}{"puct":>10}{"policy":>10}{"delta":>10}')
    print('-' * 56)
    rows = []
    for arm in ('puct', 'policy'):
        d = max(acc[arm]['decisions'], 1)
        rows.append({
            'own_at_risk (while to move)': acc[arm]['own_at_risk'] / d,
            'opp_at_risk (while to move)': acc[arm]['opp_at_risk'] / d,
            'enemy coins boxed / game': acc[arm]['boxed_inflicted'] / max(n, 1),
            'bases held at end': acc[arm]['bases_final'] / max(n, 1),
            'decisions / game': d / max(n, 1),
        })
    for k in rows[0]:
        a, b = rows[0][k], rows[1][k]
        print(f'{k:<26}{a:>10.4f}{b:>10.4f}{a - b:>+10.4f}')

    print(f'\n{"verb (share of decisions)":<26}{"puct":>10}{"policy":>10}{"delta":>10}')
    print('-' * 56)
    for vi, vname in enumerate(VERB_NAMES):
        a = acc['puct']['verbs'][vi] / max(acc['puct']['decisions'], 1)
        b = acc['policy']['verbs'][vi] / max(acc['policy']['decisions'], 1)
        if a < 1e-4 and b < 1e-4:
            continue
        print(f'{vname:<26}{a:>10.4f}{b:>10.4f}{a - b:>+10.4f}')

    print('\nHOW TO READ. The distillation-relevant question is whether search wins by\n'
          'prophylaxis or by racing harder. Prophylaxis looks like: own_at_risk *lower*\n'
          'for puct, enemy coins boxed *higher*, attack/bolster share up. Racing harder\n'
          'looks like: control/claim share up with own_at_risk flat or worse. The first\n'
          'story means ExIt distils the missing skill; the second means it distils the\n'
          'behaviour that already loses to a human.')


if __name__ == '__main__':
    main()
