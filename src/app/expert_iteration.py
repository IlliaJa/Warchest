"""Expert-iteration CLI: self-play data-gen + policy/critic distillation + the loop.

`PuctBot` (the gauntlet's strongest agent) teaches its own priors and leaf values to
fresh policy/critic nets, which then seed a stronger `PuctBot`, and so on — the
AlphaZero/ExIt loop (docs/next_steps.md). See `src/services/expert_iteration.py` for
the reusable core. Run from the project root:

    # one generation of self-play games -> a dataset
    python src/app/expert_iteration.py gen --games 200 --out data/exit/round0.npz

    # distil a dataset into new nets (writes to data/exit/)
    python src/app/expert_iteration.py distill --dataset data/exit/round0.npz

    # the full loop: gen -> distil -> re-seed puct -> repeat
    python src/app/expert_iteration.py loop --rounds 5 --games 200

ExIt artifacts live under `data/exit/` on purpose: the distilled critic predicts the
game outcome z (scale [-1,1]), NOT the shaped PPO return, so it must never become the
`data/lookahead_critic/` "latest" that the shaped bots (LookaheadCriticBot /
policy_critic / default-mode PuctBot) resolve to — that would be a scale mismatch. The
loop pairs the z-critic only with `PuctBot(value_mode='outcome')`, passing paths
explicitly. Round 0 bootstraps from the standing (shaped) PPO checkpoints with the
proven `value_mode='shaped'` search; every later round uses the z-critic in
`'outcome'` mode.
"""
import argparse
import glob
import logging
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch

from src.services.expert_iteration import (
    generate_selfplay, distill, evaluate_distillation, SelfPlayDataset,
)
from src.services.bots.puct_bot import PuctBot
from src.services.environment.obs_encoders import get_encoder
from src.services.policy.policy import Policy, Critic
from src.services.policy.checkpoint import (
    load_policy_checkpoint, load_critic_checkpoint,
    save_policy_checkpoint, save_critic_checkpoint,
)

POLICY_GLOB = 'data/warchest_ppo_*.pth'
CRITIC_GLOB = 'data/lookahead_critic/lookahead_critic_v*.pth'
EXIT_DIR = 'data/exit'

logger = logging.getLogger('warchest')


def _latest_policy_path():
    c = glob.glob(POLICY_GLOB)
    return max(c) if c else None


def _latest_critic_path():
    c = glob.glob(CRITIC_GLOB)
    if not c:
        return None

    def version(p):
        m = re.search(r'_v(\d+)\.pth$', os.path.basename(p))
        return int(m.group(1)) if m else -1

    return max(c, key=version)


def _load_policy(path, device):
    meta = load_policy_checkpoint(path, map_location=device)
    encoder = get_encoder(meta['obs_version'])
    policy = Policy(device=device, hidden_dim=meta['hidden_dim'], obs_encoder=encoder).to(device)
    policy.load_state_dict(meta['state_dict'])
    return policy, meta


def _load_critic(path, device):
    meta = load_critic_checkpoint(path, map_location=device)
    encoder = get_encoder(meta['obs_version'])
    critic = Critic(device=device, hidden_dim=meta['hidden_dim'], obs_encoder=encoder).to(device)
    critic.load_state_dict(meta['state_dict'])
    return critic, meta


def _build_bot(policy_path, critic_path, args, *, value_mode):
    return PuctBot(
        policy_path=policy_path, critic_path=critic_path, value_mode=value_mode,
        c_puct=args.c_puct, max_branching=args.max_branching, time_budget=args.time_budget,
        dirichlet_alpha=args.dirichlet_alpha, dirichlet_frac=args.dirichlet_frac,
        device=args.device, stats_log_every=0,
    )


def _run_gen(policy_path, critic_path, args, *, value_mode, out_path):
    """Self-play `args.games` games and save a dataset. Returns the dataset."""
    pmeta = load_policy_checkpoint(policy_path, map_location=args.device)
    cmeta = load_critic_checkpoint(critic_path, map_location=args.device)
    if pmeta['obs_version'] != cmeta['obs_version']:
        raise SystemExit(
            f'policy obs_version {pmeta["obs_version"]} != critic obs_version '
            f'{cmeta["obs_version"]}; expert iteration requires a matching pair '
            f'(they always match when saved from one PPO run).'
        )
    encoder = get_encoder(pmeta['obs_version'])
    bot = _build_bot(policy_path, critic_path, args, value_mode=value_mode)
    logger.info('gen: %d games, value_mode=%s, policy=%s critic=%s',
                args.games, value_mode, os.path.basename(policy_path), os.path.basename(critic_path))
    ds = generate_selfplay(bot, args.games, encoder=encoder, temperature=args.temperature,
                           temp_moves=args.temp_moves, seed=args.seed)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    ds.save(out_path)
    logger.info('gen: wrote %d samples to %s', len(ds), out_path)
    return ds


def _run_distill(dataset, policy_path, critic_path, args, *, out_policy, out_critic):
    """Distil `dataset` into new nets warm-started from the given checkpoints, save both."""
    policy, pmeta = _load_policy(policy_path, args.device)
    critic, cmeta = _load_critic(critic_path, args.device)
    if pmeta['obs_version'] != cmeta['obs_version']:
        raise SystemExit('policy/critic obs_version mismatch (see gen).')

    before = evaluate_distillation(dataset, policy, critic, device=args.device)
    res = distill(dataset, policy, critic, epochs=args.epochs, minibatch_size=args.minibatch,
                  lr_policy=args.lr, lr_critic=args.lr, device=args.device, val_frac=args.val_frac)
    after = res['val']
    logger.info('distill: held-out before %s -> after %s', _fmt(before), _fmt(after))

    os.makedirs(os.path.dirname(out_policy) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(out_critic) or '.', exist_ok=True)
    save_policy_checkpoint(policy, out_policy, obs_version=pmeta['obs_version'],
                           hidden_dim=pmeta['hidden_dim'])
    # z-scale critic: return_mean=0 / return_std=1 so PuctBot's denormalisation is
    # identity and it consumes the outcome-scale value directly (value_mode='outcome').
    save_critic_checkpoint(critic, out_critic, obs_version=cmeta['obs_version'],
                           hidden_dim=cmeta['hidden_dim'], return_mean=0.0, return_std=1.0)
    logger.info('distill: saved policy=%s critic=%s', out_policy, out_critic)
    return before, after


def _fmt(d):
    if not d:
        return '{}'
    return f"ce={d['ce']:.4f} mse={d['mse']:.4f} agree={d['agreement']:.3f}"


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_gen(args):
    policy_path = args.policy or _latest_policy_path()
    critic_path = args.critic or _latest_critic_path()
    if policy_path is None or critic_path is None:
        raise SystemExit('gen needs a policy and a critic checkpoint (none found).')
    out = args.out or os.path.join(EXIT_DIR, 'gen.npz')
    _run_gen(policy_path, critic_path, args, value_mode=args.value_mode, out_path=out)


def cmd_distill(args):
    policy_path = args.policy or _latest_policy_path()
    critic_path = args.critic or _latest_critic_path()
    if policy_path is None or critic_path is None:
        raise SystemExit('distill needs base policy and critic checkpoints (none found).')
    ds = SelfPlayDataset.load(args.dataset)
    ts = time.strftime('%Y%m%d-%H%M%S')
    out_policy = args.out_policy or os.path.join(EXIT_DIR, f'policy_{ts}.pth')
    out_critic = args.out_critic or os.path.join(EXIT_DIR, f'critic_{ts}.pth')
    _run_distill(ds, policy_path, critic_path, args, out_policy=out_policy, out_critic=out_critic)


def cmd_loop(args):
    base_policy = args.policy or _latest_policy_path()
    base_critic = args.critic or _latest_critic_path()
    if base_policy is None or base_critic is None:
        raise SystemExit('loop needs base policy and critic checkpoints (none found).')
    os.makedirs(EXIT_DIR, exist_ok=True)

    cur_policy, cur_critic, cur_mode = base_policy, base_critic, 'shaped'
    for r in range(args.rounds):
        logger.info('=== ExIt round %d/%d (mode=%s) ===', r + 1, args.rounds, cur_mode)
        ds_path = os.path.join(EXIT_DIR, f'round{r}.npz')
        ds = _run_gen(cur_policy, cur_critic, args, value_mode=cur_mode, out_path=ds_path)
        out_policy = os.path.join(EXIT_DIR, f'round{r}_policy.pth')
        out_critic = os.path.join(EXIT_DIR, f'round{r}_critic.pth')
        before, after = _run_distill(ds, cur_policy, cur_critic, args,
                                     out_policy=out_policy, out_critic=out_critic)
        logger.info('round %d done: agreement %.3f -> %.3f, critic mse %.4f -> %.4f',
                    r + 1, before.get('agreement', 0.0), after.get('agreement', 0.0),
                    before.get('mse', 0.0), after.get('mse', 0.0))
        # From here on the critic is z-scale: pair it only with outcome-mode search.
        cur_policy, cur_critic, cur_mode = out_policy, out_critic, 'outcome'
    logger.info('ExIt loop finished. Latest nets: policy=%s critic=%s (value_mode=outcome).',
                cur_policy, cur_critic)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _add_common(p):
    p.add_argument('--policy', default=None, help='Base policy .pth (default: newest data/warchest_ppo_*.pth).')
    p.add_argument('--critic', default=None, help='Base critic .pth (default: newest lookahead_critic_v*.pth).')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    # search knobs (gen)
    p.add_argument('--games', type=int, default=200, help='Self-play games per generation.')
    p.add_argument('--time-budget', type=float, default=0.1, help='PuctBot per-move search budget (s).')
    p.add_argument('--c-puct', type=float, default=1.5)
    p.add_argument('--max-branching', type=int, default=8)
    p.add_argument('--dirichlet-alpha', type=float, default=0.3,
                   help='Root Dirichlet noise for self-play exploration (0 = off).')
    p.add_argument('--dirichlet-frac', type=float, default=0.25)
    p.add_argument('--temperature', type=float, default=1.0, help='Visit-count sampling temperature.')
    p.add_argument('--temp-moves', type=int, default=12, help='Opening plies sampled before going greedy.')
    p.add_argument('--seed', type=int, default=None)
    # distill knobs
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--minibatch', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--val-frac', type=float, default=0.1)


def main():
    parser = argparse.ArgumentParser(description='Warchest expert iteration (ExIt).')
    sub = parser.add_subparsers(dest='cmd', required=True)

    g = sub.add_parser('gen', help='Self-play a generation of games into a dataset.')
    _add_common(g)
    g.add_argument('--value-mode', choices=['shaped', 'outcome'], default='shaped',
                   help="PuctBot leaf/reward mode. 'shaped' (default) with a normal PPO "
                        "critic; 'outcome' only with a z-scale (ExIt-distilled) critic.")
    g.add_argument('--out', default=None, help='Output .npz path.')
    g.set_defaults(func=cmd_gen)

    d = sub.add_parser('distill', help='Distil a dataset into new policy+critic nets.')
    _add_common(d)
    d.add_argument('--dataset', required=True, help='Dataset .npz from `gen`.')
    d.add_argument('--out-policy', default=None)
    d.add_argument('--out-critic', default=None)
    d.set_defaults(func=cmd_distill)

    lp = sub.add_parser('loop', help='Full ExIt loop: gen -> distil -> re-seed -> repeat.')
    _add_common(lp)
    lp.add_argument('--rounds', type=int, default=5)
    lp.set_defaults(func=cmd_loop)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    args.func(args)


if __name__ == '__main__':
    main()
