"""Row 2b (docs/next_iteration.md §5) — the critic-target A/B behind IDEAS.md A6.

§3.3b measured a critic trained on shaped GAE returns out-ranking a critic trained on ExIt's
`z` (the game outcome) by ~2x on within-state sibling ranking. But that comparison was not
clean: the shaped-return critic was `hidden_dim=192` and trained on a full PPO run, the `z`
arm from `eval_board_value.py fit` was `hidden_dim=96` on 120k samples — different capacity,
different data budget. Row 2b removes both confounds: the SAME `ValueArm` architecture, the
SAME `hidden_dim`, the SAME sample count and seed, trained once on `data/exit/round*.npz`
(z-key = game outcome) and once on a `ppo.py --dump-returns-dir` shard set (z-key = shaped
GAE return — same field name, per `PPOTrainer._maybe_dump_returns`, so `load_exit_dataset`
and `train_arm` need no changes to consume either). Both fits are then scored against the
SAME held-out sibling-ranking label cache (`eval_board_value.py siblings`'s output), so the
only thing that differs between the two rows of the final table is the training target.

    python src/app/eval_critic_target_ab.py \\
        --shaped-data 'data/ppo_returns/round*.npz' --labels data/la16_labels.pt

This is the experiment IDEAS.md A6 (two value heads) says to run first: A6 is only justified
if the gap below survives at matched capacity and data.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch

from src.app.eval_board_value import (
    ARMS, BUCKETS, _bucket_pairs, _demean, _pair_acc, _score, load_exit_dataset, load_policy,
    train_arm,
)
from src.app.gauntlet import _latest_policy_path

SAME_VERB_BUCKET = 'board differs, same verb'


def _score_fitted_arms(arms, sets, y_sets, pairs):
    """Score each fitted `ValueArm` against the cached sibling labels.

    Mirrors `eval_board_value.mode_siblings`'s scoring loop for the fitted-arm case (no REAL
    checkpoint, no obs re-encoding: both fits share the label cache's obs version by the
    guard in `main`).
    """
    results = {}
    same_verb = dict(BUCKETS)[SAME_VERB_BUCKET]
    for kind, arm in arms.items():
        arm.eval()
        preds = []
        with torch.no_grad():
            for sibs in sets:
                bd = torch.from_numpy(np.stack([s['board'] for s in sibs]))
                gl = torch.from_numpy(np.stack([s['global'] for s in sibs]))
                pv = torch.from_numpy(np.stack([s['priv'] for s in sibs]))
                oh = torch.zeros(len(sibs), 3)
                oh[:, 2] = 1.0
                sign = np.array([s.get('sign', -1.0) for s in sibs])
                preds.append(_demean(sign * arm(bd, gl, oh, pv).numpy()))
        s = _score(preds, y_sets)
        s['same_verb_acc'], _, s['same_verb_tied'] = _pair_acc(preds, y_sets, pairs, same_verb)
        results[kind] = s
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--policy', default=None,
                    help='policy checkpoint providing the obs encoder (default: latest)')
    ap.add_argument('--z-data', default='data/exit/round*.npz',
                    help='ExIt shards — z-key holds the game outcome')
    ap.add_argument('--shaped-data', required=True,
                    help="shards from 'ppo.py --dump-returns-dir' — z-key holds the shaped "
                         'GAE return under the same field name')
    ap.add_argument('--labels', default='data/la16_labels.pt',
                    help='sibling-ranking label cache from `eval_board_value.py siblings`')
    ap.add_argument('--arms', nargs='+', choices=ARMS, default=['globals', 'board'])
    ap.add_argument('--hidden', type=int, default=96)
    ap.add_argument('--max-samples', type=int, default=120000,
                    help='same cap applied to BOTH datasets, so neither gets a data-size edge')
    ap.add_argument('--val-rounds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eps', type=float, default=1e-6)
    args = ap.parse_args()

    device = torch.device('cpu')
    pol_path = args.policy or _latest_policy_path()
    policy, encoder = load_policy(pol_path, device)

    blob = torch.load(args.labels, map_location='cpu', weights_only=False)
    sets = blob['sets']
    print(f'label cache: {args.labels}  ({len(sets)} states, '
          f'{sum(len(s) for s in sets)} successors, obs v{blob.get("obs_version")})')
    cache_obs = blob.get('obs_version')
    if cache_obs is not None and cache_obs != getattr(encoder, 'version', None):
        raise SystemExit(f'label cache is obs v{cache_obs}, policy encoder is '
                         f'v{getattr(encoder, "version", None)} — pass --policy pointing at '
                         'a matching-era checkpoint')

    y_sets = [_demean([0.5 * (s['z0'] + s['z1']) for s in sibs]) for sibs in sets]
    pairs = _bucket_pairs(sets, eps=args.eps)

    targets = [('z (game outcome)', args.z_data), ('shaped GAE return', args.shaped_data)]
    all_results = {}
    for label, pattern in targets:
        print(f'\n=== fitting on {label}: {pattern} ===')
        tr, va = load_exit_dataset(pattern, args.max_samples, args.val_rounds, args.seed)
        if tr['boards'].shape[1] != encoder.board_channels:
            raise SystemExit(f'{pattern}: board channels do not match the policy encoder')
        arms = {}
        for kind in args.arms:
            print(f'  fitting {kind} ...')
            arm, _ = train_arm(kind, tr, va, hidden=args.hidden, epochs=args.epochs,
                               batch=args.batch, lr=args.lr,
                               policy_trunk=(policy.board_encoder if kind.startswith('polfeat')
                                             else None),
                               device=device, seed=args.seed)
            arms[kind] = arm
        all_results[label] = _score_fitted_arms(arms, sets, y_sets, pairs)

    z_label, shaped_label = targets[0][0], targets[1][0]
    print(f'\n{"=" * 88}\nHEAD-TO-HEAD — same arm, hidden_dim={args.hidden}, '
          f'max_samples={args.max_samples}, scored on the same {len(sets)}-state cache\n'
          f'{"=" * 88}')
    print(f'{"arm":<10}{"target":<20}{"corr":>8}{"spearman":>10}'
          f'{"same-verb acc":>16}{"same-verb tied":>16}')
    for kind in args.arms:
        for label in (z_label, shaped_label):
            s = all_results[label][kind]
            print(f'{kind:<10}{label:<20}{s["corr"]:8.3f}{s["spearman"]:10.3f}'
                  f'{s["same_verb_acc"]:15.1%}{s["same_verb_tied"]:16.1%}')
        z_s, sh_s = all_results[z_label][kind], all_results[shaped_label][kind]
        if abs(z_s['spearman']) > 1e-3:
            print(f'{"":<10}{"shaped / z spearman ratio":<20}'
                  f'{sh_s["spearman"] / z_s["spearman"]:7.2f}x')
        print()

    print("""HOW TO READ
  Read `board`'s row, not `globals`'s: `globals` is the control and should barely move between
  targets, since it cannot see the board under either one. `same-verb acc`/`tied` is the
  "board differs, same verb" bucket — siblings whose non-board inputs are identical, so only
  the board can rank them and a board-blind model is pinned near 50% by construction.

  `board` clears its z-target numbers by roughly the ~2x §3.3b reported -> the gap is real at
  matched capacity and data, and IDEAS.md A6 (two value heads) is justified.
  The two targets score close together -> §3.3b's original comparison was confounded by
  hidden_dim/data size, not target choice, and A6 should wait — re-open §3.3b instead.""")


if __name__ == '__main__':
    main()
