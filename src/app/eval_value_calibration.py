"""The gate in front of IDEAS.md A6 — is the calibration hack actually costing anything?

A6 proposes a second value head (`V_win`, trained on the game outcome `z`) so search gets a
calibrated win probability and `LookaheadCriticBot._calibrate_value_scale` can be deleted.
The premise is that the shipped critic's output is unusable as a probability: it is a shaped
GAE return, denormalised by the checkpoint's `return_mean`/`return_std` into reward units
(`_critic_root_values`), and `PuctBot._select` then feeds it to `sign*q + c_puct*P*sqrt(N)/(1+N)`
— a formula whose `c_puct` was tuned on the AlphaZero assumption that Q is a win probability
in [-1, 1].

But "the number is not a probability" has two very different causes, and they have very
different price tags:

  (a) only the MAPPING is wrong — the scalar ranks winners from losers perfectly well, it is
      just on the wrong scale. Fix: two Platt numbers saved into the checkpoint. No new head,
      no retraining, no new arch.
  (b) the shaped-return objective has genuinely DESTROYED win information that the trunk
      still carries. Fix: A6's second head, which reads the trunk rather than the scalar.

Running a `z`-head and finding it beats the raw critic on Brier score does NOT distinguish
these — a shaped return scored against z in {0,1} has a terrible Brier score by construction,
whatever it knows. The decomposition that does distinguish them is **AUC, which no monotone
rescaling can change**:

    as_is / platt / isotonic  share one AUC BY CONSTRUCTION (all are monotone maps of the
                              same scalar). isotonic is the best ANY post-hoc recalibration
                              of that scalar can possibly do — it bounds the whole family.
    zhead                     is the only arm that can move AUC, because it is the only one
                              that reads the trunk instead of the finished scalar.

So: `zhead` AUC ~= the shared AUC means cause (a) and A6 is over-engineering — ship the two
numbers. `zhead` AUC >> the shared AUC means cause (b) and A6's head is buying something a
rescaling cannot.

    python src/app/eval_value_calibration.py calib --critic data/warchest_critic_20260810-0802.pth
    python src/app/eval_value_calibration.py puct --critic data/warchest_critic_20260810-0802.pth

`puct` is the companion consequence check: a bad probability only matters if it actually
distorts the search, so it measures the spread of Q in the units PUCT consumes against the
exploration term it is summed with.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import torch.nn as nn

from src.app.eval_board_value import load_exit_dataset
from src.services.environment.obs_encoders import get_encoder
from src.services.policy.checkpoint import load_critic_checkpoint
from src.services.policy.policy import Critic

EPS = 1e-6


# --------------------------------------------------------------------------- #
# Metrics — implemented here rather than pulled in, sklearn is not a dependency
# --------------------------------------------------------------------------- #
def auc(p, y):
    """Mann-Whitney U / rank-sum AUC. Invariant under any monotone map of `p`.

    That invariance is the whole point of this file: it is what makes `as_is`, `platt` and
    `isotonic` share a column, so any AUC the `zhead` arm gains is attributable to reading
    the trunk rather than to being better scaled.
    """
    pos, neg = y > 0.5, y <= 0.5
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(p, kind='mergesort')
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    # Average the ranks inside each tie group, or a constant predictor scores 1.0 not 0.5.
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ece(p, y, n_bins=15):
    """Expected calibration error: mean |confidence - accuracy| over equal-width bins.

    The direct "are these numbers actually probabilities" measure, and the one that a
    monotone recalibration is supposed to drive to ~0.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def score_all(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return {
        'brier': float(((p - y) ** 2).mean()),
        'logloss': float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        'ece': ece(p, y),
        'auc': auc(p, y),
        'acc': float(((p > 0.5) == (y > 0.5)).mean()),
    }


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
def fit_platt(v, y, iters=200, lr=0.5):
    """2-parameter logistic `sigmoid(a*v + b)` by Newton-free gradient descent.

    This arm IS the cheap alternative to A6: two floats, saved next to the existing
    `return_mean`/`return_std` in the checkpoint, consumed by whatever wants a probability.
    """
    a = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    vt = torch.from_numpy(v.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32))
    opt = torch.optim.LBFGS([a, b], lr=lr, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(a * vt + b, yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(a.item()), float(b.item())


def apply_platt(v, ab):
    a, b = ab
    return 1.0 / (1.0 + np.exp(-(a * v + b)))


def fit_isotonic(v, y):
    """Pool-adjacent-violators isotonic regression. -> (sorted knots, fitted values).

    The least-squares-optimal NON-DECREASING map from the scalar to the outcome, so it is an
    upper bound on every possible post-hoc monotone recalibration of the existing critic
    output — Platt included. If this arm does not close the gap to `zhead`, no amount of
    rescaling ever will, and that is the case where A6's head is genuinely load-bearing.
    """
    order = np.argsort(v, kind='mergesort')
    xs, ys = v[order].astype(np.float64), y[order].astype(np.float64)
    # Each block holds (sum of y, count); merge left while the running means invert.
    vals, counts = [], []
    for target in ys:
        vals.append(target)
        counts.append(1.0)
        while len(vals) > 1 and vals[-2] / counts[-2] > vals[-1] / counts[-1]:
            v_last, c_last = vals.pop(), counts.pop()
            vals[-1] += v_last
            counts[-1] += c_last
    fitted = np.concatenate([np.full(int(c), s / c) for s, c in zip(vals, counts)])
    return xs, fitted


def apply_isotonic(v, model):
    xs, fitted = model
    return np.interp(v, xs, fitted)


class ZHead(nn.Module):
    """A6-lite: a small head on the FROZEN critic trunk, trained on the game outcome.

    Deliberately reads exactly what A6's proposed `V_win` head would — the pooled board
    block the critic already computes, plus globals and the privileged vector — so the
    comparison against the recalibration arms is the real A6 decision and not a strawman.
    Outputs a logit; BCE against z mapped to {0,1}.
    """

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------- #
# Critic plumbing
# --------------------------------------------------------------------------- #
def load_critic(path, device):
    meta = load_critic_checkpoint(path, map_location=device)
    enc = get_encoder(meta['obs_version'])
    critic = Critic(device=device, hidden_dim=meta['hidden_dim'], obs_encoder=enc,
                    arch=meta['arch']).to(device)
    critic.load_state_dict(meta['state_dict'])
    critic.eval()
    return critic, enc, meta


def critic_outputs(critic, data, meta, *, batch=2048, want_features=False):
    """Denormalised critic value per sample, and optionally its frozen head inputs.

    The denormalisation mirrors `LookaheadCriticBot._critic_root_values` exactly
    (`raw * return_std + return_mean`), so `v` here is the number search actually consumes,
    not the raw network output.
    """
    n = len(data['z'])
    vals = np.empty(n, dtype=np.float64)
    feats = [] if want_features else None
    scale = meta.get('return_std')
    shift = meta.get('return_mean')
    if scale is None or shift is None:
        raise SystemExit(
            f'checkpoint has no return_mean/return_std, so the scale search consumes cannot '
            f'be reproduced here. Pick a checkpoint saved after that field was added.')
    with torch.no_grad():
        for i in range(0, n, batch):
            bd = torch.from_numpy(data['boards'][i:i + batch])
            gl = torch.from_numpy(data['globals'][i:i + batch])
            oh = torch.from_numpy(data['opp_onehots'][i:i + batch])
            pv = torch.from_numpy(data['privileged'][i:i + batch])
            raw = critic.value_from_tensors(bd, gl, oh, pv)
            vals[i:i + batch] = raw.numpy() * scale + shift
            if want_features:
                pooled = critic._pooled(bd)
                feats.append(torch.cat([pooled, gl, pv], dim=-1).numpy())
    return vals, (np.concatenate(feats) if want_features else None)


def train_zhead(f_tr, y_tr, f_va, *, hidden, epochs, batch, lr, seed, device):
    torch.manual_seed(seed)
    head = ZHead(f_tr.shape[1], hidden).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    X = torch.from_numpy(f_tr)
    Y = torch.from_numpy(y_tr.astype(np.float32))
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        perm = rng.permutation(len(Y))
        head.train()
        for i in range(0, len(Y), batch):
            b = perm[i:i + batch]
            loss = nn.functional.binary_cross_entropy_with_logits(head(X[b]), Y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        print(f'    zhead epoch {ep + 1}/{epochs}  train bce={loss.item():.4f}')
    head.eval()
    with torch.no_grad():
        logits = torch.cat([head(torch.from_numpy(f_va[i:i + 4096]))
                            for i in range(0, len(f_va), 4096)])
    return torch.sigmoid(logits).numpy().astype(np.float64)


# --------------------------------------------------------------------------- #
# Mode: calib — the gate
# --------------------------------------------------------------------------- #
def mode_calib(args):
    device = torch.device('cpu')
    critic, enc, meta = load_critic(args.critic, device)
    print(f'critic {os.path.basename(args.critic)}: arch={meta["arch"]} obs=v{meta["obs_version"]} '
          f'hidden={meta["hidden_dim"]} return_mean={meta["return_mean"]:.4f} '
          f'return_std={meta["return_std"]:.4f}')

    tr, va = load_exit_dataset(args.data, args.max_samples, args.val_rounds, args.seed)
    if tr['boards'].shape[1] != enc.board_channels:
        raise SystemExit('dataset board channels do not match the critic checkpoint encoder')
    if args.max_val and len(va['z']) > args.max_val:
        # Held-out shards are not subsampled by `load_exit_dataset`, and the frozen-feature
        # cache below is pool_width+global+priv wide per sample — cap it explicitly rather
        # than discover the memory ceiling the hard way.
        idx = np.sort(np.random.default_rng(args.seed).choice(len(va['z']), args.max_val,
                                                              replace=False))
        va = {k: v[idx] for k, v in va.items()}
        print(f'held-out capped to {len(va["z"]):,} samples (--max-val)')

    # z is stored from the MOVER's perspective and the observation is ego-centric to that
    # same mover (`expert_iteration.py: label_last`), so the critic's output lines up with
    # the label directly — no sign flip.
    y_tr = ((tr['z'] + 1.0) / 2.0).astype(np.float64)
    y_va = ((va['z'] + 1.0) / 2.0).astype(np.float64)
    print(f'base rate: train {y_tr.mean():.3f}  held-out {y_va.mean():.3f}')

    print('  running critic over train ...')
    v_tr, f_tr = critic_outputs(critic, tr, meta, want_features=True)
    print('  running critic over held-out ...')
    v_va, f_va = critic_outputs(critic, va, meta, want_features=True)
    print(f'  denormalised critic value: train mean={v_tr.mean():+.4f} std={v_tr.std():.4f}  '
          f'held-out mean={v_va.mean():+.4f} std={v_va.std():.4f}')

    results = {}

    # `as_is` reads the reward-unit value as if it were already on the [-1,1] win scale.
    # That is not a strawman: it is precisely the assumption `PuctBot._select` makes when it
    # sums Q with an AlphaZero-tuned exploration term.
    results['as_is'] = score_all(np.clip((v_va + 1.0) / 2.0, 0.0, 1.0), y_va)

    ab = fit_platt(v_tr, y_tr)
    print(f'  platt fit on TRAIN only: a={ab[0]:+.4f} b={ab[1]:+.4f}')
    results['platt'] = score_all(apply_platt(v_va, ab), y_va)

    iso = fit_isotonic(v_tr, y_tr)
    results['isotonic'] = score_all(apply_isotonic(v_va, iso), y_va)

    print(f'  training zhead on frozen features ({f_tr.shape[1]} dims) ...')
    p_zhead = train_zhead(f_tr, y_tr, f_va, hidden=args.hidden, epochs=args.epochs,
                          batch=args.batch, lr=args.lr, seed=args.seed, device=device)
    results['zhead'] = score_all(p_zhead, y_va)

    print(f'\n{"=" * 78}\nHELD-OUT CALIBRATION — {len(y_va):,} samples, split by round\n{"=" * 78}')
    print(f'{"arm":<12}{"brier":>10}{"logloss":>10}{"ECE":>9}{"AUC":>9}{"acc":>9}   what it is')
    notes = {
        'as_is': 'shipped scale, read as a probability',
        'platt': '2 floats in the checkpoint',
        'isotonic': 'best possible rescaling of that scalar',
        'zhead': 'A6-lite: new head on the frozen trunk',
    }
    for k in ('as_is', 'platt', 'isotonic', 'zhead'):
        r = results[k]
        print(f'{k:<12}{r["brier"]:10.4f}{r["logloss"]:10.4f}{r["ece"]:9.4f}'
              f'{r["auc"]:9.4f}{r["acc"]:9.1%}   {notes[k]}')

    shared_auc = results['isotonic']['auc']
    d_auc = results['zhead']['auc'] - shared_auc
    d_brier = results['isotonic']['brier'] - results['zhead']['brier']
    # se(AUC) via the Hanley-McNeil approximation, so the gap is read against noise rather
    # than against zero.
    n_pos = int((y_va > 0.5).sum())
    n_neg = len(y_va) - n_pos
    a = shared_auc
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = float(np.sqrt(max(a * (1 - a) + (n_pos - 1) * (q1 - a * a)
                           + (n_neg - 1) * (q2 - a * a), 0.0) / (n_pos * n_neg)))
    print(f'\n  recalibration arms share AUC {shared_auc:.4f} (monotone maps of one scalar); '
          f'se ~ {se:.4f}')
    print(f'  zhead AUC advantage: {d_auc:+.4f}  ({d_auc / max(se, 1e-9):+.1f} se)')
    print(f'  brier: best rescaling {results["isotonic"]["brier"]:.4f} vs zhead '
          f'{results["zhead"]["brier"]:.4f}  ({d_brier:+.4f})')

    print(f"""
HOW TO READ — the AUC column decides, not brier.
  `as_is`, `platt` and `isotonic` are three monotone maps of the SAME critic scalar, so they
  share an AUC by construction and differ only in calibration. `isotonic` is the ceiling on
  every possible post-hoc rescaling. `zhead` is the only arm that reads the trunk, so it is
  the only one whose AUC can move.

  zhead AUC within ~2 se of the shared AUC -> the scalar already carries every bit of
       win-prediction the trunk has, and only the MAPPING was broken. Ship Platt's two floats
       into `save_critic_checkpoint` and delete the hack; A6's second head buys nothing that
       two numbers do not. Compare `as_is` brier against `platt`/`isotonic` to see how much
       the current mapping is costing.
  zhead AUC clearly above it -> the shaped-return objective really has destroyed win
       information the trunk still holds, no rescaling can recover it, and A6 is justified.
       Then build it KataGo-style (shared trunk, aux head) together with A7 rather than as a
       standalone arch.

  Caveat to state whenever this is quoted: `zhead` has far more parameters and a far richer
  input than a 2-parameter fit. That asymmetry is the A6 proposal itself, not a flaw in the
  comparison — but it does mean a SMALL zhead advantage is the expected outcome even under
  cause (a), which is why the verdict is keyed on the se-scaled gap and not on the sign.""")


# --------------------------------------------------------------------------- #
# Mode: puct — does the miscalibration actually distort the search?
# --------------------------------------------------------------------------- #
def mode_puct(args):
    """A bad probability only matters if it changes a move. This prices that.

    `PuctBot._select` scores each edge `sign*Q + c_puct*P*sqrt(sum N)/(1 + N)`. AlphaZero's
    c_puct is tuned for Q in [-1, 1]; here Q is a denormalised shaped return whose spread is
    whatever the reward scale happens to be. If that spread is much SMALLER than [-1,1], the
    exploration term dominates and the search drifts toward the prior (it under-uses the
    critic); much LARGER and Q swamps exploration and the tree collapses onto one line.
    """
    device = torch.device('cpu')
    critic, enc, meta = load_critic(args.critic, device)
    tr, va = load_exit_dataset(args.data, args.max_samples, args.val_rounds, args.seed)
    v, _ = critic_outputs(critic, va if args.on_heldout else tr, meta)

    spread = float(v.std())
    q_range = float(np.percentile(v, 95) - np.percentile(v, 5))
    print(f'critic {os.path.basename(args.critic)} — Q as PuctBot consumes it '
          f'(denormalised, {len(v):,} states)')
    print(f'  mean {v.mean():+.4f}   std {spread:.4f}   p5..p95 span {q_range:.4f}   '
          f'min {v.min():+.4f}   max {v.max():+.4f}')
    print(f'  an AlphaZero-scale Q (win probability in [-1,1]) would have a p5..p95 span '
          f'near 1.0-2.0 for comparison')

    print(f'\nexploration term  c_puct * P * sqrt(sum N) / (1 + N)   at c_puct={args.c_puct}')
    print(f'{"prior P":>9}{"N=0, sumN=1":>14}{"N=1, sumN=8":>14}{"N=4, sumN=32":>15}')
    for p in (0.05, 0.2, 0.5):
        u0 = args.c_puct * p * 1.0 / 1.0
        u1 = args.c_puct * p * np.sqrt(8) / 2.0
        u4 = args.c_puct * p * np.sqrt(32) / 5.0
        print(f'{p:>9.2f}{u0:>14.4f}{u1:>14.4f}{u4:>15.4f}')

    u_typ = args.c_puct * 0.2 * np.sqrt(8) / 2.0
    print(f'\n  Q spread / typical U  =  {spread:.4f} / {u_typ:.4f}  =  {spread / u_typ:.2f}')
    print(f"""
HOW TO READ
  Ratio near 1 -> Q and exploration are commensurate and c_puct is doing what it was tuned
       to do. The miscalibration is cosmetic for PUCT and A6's search argument is weak.
  Ratio << 1 -> the critic barely moves selection; the tree is following the policy prior and
       whatever the critic knows is being drowned. Note this is fixable by rescaling alone
       (or by re-tuning c_puct) and does NOT by itself justify a second head — cross-check
       against `calib`'s AUC column before concluding anything.
  Ratio >> 1 -> Q dominates, exploration collapses, and the search is effectively a greedy
       critic walk.""")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--critic', required=True, help='critic checkpoint to test')
    common.add_argument('--data', default='data/exit/**/round*.npz',
                        help='shards whose z-key is the GAME OUTCOME (not shaped returns)')
    common.add_argument('--max-samples', type=int, default=60000)
    common.add_argument('--val-rounds', type=int, default=5)
    common.add_argument('--seed', type=int, default=0)

    c = sub.add_parser('calib', parents=[common], help='the A6 gate: rescaling vs a z-head')
    c.add_argument('--max-val', type=int, default=40000,
                   help='cap held-out samples; the frozen-feature cache is wide')
    c.add_argument('--hidden', type=int, default=128)
    c.add_argument('--epochs', type=int, default=4)
    c.add_argument('--batch', type=int, default=256)
    c.add_argument('--lr', type=float, default=3e-4)
    c.set_defaults(func=mode_calib)

    p = sub.add_parser('puct', parents=[common], help='is the scale actually distorting PUCT')
    p.add_argument('--c-puct', type=float, default=1.5, help="PuctBot's default")
    p.add_argument('--on-heldout', action='store_true')
    p.set_defaults(func=mode_puct)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
