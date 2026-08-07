"""Q1 — does the board carry predictive power for V(s)? (docs/next_iteration.md, P0-Q1)

The whole of P0 assumes the critic's board pathway is worth fixing. If V(s) is
essentially a function of base counts and coin economy — plausible for an uncontested
~11-round race (docs/next_iteration.md §1.4) — then no readout, norm, shared trunk or
auxiliary head can help, because there is nothing for them to expose, and P0 collapses
into P1 (make the game contested). This measures that.

Four modes, cheapest first. Run them in this order; each can settle the question alone.

  distinguish   NO training, NO labels, seconds. Enumerates the legal successors of real
                decision states and asks how often two siblings carry an IDENTICAL
                non-board input (globals ++ opp_onehot ++ privileged). A globals-only
                value function assigns those pairs the same value BY CONSTRUCTION, so it
                cannot rank them — the pair-collision rate is a hard lower bound on how
                board-blind such a critic must be. Advantages in PPO are exactly a
                within-state ranking problem, so this is the quantity that matters.

  fit           Supervised value regression on the ExIt datasets already on disk
                (data/exit/round*.npz — ~30 rounds x ~16k (state -> z) samples). Three
                arms with an identical head:
                  globals  — board block removed (the control)
                  board    — full Critic: own HexConv trunk + split_pool
                  polfeat  — FROZEN policy trunk features, split_pool, detached
                             (a direct simulation of the shared-encoder + stop-gradient
                             design in P0 item 5, i.e. Q1b)
                Held-out split is BY ROUND, not by sample, so consecutive states from one
                trajectory cannot leak across the split.

  siblings      The labelled within-state test. Generates sibling successor sets, labels
                each successor by Monte-Carlo playout (paired seeds across siblings, so
                the within-state differences are common-random-number reduced), then
                scores the fitted arms on the DEMEANED-per-state residual plus pairwise
                ranking accuracy. Also reports a reliability ceiling from two independent
                label halves, so an R^2 can be read against what is achievable at all.

  rank          Trains ON the within-state objective instead of measuring a model that
                was not (docs/next_iteration.md P0'). Margin-weighted pairwise ranking loss
                over the siblings of a state, held out by STATE. Needs a label cache from
                `siblings`, and ALWAYS pass --init-from: without it the ranking arms train
                on a few thousand successors while the regression baseline they are
                compared against saw 120k, and a flat result is confounded with the data
                gap rather than informative about the objective.

Read the result as: pooled R^2 is dominated by ACROSS-state variance (game phase, base
count) which globals explain trivially, so the board's marginal share there will look
small even when it is decisive. The within-state numbers are the ones that matter — and
note they answer a DIFFERENT question from "can you tell who is winning by looking":
across-state value IS predictable and the board does help it (+0.039 R^2).

READ THE BUCKETED TABLE, NOT THE POOLED ONE. The within-state question is really TWO
disjoint sub-problems, and averaging them understates both:

  * ~30% of sibling pairs have IDENTICAL boards (recruit vs recruit vs pass: the coin
    leaves the hand, the board never moves). Only globals can rank those.
  * ~5% have identical non-board inputs and differing boards. Only the board can rank
    those — a globals-only model assigns them the same value BY CONSTRUCTION.

Pooling the two makes every evaluator look equally mediocre. On the pooled metric the
board arm beats a globals-only control by 1.1pp (55.9% vs 54.8%) and the honest reading is
"the board adds nothing"; restricted to the pairs where the board is the only thing that
differs it is 61.0% against a structural 49.2%. That dilution produced this document's
retracted §3.1 conclusion. `distinguish` mode always bucketed; the labelled modes did not,
which is the bug this table fixes.

    python src/app/eval_board_value.py distinguish --games 40 --stride 3
    python src/app/eval_board_value.py fit --max-samples 120000 --epochs 3
    python src/app/eval_board_value.py siblings --states 80 --playouts 16
    python src/app/eval_board_value.py siblings --states 700 --playouts 8 --relabel \
        --labels data/rank_labels.pt          # bigger set for training, ~12 min on 12 cores
    python src/app/eval_board_value.py rank --labels data/rank_labels.pt \
        --init-from data/board_value_probe.pt --epochs 12
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import torch.nn as nn

from src.app.gauntlet import _latest_policy_path
from src.services.bots.evaluation import HeuristicEvaluator
from src.services.bots.lookahead_bot import _clone_state
from src.services.environment.game_record import game_state_from_dict, game_state_to_dict
from src.services.environment.obs_encoders import get_encoder
from src.services.environment.warchest_env import VERB_OF_ACTION, WarChestEnv
from src.services.gauntlet import checkpoint_agent, greedy_fast_agent
from src.services.policy.checkpoint import load_policy_checkpoint
from src.services.policy.policy import HexConv2d, Policy, _split_pool

# kind -> (trunk source, readout, keep the non-board context block).
# 'pool' is today's `_split_pool` flank average; 'flat' is the location-preserving readout
# P0 item 1 proposes (1x1 conv to FLAT_K channels, every cell kept). Comparing the two
# tests whether the flank average is what destroys sibling discrimination.
# `board_solo` drops globals/opp/privileged entirely: it is the direct test of "I can look
# at the board and say who is winning" — can a model, given ONLY the board, do the same?
ARM_SPEC = {
    'globals': (None, None, True),
    'board': ('own', 'pool', True),
    'board_xy': ('own', 'flat', True),
    'polfeat': ('frozen', 'pool', True),
    'polfeat_xy': ('frozen', 'flat', True),
    'board_solo': ('own', 'flat', False),
}
ARMS = tuple(ARM_SPEC)
FLAT_K = 8
WORK_DEFAULT = 'data/board_value_probe.pt'


# --------------------------------------------------------------------------- #
# The arms — identical head, different board block
# --------------------------------------------------------------------------- #
class ValueArm(nn.Module):
    """A Critic stripped to exactly the question: what does the board block add?

    `globals` has no board block at all; `board*` learns its own HexConv trunk;
    `polfeat*` reads a frozen, detached policy trunk (the shared-encoder + stop-gradient
    design). The `_xy` variants swap the flank-average readout for a location-preserving
    one. Everything after the concatenation is identical across arms.
    """

    def __init__(self, kind, *, board_channels, global_dim, priv_dim, hidden,
                 policy_trunk=None):
        super().__init__()
        if kind not in ARM_SPEC:
            raise ValueError(f'kind must be one of {ARMS}, got {kind!r}')
        self.kind = kind
        src, self.readout, self.use_ctx = ARM_SPEC[kind]
        self.trunk = None
        self.reduce = None
        board_block = 0
        if src == 'own':
            self.trunk = nn.Sequential(
                HexConv2d(board_channels, 32), nn.ReLU(),
                HexConv2d(32, hidden), nn.ReLU(),
                HexConv2d(hidden, hidden), nn.ReLU(),
            )
            width = hidden
        elif src == 'frozen':
            if policy_trunk is None:
                raise ValueError(f'{kind} arm needs a frozen policy trunk')
            self.trunk = policy_trunk
            for p in self.trunk.parameters():
                p.requires_grad_(False)
            self.trunk.eval()
            width = _trunk_width(policy_trunk)
        if self.readout == 'pool':
            board_block = 2 * width
        elif self.readout == 'flat':
            self.reduce = nn.Conv2d(width, FLAT_K, kernel_size=1)
            board_block = FLAT_K * 7 * 7
        self.frozen = src == 'frozen'
        head_in = board_block + ((global_dim + 3 + priv_dim) if self.use_ctx else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, board, glob, opp, priv):
        parts = []
        if self.trunk is not None:
            if self.frozen:
                with torch.no_grad():
                    feat = self.trunk(board)
                feat = feat.detach()
            else:
                feat = self.trunk(board)
            parts.append(_split_pool(feat) if self.readout == 'pool'
                         else self.reduce(feat).flatten(1))
        if self.use_ctx:
            parts += [glob, opp, priv]
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]


def _trunk_width(trunk):
    return [m for m in trunk if isinstance(m, HexConv2d)][-1].conv.out_channels


def load_policy(path, device):
    ck = load_policy_checkpoint(path)
    enc = get_encoder(ck['obs_version'])
    pol = Policy(device, hidden_dim=ck['hidden_dim'], obs_encoder=enc)
    pol.load_state_dict(ck['state_dict'])
    pol.eval()
    return pol, enc


# --------------------------------------------------------------------------- #
# Mode: distinguish — no training, no labels
# --------------------------------------------------------------------------- #
def enumerate_siblings(env, encoder, *, max_siblings, rng):
    """Encodings of every legal successor of `env`, whoever it leaves to act.

    Pending-tactic continuations and terminal/truncated states are dropped: the first are
    not comparable, the second are trivially separable and would flatter both arms. That
    filter keeps ~97% of legal successors, but it removes **100% of tactic initiations**
    (they always leave a pending choice), so this instrument is structurally blind to the
    tactic verb — a bounded limitation to state whenever its results are quoted.

    Successors are NOT restricted to one mover: ~17% of sibling sets mix "opponent to
    move" with "I move again", because some actions do not end the turn. That is fine —
    the label is absolute (player 1's frame) and `sign` maps each ego-centric prediction
    into it — but it means a sibling set is not a fixed ply parity.
    """
    acts = list(env.get_possible_actions())
    if len(acts) > max_siblings:
        acts = list(rng.choice(acts, size=max_siblings, replace=False))
    state = _clone_state(env.state)
    sim = WarChestEnv(save_game_history=False)
    evaluator = HeuristicEvaluator()
    out = []
    for a in acts:
        sim.set_state(_clone_state(state))
        try:
            _, _, term, trunc, _ = sim.step(int(a))
        except Exception:
            continue
        if term or trunc or sim.state.pending is not None:
            continue
        obs = encoder.encode(sim)
        out.append({
            'action': int(a),
            'board': obs['board'],
            'global': obs['global'],
            'priv': sim.get_privileged_features(),
            # Stored as the recorder's plain-dict form, not a live GameState: unit
            # classes are built dynamically so a live state neither pickles to a cache
            # nor crosses a process boundary.
            'state': game_state_to_dict(sim.state),
            # Every observation here is EGO-CENTRIC to the successor's active player, but
            # the playout label is always from player 1's perspective. Without this factor
            # roughly half the sibling sets are scored with an inverted predictor, which
            # cancels the signal to ~0 when pooled across states.
            'sign': 1.0 if sim.active_player == 1 else -1.0,
            # Hand-written board-aware control (LookaheadBot's leaf), already in p1's frame.
            'heur': float(evaluator.evaluate(sim, 1)),
        })
    return out


def collect_sibling_sets(agent, encoder, *, n_games, max_siblings, min_siblings, seed,
                         stride=3, cap=None):
    """Sibling sets sampled along real self-play trajectories."""
    rng = np.random.default_rng(seed)
    sets = []
    for g in range(n_games):
        env = WarChestEnv(save_game_history=False)
        env.reset(seed=seed + g)
        for t in range(600):
            if not env.get_possible_actions():
                break
            if t % stride == 0 and env.state.pending is None:
                sibs = enumerate_siblings(env, encoder, max_siblings=max_siblings, rng=rng)
                if len(sibs) >= min_siblings:
                    sets.append(sibs)
                    if cap and len(sets) >= cap:
                        return sets
            _, _, term, trunc, _ = env.step(agent.act(env))
            if term or trunc:
                break
    return sets


def mode_distinguish(args):
    device = torch.device('cpu')
    pol_path = args.policy or _latest_policy_path()
    agent = checkpoint_agent(pol_path, device)
    _, encoder = load_policy(pol_path, device)
    print(f'policy: {os.path.basename(pol_path)}   encoder: v{encoder.version if hasattr(encoder, "version") else "?"}')

    t0 = time.time()
    sets = collect_sibling_sets(agent, encoder, n_games=args.games,
                                max_siblings=args.max_siblings, min_siblings=3,
                                seed=args.seed, stride=args.stride)
    n_sib = sum(len(s) for s in sets)
    print(f'{len(sets)} sibling sets, {n_sib} successors, '
          f'{n_sib / max(len(sets), 1):.1f} per set  ({time.time() - t0:.0f}s)\n')

    # Pairs are bucketed, because the aggregate is diluted: `recruit`/`pass`/
    # `claim_initiative` successors leave the board untouched and move only globals, so
    # they are trivially separable without the board and say nothing about the question.
    # The pairs that decide it are the ones whose BOARDS differ — and, sharpest of all,
    # the ones that also play the SAME verb (two directions for one unit: the same coin
    # leaves the hand, so the globals are near-identical and only the board differs).
    tot = coll_nb = coll_bd = 0
    bd_diff = bd_diff_coll = 0
    same_verb = same_verb_coll = 0
    board_l1, nonboard_l1 = [], []
    per_set_frac = []
    for sibs in sets:
        nb = np.stack([np.concatenate([s['global'], s['priv']]) for s in sibs])
        bd = np.stack([s['board'].ravel() for s in sibs])
        verbs = [VERB_OF_ACTION[s['action']] for s in sibs]
        k = len(sibs)
        local_pairs = local_coll = 0
        for i in range(k):
            for j in range(i + 1, k):
                dnb = float(np.abs(nb[i] - nb[j]).max())
                dbd = float(np.abs(bd[i] - bd[j]).max())
                nb_same, bd_same = dnb <= args.eps, dbd <= args.eps
                tot += 1
                local_pairs += 1
                nonboard_l1.append(float(np.abs(nb[i] - nb[j]).sum()))
                board_l1.append(float(np.abs(bd[i] - bd[j]).sum()))
                coll_nb += nb_same
                coll_bd += bd_same
                local_coll += nb_same
                if not bd_same:
                    bd_diff += 1
                    bd_diff_coll += nb_same
                    if verbs[i] == verbs[j]:
                        same_verb += 1
                        same_verb_coll += nb_same
        per_set_frac.append(local_coll / max(local_pairs, 1))

    def pct(a, b):
        return f'{a / max(b, 1):7.2%}'

    print(f'PAIR COLLISIONS among sibling successors (identical to within eps={args.eps})')
    print(f'  all pairs                                    n={tot:<7} '
          f'non-board identical {pct(coll_nb, tot)}   board identical {pct(coll_bd, tot)}')
    print(f'  pairs whose BOARDS differ                    n={bd_diff:<7} '
          f'non-board identical {pct(bd_diff_coll, bd_diff)}   <- board could matter here')
    print(f'  ...and that play the SAME verb               n={same_verb:<7} '
          f'non-board identical {pct(same_verb_coll, same_verb)}   <- the sharpest bucket')
    print(f'  mean per-state collision fraction (all pairs)  {np.mean(per_set_frac):7.2%}')
    print(f'  mean L1 distance between siblings — board {np.mean(board_l1):8.3f}   '
          f'non-board {np.mean(nonboard_l1):8.3f}')
    print("""
HOW TO READ — the "SAME verb, boards differ" line is the one to read.
  High (>40%)  a globals-only critic CANNOT rank a large share of the choices PPO has to
               make; the board is load-bearing for advantages and P0 is justified whatever
               the pooled fit says.
  Low  (<10%)  globals numerically separate almost every sibling. That is NOT the same as
               ranking them CORRECTLY — separability is necessary, not sufficient — so this
               mode cannot clear P0 on its own. Run `fit`, then `siblings`.""")


# --------------------------------------------------------------------------- #
# Mode: fit — supervised value regression on the ExIt datasets
# --------------------------------------------------------------------------- #
def load_exit_dataset(pattern, max_samples, val_rounds, seed):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f'no dataset files matched {pattern!r}')

    def rkey(p):
        base = os.path.basename(p)
        digits = ''.join(c for c in base if c.isdigit())
        return int(digits) if digits else 0

    files.sort(key=rkey)
    val_files = files[-val_rounds:]
    train_files = files[:-val_rounds]
    print(f'{len(files)} dataset files — train on {len(train_files)}, '
          f'held out (by round, no trajectory leakage): {[os.path.basename(f) for f in val_files]}')

    def stack(fs):
        keys = ('boards', 'globals', 'opp_onehots', 'privileged', 'z')
        parts = {k: [] for k in keys}
        for f in fs:
            d = np.load(f)
            for k in keys:
                parts[k].append(d[k])
        return {k: np.concatenate(v) for k, v in parts.items()}

    tr, va = stack(train_files), stack(val_files)
    if max_samples and len(tr['z']) > max_samples:
        idx = np.random.default_rng(seed).choice(len(tr['z']), max_samples, replace=False)
        tr = {k: v[idx] for k, v in tr.items()}
    print(f'train {len(tr["z"]):,} samples   held-out {len(va["z"]):,} samples   '
          f'z std={va["z"].std():.4f}')
    return tr, va


def train_arm(kind, tr, va, *, hidden, epochs, batch, lr, policy_trunk, device, seed):
    torch.manual_seed(seed)
    arm = ValueArm(kind, board_channels=tr['boards'].shape[1],
                   global_dim=tr['globals'].shape[1], priv_dim=tr['privileged'].shape[1],
                   hidden=hidden, policy_trunk=policy_trunk).to(device)
    n_par = sum(p.numel() for p in arm.trainable())
    opt = torch.optim.Adam(arm.trainable(), lr=lr)
    n = len(tr['z'])
    T = {k: torch.from_numpy(v) for k, v in tr.items()}
    V = {k: torch.from_numpy(v) for k, v in va.items()}
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(n)
        arm.train()
        if arm.kind == 'polfeat':
            arm.trunk.eval()
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            pred = arm(T['boards'][b], T['globals'][b], T['opp_onehots'][b],
                       T['privileged'][b])
            loss = ((pred - T['z'][b]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        arm.eval()
        with torch.no_grad():
            pv = torch.cat([arm(V['boards'][i:i + 2048], V['globals'][i:i + 2048],
                                V['opp_onehots'][i:i + 2048], V['privileged'][i:i + 2048])
                            for i in range(0, len(V['z']), 2048)])
            mse = float(((pv - V['z']) ** 2).mean())
        print(f'    {kind:<8} epoch {ep + 1}/{epochs}  held-out mse={mse:.4f}  '
              f'({time.time() - t0:.0f}s)')
    with torch.no_grad():
        pv = torch.cat([arm(V['boards'][i:i + 2048], V['globals'][i:i + 2048],
                            V['opp_onehots'][i:i + 2048], V['privileged'][i:i + 2048])
                        for i in range(0, len(V['z']), 2048)]).numpy()
    z = va['z']
    return arm, {'mse': float(((pv - z) ** 2).mean()), 'mae': float(np.abs(pv - z).mean()),
                 'r2': float(1 - ((pv - z) ** 2).mean() / z.var()), 'params': n_par}


def mode_fit(args):
    device = torch.device('cpu')
    pol_path = args.policy or _latest_policy_path()
    policy, encoder = load_policy(pol_path, device)
    tr, va = load_exit_dataset(args.data, args.max_samples, args.val_rounds, args.seed)
    if tr['boards'].shape[1] != encoder.board_channels:
        raise SystemExit('dataset board channels do not match the policy checkpoint encoder')

    results, arms = {}, {}
    for kind in (args.arms or list(ARMS)):
        print(f'  fitting {kind} ...')
        arm, res = train_arm(kind, tr, va, hidden=args.hidden, epochs=args.epochs,
                             batch=args.batch, lr=args.lr,
                             policy_trunk=policy.board_encoder if kind.startswith('polfeat') else None,
                             device=device, seed=args.seed)
        results[kind], arms[kind] = res, arm

    print(f'\nHELD-OUT (pooled, across-state) — z var = {va["z"].var():.4f}')
    print(f'{"arm":<10}{"mse":>10}{"mae":>10}{"R2":>10}{"trainable params":>19}')
    for k, r in results.items():
        print(f'{k:<10}{r["mse"]:10.4f}{r["mae"]:10.4f}{r["r2"]:10.4f}{r["params"]:19,}')
    if 'globals' in results and 'board' in results:
        g, b = results['globals']['r2'], results['board']['r2']
        print(f'\n  board - globals pooled R2 gap: {b - g:+.4f}')
    print("""
  Do NOT decide P0 on this table alone: pooled variance is dominated by game phase and
  base count, which globals explain for free. Run `siblings` for the number that matters.""")

    torch.save({'arms': {k: a.state_dict() for k, a in arms.items()},
                'hidden': args.hidden, 'policy_path': pol_path,
                'dims': {'board_channels': tr['boards'].shape[1],
                         'global_dim': tr['globals'].shape[1],
                         'priv_dim': tr['privileged'].shape[1]}}, args.work)
    print(f'\nfitted arms saved to {args.work} (used by `siblings`)')


# --------------------------------------------------------------------------- #
# Mode: siblings — labelled within-state test
# --------------------------------------------------------------------------- #
OUTCOMES = ('decisive', 'truncated', 'ply-cap', 'ERROR')


def playout(state, agent, *, seed, max_plies=400):
    """Play a state to the end. -> (+1 if player 1 wins / -1 / 0, outcome-kind).

    The outcome kind is returned, not swallowed: an aborted playout also scores 0.0, which
    is indistinguishable from a genuine draw in the label but is silent corruption rather
    than information. `_label_sets` aggregates and reports the rate.
    """
    env = WarChestEnv(save_game_history=False)
    env.set_state(_clone_state(state))
    np.random.seed(seed)
    for _ in range(max_plies):
        if not env.get_possible_actions():
            return (-1.0 if env.active_player == 1 else 1.0), 'decisive'
        pid = env.active_player
        try:
            a = agent.act(env)
            _, _, term, trunc, info = env.step(a)
            if not info['action'].is_valid:
                _, _, term, trunc, _ = env.make_random_step()
        except Exception:
            return 0.0, 'ERROR'
        if term:
            return (1.0 if pid == 1 else -1.0), 'decisive'
        if trunc:
            return 0.0, 'truncated'
    return 0.0, 'ply-cap'


# --------------------------------------------------------------------------- #
# Monte-Carlo labelling, parallel across processes
# --------------------------------------------------------------------------- #
_W = {}


def _worker_init(pol_path, playout_bot):
    torch.set_num_threads(1)  # 12 workers each spawning 8 BLAS threads would thrash
    if playout_bot == 'greedy':
        _W['agent'] = greedy_fast_agent()
    elif playout_bot == 'lookahead':
        # Built inside the worker on purpose: LookaheadBot monkeypatches its sim env's
        # `_draw_one` and is therefore unpicklable, so it can never be sent across.
        from src.services.gauntlet import lookahead_agent
        _W['agent'] = lookahead_agent(time_budget=0.05)
    else:
        _W['agent'] = checkpoint_agent(pol_path, torch.device('cpu'))


def _worker_label(task):
    """(index, state-dict, seeds) -> (index, mean p1-frame outcome, per-kind outcome counts)."""
    idx, sd, seeds = task
    state = game_state_from_dict(sd)
    res = [playout(state, _W['agent'], seed=s) for s in seeds]
    counts = tuple(sum(1 for _, k in res if k == kind) for kind in OUTCOMES)
    return idx, float(np.mean([z for z, _ in res])), counts


def _label_sets(sets, pol_path, args):
    """Fill `z0`/`z1` on every successor. Two independent halves give the reliability ceiling.

    Siblings of one state share the playout seed stream (common random numbers), so the
    *within-state differences* — the only thing scored — carry far less noise than the
    absolute values do.
    """
    # Unit classes are built dynamically (`units/__init__.py`), so a live GameState is
    # unpicklable and cannot cross a process boundary. Ship the plain-dict form the game
    # recorder already defines and rebuild it inside the worker.
    tasks = []
    for si, sibs in enumerate(sets):
        sds = [sb['state'] for sb in sibs]
        for half in (0, 1):
            for bi, sd in enumerate(sds):
                seeds = [args.seed + 7919 * si + 104729 * half + r
                         for r in range(args.playouts)]
                tasks.append(((si, half, bi), sd, seeds))
    n_workers = max(1, min(args.n_workers, len(tasks)))
    t0 = time.time()
    print(f'labelling {len(tasks)} successor-halves x {args.playouts} playouts '
          f'on {n_workers} workers')
    if n_workers == 1:
        _worker_init(pol_path, args.playout_bot)
        results = [_worker_label(t) for t in tasks]
    else:
        from concurrent.futures import ProcessPoolExecutor
        results = []
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init,
                                 initargs=(pol_path, args.playout_bot)) as ex:
            for i, r in enumerate(ex.map(_worker_label, tasks, chunksize=4)):
                results.append(r)
                if i % 500 == 0:
                    print(f'  {i}/{len(tasks)} ({time.time() - t0:.0f}s)', flush=True)
    totals = np.zeros(len(OUTCOMES), dtype=np.int64)
    for (si, half, bi), z, counts in results:
        sets[si][bi][f'z{half}'] = z
        totals += np.asarray(counts, dtype=np.int64)
    n = int(totals.sum())
    print(f'labelling done ({time.time() - t0:.0f}s)   playout outcomes over {n}: '
          + '  '.join(f'{k} {int(c)} ({c / max(n, 1):.1%})' for k, c in zip(OUTCOMES, totals)))
    n_err = int(totals[OUTCOMES.index('ERROR')])
    if n_err:
        print(f'  !! {n_err / max(n, 1):.2%} of playouts ABORTED on an exception and were '
              f'labelled 0.0 — that is a draw in the label but silent corruption in fact. '
              f'Non-trivial rates invalidate the label precision this run reports.')


def _demean(vals):
    v = np.asarray(vals, dtype=np.float64)
    return v - v.mean()


def _score(preds_per_set, y_per_set):
    """Within-state scores: pooled R^2/corr, per-state Spearman, pairwise + top-1 accuracy."""
    p, y = np.concatenate(preds_per_set), np.concatenate(y_per_set)
    ok = tot = ties = top1 = top1_chance = n_states = 0
    rhos = []
    for pp, yy in zip(preds_per_set, y_per_set):
        k = len(pp)
        for i in range(k):
            for j in range(i + 1, k):
                if yy[i] == yy[j]:
                    continue
                tot += 1
                d = (pp[i] - pp[j]) * (yy[i] - yy[j])
                # A predictor that TIES two siblings has expressed no preference; scoring
                # that as a miss punishes coarse evaluators for the wrong reason. The
                # hand-written heuristic ties on ~68% of pairs, which turned its accuracy
                # into 20% (far "below chance") — an artefact, not a finding.
                if d == 0:
                    ties += 1
                    ok += 0.5
                else:
                    ok += float(d > 0)
        if pp.std() > 0 and yy.std() > 0:
            rp = np.argsort(np.argsort(pp)).astype(float)
            ry = np.argsort(np.argsort(yy)).astype(float)
            rhos.append(float(np.corrcoef(rp, ry)[0, 1]))
        n_states += 1
        top1 += int(yy[int(np.argmax(pp))] == yy.max())
        top1_chance += float(np.mean(yy == yy.max()))
    return {
        'r2': 1 - ((p - y) ** 2).mean() / y.var(),
        'corr': float(np.corrcoef(p, y)[0, 1]) if p.std() > 0 else float('nan'),
        'spearman': float(np.mean(rhos)) if rhos else float('nan'),
        'pair': ok / max(tot, 1),
        'tied': ties / max(tot, 1),
        'top1': top1 / max(n_states, 1),
        'top1_chance': top1_chance / max(n_states, 1),
    }


# --------------------------------------------------------------------------- #
# Pair buckets — the within-state question is TWO disjoint sub-problems
# --------------------------------------------------------------------------- #
# `distinguish` mode always bucketed its pairs; the labelled modes pooled them, and that
# dilution is what produced the (now retracted) "the board adds nothing within a state"
# reading. On pairs whose boards are identical only globals can rank; on pairs whose
# non-board inputs are identical only the board can, and a globals-only model is pinned to
# 50% there BY CONSTRUCTION. Averaging the two understates both.
BUCKETS = (
    ('all pairs', lambda bd, nb, sv: True),
    ('board differs', lambda bd, nb, sv: bd),
    ('board differs, non-board SAME', lambda bd, nb, sv: bd and not nb),
    ('board differs, same verb', lambda bd, nb, sv: bd and sv),
    ('board IDENTICAL', lambda bd, nb, sv: not bd),
)


def _bucket_pairs(sets, eps=1e-6):
    """Per-set `[(i, j, board_differs, nonboard_differs, same_verb)]` over sibling pairs."""
    out = []
    for sibs in sets:
        nb = np.stack([np.concatenate([s['global'], s['priv']]) for s in sibs])
        bd = np.stack([s['board'].ravel() for s in sibs])
        vb = [VERB_OF_ACTION[s['action']] for s in sibs]
        ps = []
        for i in range(len(sibs)):
            for j in range(i + 1, len(sibs)):
                ps.append((i, j,
                           float(np.abs(bd[i] - bd[j]).max()) > eps,
                           float(np.abs(nb[i] - nb[j]).max()) > eps,
                           vb[i] == vb[j]))
        out.append(ps)
    return out


def _pair_acc(preds_per_set, y_per_set, pairs_per_set, keep):
    """Pairwise accuracy over the pairs `keep` selects. -> (acc, n_pairs, tie_rate).

    Ties score 0.5 and the tie rate is returned alongside: a coarse evaluator that cannot
    express a preference must not be scored as if it had answered wrongly, and the tie rate
    is itself the interesting number (a globals-only model ties ~90% of the board-only
    bucket, which is the whole point of the bucket).
    """
    ok = tot = ties = 0
    for pp, yy, ps in zip(preds_per_set, y_per_set, pairs_per_set):
        for i, j, bd, nb, sv in ps:
            if not keep(bd, nb, sv) or yy[i] == yy[j]:
                continue
            tot += 1
            d = (pp[i] - pp[j]) * (yy[i] - yy[j])
            if d == 0:
                ties += 1
                ok += 0.5
            else:
                ok += float(d > 0)
    return ok / max(tot, 1), tot, ties / max(tot, 1)


def _report_buckets(evals, y_sets, sets, eps=1e-6):
    """Bucketed pairwise-accuracy table: rows are buckets, columns are evaluators."""
    pairs = _bucket_pairs(sets, eps=eps)
    n_bd_only = sum(1 for ps in pairs for _, _, bd, nb, _ in ps if bd and not nb)
    n_tot = sum(len(ps) for ps in pairs)
    print(f'\nBUCKETED PAIRWISE ACCURACY (ties = 0.5, "t" = tie rate) — {n_tot} sibling pairs, '
          f'{n_bd_only} ({n_bd_only / max(n_tot, 1):.1%}) board-only')
    print(f'{"bucket":<32}{"n":>7}' + ''.join(f'{short:>15}' for short, _, _ in evals))
    for name, keep in BUCKETS:
        cells, n_pairs = '', 0
        for _, _, preds in evals:
            a, n_pairs, ti = _pair_acc(preds, y_sets, pairs, keep)
            cells += f'{a:8.1%} t{ti:<5.0%}'
        print(f'{name:<32}{n_pairs:>7}{cells}')


def mode_siblings(args):
    device = torch.device('cpu')
    blob = torch.load(args.work, map_location='cpu', weights_only=False)
    pol_path = blob['policy_path']
    policy, encoder = load_policy(pol_path, device)
    cache = args.labels or (os.path.splitext(args.work)[0] + '_labels.pt')

    if os.path.exists(cache) and not args.relabel:
        sets = torch.load(cache, map_location='cpu', weights_only=False)['sets']
        print(f'loaded cached labels: {cache}  ({len(sets)} sets, '
              f'{sum(len(s) for s in sets)} successors) — pass --relabel to regenerate')
    else:
        agent = (greedy_fast_agent() if args.playout_bot == 'greedy'
                 else checkpoint_agent(pol_path, device))
        print(f'policy {os.path.basename(pol_path)}   playout bot: {args.playout_bot}')
        t0 = time.time()
        sets = collect_sibling_sets(checkpoint_agent(pol_path, device), encoder,
                                    n_games=args.games, max_siblings=args.max_siblings,
                                    min_siblings=4, seed=args.seed, stride=args.stride,
                                    cap=args.states)
        print(f'{len(sets)} sibling sets, {sum(len(s) for s in sets)} successors '
              f'({time.time() - t0:.0f}s)')
        _label_sets(sets, pol_path, args)
        # `state` is kept deliberately: a critic from a different obs era encodes the same
        # position into different tensors, so scoring one means re-encoding from the raw
        # state. Without this the cache silently locks you to one obs version.
        torch.save({'sets': sets, 'playouts': args.playouts,
                    'obs_version': getattr(encoder, 'version', None)}, cache)
        print(f'labels cached to {cache} — re-score other critics without re-running playouts')

    # --- reliability: agreement of two independent label halves ---
    a = np.concatenate([_demean([s['z0'] for s in sibs]) for sibs in sets])
    b = np.concatenate([_demean([s['z1'] for s in sibs]) for sibs in sets])
    rel = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')
    # Spearman-Brown: the scored label is the MEAN of both halves, so its reliability is
    # 2r/(1+r), not r. That is the achievable R^2; the achievable corr is its square root.
    rel_avg = 2 * rel / (1 + rel)
    y_sets = [_demean([0.5 * (s['z0'] + s['z1']) for s in sibs]) for sibs in sets]
    y = np.concatenate(y_sets)
    var_true = rel_avg * y.var()
    print(f'\nLABEL RELIABILITY  corr(half A, half B) = {rel:.3f} on the within-state '
          f'residual (playouts/half as cached)')
    if not (rel > 0):
        # A negative half-to-half correlation means the two label halves disagree more than
        # chance: at this sample size there is no measurable within-state signal at all. The
        # Spearman-Brown ceiling and every `corr` below are then undefined, not small — say so
        # instead of printing `nan` and letting a reader treat the table as a null result.
        print(f'  *** NO MEASURABLE SIGNAL: reliability is <= 0, so the ceiling is undefined '
              f'and every score below is noise around a target that does not exist at this '
              f'sample size ({len(sets)} states x {sum(len(s) for s in sets)} successors). '
              f'Raise --states and especially --playouts; do not read the tables.')
    else:
        print(f'  ceiling for the AVERAGED label (Spearman-Brown 2r/(1+r)): '
              f'R2 <= {rel_avg:.3f}, corr <= {np.sqrt(rel_avg):.3f}')
        print(f'  measured within-state residual var {y.var():.4f} = true signal '
              f'{var_true:.4f} (std {np.sqrt(var_true):.3f}) + label noise '
              f'{y.var() - var_true:.4f}')

    dims = blob['dims']
    scored = []
    for kind in ARMS:
        if kind not in blob['arms']:
            continue
        arm = ValueArm(kind, board_channels=dims['board_channels'],
                       global_dim=dims['global_dim'], priv_dim=dims['priv_dim'],
                       hidden=blob['hidden'],
                       policy_trunk=policy.board_encoder if kind.startswith('polfeat') else None)
        arm.load_state_dict(blob['arms'][kind])
        arm.eval()
        scored.append((kind, arm, kind))

    for path in (args.critic_path or []) if isinstance(args.critic_path, list) \
            else ([args.critic_path] if args.critic_path else []):
        from src.services.policy.checkpoint import load_critic_checkpoint
        from src.services.policy.policy import Critic
        ck = load_critic_checkpoint(path)
        cenc = get_encoder(ck['obs_version'])
        real = Critic(device, hidden_dim=ck['hidden_dim'], obs_encoder=cenc, arch=ck['arch'])
        real.load_state_dict(ck['state_dict'])
        real.eval()
        # Fingerprint the weights. Two differently-named checkpoints are routinely the
        # same file here (lookahead_critic_v4.pth is a byte-identical copy of
        # warchest_critic_20260727-0506.pth), and scoring one twice looks like a
        # replication when it is not.
        fp = hash(tuple(float(v.sum()) for v in ck['state_dict'].values())) & 0xFFFFFF
        print(f'  critic {os.path.basename(path)}: obs v{ck["obs_version"]} '
              f'hidden={ck["hidden_dim"]} weight-fingerprint={fp:06x}')
        if cenc.global_dim != dims['global_dim']:
            if 'state' not in sets[0][0]:
                print(f'  ! cannot score it: obs v{ck["obs_version"]} needs re-encoding but '
                      f'this label cache predates state retention — re-run with --relabel')
                continue
            print(f'  re-encoding {sum(len(s) for s in sets)} successors under obs '
                  f'v{ck["obs_version"]}')
        # Trim from the RIGHT: these filenames share a long prefix, so a left-truncated
        # label renders two different checkpoints as the same row.
        short = os.path.splitext(os.path.basename(path))[0][-13:]
        scored.append((f'REAL:{short}', (real, cenc), short))

    print(f'\nWITHIN-STATE (per-state mean removed) — {len(y)} successors in {len(sets)} '
          f'states, {len(y) / len(sets):.1f} per state')
    print(f'{"arm":<30}{"R2":>9}{"corr":>8}{"spearman":>10}{"pair-acc":>10}'
          f'{"tied":>7}{"top1":>8}{"chance":>8}')

    evals = []  # (short label, long label, preds per set) — reused by the bucketed table

    def report(name, short, preds):
        s = _score(preds, y_sets)
        print(f'{name:<30}{s["r2"]:9.3f}{s["corr"]:8.3f}{s["spearman"]:10.3f}'
              f'{s["pair"]:9.1%}{s["tied"]:7.0%}{s["top1"]:8.1%}{s["top1_chance"]:8.1%}')
        evals.append((short, name, preds))

    # Control: the hand-written, board-aware leaf every LookaheadBot uses. If THIS ranks
    # siblings while every learned critic sits at chance, the signal exists and is
    # learnable — the failure is the critic, not the game.
    if all('heur' in s for sibs in sets for s in sibs):
        report('HEURISTIC (LookaheadBot leaf)', 'HEURISTIC',
               [_demean([s['heur'] for s in sibs]) for sibs in sets])

    sim = WarChestEnv(save_game_history=False)
    for name, model, short in scored:
        enc_for = None
        if isinstance(model, tuple):
            model, cenc = model
            if cenc.global_dim != dims['global_dim']:
                enc_for = cenc  # different obs era: rebuild the inputs from raw states
        preds = []
        with torch.no_grad():
            for sibs in sets:
                if enc_for is None:
                    bd = torch.from_numpy(np.stack([s['board'] for s in sibs]))
                    gl = torch.from_numpy(np.stack([s['global'] for s in sibs]))
                    pv = torch.from_numpy(np.stack([s['priv'] for s in sibs]))
                else:
                    obs = []
                    for s in sibs:
                        sim.set_state(game_state_from_dict(s['state']))
                        o = enc_for.encode(sim)
                        obs.append((o['board'], o['global'], enc_for.encode_privileged(sim)))
                    bd = torch.from_numpy(np.stack([o[0] for o in obs]))
                    gl = torch.from_numpy(np.stack([o[1] for o in obs]))
                    pv = torch.from_numpy(np.stack([o[2] for o in obs]))
                oh = torch.zeros(len(sibs), 3)
                oh[:, 2] = 1.0
                sign = np.array([s.get('sign', -1.0) for s in sibs])
                v = (model.value_from_tensors(bd, gl, oh, pv) if name.startswith('REAL:')
                     else model(bd, gl, oh, pv))
                preds.append(_demean(sign * v.numpy()))
        report(name, short, preds)

    _report_buckets(evals, y_sets, sets, eps=args.eps)

    ceil = f'{np.sqrt(rel_avg):.3f}' if rel > 0 else 'UNDEFINED — reliability <= 0'
    print(f"""
HOW TO READ — the BUCKETED table decides, the pooled one above dilutes.
  corr is the scale-free number: read it against the ceiling above ({ceil}),
  never against 1.0. R2 can go sharply negative purely from MISCALIBRATION — an arm trained
  on pooled z has a prediction spread sized to across-state variance, which dwarfs the
  within-state residual — so a negative R2 alongside corr ~ 0 means "no signal", not
  "actively wrong".
  `board differs, non-board SAME` is the only bucket that isolates the board: a globals-only
  model is pinned near 50% there with a ~90% tie rate BY CONSTRUCTION, so anything above
  ~55% from a board arm is board-carried within-state signal. `board IDENTICAL` is the
  mirror image (recruit/pass economy choices) and is where a globals arm earns its pooled
  score. Averaging the two — the `all pairs` row — understates both, which is exactly how
  this document once concluded the board adds nothing within a state.
  A dead critic trunk shows up here as a ~90% tie rate on the board-only bucket: the pooled
  block is identically zero, so those siblings get identical values and it CANNOT rank them.""")


# --------------------------------------------------------------------------- #
# Mode: rank — train ON the within-state objective (docs/next_iteration.md P0')
# --------------------------------------------------------------------------- #
def _set_tensors(sibs, use_ctx=True):
    bd = torch.from_numpy(np.stack([s['board'] for s in sibs]))
    gl = torch.from_numpy(np.stack([s['global'] for s in sibs]))
    pv = torch.from_numpy(np.stack([s['priv'] for s in sibs]))
    oh = torch.zeros(len(sibs), 3)
    oh[:, 2] = 1.0
    sg = torch.from_numpy(np.array([s.get('sign', -1.0) for s in sibs], dtype=np.float32))
    y = torch.from_numpy(_demean([0.5 * (s['z0'] + s['z1']) for s in sibs]).astype(np.float32))
    return bd, gl, pv, oh, sg, y


def rank_loss(pred, y, *, kind='rank', temp=1.0):
    """Within-state loss over one state's siblings. `pred`/`y` are 1-D, already our-frame.

    'rank'     margin-weighted pairwise logistic (RankNet). Every ordered pair contributes
               `softplus(-(s_i - s_j)/T)` weighted by |y_i - y_j|, so near-ties — which are
               most pairs, and mostly label noise — barely count, and the clear
               better/worse pairs dominate. Scale-free: it never asks the model to match
               the magnitude of a noisy Monte-Carlo estimate, only its order.
    'listwise' cross-entropy between softmax(pred) and softmax(y) over the siblings.
    'residual' plain MSE against the demeaned label — the most direct, but it inherits the
               label's noise scale.
    """
    if kind == 'residual':
        return ((pred - pred.mean()) - y) ** 2
    if kind == 'listwise':
        tgt = torch.softmax(y / max(temp, 1e-6), dim=0)
        return -(tgt * torch.log_softmax(pred / max(temp, 1e-6), dim=0))
    d_p = pred.unsqueeze(1) - pred.unsqueeze(0)
    d_y = y.unsqueeze(1) - y.unsqueeze(0)
    w = d_y.abs()
    iu = torch.triu(torch.ones_like(w, dtype=torch.bool), diagonal=1)
    sign = torch.sign(d_y)
    per_pair = torch.nn.functional.softplus(-sign * d_p / max(temp, 1e-6)) * w
    denom = w[iu].sum().clamp_min(1e-6)
    return per_pair[iu].sum() / denom


def mode_rank(args):
    device = torch.device('cpu')
    cache = args.labels or (os.path.splitext(args.work)[0] + '_labels.pt')
    if not os.path.exists(cache):
        raise SystemExit(f'no label cache at {cache} — run `siblings` first (it caches labels)')
    sets = torch.load(cache, map_location='cpu', weights_only=False)['sets']
    pol_path = args.policy or torch.load(args.work, map_location='cpu',
                                         weights_only=False)['policy_path']
    policy, encoder = load_policy(pol_path, device)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(sets))
    n_val = max(1, int(len(sets) * args.val_frac))
    val_idx, tr_idx = order[:n_val], order[n_val:]
    print(f'{len(sets)} labelled states — train {len(tr_idx)}, held-out {len(val_idx)} '
          f'(split by STATE, so held-out positions are unseen)')

    # Reliability ceiling on the held-out split only.
    a = np.concatenate([_demean([s['z0'] for s in sets[i]]) for i in val_idx])
    b = np.concatenate([_demean([s['z1'] for s in sets[i]]) for i in val_idx])
    rel = float(np.corrcoef(a, b)[0, 1])
    rel_avg = 2 * rel / (1 + rel)
    print(f'held-out label reliability r={rel:.3f} -> ceiling corr <= {np.sqrt(rel_avg):.3f}\n')

    tr = [_set_tensors(sets[i]) for i in tr_idx]
    va_sets = [sets[i] for i in val_idx]
    y_val = [_demean([0.5 * (s['z0'] + s['z1']) for s in sibs]) for sibs in va_sets]

    dims = {'board_channels': tr[0][0].shape[1], 'global_dim': tr[0][1].shape[1],
            'priv_dim': tr[0][2].shape[1]}
    # Warm-start from the regression-fitted arms. Without this the ranking arms see only
    # ~3k successors while the regression baseline they are compared against saw 120k — so
    # a flat result is confounded with a 40x data gap and settles nothing about the
    # objective. Warm-starting gives the representation the big dataset and lets the
    # ranking loss shape only what it is actually being tested on.
    kinds = args.arms or ['globals', 'board', 'board_xy', 'polfeat_xy', 'board_solo']
    init_blob = None
    if args.init_from:
        init_blob = torch.load(args.init_from, map_location='cpu', weights_only=False)
        print(f'warm-starting from {args.init_from} '
              f'(arms: {", ".join(init_blob["arms"])})')
        if init_blob['hidden'] != args.hidden:
            raise SystemExit(f'--init-from was fitted at hidden={init_blob["hidden"]}; '
                             f'pass --hidden {init_blob["hidden"]}')
        # Validated BEFORE any arm trains. Silently cold-starting one arm while its
        # comparators are warm-started is the exact confound --init-from exists to remove:
        # a cold arm sees ~3k successors against the 120k the regression baseline saw, and
        # its flat result then gets read as "this board pathway cannot fit anything".
        # `board_solo` was scored that way once — it was never in the `fit` work file — and
        # its "train corr 0.003" entered docs/next_iteration.md §3.1 as evidence about the
        # board. Refuse rather than warn.
        missing = [k for k in kinds if k not in init_blob['arms']]
        if missing:
            raise SystemExit(
                f'arm(s) {", ".join(missing)} are not in {args.init_from} (it has: '
                f'{", ".join(init_blob["arms"])}), so they would be COLD-STARTED while the '
                f'rest are warm-started — a 40x data gap masquerading as a result. Re-run '
                f'`fit --arms {" ".join(missing)}` first, or drop them from --arms.')
    tr_sets = [sets[i] for i in tr_idx]
    y_tr = [_demean([0.5 * (s['z0'] + s['z1']) for s in sibs]) for sibs in tr_sets]

    va_pairs = _bucket_pairs(va_sets, eps=args.eps)
    board_only = dict(BUCKETS)['board differs, non-board SAME']

    def score_arm(arm, sets_, y_, pairs_=None):
        arm.eval()
        preds = []
        with torch.no_grad():
            for sibs in sets_:
                bd, gl, pv, oh, sg, _ = _set_tensors(sibs)
                preds.append(_demean((sg * arm(bd, gl, oh, pv)).numpy()))
        s = _score(preds, y_)
        # The board-only bucket is the only one that isolates the board; the pooled
        # `pair-acc` next to it averages it against pairs the board cannot speak to.
        if pairs_ is not None:
            a, n, ti = _pair_acc(preds, y_, pairs_, board_only)
            s['pair_board'], s['n_board'], s['tied_board'] = a, n, ti
        return s

    if len(val_idx) < 100:
        print(f'!! held-out is only {len(val_idx)} states (~{len(val_idx) * 7} successors, '
              f'se(corr) ~ {1 / np.sqrt(max(len(val_idx) * 7, 1)):.3f}). The `corr` column '
              f'picks the epoch BY that same held-out set, so it is optimistically biased at '
              f'this size — read `final corr` and treat differences below ~2x se as noise.')
    n_bd = _pair_acc([np.zeros(len(s)) for s in va_sets], y_val, va_pairs, board_only)[1]
    print(f'{"arm":<12}{"corr":>8}{"spearman":>10}{"pair-acc":>10}{"bd-only":>9}'
          f'{"top1":>8}{"chance":>8}{"train corr":>12}{"  best ep":>10}{"final corr":>12}')
    print(f'{"":<12}{"":>8}{"":>10}{"(all pairs)":>10}{f"(n={n_bd})":>9}')
    for kind in kinds:
        torch.manual_seed(args.seed)
        arm = ValueArm(kind, hidden=args.hidden, **dims,
                       policy_trunk=policy.board_encoder if kind.startswith('polfeat') else None)
        if init_blob is not None:
            arm.load_state_dict(init_blob['arms'][kind])
        opt = torch.optim.Adam(arm.trainable(), lr=args.lr)
        # ~3k training successors against a 200k-parameter model overfits within a few
        # epochs, so select the epoch on the held-out split instead of taking the last.
        best = (score_arm(arm, va_sets, y_val, va_pairs), 0,
                score_arm(arm, tr_sets, y_tr)['corr'])
        final = best[0]
        for ep in range(1, args.epochs + 1):
            arm.train()
            if arm.frozen:
                arm.trunk.eval()
            for i in rng.permutation(len(tr)):
                bd, gl, pv, oh, sg, y = tr[i]
                pred = sg * arm(bd, gl, oh, pv)          # to player-1 frame, as the label is
                loss = rank_loss(pred, y, kind=args.loss, temp=args.temp)
                if loss.dim() > 0:
                    loss = loss.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            s_val = score_arm(arm, va_sets, y_val, va_pairs)
            final = s_val
            if s_val['corr'] > best[0]['corr']:
                best = (s_val, ep, score_arm(arm, tr_sets, y_tr)['corr'])
        s, best_ep, tr_corr = best
        print(f'{kind:<12}{s["corr"]:8.3f}{s["spearman"]:10.3f}{s["pair"]:9.1%}'
              f'{s["pair_board"]:8.1%}{s["top1"]:8.1%}{s["top1_chance"]:8.1%}'
              f'{tr_corr:12.3f}{best_ep:10d}{final["corr"]:12.3f}')
    print(f"""
HOW TO READ — this is the P0' fork (docs/next_iteration.md).
  `best ep` = 0 means training NEVER beat the starting point: the ranking loss did nothing
  (or, without --init-from, that a randomly-initialised net was as good as a trained one —
  which means the run is too small to conclude anything).
  `train corr` far above the held-out corr = overfitting; the training set here is only
  ~7 successors x however many states, which is tiny for a conv net.
  Compare the held-out corr against the ceiling printed above, never against 1.0, AND
  against what the same arm scores under plain regression (`siblings` mode on these labels).
  Clears the regression baseline by a wide margin -> the objective was the bug: add the
       term to the PPO critic loss and to ExIt's.
  Flat or worse, WITH --init-from and a healthy `best ep` -> the objective is not the bug;
       within-state value is not learnable from the state at this data budget. Go to P1 and
       re-measure once the population contests bases.
  Watch `board` / `board_xy` / `board_solo` against `globals`: a ranking objective strips
  out the across-state variance that globals explain for free, so this is the fair test of
  whether the board carries the signal you can see by eye.
  `bd-only` is pairwise accuracy restricted to siblings whose boards differ and whose
  non-board inputs are IDENTICAL — the only bucket in which the board is the sole available
  discriminator, and where `globals` is pinned near 50%. It is the column to read; `pair-acc`
  next to it averages that bucket against pairs the board cannot speak to. Run `siblings`
  on the same labels for the full bucket breakdown.""")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--policy', default=None, help='policy checkpoint (default: latest)')
    common.add_argument('--seed', type=int, default=0)
    common.add_argument('--work', default=WORK_DEFAULT)

    d = sub.add_parser('distinguish', parents=[common], help='no training, no labels, seconds')
    d.add_argument('--games', type=int, default=20)
    d.add_argument('--stride', type=int, default=3, help='sample every Nth decision point')
    d.add_argument('--max-siblings', type=int, default=16)
    d.add_argument('--eps', type=float, default=1e-6)
    d.set_defaults(func=mode_distinguish)

    f = sub.add_parser('fit', parents=[common], help='supervised value regression')
    f.add_argument('--data', default='data/exit/round*.npz')
    f.add_argument('--max-samples', type=int, default=120000)
    f.add_argument('--val-rounds', type=int, default=5)
    f.add_argument('--epochs', type=int, default=3)
    f.add_argument('--batch', type=int, default=256)
    f.add_argument('--hidden', type=int, default=96)
    f.add_argument('--lr', type=float, default=3e-4)
    f.add_argument('--arms', nargs='+', choices=ARMS, default=None)
    f.set_defaults(func=mode_fit)

    s = sub.add_parser('siblings', parents=[common], help='labelled within-state test')
    s.add_argument('--games', type=int, default=40)
    s.add_argument('--states', type=int, default=80, help='cap on sibling sets')
    s.add_argument('--stride', type=int, default=5)
    s.add_argument('--max-siblings', type=int, default=8)
    s.add_argument('--playouts', type=int, default=4, help='per label half; 2 halves are run')
    s.add_argument('--playout-bot', choices=('policy', 'greedy', 'lookahead'),
                   default='policy',
                   help="who plays out the successor. 'policy' answers 'does this choice "
                        "matter given both sides then play like the current policy' — which "
                        "is the WRONG question if the policy cannot convert a positional "
                        "advantage. 'lookahead' is a non-policy reference that resolves 4-6 "
                        "plies, so positional value can actually show up in the label.")
    s.add_argument('--labels', default=None,
                   help='label cache path (default: <work>_labels.pt). Reused unless --relabel')
    s.add_argument('--relabel', action='store_true', help='regenerate labels from scratch')
    s.add_argument('--critic-path', nargs='+', default=None,
                   help='also score a REAL trained Critic checkpoint against the same labels')
    s.add_argument('--eps', type=float, default=1e-6,
                   help='tolerance for calling two siblings identical when bucketing pairs')
    s.add_argument('--n-workers', type=int, default=max(1, (os.cpu_count() or 2) - 4))
    s.set_defaults(func=mode_siblings)

    r = sub.add_parser('rank', parents=[common],
                       help="train ON the within-state objective — the P0' fork")
    r.add_argument('--labels', default=None, help='label cache (default: <work>_labels.pt)')
    r.add_argument('--loss', choices=('rank', 'listwise', 'residual'), default='rank')
    r.add_argument('--temp', type=float, default=1.0, help='logit temperature in the loss')
    r.add_argument('--val-frac', type=float, default=0.25)
    r.add_argument('--epochs', type=int, default=30)
    r.add_argument('--hidden', type=int, default=96)
    r.add_argument('--lr', type=float, default=1e-3)
    r.add_argument('--arms', nargs='+', choices=ARMS, default=None)
    r.add_argument('--init-from', default=None,
                   help='warm-start each arm from a `fit` work file (data/board_value_probe.pt) '
                        'so the ranking loss is not also being asked to learn the '
                        'representation from ~3k samples')
    r.add_argument('--eps', type=float, default=1e-6,
                   help='tolerance for calling two siblings identical when bucketing pairs')
    r.set_defaults(func=mode_rank)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
