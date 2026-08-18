"""Expert-iteration CLI: self-play data-gen + policy/critic distillation + the loop.

`PuctBot` (the gauntlet's strongest agent) teaches its own priors and leaf values to
fresh policy/critic nets, which then seed a stronger `PuctBot`, and so on — the
AlphaZero/ExIt loop (docs/history.md — expert iteration). See `src/services/expert_iteration.py` for
the reusable core and `src/services/selfplay_collector.py` for the parallel self-play
worker pool (mirrors `rollout_collector.py`'s design for PPO rollout collection). Run
from the project root:

    # one generation of self-play games -> a dataset (parallel by default)
    python src/app/expert_iteration.py gen --games 200 --out data/exit/round0.npz

    # distil a dataset into new nets (writes to data/exit/)
    python src/app/expert_iteration.py distill --dataset data/exit/round0.npz

    # the full loop: gen -> distil -> re-seed puct -> gauntlet-check -> repeat
    python src/app/expert_iteration.py loop --rounds 5 --games 200

ExIt artifacts live under `data/exit/` on purpose: the distilled critic predicts the
game outcome z (scale [-1,1]), NOT the shaped PPO return, so it must never become the
`data/lookahead_critic/` "latest" that the shaped bots (LookaheadCriticBot /
policy_critic / default-mode PuctBot) resolve to — that would be a scale mismatch. The
loop pairs the z-critic only with `PuctBot(value_mode='outcome')`, passing paths
explicitly. Round 0 bootstraps from the standing (shaped) PPO checkpoints with the
proven `value_mode='shaped'` search; every later round uses the z-critic in
`'outcome'` mode.

After every round `loop` runs a small round-robin gauntlet (base policy vs. every
round's distilled policy so far, raw — no search, so it's cheap) via the existing
`services/gauntlet.py` + `services/gauntlet_parallel.py` machinery, and logs the same
win-rate/Bradley-Terry report `app/gauntlet.py` prints — this is the actual answer to
"did this round make the policy stronger", not just whether CE/MSE went down.
"""
import argparse
import glob
import logging
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch

from src.services.expert_iteration import (
    generate_selfplay, summarize_game_stats, distill, evaluate_distillation, SelfPlayDataset,
)
from src.services.selfplay_collector import ParallelSelfPlayCollector
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


def setup_run_logger(run_id):
    """File + console logging on the shared 'warchest' logger (mirrors ppo.py).

    Everything `gen`/`distill`/`loop` and the services core log goes to
    `logs/exit_{run_id}.log` at DEBUG and to the console at INFO — so a long
    background run leaves a full on-disk record without needing a shell redirect.
    Returns the log-file path.
    """
    os.makedirs('logs', exist_ok=True)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    path = f'logs/exit_{run_id}.log'
    fh = logging.FileHandler(path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return path


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
    policy = Policy(device=device, hidden_dim=meta['hidden_dim'], obs_encoder=encoder,
                    arch=meta['arch']).to(device)
    policy.load_state_dict(meta['state_dict'])
    return policy, meta


def _load_critic(path, device):
    meta = load_critic_checkpoint(path, map_location=device)
    encoder = get_encoder(meta['obs_version'])
    critic = Critic(device=device, hidden_dim=meta['hidden_dim'], obs_encoder=encoder,
                    arch=meta['arch']).to(device)
    critic.load_state_dict(meta['state_dict'])
    return critic, meta


def _build_bot(policy_path, critic_path, args, *, value_mode):
    return PuctBot(
        policy_path=policy_path, critic_path=critic_path, value_mode=value_mode,
        c_puct=args.c_puct, max_branching=args.max_branching, time_budget=args.time_budget,
        dirichlet_alpha=args.dirichlet_alpha, dirichlet_frac=args.dirichlet_frac,
        device=args.device, stats_log_every=0,
    )


def _run_gen(policy_path, critic_path, args, *, value_mode, out_path, collector=None, desc='self-play'):
    """Self-play `args.games` games and save a dataset. Returns the dataset.

    Parallel (`args.n_workers > 1`, the default) uses `ParallelSelfPlayCollector` — a
    caller driving several rounds (`cmd_loop`) passes its own persistent `collector` so
    workers are spawned once for the whole run, mirroring how `ppo.py` reuses one
    `ParallelRolloutCollector` across every training batch; a standalone `gen` call
    builds and tears down its own. `args.n_workers <= 1` falls back to the sequential
    in-process path (`generate_selfplay`) — useful for debugging one game at a time.
    """
    pmeta = load_policy_checkpoint(policy_path, map_location=args.device)
    cmeta = load_critic_checkpoint(critic_path, map_location=args.device)
    if pmeta['obs_version'] != cmeta['obs_version']:
        raise SystemExit(
            f'policy obs_version {pmeta["obs_version"]} != critic obs_version '
            f'{cmeta["obs_version"]}; expert iteration requires a matching pair '
            f'(they always match when saved from one PPO run).'
        )
    logger.info('gen: %d games requested (n_workers=%d), value_mode=%s, policy=%s critic=%s',
                args.games, args.n_workers, value_mode,
                os.path.basename(policy_path), os.path.basename(critic_path))

    t0 = time.perf_counter()
    if args.n_workers > 1:
        own_collector = collector is None
        if own_collector:
            collector = ParallelSelfPlayCollector(args.n_workers, seed_base=args.seed or 0)
        try:
            ds, game_stats, timing = collector.collect(
                policy_path=policy_path, critic_path=critic_path, value_mode=value_mode,
                n_games=args.games, c_puct=args.c_puct, max_branching=args.max_branching,
                time_budget=args.time_budget, dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_frac=args.dirichlet_frac, temperature=args.temperature,
                temp_moves=args.temp_moves, max_turns=2000, desc=desc,
            )
        finally:
            if own_collector:
                collector.shutdown()
        logger.info('gen: rollout wall=%.1fs (worker critical path), ipc=%.1fs',
                    timing['rollout'], timing['ipc'])
    else:
        encoder = get_encoder(pmeta['obs_version'])
        bot = _build_bot(policy_path, critic_path, args, value_mode=value_mode)
        ds, game_stats = generate_selfplay(bot, args.games, encoder=encoder, temperature=args.temperature,
                                           temp_moves=args.temp_moves, seed=args.seed, desc=desc)

    s = summarize_game_stats(game_stats)
    logger.info(
        'gen: %d games, %d samples, wall=%.1fs — turns/game avg=%.1f (min=%d max=%d), '
        'decisive=%.0f%%, mean_legal_actions=%.1f, mean_visit_entropy=%.3f nats, '
        'policy/search agreement=%.3f',
        s['n_games'], s['n_samples'], time.perf_counter() - t0,
        s['turns_mean'], s['turns_min'], s['turns_max'],
        100 * s['decisive_frac'], s['mean_legal_actions'], s['mean_visit_entropy'],
        s['mean_agreement'],
    )
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    ds.save(out_path)
    logger.info('gen: wrote dataset to %s', out_path)
    return ds


def _run_distill(dataset, policy_path, critic_path, args, *, out_policy, out_critic):
    """Distil `dataset` into new nets warm-started from the given checkpoints, save both."""
    policy, pmeta = _load_policy(policy_path, args.device)
    critic, cmeta = _load_critic(critic_path, args.device)
    if pmeta['obs_version'] != cmeta['obs_version']:
        raise SystemExit('policy/critic obs_version mismatch (see gen).')

    freeze = getattr(args, 'freeze_critic', False)
    logger.info('distill: %d samples, %d epochs, minibatch=%d, lr=%.1e, critic=%s',
                len(dataset), args.epochs, args.minibatch, args.lr,
                'FROZEN (policy-only distillation)' if freeze else 'trained on z')
    before = evaluate_distillation(dataset, policy, critic, device=args.device)
    if before.get('agreement', 0.0) >= 0.9:
        logger.warning(
            'distill: pre-distill policy/search agreement is %.3f (>= 0.90) — the search is '
            'reproducing the prior it was seeded with, so the distillation target is the policy '
            'itself and this round cannot teach it anything (independent_opponents.md §1 fact 2; '
            'the mechanism is IDEAS.md R.0.2). Raise --time-budget and re-run `preflight` before '
            'spending the compute.', before['agreement'])
    res = distill(dataset, policy, critic, epochs=args.epochs, minibatch_size=args.minibatch,
                  lr_policy=args.lr, lr_critic=args.lr, device=args.device,
                  val_frac=args.val_frac, train_critic=not freeze)
    after = res['val']
    logger.info('distill: held-out n_val=%d', res['n_val'])
    logger.info('distill: before %s', _fmt(before))
    logger.info('distill: after  %s', _fmt(after))
    if after.get('visit_entropy', 0.0) > before.get('policy_entropy', 0.0):
        logger.warning(
            'distill: visit_entropy (%.3f) > pre-distill policy_entropy (%.3f) — the search '
            'target is LESS decisive than the policy already was; distilling toward it will '
            'flatten rather than sharpen the policy (see evaluate_distillation docstring). '
            'Usually means too few simulations/move relative to branching — consider raising '
            '--time-budget or narrowing --max-branching.',
            after['visit_entropy'], before['policy_entropy'],
        )

    os.makedirs(os.path.dirname(out_policy) or '.', exist_ok=True)
    # Distillation fine-tunes the loaded nets in place, so the saved copies are the same
    # architecture they came in as — never the current default.
    save_policy_checkpoint(policy, out_policy, obs_version=pmeta['obs_version'],
                           hidden_dim=pmeta['hidden_dim'], arch=pmeta['arch'])
    if freeze:
        # Nothing to write: the critic is untouched, and re-saving it would be the one place
        # a scale bug could enter (the z-scale branch below overwrites return_mean/return_std,
        # which a shaped-return critic must keep). Callers get the original path back.
        critic_in_force = critic_path
        logger.info('distill: saved policy=%s; critic frozen at %s', out_policy, critic_path)
    else:
        os.makedirs(os.path.dirname(out_critic) or '.', exist_ok=True)
        # z-scale critic: return_mean=0 / return_std=1 so PuctBot's denormalisation is
        # identity and it consumes the outcome-scale value directly (value_mode='outcome').
        save_critic_checkpoint(critic, out_critic, obs_version=cmeta['obs_version'],
                               hidden_dim=cmeta['hidden_dim'], arch=cmeta['arch'],
                               return_mean=0.0, return_std=1.0)
        critic_in_force = out_critic
        logger.info('distill: saved policy=%s critic=%s', out_policy, out_critic)
    return before, after, critic_in_force


def _check_critic_staleness(critic_path):
    """Warn when a newer critic exists that this CLI will not pick up.

    `_latest_critic_path` resolves `data/lookahead_critic/lookahead_critic_v{N}.pth`, a
    hand-maintained directory, while PPO writes `data/warchest_critic_<stamp>.pth`. Those
    drift: as of 2026-08-16 the newest file here was `v5` (`critic_v2`, 2026-08-08) while
    PPO had since shipped `critic_v4` (2026-08-10), so every gauntlet and ExIt search for
    a week ran a critic one generation behind without saying so (IDEAS.md R.3, last row).
    A silent stale leaf is the single easiest way to spend a training run measuring the
    wrong thing, so it is checked out loud rather than left to the reader.
    """
    newest_ppo = sorted(glob.glob('data/warchest_critic_*.pth'), key=os.path.getmtime)
    if not newest_ppo or critic_path is None:
        return
    newest = newest_ppo[-1]
    if os.path.getmtime(newest) > os.path.getmtime(critic_path):
        logger.warning(
            'STALE CRITIC: using %s (mtime %s) but %s is newer (mtime %s). '
            '`--critic` resolves data/lookahead_critic/, which PPO does not write. '
            'Either pass --critic %s explicitly, or copy it in as the next '
            'lookahead_critic_v{N}.pth.',
            os.path.basename(critic_path),
            time.strftime('%Y-%m-%d', time.localtime(os.path.getmtime(critic_path))),
            os.path.basename(newest),
            time.strftime('%Y-%m-%d', time.localtime(os.path.getmtime(newest))),
            newest)


def cmd_preflight(args):
    """Check every gate that decides whether an ExIt run can work, before paying for one.

    Four gates, cheapest first. Each one has cost a run at least once in this project's
    history, and each is a single number:

      1. **Staleness** — is the critic in force actually the newest one (R.3, last row)?
      2. **Teacher budget** — how many node expansions does `--time-budget` really buy?
         Below ~50 on a mean-10.5 branching tree the visit counts are the prior plus noise
         and the distillation target is the student (IDEAS.md R.0.2).
      3. **Teacher strength** — does the search actually beat the raw policy it is seeded
         with, head to head? This is `next_iteration.md` §5 row 9. A teacher under ~0.60
         is not worth distilling, and this is the gate that was never checked before the
         30-round run that got monotonically weaker.
      4. **Target sharpness** — on a handful of real self-play games, is the visit-count
         entropy *below* the policy's own? If it is above, distilling flattens the policy
         (the recorded `--dirichlet-frac 0.25` failure). This pair is coupled to the
         budget, so it must be re-read whenever either moves.
    """
    from src.services.bots.puct_bot import PuctBot
    from src.services.gauntlet import build_agent
    from src.services.gauntlet_parallel import round_robin_parallel
    from src.services.environment.warchest_env import WarChestEnv

    policy_path = args.policy or _latest_policy_path()
    critic_path = args.critic or _latest_critic_path()
    if policy_path is None or critic_path is None:
        raise SystemExit('preflight needs a policy and a critic checkpoint (none found).')
    pmeta = load_policy_checkpoint(policy_path, map_location='cpu')
    cmeta = load_critic_checkpoint(critic_path, map_location='cpu')
    verdicts = []

    logger.info('--- gate 1: checkpoints ---')
    logger.info('policy = %s (arch %s, obs v%d, hidden %d)', policy_path,
                pmeta['arch'], pmeta['obs_version'], pmeta['hidden_dim'])
    logger.info('critic = %s (arch %s, obs v%d, hidden %d)', critic_path,
                cmeta['arch'], cmeta['obs_version'], cmeta['hidden_dim'])
    _check_critic_staleness(critic_path)

    logger.info('--- gate 2: what --time-budget %.2fs actually buys ---', args.time_budget)
    bot = PuctBot(policy_path=policy_path, critic_path=critic_path, c_puct=args.c_puct,
                  max_branching=args.max_branching, time_budget=args.time_budget,
                  device='cpu', stats_log_every=0)
    env = WarChestEnv(save_game_history=False)
    np.random.seed(args.seed or 0)
    env.reset()
    expansions, depths, legal = [], [], []
    for _ in range(args.preflight_moves):
        legal.append(len(env.get_possible_actions()))
        action = bot.act(env)
        st = bot.last_stats or {}
        expansions.append(st.get('nodes_visited', 0))
        depths.append(st.get('depth_reached', 0))
        _, _, term, trunc, info = env.step(action)
        if not info['action'].is_valid:
            _, _, term, trunc, info = env.make_random_step()
        if term or trunc:
            break
    mean_exp = float(np.mean(expansions)) if expansions else 0.0
    logger.info('expansions/move %.1f (min %d max %d) | max depth %.1f | legal at root %.1f',
                mean_exp, min(expansions or [0]), max(expansions or [0]),
                float(np.mean(depths or [0])), float(np.mean(legal or [0])))
    if mean_exp < 50:
        logger.warning(
            'THIN SEARCH: %.1f expansions/move against ~%.1f legal actions at the root. The '
            'root expansion alone consumes one per child, so the visit counts carry roughly '
            'one bit per child and the target is the prior. Raise --time-budget.',
            mean_exp, float(np.mean(legal or [0])))
        verdicts.append(('search depth', 'FAIL', f'{mean_exp:.0f} expansions/move, want >= 50'))
    else:
        verdicts.append(('search depth', 'ok', f'{mean_exp:.0f} expansions/move'))

    logger.info('--- gate 3: teacher vs student, %d games ---', args.preflight_games)
    specs = [
        {'kind': 'policy', 'path': policy_path},
        {'kind': 'puct', 'name': 'puct', 'kwargs': {
            'policy_path': policy_path, 'critic_path': critic_path,
            'c_puct': args.c_puct, 'max_branching': args.max_branching,
            'time_budget': args.time_budget, 'stats_log_every': 0,
        }},
    ]
    device = torch.device('cpu')
    names = [build_agent(sp, device=device).name for sp in specs]
    out = round_robin_parallel(specs, names, k_games=args.preflight_games,
                               seed=args.seed or 0, n_workers=max(1, args.n_workers))
    _log_gauntlet_report(out)
    teacher_wr = float(out['win_rate'][1, 0])
    se = (teacher_wr * (1 - teacher_wr) / max(args.preflight_games, 1)) ** 0.5
    logger.info('teacher WR vs student = %.3f +/- %.3f', teacher_wr, se)
    if teacher_wr < 0.60:
        logger.warning(
            'WEAK TEACHER: the search wins %.3f against the policy that supplies its own '
            'priors. Distilling it can only teach what it already knows. Fix gate 2 first.',
            teacher_wr)
        verdicts.append(('teacher strength', 'FAIL', f'{teacher_wr:.3f}, want >= 0.60'))
    else:
        verdicts.append(('teacher strength', 'ok', f'{teacher_wr:.3f} +/- {se:.3f}'))

    logger.info('--- gate 4: is the target sharper than the policy? (%d self-play games) ---',
                args.preflight_selfplay_games)
    encoder = get_encoder(pmeta['obs_version'])
    sp_bot = _build_bot(policy_path, critic_path, args, value_mode=args.value_mode)
    ds, game_stats = generate_selfplay(
        sp_bot, args.preflight_selfplay_games, encoder=encoder,
        temperature=args.temperature, temp_moves=args.temp_moves, seed=args.seed,
        desc='preflight self-play')
    gs = summarize_game_stats(game_stats)
    policy, _ = _load_policy(policy_path, 'cpu')
    critic, _ = _load_critic(critic_path, 'cpu')
    ev = evaluate_distillation(ds, policy, critic, device='cpu')
    logger.info('self-play: %d samples, agreement %.3f, mean_visit_entropy %.3f nats',
                gs['n_samples'], gs['mean_agreement'], gs['mean_visit_entropy'])
    logger.info('pre-distill %s', _fmt(ev))
    if ev['visit_entropy'] > ev['policy_entropy']:
        logger.warning(
            'FLATTENING TARGET: visit_entropy %.3f > policy_entropy %.3f — distilling this '
            'target makes the policy less decisive, the opposite of the intended effect. '
            'Lower --dirichlet-frac or raise --time-budget.',
            ev['visit_entropy'], ev['policy_entropy'])
        verdicts.append(('target sharpness', 'FAIL',
                         f"visit {ev['visit_entropy']:.3f} > policy {ev['policy_entropy']:.3f}"))
    else:
        verdicts.append(('target sharpness', 'ok',
                         f"visit {ev['visit_entropy']:.3f} <= policy {ev['policy_entropy']:.3f}"))
    if ev['agreement'] >= 0.9:
        verdicts.append(('teacher divergence', 'FAIL',
                         f"agreement {ev['agreement']:.3f}, want < 0.90"))
    else:
        verdicts.append(('teacher divergence', 'ok', f"agreement {ev['agreement']:.3f}"))

    logger.info('=== preflight verdict ===')
    for gate, status, detail in verdicts:
        logger.info('  [%s] %-20s %s', status.upper(), gate, detail)
    failed = [g for g, st, _ in verdicts if st == 'FAIL']
    if failed:
        logger.warning('NOT ready to run: %s. Fix these before spending a loop.',
                       ', '.join(failed))
    else:
        logger.info('All gates pass. A `loop` run at these settings is justified.')
    return verdicts


def _fmt(d):
    if not d:
        return '{}'
    return (f"ce={d['ce']:.4f} mse={d['mse']:.4f} agree={d['agreement']:.3f} "
            f"policy_entropy={d['policy_entropy']:.3f} visit_entropy={d['visit_entropy']:.3f}")


def _run_post_round_gauntlet(field_specs, args):
    """Round-robin the accumulated field (base policy + every ExIt round's distilled
    policy so far) via the existing gauntlet machinery, raw-policy agents only (no
    search, so this stays cheap even after several rounds). Logs the same win-rate
    matrix / Bradley-Terry report `app/gauntlet.py --bots policy` prints, through the
    logger so it lands in this run's log file too.

    This is the real answer to "did this round help": CE/MSE falling only means the
    nets fit the search's targets better, not that the resulting policy is stronger.
    """
    from src.services.gauntlet import build_agent
    from src.services.gauntlet_parallel import round_robin_parallel

    device = torch.device('cpu')
    names = [build_agent(s, device=device).name for s in field_specs]
    n_workers = max(1, args.n_workers)
    logger.info('post-round gauntlet: %d-agent field, k_games=%d, n_workers=%d',
                len(field_specs), args.gauntlet_k_games, n_workers)
    out = round_robin_parallel(field_specs, names, k_games=args.gauntlet_k_games,
                               seed=args.seed or 0, n_workers=n_workers)
    _log_gauntlet_report(out)
    return out


def _log_gauntlet_report(out):
    """`_print_report`-equivalent (`app/gauntlet.py`), but through the logger."""
    names = out['names']
    wr = out['win_rate']
    w = max(len(n) for n in names)

    logger.info('Win-rate matrix (row vs column):')
    logger.info(' ' * (w + 2) + '  '.join(f'{n[:6]:>6}' for n in names))
    for i, n in enumerate(names):
        cells = ['   -  ' if i == j else f'{wr[i, j]:6.2f}' for j in range(len(names))]
        logger.info(f'{n:>{w}}  ' + '  '.join(cells))

    logger.info('Bradley-Terry ranking (Elo-scaled, field mean = 1000):')
    for n, r in sorted(out['ratings'].items(), key=lambda kv: -kv[1]):
        logger.info(f'  {n:>{w}}  {r:7.1f}')

    frac = out['intransitive_fraction']
    logger.info('Intransitive-triple fraction: %.3f%s', frac,
               '' if frac == 0 else '  (cycles present)')


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

    # One persistent worker pool for the whole loop (mirrors ppo.py's
    # `_lazy_init_collector`: spawned once, reused every round — each round's workers
    # just rebuild their PuctBot from that round's freshly distilled checkpoints
    # instead of respawning processes).
    collector = ParallelSelfPlayCollector(args.n_workers, seed_base=args.seed or 0) \
        if args.n_workers > 1 else None
    # Accumulated round-robin field for the post-round gauntlet: base + every round's
    # distilled policy so far, so each round's report shows the whole trend, not just
    # this round vs. base.
    field_specs = [{'kind': 'policy', 'path': base_policy}]

    cur_policy, cur_critic, cur_mode = base_policy, base_critic, 'shaped'
    try:
        for r in range(args.rounds):
            round_t0 = time.perf_counter()
            logger.info('=== ExIt round %d/%d (mode=%s) ===', r + 1, args.rounds, cur_mode)
            ds_path = os.path.join(EXIT_DIR, f'round{r}.npz')
            ds = _run_gen(cur_policy, cur_critic, args, value_mode=cur_mode, out_path=ds_path,
                         collector=collector, desc=f'round {r + 1}/{args.rounds} self-play')
            out_policy = os.path.join(EXIT_DIR, f'round{r}_policy.pth')
            out_critic = os.path.join(EXIT_DIR, f'round{r}_critic.pth')
            before, after, critic_in_force = _run_distill(
                ds, cur_policy, cur_critic, args,
                out_policy=out_policy, out_critic=out_critic)

            field_specs.append({'kind': 'policy', 'path': out_policy})
            if not args.skip_gauntlet:
                _run_post_round_gauntlet(field_specs, args)

            logger.info(
                'round %d/%d done in %.1fs — agreement %.3f -> %.3f, critic mse %.4f -> %.4f',
                r + 1, args.rounds, time.perf_counter() - round_t0,
                before.get('agreement', 0.0), after.get('agreement', 0.0),
                before.get('mse', 0.0), after.get('mse', 0.0),
            )
            # With --freeze-critic the critic never changes scale, so the search stays in
            # the 'shaped' mode that produced the only ExIt round that helped. Otherwise the
            # critic is now z-scale from this round on and must be paired with outcome mode.
            cur_policy = out_policy
            cur_critic = critic_in_force
            cur_mode = cur_mode if args.freeze_critic else 'outcome'
    finally:
        if collector is not None:
            collector.shutdown()
    logger.info('ExIt loop finished. Latest nets: policy=%s critic=%s (value_mode=%s).',
                cur_policy, cur_critic, cur_mode)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _add_common(p):
    p.add_argument('--policy', default=None, help='Base policy .pth (default: newest data/warchest_ppo_*.pth).')
    p.add_argument('--critic', default=None, help='Base critic .pth (default: newest lookahead_critic_v*.pth).')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--n-workers', type=int, default=min(os.cpu_count() or 4, 8),
                   help='Parallel self-play worker processes (mirrors ppo.py\'s rollout '
                        'workers). 1 = sequential in-process. Also the worker count for '
                        'the post-round gauntlet check in `loop`. Default: min(cpu_count, 8).')
    # search knobs (gen)
    p.add_argument('--games', type=int, default=200, help='Self-play games per generation.')
    p.add_argument('--time-budget', type=float, default=1.0,
                   help='PuctBot per-move search budget (s). Raised 0.1 -> 1.0 2026-08-16: at '
                        '0.1 s the search performs ~14-30 node expansions on a tree whose mean '
                        'branching is 10.5, so the root alone eats 8 of them and the visit '
                        'counts are the policy prior plus about one bit per child (IDEAS.md '
                        'R.0.2). That is the arithmetic behind the recorded 0.94-0.95 '
                        'teacher/student agreement — not a self-play subtlety. 1.0 s buys ~97 '
                        'expansions and measures 0.74 against the raw policy vs 0.66 at 0.1 s '
                        '(R.0.1). Generation is offline, so this is the cheap axis; run '
                        '`preflight` if you change it.')
    p.add_argument('--c-puct', type=float, default=1.5)
    p.add_argument('--max-branching', type=int, default=8)
    p.add_argument('--dirichlet-alpha', type=float, default=0.3,
                   help='Root Dirichlet noise for self-play exploration (0 = off).')
    p.add_argument('--dirichlet-frac', type=float, default=0.03,
                   help="Root noise mixing fraction. 0.25 (AlphaZero's own default, this "
                        "CLI's old default) measured mean_visit_entropy=0.87 nats at the "
                        "0.1 s budget then in force — which is ~14-30 expansions, not the "
                        "'100-300 sims/move' this help used to claim (IDEAS.md R.0.2) — "
                        "nearly double the pre-distill "
                        "policy's own entropy (~0.6), so distillation flattened the policy "
                        "every round instead of sharpening it (see evaluate_distillation's "
                        "docstring / docs/bots.md's ExIt section). AlphaZero's ~800 sims/move "
                        "lets Q signal outcompete a 25%%-noisy prior; at our budget it can't. "
                        "0.03 measured 0.586 nats (essentially matching frac=0, which measured "
                        "0.508) — keeps a little self-play move diversity without recreating "
                        "the collapse. Lower only if you still see the visit_entropy-above-"
                        "policy_entropy warning after a round. NOTE the coupling: that "
                        "measurement was taken at --time-budget 0.1. Noise fraction and search "
                        "budget are not independent — 0.25 is affordable exactly when Q has "
                        "enough visits to outcompete a noisy prior, so re-test 0.25 at the new "
                        "1.0 s default (via `preflight`, which reports the entropy pair before "
                        "a full generation) rather than assuming either value transfers.")
    p.add_argument('--temperature', type=float, default=1.0, help='Visit-count sampling temperature.')
    p.add_argument('--temp-moves', type=int, default=12, help='Opening plies sampled before going greedy.')
    p.add_argument('--seed', type=int, default=None)
    # distill knobs
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--minibatch', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--val-frac', type=float, default=0.1)
    p.add_argument('--freeze-critic', action='store_true',
                   help="Distil the POLICY only; leave the critic bit-identical and keep the "
                        "search in value_mode='shaped' for every round. This is the "
                        "configuration the record argues for: the only ExIt round that made "
                        "the policy stronger was round 0 — the one round still running the PPO "
                        "shaped-return critic (agreement 0.77, BT 1089, best of 30) — and every "
                        "round on the self-distilled z-critic got monotonically weaker "
                        "(independent_opponents.md §1). next_iteration.md row 2b then measured "
                        "`z` as the worse target independently. Freezing leaves the loop exactly "
                        "one moving part, so a negative result is attributable.")

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

    lp = sub.add_parser('loop', help='Full ExIt loop: gen -> distil -> re-seed -> gauntlet-check -> repeat.')
    _add_common(lp)
    lp.add_argument('--rounds', type=int, default=3)
    lp.add_argument('--gauntlet-k-games', type=int, default=20,
                    help='Games per pair in the post-round gauntlet check. Default 20.')
    lp.add_argument('--skip-gauntlet', action='store_true',
                    help='Skip the post-round gauntlet check (faster iteration).')
    lp.set_defaults(func=cmd_loop)

    pf = sub.add_parser('preflight',
                        help='Check the four gates that decide whether an ExIt run can work, '
                             'before paying for one (see cmd_preflight).')
    _add_common(pf)
    pf.add_argument('--value-mode', choices=['shaped', 'outcome'], default='shaped',
                    help='Search mode to check; match what the run will use.')
    pf.add_argument('--preflight-moves', type=int, default=12,
                    help='Decisions to measure expansions/move over (gate 2). Run this on an '
                         'otherwise idle box: the count is time-budgeted, so CPU contention '
                         'deflates it (measured 30/move idle, 14/move under 12 busy workers, '
                         '2/move under a loaded box at the same 0.1 s budget).')
    pf.add_argument('--preflight-games', type=int, default=60,
                    help='Teacher-vs-student games (gate 3). se(WR) ~ 6 pp at 60.')
    pf.add_argument('--preflight-selfplay-games', type=int, default=6,
                    help='Self-play games for the entropy/agreement read (gate 4).')
    pf.set_defaults(func=cmd_preflight)

    args = parser.parse_args()
    run_id = time.strftime('%Y%m%d-%H%M%S')
    log_path = setup_run_logger(run_id)
    logger.info('expert iteration: cmd=%s, logging to %s', args.cmd, log_path)
    args.func(args)


if __name__ == '__main__':
    main()
