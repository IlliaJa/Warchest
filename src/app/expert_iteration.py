"""Expert-iteration CLI: self-play data-gen + policy/critic distillation + the loop.

`PuctBot` (the gauntlet's strongest agent) teaches its own priors and leaf values to
fresh policy/critic nets, which then seed a stronger `PuctBot`, and so on — the
AlphaZero/ExIt loop (docs/history.md — expert iteration). See `src/services/expert_iteration.py` for
the reusable core and `src/services/selfplay_collector.py` for the parallel self-play
worker pool (mirrors `rollout_collector.py`'s design for PPO rollout collection). Run
from the project root:

    # one generation of self-play games -> a dataset (parallel by default)
    python src/app/expert_iteration.py gen --games 200

    # distil a dataset into new nets
    python src/app/expert_iteration.py distill --dataset data/exit/<run>/gen.npz

    # the full loop: gen -> distil -> re-seed puct -> gauntlet-check -> repeat
    python src/app/expert_iteration.py loop --rounds 5 --games 200

Every invocation writes its artifacts into a fresh `data/exit/{launch timestamp}/`
(`--run-dir` overrides it) together with a `meta.json` recording the invocation, so a
new run can never overwrite an earlier run's datasets or checkpoints — it used to write
`round{r}.npz` / `round{r}_policy.pth` flat into `data/exit/`, which silently destroyed
the previous loop's nets. The run id matches `logs/exit_{run_id}.log`.

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
"did this round make the policy stronger", not just whether CE/MSE went down. That
report also drives a promotion gate (`--promote-threshold`, default win rate 0.5
against the checkpoint the round's self-play was generated from): a round that loses
is REJECTED and the next round retries from the same checkpoint instead of building
on a regression. Before this gate existed, a 3-round run accepted every round
unconditionally and got monotonically weaker while its own gauntlet report showed
it (base beat every round, 0.70/0.78/0.82 — docs/IDEAS.md R.10.9), which is also why
the CE-target visit distribution is sharpened (`--visit-temp`, see
`_sharpen_target`) rather than used raw: at this project's search budget it is
measurably less decisive than the policy already distilling toward it, and
unsharpened distillation was the mechanism behind that same regression.

Both fixes above stop a round from *overwriting* the last one, but a round-0-only
regression can still recur even with a correctly-sharpened target and a working gate,
because `distill()` used to see only the round it was just handed — one network's
~25k-sample self-play slice, refit for several epochs, with nothing from any earlier
round to anchor it (docs/IDEAS.md R.10.5a). `--replay-rounds` (default 3, see
`services.expert_iteration.ReplayWindow`) closes that: each round now trains on the
concatenation of the last few rounds' datasets, a sliding window over generations the
way every AlphaZero-family trainer keeps one, rather than one round's narrow slice.
"""
import argparse
import glob
import json
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
    ReplayWindow,
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


def resolve_run_dir(args, run_id):
    """Per-run output directory: `data/exit/{run_id}/` unless --run-dir says otherwise.

    Every artifact a run writes (datasets, distilled policy/critic checkpoints, the
    run's `meta.json`) goes in here. Runs used to write `data/exit/round{r}.npz` +
    `round{r}_policy.pth` flat, so a second `loop` silently overwrote the first one's
    checkpoints — including a `PuctBot` seed someone still wanted. The run id is the
    same launch timestamp as `logs/exit_{run_id}.log`, so a directory and its log
    always pair up by name. The directory itself is only created once something is
    written into it, so `preflight` (which writes nothing) leaves no empty dir behind.
    """
    return getattr(args, 'run_dir', None) or os.path.join(EXIT_DIR, run_id)


def write_run_meta(args, run_dir, run_id, log_path):
    """Drop the invocation into `{run_dir}/meta.json` so a stale run dir is identifiable."""
    os.makedirs(run_dir, exist_ok=True)
    meta = {
        'run_id': run_id,
        'cmd': args.cmd,
        'argv': sys.argv[1:],
        'log': log_path,
        'args': {k: v for k, v in sorted(vars(args).items()) if k != 'func'},
    }
    path = os.path.join(run_dir, 'meta.json')
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    return path


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
        # Blind + PIMC by default for data-gen: the teacher used to read the opponent's
        # real hand (PuctBot's own default), so 15.4 % of its moves depended on a variable
        # the student's observation does not contain — irreducible label noise as large as
        # the whole improvement signal (docs/IDEAS.md R.10.4, M3).
        see_opponent_hand=not args.blind_teacher,
        n_determinizations=args.n_determinizations,
        forced_playouts_k=args.forced_playouts_k,
        device=args.device, stats_log_every=0,
    )


def _run_gen(policy_path, critic_path, args, *, value_mode, out_path, collector=None,
             desc='self-play', round_seed=None):
    """Self-play `args.games` games and save a dataset. Returns the dataset.

    Parallel (`args.n_workers > 1`, the default) uses `ParallelSelfPlayCollector` — a
    caller driving several rounds (`cmd_loop`) passes its own persistent `collector` so
    workers are spawned once for the whole run, mirroring how `ppo.py` reuses one
    `ParallelRolloutCollector` across every training batch; a standalone `gen` call
    builds and tears down its own. `args.n_workers <= 1` falls back to the sequential
    in-process path (`generate_selfplay`) — useful for debugging one game at a time.

    `round_seed` overrides `args.seed` for this round's sequential RNG. The parallel
    workers seed once at spawn and let their streams run on, so their rounds differ by
    construction, but `generate_selfplay` re-seeds per call — without this, every round
    of a `loop` at `--n-workers 1` would replay the identical games (docs/IDEAS.md
    R.10.7).
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
    logger.info('gen: teacher = %s hand, n_determinizations=%d, forced_playouts_k=%.1f, '
                'apprentice_frac=%.2f',
                'BLIND to the opponent\'s' if args.blind_teacher else 'READING the opponent\'s',
                args.n_determinizations, args.forced_playouts_k, args.apprentice_frac)

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
                temp_moves=args.temp_moves, max_turns=2000,
                see_opponent_hand=not args.blind_teacher,
                n_determinizations=args.n_determinizations,
                forced_playouts_k=args.forced_playouts_k,
                apprentice_frac=args.apprentice_frac, desc=desc,
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
                                           temp_moves=args.temp_moves,
                                           seed=round_seed if round_seed is not None else args.seed,
                                           apprentice_frac=args.apprentice_frac, desc=desc)

    s = summarize_game_stats(game_stats)
    logger.info(
        'gen: %d games, %d samples, wall=%.1fs — turns/game avg=%.1f (min=%d max=%d), '
        'decisive=%.0f%%, mean_legal_actions=%.1f, mean_visit_entropy=%.3f nats, '
        'policy/search agreement=%.3f, apprentice plies=%.0f%%',
        s['n_games'], s['n_samples'], time.perf_counter() - t0,
        s['turns_mean'], s['turns_min'], s['turns_max'],
        100 * s['decisive_frac'], s['mean_legal_actions'], s['mean_visit_entropy'],
        s['mean_agreement'], 100 * s['apprentice_frac'],
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
    logger.info('distill: %d samples, %d epochs, minibatch=%d, lr=%.1e, visit_temp=%.2f, '
                'kl_coeff=%.2f, critic=%s',
                len(dataset), args.epochs, args.minibatch, args.lr, args.visit_temp,
                args.kl_coeff, 'FROZEN (policy-only distillation)' if freeze else 'trained on z')
    before = evaluate_distillation(dataset, policy, critic, device=args.device,
                                   visit_temp=args.visit_temp)
    if before.get('agreement', 0.0) >= 0.9:
        logger.warning(
            'distill: pre-distill policy/search agreement is %.3f (>= 0.90) — the search is '
            'reproducing the prior it was seeded with, so the distillation target is the policy '
            'itself and this round cannot teach it anything (independent_opponents.md §1 fact 2; '
            'the mechanism is IDEAS.md R.0.2). Raise --time-budget and re-run `preflight` before '
            'spending the compute.', before['agreement'])
    res = distill(dataset, policy, critic, epochs=args.epochs, minibatch_size=args.minibatch,
                  lr_policy=args.lr, lr_critic=args.lr, device=args.device,
                  val_frac=args.val_frac, train_critic=not freeze, visit_temp=args.visit_temp,
                  kl_coeff=args.kl_coeff, early_stop=not args.no_early_stop,
                  patience=args.patience)
    after = res['val']
    logger.info('distill: held-out n_val=%d samples from %d games; ran %d/%d epochs, kept '
                'epoch %s', res['n_val'], res['n_val_games'], res['epochs_run'], args.epochs,
                res['best_epoch'] if res['best_epoch'] else 'n/a (early stopping off)')
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
    logger.info('teacher = puct @ %.2fs, %s the opponent hand, n_determinizations=%d, '
                'forced_playouts_k=%.1f | data-gen apprentice_frac=%.2f, visit_temp=%.2f',
                args.time_budget, 'BLIND to' if args.blind_teacher else 'READING',
                args.n_determinizations, args.forced_playouts_k, args.apprentice_frac,
                args.visit_temp)

    logger.info('--- gate 2: what --time-budget %.2fs actually buys ---', args.time_budget)
    # Every gate below runs the *data-gen* teacher configuration (blindness, PIMC
    # determinizations, forced playouts), not a default PuctBot: a gate measured on a
    # different bot than the one that will generate the data is not a gate.
    bot = PuctBot(policy_path=policy_path, critic_path=critic_path, c_puct=args.c_puct,
                  max_branching=args.max_branching, time_budget=args.time_budget,
                  see_opponent_hand=not args.blind_teacher,
                  n_determinizations=args.n_determinizations,
                  forced_playouts_k=args.forced_playouts_k,
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
            'see_opponent_hand': not args.blind_teacher,
            'n_determinizations': args.n_determinizations,
            'forced_playouts_k': args.forced_playouts_k,
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
        apprentice_frac=args.apprentice_frac, desc='preflight self-play')
    gs = summarize_game_stats(game_stats)
    policy, _ = _load_policy(policy_path, 'cpu')
    critic, _ = _load_critic(critic_path, 'cpu')
    ev = evaluate_distillation(ds, policy, critic, device='cpu', visit_temp=args.visit_temp)
    logger.info('self-play: %d samples, agreement %.3f, mean_visit_entropy %.3f nats, '
                'apprentice plies %.0f%%',
                gs['n_samples'], gs['mean_agreement'], gs['mean_visit_entropy'],
                100 * gs['apprentice_frac'])
    logger.info('pre-distill (visit_temp=%.2f) %s', args.visit_temp, _fmt(ev))
    if ev['visit_entropy'] > ev['policy_entropy']:
        logger.warning(
            'FLATTENING TARGET: visit_entropy %.3f > policy_entropy %.3f (at visit_temp=%.2f) '
            '— distilling this target makes the policy less decisive, the opposite of the '
            'intended effect. Lower --visit-temp further, lower --dirichlet-frac, or raise '
            '--time-budget.',
            ev['visit_entropy'], ev['policy_entropy'], args.visit_temp)
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
    """Round-robin `field_specs` (the run base, the round's own base, and the round's
    distilled policy — see `cmd_loop`) via the existing gauntlet machinery, raw-policy
    agents only (no search, so 100+ games per pair is still cheap). Logs the same win-rate
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
    out = args.out or os.path.join(args.run_dir, 'gen.npz')
    _run_gen(policy_path, critic_path, args, value_mode=args.value_mode, out_path=out)


def cmd_distill(args):
    policy_path = args.policy or _latest_policy_path()
    critic_path = args.critic or _latest_critic_path()
    if policy_path is None or critic_path is None:
        raise SystemExit('distill needs base policy and critic checkpoints (none found).')
    ds = SelfPlayDataset.load(args.dataset)
    out_policy = args.out_policy or os.path.join(args.run_dir, 'policy.pth')
    out_critic = args.out_critic or os.path.join(args.run_dir, 'critic.pth')
    _run_distill(ds, policy_path, critic_path, args, out_policy=out_policy, out_critic=out_critic)


def cmd_loop(args):
    base_policy = args.policy or _latest_policy_path()
    base_critic = args.critic or _latest_critic_path()
    if base_policy is None or base_critic is None:
        raise SystemExit('loop needs base policy and critic checkpoints (none found).')

    # One persistent worker pool for the whole loop (mirrors ppo.py's
    # `_lazy_init_collector`: spawned once, reused every round — each round's workers
    # just rebuild their PuctBot from that round's freshly distilled checkpoints
    # instead of respawning processes).
    collector = ParallelSelfPlayCollector(args.n_workers, seed_base=args.seed or 0) \
        if args.n_workers > 1 else None
    # The post-round gauntlet field is deliberately SMALL: the run's original base, the
    # checkpoint this round was generated from, and the round's own output — at most three
    # agents, exactly the two comparisons the promotion gate needs. The field used to
    # accumulate every round, which spent the whole games budget on pairs nothing reads
    # and produced fields the tool itself flagged as intransitive (0.100-0.200) at
    # k=20-30, se ~ 9-11 pp. Three agents at k=100+ costs less and resolves ~4 pp
    # (docs/IDEAS.md R.10.11's closing note, R.9.3).
    base_spec = {'kind': 'policy', 'path': base_policy}
    if args.skip_gauntlet:
        logger.warning(
            '--skip-gauntlet disables the promotion gate: every round will be accepted '
            'unconditionally regardless of whether it actually beat the checkpoint it '
            'was generated from (this is exactly the bug that let a 3-round run get '
            'monotonically weaker while its own reports showed it — docs/IDEAS.md R.10.9).')

    # Sliding window over the last --replay-rounds rounds' self-play data (docs/IDEAS.md
    # R.10.5a, R.10.8 item 2): each round still writes its own round{r}.npz, but distill()
    # now trains on the concatenation of the last few rounds rather than this round alone,
    # so one round's narrow, single-network self-play slice can't fully overwrite what
    # earlier rounds taught. --replay-rounds 1 reproduces the old one-round-only behaviour.
    replay = ReplayWindow(args.replay_rounds)

    cur_policy, cur_critic, cur_mode = base_policy, base_critic, 'shaped'
    try:
        for r in range(args.rounds):
            round_t0 = time.perf_counter()
            logger.info('=== ExIt round %d/%d (mode=%s, base=%s) ===',
                        r + 1, args.rounds, cur_mode, os.path.basename(cur_policy))
            ds_path = os.path.join(args.run_dir, f'round{r}.npz')
            ds = _run_gen(cur_policy, cur_critic, args, value_mode=cur_mode, out_path=ds_path,
                         collector=collector, desc=f'round {r + 1}/{args.rounds} self-play',
                         round_seed=None if args.seed is None else args.seed + 1000 * r)
            replay.push(ds)
            train_ds = replay.concat()
            logger.info('round %d/%d: training on %d samples buffered over %d round(s) '
                        '(this round contributed %d)', r + 1, args.rounds, len(train_ds),
                        len(replay), len(ds))
            out_policy = os.path.join(args.run_dir, f'round{r}_policy.pth')
            out_critic = os.path.join(args.run_dir, f'round{r}_critic.pth')
            before, after, critic_in_force = _run_distill(
                train_ds, cur_policy, cur_critic, args,
                out_policy=out_policy, out_critic=out_critic)

            # base first, then this round's own base (when it isn't the run base), then the
            # new checkpoint — so index 0 is always the run base and -1 is always the round.
            field_specs = [base_spec]
            if cur_policy != base_policy:
                field_specs.append({'kind': 'policy', 'path': cur_policy})
            field_specs.append({'kind': 'policy', 'path': out_policy})
            new_idx = len(field_specs) - 1
            cur_idx = new_idx - 1
            promoted = True
            wr_vs_cur = wr_vs_base = None
            if not args.skip_gauntlet:
                out = _run_post_round_gauntlet(field_specs, args)
                wr_vs_cur = float(out['win_rate'][new_idx, cur_idx])
                # Also gate against the run's *original* base (field_specs[0]). The
                # vs-predecessor check alone answers "did this round improve on last
                # round's checkpoint", which a chain of marginal wins can satisfy without
                # ever improving on where the run started: the 2026-08-19 8-round run
                # promoted four times at 0.533/0.533/0.500/0.500 against successive
                # predecessors, and its own final field then read the result as *losing*
                # to the original base 0.40 (IDEAS.md R.10.11). Skipped while cur_idx is
                # still 0 — that is the same comparison.
                if cur_idx != 0:
                    wr_vs_base = float(out['win_rate'][new_idx, 0])
                promoted = (wr_vs_cur >= args.promote_threshold
                            and (wr_vs_base is None or wr_vs_base >= args.promote_threshold))
                logger.info('round %d/%d vs its own base: %.3f%s (promote threshold %.2f) -> %s',
                            r + 1, args.rounds, wr_vs_cur,
                            '' if wr_vs_base is None else f', vs run base: {wr_vs_base:.3f}',
                            args.promote_threshold, 'PROMOTED' if promoted else 'REJECTED')

            logger.info(
                'round %d/%d done in %.1fs — agreement %.3f -> %.3f, critic mse %.4f -> %.4f',
                r + 1, args.rounds, time.perf_counter() - round_t0,
                before.get('agreement', 0.0), after.get('agreement', 0.0),
                before.get('mse', 0.0), after.get('mse', 0.0),
            )
            if not promoted:
                failed = 'its predecessor' if wr_vs_cur < args.promote_threshold else 'the run base'
                logger.warning(
                    'round %d/%d REJECTED: round%d_policy did not clear %.2f against %s '
                    '(vs predecessor %s %.3f, vs run base %s). Keeping %s as the base for the '
                    'next round instead of building on a regression.',
                    r + 1, args.rounds, r, args.promote_threshold, failed,
                    os.path.basename(cur_policy), wr_vs_cur,
                    'n/a' if wr_vs_base is None else f'{wr_vs_base:.3f}',
                    os.path.basename(cur_policy))
                continue

            # With --freeze-critic the critic never changes scale, so the search stays in
            # the 'shaped' mode that produced the only ExIt round that helped. Otherwise the
            # critic is now z-scale from this round on and must be paired with outcome mode.
            cur_policy = out_policy
            cur_critic = critic_in_force
            cur_mode = cur_mode if args.freeze_critic else 'outcome'
    finally:
        if collector is not None:
            collector.shutdown()
    logger.info('ExIt loop finished in %s. Best promoted nets: policy=%s critic=%s (value_mode=%s).',
                args.run_dir, cur_policy, cur_critic, cur_mode)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _add_common(p):
    p.add_argument('--policy', default=None, help='Base policy .pth (default: newest data/warchest_ppo_*.pth).')
    p.add_argument('--critic', default=None, help='Base critic .pth (default: newest lookahead_critic_v*.pth).')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--run-dir', default=None,
                   help='Directory for this run\'s artifacts (datasets, distilled nets, '
                        'meta.json). Default: data/exit/{launch timestamp}/ — one fresh '
                        'directory per invocation, so a new run can never overwrite an '
                        'earlier one\'s checkpoints. Pass an existing path to add to it.')
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
    p.add_argument('--blind-teacher', action='store_true', default=True,
                   help="Data-gen teacher does NOT read the opponent's hand (the default "
                        "since 2026-08-20; --cheating-teacher restores PuctBot's own "
                        "see_opponent_hand=True default). Nothing in ExIt ever turned it off, "
                        "so every recorded run distilled a search that conditioned on a "
                        "variable the student's observation does not contain: cheat and blind "
                        "searches pick different moves on 15.4%% of positions (TV 0.123) "
                        "against the 13.1%% on which the search improves on its prior at all "
                        "— irreducible label noise at least as large as the entire signal, "
                        "sitting exactly on top of it (docs/IDEAS.md R.10.4, M3). Blind search "
                        "measured ~0 weaker in strength (search_under_uncertainty.md §8.1), so "
                        "gate 3 is the check that this holds here too.")
    p.add_argument('--cheating-teacher', dest='blind_teacher', action='store_false',
                   help='Let the data-gen teacher read the opponent real hand (the pre-'
                        '2026-08-20 behaviour). For reproducing old runs.')
    p.add_argument('--n-determinizations', type=int, default=3,
                   help='PIMC votes per teacher move. With a blind root each search samples '
                        'its own re-split of the opponent hidden coins, so n=1 is single-'
                        'determinization search — the strategy-fusion failure mode '
                        'docs/search_under_uncertainty.md §2 is about. The budget is SPLIT '
                        '(0.8 primary + 0.2 spread over the hedges, LookaheadCriticBot.act), '
                        'so this costs no extra wall-clock; it trades some primary depth for a '
                        'second opinion. 1 with --cheating-teacher reproduces the old teacher.')
    p.add_argument('--forced-playouts-k', type=float, default=0.0,
                   help="KataGo forced playouts at the search root: every child is guaranteed "
                        "ceil(sqrt(k * P(a) * N)) visits, and the same allocation is subtracted "
                        "back out of the recorded target (policy target pruning). 0 = off. "
                        "Aimed at the one quantity nothing else here touches: the prior puts "
                        "0.817 on its top move and only ~3.3 of the 8 kept children have a "
                        "first-visit U bonus big enough for PUCT to reach at all, so the visit "
                        "counts largely re-encode the prior and the search cannot promote a "
                        "move the prior ranked low (docs/IDEAS.md R.10 M1). k=2 is KataGo's "
                        "value; it reserves ~60 of 300 expansions at 8 children. Re-read "
                        "preflight gates 2-4 when changing it — it makes the target sharper "
                        "(pruning) AND spends budget on breadth, so both the entropy pair and "
                        "teacher strength can move.")
    p.add_argument('--apprentice-frac', type=float, default=0.0,
                   help="Fraction of recorded plies PLAYED by the raw policy (sampled from its "
                        "own priors) instead of by the search. The search still runs and still "
                        "supplies the target at every ply, so this changes only which states "
                        "the dataset covers, at identical cost. 0.0 is AlphaZero's convention "
                        "(the expert plays); 1.0 is Expert Iteration's original one (Anthony "
                        "et al. 2017 — apprentice plays, expert labels), which exists because "
                        "a policy trained only on states the stronger player reaches has no "
                        "target for the positions its own weaker play walks into (the "
                        "compounding-error argument against plain behaviour cloning). The "
                        "symptom it targets: a round's in-sample agreement with the teacher "
                        "rises 0.744 -> 0.861 while its strength does not move (IDEAS.md "
                        "R.10.12). 0.5 mixes both distributions.")
    # distill knobs
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--minibatch', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--val-frac', type=float, default=0.1,
                   help='Held-out share, split by GAME rather than by sample (~84 samples '
                        'share a game trajectory and its outcome, so a per-sample split puts '
                        'near-duplicates on both sides — docs/IDEAS.md R.10.5c).')
    p.add_argument('--no-early-stop', action='store_true',
                   help='Run all --epochs and keep the last one, i.e. treat the held-out split '
                        'as a report only. That was the behaviour through 2026-08-19: val_frac '
                        'was computed, logged and never used to stop, so ~350 unconstrained '
                        'Adam steps ran regardless of what it said (R.10.5c / R.10.8 item 3).')
    p.add_argument('--patience', type=int, default=2,
                   help='Epochs without held-out improvement before distillation stops early. '
                        'The best-scoring epoch weights are restored either way.')
    p.add_argument('--visit-temp', type=float, default=0.5,
                   help="Sharpening exponent applied to the recorded visit distribution "
                        "before it is used as the CE target (`t = t**(1/visit_temp)`, "
                        "renormalised; 1.0 = off, the raw AlphaZero convention). Exists "
                        "because at this project's search budget the raw visit counts are "
                        "measurably LESS decisive than the policy already distilling toward "
                        "them: on `data/exit/round0.npz` (2026-08-18, --time-budget 1.0) mean "
                        "visit entropy was 0.720 nats against a pre-distill policy entropy of "
                        "0.469, and one round of unsharpened distillation dragged the policy's "
                        "own entropy up to 0.875 — the loop making the model measurably worse "
                        "(base beat every one of 3 rounds by a widening margin, 0.70/0.78/0.82; "
                        "docs/IDEAS.md R.10.9). 0.5 measured 0.304 nats on that same dataset, a "
                        "clear margin under 0.469; canonical high-sim-count AlphaZero doesn't "
                        "need this knob because its search naturally concentrates visits via "
                        "PUCT's positive feedback loop before ~800 sims are spent, which ~97 "
                        "expansions at max_branching=8 does not have time to do. Re-check via "
                        "`preflight`'s gate 4 whenever --time-budget or --max-branching change.")
    p.add_argument('--kl-coeff', type=float, default=0.0,
                   help="Trust-region penalty added to the policy loss: "
                        "kl_coeff * KL(policy_at_round_start || policy). 0.0 (default) is the "
                        "old, unregularised CE-only distillation. Complements --replay-rounds: "
                        "the replay window controls which data a round trains on, this controls "
                        "how far plain CE (which has no trust region, unlike PPO's clip) is "
                        "allowed to move the policy on it in `epochs` passes at `lr` "
                        "(docs/IDEAS.md R.10.10). Untuned — calibrate with a quick `distill` + "
                        "gauntlet check on an existing dataset before trusting a value here.")
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
    lp.add_argument('--gauntlet-k-games', type=int, default=100,
                    help='Games per pair in the post-round gauntlet check, which also decides '
                         'promotion (see --promote-threshold). Raised 20 -> 100 on 2026-08-20: '
                         'se(WR) is ~11 pp at 20 and ~5 pp at 100, and the whole 8-round run of '
                         '2026-08-19 promoted on 0.533/0.533/0.500/0.500 measured at k=30 '
                         '(se ~ 9 pp) — every one of those decisions was inside its own noise, '
                         'and the run\'s own final field then read the result as a loss to base '
                         '(docs/IDEAS.md R.10.11). The field is only 2-3 raw-policy agents, so '
                         'k=100 is seconds of compute against a ~50-minute round.')
    lp.add_argument('--promote-threshold', type=float, default=0.5,
                    help="Minimum win rate a round's distilled policy must score against the "
                         "checkpoint its self-play was generated from (not the run's original "
                         "base) to be accepted; otherwise the round is REJECTED and the next "
                         "round retries self-play from the same, unpromoted checkpoint instead "
                         "of building on a regression. Without this gate, `cur_policy = "
                         "out_policy` happened unconditionally and produced a 3-round run "
                         "where the base beat every round by a widening margin (0.70 / 0.78 / "
                         "0.82 — docs/IDEAS.md R.10.9) while the loop kept building the next "
                         "round on top of the previous one's regression regardless.")
    lp.add_argument('--skip-gauntlet', action='store_true',
                    help='Skip the post-round gauntlet check. Also disables the promotion '
                         'gate above (there is no win-rate to gate on) — every round is then '
                         'accepted unconditionally, which is the pre-fix behaviour.')
    lp.add_argument('--replay-rounds', type=int, default=6,
                    help="Train each round on the concatenation of the last N rounds' self-play "
                         "data (a sliding window, oldest dropped first) instead of this round's "
                         "~25k-sample slice alone. 1 reproduces the old one-round-only behaviour. "
                         "Ships docs/IDEAS.md R.10.8 item 2's replay-window half (the promotion-gate "
                         "half shipped 2026-08-19): distilling on one round only means each fine-tune "
                         "sees a single network's narrow self-play distribution and nothing else, "
                         "which the record ties to the same churn signature as an untrained critic "
                         "(independent_opponents.md §1: held-out MSE re-fit within a round but did "
                         "not generalise to the next one). Cost is RAM only (~0.5 GB/300-game round, "
                         "`SelfPlayDataset.concat`), so the default keeps 6 rounds in the window "
                         "(raised from 3 on 2026-08-20: one round is ~20k samples for a 1.8M-"
                         "parameter net, and round 1 cannot exercise the window at all).")
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
    args.run_dir = resolve_run_dir(args, run_id)
    if args.cmd == 'preflight':
        # Preflight writes no artifacts; do not leave an empty run directory behind.
        logger.info('expert iteration: cmd=%s, logging to %s', args.cmd, log_path)
    else:
        meta_path = write_run_meta(args, args.run_dir, run_id, log_path)
        logger.info('expert iteration: cmd=%s, run_dir=%s, logging to %s (invocation in %s)',
                    args.cmd, args.run_dir, log_path, meta_path)
    args.func(args)


if __name__ == '__main__':
    main()
