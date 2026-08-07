"""Does the critic actually USE its privileged features? (docs/search_under_uncertainty.md §8.1)

`eval_info_value.py` measured that seeing the opponent's hand is worth ~0 to the search.
That has two very different explanations:

  (a) hidden information genuinely matters little in Warchest, or
  (b) our critic never learned to use it.

This tells them apart. The `Critic` is handed a `priv_dim` vector of the opponent's TRUE
hidden coin split (hand / bag / face-down discard per coin — `encode_privileged`), which
is concatenated straight into its value head (`Critic._forward`). If the head has learned
to read it, corrupting that block must move the value; if the value barely moves, the
whole privileged-critic design is inert and (b) is the answer — a far more actionable
finding than anything in §6.

Three readings, cheapest first:

  weights     First-order sensitivity of the value head's first layer to each input
              block, computed as ||W[:, block] * std(x_block)||_F over real states.
              Scale-corrected, so the privileged block is comparable against the board
              and global blocks. Costs milliseconds and needs no games.

  values      Critic value under the true privileged vector vs three corruptions:
              zeroed, permuted across the batch (real vectors, wrong states), and
              fair-resplit (the uniform re-deal `LookaheadBot._prepare_root` uses in
              blind mode — i.e. exactly what the blind arm of eval_info_value feeds it).
              Reports std(delta)/std(v) and Pearson r.

  decisions   The one that matters. The search never uses V(s) in isolation — it uses it
              to RANK sibling children. So for each sampled decision point this scores
              every legal child both ways and reports how often the argmax child changes
              and how the full ranking correlates (Spearman). A critic whose values shift
              but whose *ranking* does not is, for search purposes, still inert.

    python src/app/eval_privileged_ablation.py --games 12
    python src/app/eval_privileged_ablation.py --games 30 --critic-path data/lookahead_critic/lookahead_critic_v4.pth
"""
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch

from src.app.gauntlet import _latest_critic_path, _latest_policy_path, CRITIC_GLOB
from src.services.bots.lookahead_bot import LookaheadBot, _clone_state
from src.services.bots.lookahead_critic_bot import LookaheadCriticBot
from src.services.environment.warchest_env import WarChestEnv

# Corruptions applied to the privileged block. 'true' is the reference.
MODES = ('zero', 'permute', 'resplit')


# --------------------------------------------------------------------------- #
# Critic plumbing
# --------------------------------------------------------------------------- #
def _encode(bot, states):
    """`(boards, globals, privs)` stacked arrays for `states`, as `_critic_values_raw` builds them."""
    boards, globals_, privs = [], [], []
    for state in states:
        bot._sim_env.set_state(state)
        obs = bot._sim_env.generate_observation()
        boards.append(obs['board'])
        globals_.append(obs['global'])
        privs.append(bot._sim_env.get_privileged_features())
    return np.stack(boards), np.stack(globals_), np.stack(privs)


def _values(bot, boards, globals_, privs):
    """Raw critic values for pre-encoded arrays (mirrors `LookaheadCriticBot._critic_values_raw`)."""
    batch = {
        'board': torch.from_numpy(boards).to(bot.device),
        'global': torch.from_numpy(globals_).to(bot.device),
        'opp_onehot': bot._opp_onehot.unsqueeze(0).expand(len(boards), -1),
        'privileged': torch.from_numpy(privs).to(bot.device),
    }
    with torch.inference_mode():
        return bot._critic.value_batch(batch).cpu().numpy()


def _resplit_privs(bot, states, root_player):
    """Privileged vectors of the same states after the blind-mode uniform re-deal.

    This is the exact input the *blind* arm of `eval_info_value.py` feeds the critic, so
    the `resplit` row answers "how much does the critic notice the lie we tell it in fair
    mode" — the corruption with real operational meaning, as opposed to zero/permute
    which are diagnostic extremes.
    """
    out = []
    for state in states:
        s = _clone_state(state)
        LookaheadBot._resplit_hidden(s, 3 - root_player)
        bot._sim_env.set_state(s)
        out.append(bot._sim_env.get_privileged_features())
    return np.stack(out)


def _corrupt(privs, mode, rng):
    if mode == 'zero':
        return np.zeros_like(privs)
    if mode == 'permute':
        # Real privileged vectors attached to the wrong states: keeps the marginal
        # distribution intact and destroys only the state-privileged correlation, so a
        # critic that merely rescales by "how many coins are hidden" is not credited.
        return privs[rng.permutation(len(privs))]
    raise ValueError(mode)


# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #
def _collect_decisions(bot, driver, n_games, sample_every, seed, max_decisions):
    """Play games with `driver` on both sides; capture `(root_player, [child states])`.

    Children (not just visited states) are what the search actually ranks, so the
    decision-level metric needs the whole sibling set of a real decision point. Sampling
    every `sample_every`-th ply keeps successive decision points from being near-copies.
    """
    decisions = []
    ply_total = 0
    for g in range(n_games):
        env = WarChestEnv(save_game_history=False)
        np.random.seed(seed + g)
        env.reset()
        for ply in range(2000):
            if len(decisions) >= max_decisions:
                return decisions
            pid = env.active_player
            if ply_total % sample_every == 0:
                legal = env.get_possible_actions()
                if len(legal) > 1:
                    root_state, queues = bot._prepare_root(env, pid)
                    kids = []
                    for a in legal:
                        s = _clone_state(root_state)
                        q = {1: list(queues[1]), 2: list(queues[2])}
                        res = bot._apply(s, q, a)
                        if not res.finishes_game:
                            kids.append(s)
                    if len(kids) > 1:
                        decisions.append((pid, kids))
            ply_total += 1
            action = driver.act(env)
            _, _, terminated, truncated, info = env.step(action)
            if not info['action'].is_valid:
                _, _, terminated, truncated, info = env.make_random_step()
            if terminated or truncated:
                break
    return decisions


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _pearson(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    """Rank correlation without scipy: Pearson on ranks (ties broken by order — fine here,
    exact ties in float critic outputs are vanishingly rare)."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return _pearson(ra, rb)


def _report_weights(bot):
    """First-order sensitivity of the value head to each input block, scale-corrected.

    `Critic._forward` concatenates [pooled | global | opp_onehot | privileged] and feeds
    `head[0]`, so each block owns a contiguous column slice of that layer's weight. A raw
    weight norm is not comparable across blocks (different input scales), so each column
    is multiplied by that input's observed std before the norm is taken.
    """
    critic = bot._critic
    w = critic.head[0].weight.detach().cpu().numpy()   # [hidden, head_in]
    hidden = w.shape[0]
    n_pool = 2 * hidden
    blocks = [
        ('board (pooled)', 0, n_pool),
        ('global', n_pool, n_pool + critic.global_dim),
        ('opp_onehot', n_pool + critic.global_dim, n_pool + critic.global_dim + critic.OPP_DIM),
        ('privileged', w.shape[1] - critic.priv_dim, w.shape[1]),
    ]
    return w, blocks


def main():
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(
        description="Test whether the Critic's privileged features affect its output at all.")
    ap.add_argument('--games', type=int, default=12,
                    help='Games played to harvest decision points. Default 12.')
    ap.add_argument('--sample-every', type=int, default=7,
                    help='Capture a decision point every N plies (decorrelation). Default 7.')
    ap.add_argument('--max-decisions', type=int, default=400)
    ap.add_argument('--driver-time-budget', type=float, default=0.02,
                    help='Per-move budget of the LookaheadBot generating the games. Default 0.02.')
    ap.add_argument('--critic-path', default=None, help=f'Default: newest {CRITIC_GLOB}.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    critic_path = args.critic_path or _latest_critic_path()
    if critic_path is None:
        raise SystemExit(f'no critic checkpoint matching {CRITIC_GLOB}')

    bot = LookaheadCriticBot(critic_path=critic_path, time_budget=0.0,
                             stats_log_every=0, device=args.device)
    driver = LookaheadBot(time_budget=args.driver_time_budget)
    rng = np.random.default_rng(args.seed)

    print(f'Critic: {critic_path}  (priv_dim={bot._critic.priv_dim}, '
          f'global_dim={bot._critic.global_dim})')

    # --- 1. weights -------------------------------------------------------- #
    t0 = time.perf_counter()
    decisions = _collect_decisions(bot, driver, args.games, args.sample_every,
                                   args.seed, args.max_decisions)
    if not decisions:
        raise SystemExit('no decision points collected — raise --games')
    flat_states = [s for _, kids in decisions for s in kids]
    root_player = decisions[0][0]
    boards, globals_, privs = _encode(bot, flat_states)
    print(f'{len(decisions)} decision points, {len(flat_states)} states '
          f'({time.perf_counter() - t0:.0f}s)')

    w, blocks = _report_weights(bot)
    # Board block std is measured through the encoder — and while we are running it,
    # check the trunk is alive at all. A stack of HexConv2d+ReLU whose last ReLU
    # receives only negative pre-activations outputs exactly zero for every input, and
    # then `_split_pool` feeds `head[0]` a block of hard zeros: the critic is blind to
    # the board and nothing else in this report can be read without knowing that.
    with torch.inference_mode():
        board_t = torch.from_numpy(boards).to(bot.device)
        pre = board_t
        for layer in list(bot._critic.board_encoder)[:-1]:
            pre = layer(pre)
        feat = bot._critic.board_encoder(board_t)
        from src.services.policy.policy import _split_pool
        pooled = _split_pool(feat).cpu().numpy()
        pooled_std = pooled.std(axis=0)
        alive = float((pre > 0).float().mean())

    print(f'\n0. TRUNK HEALTH — final ReLU pre-activations > 0: {alive:.4%}  |  '
          f'pooled |max|={np.abs(pooled).max():.4g}')
    if np.abs(pooled).max() == 0.0:
        print('  *** DEAD SPATIAL TRUNK: the board encoder outputs exactly zero for every')
        print('  *** state, so `head[0]` sees a block of hard zeros and this critic is')
        print('  *** BLIND TO THE BOARD — it is a function of global + privileged only.')
        print('  *** Every number below is measured on a crippled critic. Fix this first.')
    stds = np.concatenate([
        pooled_std,
        globals_.std(axis=0),
        np.zeros(bot._critic.OPP_DIM) + 1e-9,  # constant one-hot within a run
        privs.std(axis=0),
    ])
    print('\n1. HEAD SENSITIVITY — ||W[:, block] * std(input)||_F, scale-corrected')
    total = 0.0
    rows = []
    for name, lo, hi in blocks:
        s = float(np.linalg.norm(w[:, lo:hi] * stds[None, lo:hi]))
        rows.append((name, hi - lo, s))
        total += s
    for name, dim, s in rows:
        print(f'  {name:<16} dim={dim:<5} sensitivity={s:8.4f}   {s / total * 100:5.1f}% of total')
    print('  (a near-zero privileged share means the head ignores the block outright)')

    # --- 2. values --------------------------------------------------------- #
    v_true = _values(bot, boards, globals_, privs)
    print(f'\n2. VALUE SHIFT UNDER CORRUPTION  (n={len(flat_states)}, '
          f'std(V)={v_true.std():.4f})')
    variants = {m: _corrupt(privs, m, rng) for m in ('zero', 'permute')}
    variants['resplit'] = _resplit_privs(bot, flat_states, root_player)
    v_alt = {}
    for mode, p in variants.items():
        v = _values(bot, boards, globals_, p)
        v_alt[mode] = v
        d = v - v_true
        print(f'  {mode:<10} std(delta)/std(V)={d.std() / max(v_true.std(), 1e-12):6.3f}   '
              f'mean|delta|={np.abs(d).mean():.4f}   r(V_true, V_alt)={_pearson(v_true, v):.4f}')

    # --- 3. decisions ------------------------------------------------------ #
    print(f'\n3. DECISION-LEVEL — does the RANKING of sibling children change? '
          f'({len(decisions)} points)')
    off = 0
    idx_ranges = []
    for _, kids in decisions:
        idx_ranges.append((off, off + len(kids)))
        off += len(kids)
    for mode in MODES:
        flips, rhos = 0, []
        for lo, hi in idx_ranges:
            a, b = v_true[lo:hi], v_alt[mode][lo:hi]
            if int(np.argmax(a)) != int(np.argmax(b)):
                flips += 1
            rhos.append(_spearman(a, b))
        rho = float(np.nanmean(rhos))
        print(f'  {mode:<10} argmax child changed in {flips / len(idx_ranges):6.1%} of decisions   '
              f'mean Spearman={rho:.4f}')

    print('\nHOW TO READ')
    print('  Privileged share <~2% in (1), std(delta)/std(V) <~0.05 and r >0.99 in (2),')
    print('  and argmax flips <~5% in (3) => the critic ignores the privileged block.')
    print('  Then eval_info_value\'s null measures OUR CRITIC, not the game, and the')
    print('  honest next step is to fix/retrain the critic before retiring §6.')
    print('  Large shifts with few argmax flips => it reads the block but the information')
    print('  does not change which move looks best — that IS the game answering.')


if __name__ == '__main__':
    main()
