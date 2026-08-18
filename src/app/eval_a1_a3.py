"""Did A1 (unit-type embedding) and A3 (FiLM) actually do anything? — docs/IDEAS.md A1 + A3.

Motivation: the first `policy_factored_v2` run came back at ~50 % head-to-head against the
`policy_factored_v1` baseline. That number on its own says almost nothing, for two separate
reasons, and this script exists to replace it with measurements that can come back negative.

**Reason 1 — a pooled win rate has no power.** At p = 0.5 the standard error is
sqrt(0.25/n): 200 games gives +/- 7 pp at 95 %, so "50 %" is consistent with anything from
43 % to 57 %. `comps` prints the CI and the minimum detectable effect for whatever n was
actually run, so the result is stated as a bound rather than as a point.

**Reason 2 — neither A1 nor A3 predicts a pooled gain in the first place.** A1's claim is
about *generalisation across draft compositions*: shared, rules-grounded columns should help
where a per-type slot is under-trained, which is a claim about the tail, and a tail effect is
diluted by every ordinary game in the pool. A3's claim is about *hand-conditioning of the
board features*, which is a claim about the function the net computes, not directly about
strength at all.

So the three subcommands run in increasing order of cost, and the cheap ones come first
because they can invalidate the expensive one:

  `weights`  (instant, no games) — is A3 even switched on? `FiLM`'s output layer is
             zero-initialised, so an untrained or weakly-trained net is *exactly* the
             identity and v2 collapses to "v1 plus an embedding". The decisive number is not
             |gamma| but the **per-channel spread of gamma across observations**: a gamma that
             is large but constant is a learned per-channel gain, not conditioning, and buys
             none of A3's argument. Also reports whether A1's learned rows moved off their
             init at all, and which frozen columns the trunk leans on.

  `hand`     (seconds, no games) — A3's claim, measured directly and with a control that
             cannot be argued with. Hold a board fixed, substitute other states' hands, and
             ask how much the within-verb preference over the 49 cells moves. On v1 this is
             **provably exactly zero**: `policy_head` is a 1x1 conv over `[feat, broadcast(g)]`,
             so the globals contribute `W_g @ g` identically at every cell and the term
             cancels in a softmax across cells. If the v1 arm reports anything but 0 the
             measurement is broken, not the network.

  `comps`    (minutes+, games) — A1's claim, as a controlled experiment rather than an
             observational bucketing. `force_units` pins a whole 4-unit composition, so
             instead of hoping rare drafts turn up, the archetypes are *constructed* out of
             the frozen attribute columns themselves (all-vanilla, all-tactic, ranged
             strikers, support/grant, ...). Both arms play the same forced archetypes.

Usage — note that `--new` / `--old` / `--seed` are global and go BEFORE the subcommand:
    python src/app/eval_a1_a3.py weights
    python src/app/eval_a1_a3.py hand --states 60 --swaps 12
    python src/app/eval_a1_a3.py comps --games 400
    python src/app/eval_a1_a3.py --seed 7 comps --games 1200 --archetype tempo
    python src/app/eval_a1_a3.py --new data/a.pth --old data/b.pth comps --out-csv comps.csv

A single archetype with `--archetype` is the confirmatory form: one pre-chosen row carries no
multiple-comparison penalty, so it is how a starred lead from the all-archetype table gets
either confirmed or dropped. Use a fresh `--seed` for it, or it is the same games again.

Both checkpoints default to the newest `policy_factored_v2` / `policy_factored_v1` pair in
data/. Everything reports a 95 % interval; nothing here reports a bare point estimate.
"""
import argparse
import csv
import glob
import itertools
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from rich.progress import track

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.services.environment.obs_encoders import get_encoder
from src.services.environment.roster import UNIT_TYPES
from src.services.environment.warchest_env import (
    BOARD_DIM, N_VERBS, SPATIAL_SIZE, WarChestEnv,
)
from src.services.gauntlet import PolicyAgent
from src.services.policy.checkpoint import load_policy_checkpoint
from src.services.policy.policy import POLICY_ARCH_V1, POLICY_ARCH_V2, Policy
from src.services.policy.unit_embedding import FROZEN_ATTR_NAMES, LEARNED_INIT_STD

DEV = torch.device('cpu')
Z95 = 1.959963985


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _latest_of_arch(arch):
    """Newest data/warchest_ppo_*.pth whose recorded arch matches, or None."""
    best = None
    for path in sorted(glob.glob('data/warchest_ppo_*.pth')):
        try:
            meta = load_policy_checkpoint(path)
        except Exception:
            continue
        if meta['arch'] == arch:
            best = path
    return best


def load_arm(path, label):
    meta = load_policy_checkpoint(path, map_location=DEV)
    enc = get_encoder(meta['obs_version'])
    pol = Policy(DEV, hidden_dim=meta['hidden_dim'], obs_encoder=enc, arch=meta['arch']).to(DEV)
    pol.load_state_dict(meta['state_dict'])
    pol.eval()
    print(f'{label}: {os.path.basename(path)}  arch={meta["arch"]} '
          f'obs=v{meta["obs_version"]} hidden={meta["hidden_dim"]}')
    return pol, enc, meta


def resolve_arms(args):
    new = args.new or _latest_of_arch(POLICY_ARCH_V2)
    old = args.old or _latest_of_arch(POLICY_ARCH_V1)
    if new is None or old is None:
        raise SystemExit(
            f'need one checkpoint of each arch in data/ (found new={new}, old={old}); '
            f'pass --new/--old explicitly')
    return load_arm(new, 'new'), load_arm(old, 'old')


def wilson(k, n):
    """95 % Wilson interval for k/n — honest at the small n and extreme p a bucket hits."""
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    p = k / n
    d = 1 + Z95 ** 2 / n
    centre = (p + Z95 ** 2 / (2 * n)) / d
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n ** 2)) / d
    return p, centre - half, centre + half


def newcombe_diff(k1, n1, k2, n2, z=Z95):
    """Interval for p1 - p2 (Newcombe's square-and-add of two Wilson intervals).

    Used instead of a Wald interval because the interesting buckets land at 0/n or n/n,
    where Wald reports a zero-width interval and would turn "no data either way" into
    "certainly no difference". `z` is exposed so the caller can also ask for a
    family-wise-corrected interval when several buckets are tested at once.
    """
    if n1 == 0 or n2 == 0:
        return float('nan'), float('nan'), float('nan')
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    if z != Z95:  # rescale each Wilson half-width to the requested level
        s = z / Z95
        l1, u1 = p1 - (p1 - l1) * s, p1 + (u1 - p1) * s
        l2, u2 = p2 - (p2 - l2) * s, p2 + (u2 - p2) * s
    return (d,
            d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def _bonferroni_z(m):
    """z for a two-sided family-wise 0.05 across `m` comparisons (Bonferroni).

    Six archetypes tested at 0.05 each means a ~26 % chance that at least one row lights up
    with both nets identical, which is exactly how a spread gets read into noise. The
    corrected column is the one to believe when several rows are on screen.
    """
    from statistics import NormalDist
    return NormalDist().inv_cdf(1 - 0.05 / (2 * max(m, 1)))


# --------------------------------------------------------------------------- #
# Shared: collect real observations to probe on
# --------------------------------------------------------------------------- #
def collect_states(policy, n, *, seed=0, max_t=400):
    """Play games with `policy` on both seats and snapshot decision-point observations.

    On-distribution states matter here: probing a net on `torch.randn` boards measures its
    behaviour somewhere it never has to be right.
    """
    rng = np.random.default_rng(seed)
    env = WarChestEnv(save_game_history=False, debug_mode=False)
    out = []
    while len(out) < n:
        np.random.seed(int(rng.integers(1 << 30)))
        obs, _ = env.reset()
        for _ in range(max_t):
            out.append({k: np.array(v, copy=True) for k, v in obs.items()
                        if k in ('board', 'global', 'valid_action_mask')})
            if len(out) >= n:
                break
            action, _, _ = policy.act(obs)
            # Same ego-frame remap `PolicyAgent.act` does — without it every P2 ply is
            # mirrored, fails validation and degrades into a random move, so the states
            # collected here would drift off the distribution they are meant to represent.
            if env.active_player == 2:
                action = WarChestEnv.remap_action(action)
            obs, _, terminated, truncated, info = env.step(action)
            if not info['action'].is_valid:
                obs, _, terminated, truncated, info = env.make_random_step()
            if terminated or truncated:
                break
    return out[:n]


def _stack(states, key, dtype=torch.float32):
    return torch.from_numpy(np.stack([s[key] for s in states])).to(dtype)


# --------------------------------------------------------------------------- #
# `weights` — is A3 switched on, did A1's learned half move
# --------------------------------------------------------------------------- #
def film_report(policy, states):
    """Per-block gamma statistics on real observations.

    `across_obs_std` is the one that matters. FiLM's argument is that the *same* channel is
    worth different amounts under different hands; a channel whose gamma is identical on
    every observation is a constant gain that a plain per-channel weight already had. So:

        mean|gamma|        how far the module is from its zero-init identity at all
        across_obs_std     how much of that varies WITH the hand — the conditioning itself
        attenuated         share of (channel, obs) pairs with |1 + gamma| < 0.5, i.e. the
                           switching-off behaviour that motivated multiplying over adding
    """
    g = policy._embed_globals(_stack(states, 'global'))
    rows = []
    with torch.no_grad():
        for i, film in enumerate(policy.films):
            gamma, beta = film.net(g).chunk(2, dim=-1)  # [N, C] each
            rows.append({
                'block': i + 1,
                'channels': gamma.shape[1],
                'mean_abs_gamma': float(gamma.abs().mean()),
                'across_obs_std': float(gamma.std(dim=0).mean()),
                'across_chan_std': float(gamma.mean(dim=0).std()),
                'attenuated': float(((1.0 + gamma).abs() < 0.5).float().mean()),
                'mean_abs_beta': float(beta.abs().mean()),
            })
    return rows


def embedding_report(policy):
    """Did the learned rows move off init, and did the three planted collisions separate?

    The frozen table is deliberately non-injective: Swordsman/Berserker/Mercenary,
    Knight/Pikeman and Ensign/Marshall share a frozen row on purpose, and the learned block
    is what is supposed to pull them apart as data arrives. If those pairs are still at
    init-scale distance, the learned half never engaged and v2 is running on frozen
    attributes alone.
    """
    emb = policy.type_emb.emb
    with torch.no_grad():
        learned = emb.learned.detach()
        frozen = emb.frozen.detach()
    name = {i: u.name for i, u in enumerate(UNIT_TYPES)}
    # Expected distance between two independent rows at init: sqrt(2 * d) * std.
    init_scale = math.sqrt(2 * learned.shape[1]) * LEARNED_INIT_STD

    groups = defaultdict(list)
    for i in range(frozen.shape[0]):
        groups[frozen[i].numpy().tobytes()].append(i)
    collided = [idxs for idxs in groups.values() if len(idxs) > 1]

    pairwise = torch.cdist(learned, learned)
    off_diag = pairwise[~torch.eye(len(learned), dtype=torch.bool)]
    return {
        'row_norm_mean': float(learned.norm(dim=1).mean()),
        'row_norm_init_expected': math.sqrt(learned.shape[1]) * LEARNED_INIT_STD,
        'pairwise_mean': float(off_diag.mean()),
        'pairwise_init_expected': init_scale,
        'collisions': [
            {
                'members': [name[i] for i in idxs],
                'mean_dist': float(np.mean([float(pairwise[a, b])
                                            for a, b in itertools.combinations(idxs, 2)])),
            }
            for idxs in collided
        ],
    }


def frozen_usage_report(policy, states):
    """Which frozen columns the network actually leans on.

    Gradient of the summed spatial logits w.r.t. each frozen column, accumulated over real
    states. A column with ~0 sensitivity is being ignored — worth knowing before adding more.
    """
    emb = policy.type_emb.emb
    board = _stack(states, 'board')
    glob = _stack(states, 'global')
    emb.frozen.requires_grad_(True)
    try:
        flat, _ = policy._features(board, glob)
        flat[:, :SPATIAL_SIZE].abs().sum().backward()
        grad = emb.frozen.grad.abs().sum(dim=0).detach().clone()
    finally:
        emb.frozen.grad = None
        emb.frozen.requires_grad_(False)
    total = float(grad.sum()) or 1.0
    return sorted(
        [(FROZEN_ATTR_NAMES[i], float(grad[i]) / total) for i in range(len(FROZEN_ATTR_NAMES))],
        key=lambda kv: -kv[1])


def cmd_weights(args):
    (new, _, _), (old, _, _) = resolve_arms(args)
    if new.type_emb is None:
        raise SystemExit('--new is not a policy_factored_v2 checkpoint; nothing to inspect')
    states = collect_states(new, args.states, seed=args.seed)
    print(f'\nprobing on {len(states)} on-distribution states\n')

    print('== A3: is FiLM switched on? ==')
    print('  (zero-init means an untrained FiLM reads all-zero and v2 == v1 + embedding)')
    print(f'{"block":>6} {"chans":>6} {"mean|g|":>9} {"std across obs":>15} '
          f'{"std across chan":>16} {"attenuated":>11} {"mean|b|":>9}')
    rows = film_report(new, states)
    for r in rows:
        print(f'{r["block"]:>6} {r["channels"]:>6} {r["mean_abs_gamma"]:>9.4f} '
              f'{r["across_obs_std"]:>15.4f} {r["across_chan_std"]:>16.4f} '
              f'{r["attenuated"]:>10.1%} {r["mean_abs_beta"]:>9.4f}')
    worst = max(r['across_obs_std'] for r in rows)
    if worst < 1e-3:
        print('\n  VERDICT: FiLM is INERT — gamma does not vary with the observation. A3 did '
              '\n  not happen in this run; whatever it measured is A1 alone.')
    elif max(r['mean_abs_gamma'] for r in rows) < 1e-2:
        print('\n  VERDICT: FiLM barely left its zero init. Treat A3 as untested.')
    else:
        print(f'\n  VERDICT: FiLM is active and observation-dependent (max across-obs std '
              f'{worst:.4f}).\n  `hand` quantifies what that buys.')

    print('\n== A1: did the learned rows move off init? ==')
    e = embedding_report(new)
    print(f'  row norm      {e["row_norm_mean"]:.4f}   (init expectation '
          f'{e["row_norm_init_expected"]:.4f})')
    print(f'  pairwise dist {e["pairwise_mean"]:.4f}   (init expectation '
          f'{e["pairwise_init_expected"]:.4f})')
    print('  planted collisions — the frozen block ties these, the learned block must split them:')
    for c in e['collisions']:
        ratio = c['mean_dist'] / e['pairwise_init_expected']
        flag = 'still at init scale' if ratio < 1.2 else 'separated'
        print(f'    {"+".join(c["members"]):<38} dist {c["mean_dist"]:.4f} '
              f'({ratio:.2f}x init)  {flag}')

    print('\n== A1: which frozen columns the trunk leans on (share of |grad|) ==')
    for name, share in frozen_usage_report(new, states):
        bar = '#' * int(round(share * 60))
        print(f'  {name:<26} {share:>6.1%} {bar}')
    print('\n(old arm is loaded only to confirm the pair; it has no FiLM or embedding.)')
    assert old.type_emb is None or old.arch == POLICY_ARCH_V1


# --------------------------------------------------------------------------- #
# `hand` — A3's claim, with a provable v1 control
# --------------------------------------------------------------------------- #
def hand_sensitivity(policy, states, swaps, rng):
    """How much does the within-verb preference over cells move when only the hand changes?

    For each state we keep the board and the legal-action mask fixed and substitute the
    `global` vector of `swaps` other states. Holding the mask fixed is deliberate: the hand
    obviously changes which actions are legal, and that channel of influence exists on both
    architectures. What is being isolated here is whether the hand changes the *network's*
    spatial preference, which is the thing A3 added.

    -> (mean total-variation distance, mean top-1 cell flip rate)
    """
    boards = _stack(states, 'board')
    globs = _stack(states, 'global')
    n = len(states)
    tv_all, flip_all = [], []
    with torch.no_grad():
        for i in range(n):
            b = boards[i:i + 1].expand(swaps + 1, -1, -1, -1)
            picks = rng.choice(n, size=swaps, replace=False)
            g = torch.cat([globs[i:i + 1], globs[picks]], dim=0)
            flat, _ = policy._features(b, g)
            sp = flat[:, :SPATIAL_SIZE].view(swaps + 1, N_VERBS, BOARD_DIM * BOARD_DIM)
            p = torch.softmax(sp, dim=-1)          # within-verb, across the 49 cells
            base = p[0:1]
            tv_all.append(float((p[1:] - base).abs().sum(dim=-1).mul(0.5).mean()))
            top = p.argmax(dim=-1)                 # [swaps+1, N_VERBS]
            flip_all.append(float((top[1:] != top[0:1]).float().mean()))
    return float(np.mean(tv_all)), float(np.mean(flip_all))


def cmd_hand(args):
    (new, _, _), (old, _, _) = resolve_arms(args)
    states = collect_states(new, args.states, seed=args.seed)
    print(f'\nswapping {args.swaps} foreign hands onto each of {len(states)} boards, '
          f'mask held fixed\n')
    print(f'{"arm":>6} {"arch":>22} {"TV distance":>13} {"top-1 flip":>12}')
    for label, pol in (('new', new), ('old', old)):
        tv, flip = hand_sensitivity(pol, states, args.swaps, np.random.default_rng(args.seed))
        print(f'{label:>6} {pol.arch:>22} {tv:>13.5f} {flip:>11.2%}')
    print('\n  The old arm MUST read exactly 0.00000 / 0.00 %: on policy_factored_v1 the')
    print('  globals enter `policy_head` as a value broadcast to every cell, so within a verb')
    print('  they shift all 49 logits by the same constant and cancel in the softmax. A')
    print('  non-zero old-arm reading means this measurement is wrong, not the network.')
    print('  The new arm\'s number is A3 in one figure: how much the hand re-ranks the board.')


# --------------------------------------------------------------------------- #
# `comps` — A1's claim, on constructed attribute-extreme compositions
# --------------------------------------------------------------------------- #
def _ids(*names):
    by_name = {u.name: u.id for u in UNIT_TYPES}
    return tuple(by_name[n] for n in names)


# Archetypes built out of the frozen columns themselves, so a difference here is a
# difference on the axis A1 claims to have installed. Each is 4 distinct types, and the
# opponent's 4 are drawn from the remaining 12 at random.
ARCHETYPES = {
    # no tactic at all — the plainest possible read of the roster
    'vanilla': _ids('Swordsman', 'Knight', 'Berserker', 'Pikeman'),
    # every member has a tactic, and no two share a mechanic
    'tactic': _ids('Cavalry', 'Light Cavalry', 'Footman', 'Royal Guard'),
    # tactic_ranged_strike + tactic_charge_strike; two of them cannot attack normally
    'ranged': _ids('Archer', 'Crossbowman', 'Lancer', 'Cavalry'),
    # tactic_targets_friendly + tactic_relocates_self — force projection, no direct damage
    'support': _ids('Ensign', 'Marshall', 'Light Cavalry', 'Scout'),
    # every defensive-trait unit in the roster, padded to 4
    'defensive': _ids('Knight', 'Pikeman', 'Royal Guard', 'Swordsman'),
    # gives_extra_tempo across the board
    'tempo': _ids('Swordsman', 'Berserker', 'Mercenary', 'Warrior Priest'),
}


def play_forced(agent_p1, agent_p2, *, seed, force_units, max_turns=2000):
    """One game with a pinned composition. `gauntlet.play_game` plus `reset` options.

    Agents are `PolicyAgent`s, not bare `Policy` objects, and that is load-bearing rather
    than tidiness: the observation is ego-centric, so a policy's chosen action id is in the
    *rotated* frame whenever P2 is to move and has to go back through
    `WarChestEnv.remap_action` before `step`. Skipping that does not raise — every P2 move
    lands mirrored, fails validation, and silently becomes a random legal move, which reads
    as "P1 wins almost every game" rather than as a bug. `stats['invalid']` is returned so
    that failure can never pass quietly again.
    """
    env = WarChestEnv(save_game_history=False)
    np.random.seed(seed)
    # torch's global RNG as well, not just numpy's: the draft comes from numpy but every
    # action is a `Categorical.sample()` off the torch generator, so seeding only numpy
    # leaves the games unreproducible — two runs at the same --seed then disagree on which
    # archetype looks significant, which is indistinguishable from a real effect moving.
    torch.manual_seed(seed)
    env.reset(options={'force_units': force_units} if force_units else None)
    agents = {1: agent_p1, 2: agent_p2}
    invalid = plies = 0
    for _ in range(max_turns):
        pid = env.active_player
        action = agents[pid].act(env)
        _, _, terminated, truncated, info = env.step(action)
        plies += 1
        if not info['action'].is_valid:
            invalid += 1
            _, _, terminated, truncated, info = env.make_random_step()
        if terminated:
            return pid, invalid, plies
        if truncated:
            return 0, invalid, plies
    return 0, invalid, plies


def cmd_comps(args):
    (new, new_enc, _), (old, old_enc, _) = resolve_arms(args)
    new_agent = PolicyAgent('new', new, new_enc)
    old_agent = PolicyAgent('old', old, old_enc)
    names = list(ARCHETYPES) if args.archetype == 'all' else [args.archetype]
    # The schedule cycles holder x seat, so a bucket needs a multiple of 4 to be balanced;
    # anything less silently leaves one arm never holding the archetype and its cell empty.
    per = max(4, 4 * round(args.games / len(names) / 4))
    total = per * len(names)
    if total != args.games:
        print(f'\n  rounding {args.games} -> {total} games so each archetype gets a multiple '
              f'of 4 (holder x seat)')
    print(f'\n{per} games per archetype ({per // 2} per holder), {len(names)} archetypes\n')

    rows = []
    tot_invalid = tot_plies = 0
    for arch_name in names:
        comp = ARCHETYPES[arch_name]
        # Each archetype is played from BOTH sides. Forcing it only onto the new arm would
        # confound "v2 plays this composition worse" with "this composition is weak": a deck
        # that loses to anything would read as a v2 regression. Running the mirror gives the
        # old arm the same deck against the same kind of opposition, so the composition's own
        # strength cancels and what is left is the difference between the two nets on it.
        holder = {'new': [0, 0], 'old': [0, 0]}  # label -> [wins, decided]
        for gi in track(range(per), description=f'{arch_name:<10}'):
            seed = args.seed * 1_000_003 + gi
            new_holds = (gi // 2) % 2 == 0   # alternate the holder every 2 games...
            holder_is_p1 = gi % 2 == 0       # ...and the seat every game
            label = 'new' if new_holds else 'old'
            held_by = new_agent if new_holds else old_agent
            other = old_agent if new_holds else new_agent
            force = {1 if holder_is_p1 else 2: list(comp)}
            p1, p2 = (held_by, other) if holder_is_p1 else (other, held_by)
            res, invalid, plies = play_forced(p1, p2, seed=seed, force_units=force,
                                              max_turns=args.max_turns)
            tot_invalid += invalid
            tot_plies += plies
            if res == 0:
                continue  # draws are excluded from both numerator and denominator
            holder[label][1] += 1
            if (res == 1) == holder_is_p1:
                holder[label][0] += 1
        nw, nd = holder['new']
        ow, od = holder['old']
        rows.append((arch_name, nw, nd, ow, od) + newcombe_diff(nw, nd, ow, od))

    frac_invalid = tot_invalid / max(tot_plies, 1)
    if frac_invalid > 0.02:
        print(f'\n  WARNING: {frac_invalid:.1%} of plies were illegal and fell back to a '
              f'random legal move.\n  Above a fraction of a percent this is a protocol bug '
              f'(ego-frame remap, mask, obs version),\n  not policy noise — the numbers below '
              f'are measuring the fallback, not the nets.')

    print(f'\nWR = win rate of whichever net was DEALT the archetype, vs the other net on a '
          f'random fill.\n')
    zc = _bonferroni_z(len(rows))
    print(f'{"archetype":<12} {"new holds":>18} {"old holds":>18} {"difference":>10} '
          f'{"95% CI on diff":>20}')
    for name, nw, nd, ow, od, d, lo, hi in rows:
        pn = f'{nw}/{nd} = {nw / nd:.0%}' if nd else 'n/a'
        po = f'{ow}/{od} = {ow / od:.0%}' if od else 'n/a'
        _, clo, chi = newcombe_diff(nw, nd, ow, od, z=zc)
        sig = '  **' if (clo > 0 or chi < 0) else ('  *' if (lo > 0 or hi < 0) else '')
        print(f'{name:<12} {pn:>18} {po:>18} {d:>+9.1%} '
              f'{f"[{lo:+.1%}, {hi:+.1%}]":>20}{sig}')

    tnw = sum(r[1] for r in rows)
    tnd = sum(r[2] for r in rows)
    tow = sum(r[3] for r in rows)
    tod = sum(r[4] for r in rows)
    d, lo, hi = newcombe_diff(tnw, tnd, tow, tod)
    print(f'{"POOLED":<12} {f"{tnw}/{tnd} = {tnw / max(tnd,1):.0%}":>18} '
          f'{f"{tow}/{tod} = {tow / max(tod,1):.0%}":>18} {d:>+9.1%} '
          f'{f"[{lo:+.1%}, {hi:+.1%}]":>20}')

    print(f'\n  *  = per-row 95 % interval excludes 0.  ** = still excludes 0 after a '
          f'Bonferroni\n       correction across the {len(rows)} archetypes. With {len(rows)} rows '
          f'tested at 0.05 each there is a\n       ~{1 - 0.95 ** len(rows):.0%} chance of at '
          f'least one bare `*` even when the nets are identical,\n       so a single starred row '
          f'is a lead to re-run at higher n, not a result.')
    print(f'  A positive difference means the NEW net gets more out of that archetype than '
          f'the old one.\n  A1 predicts a spread across rows; every row at ~0 with tight '
          f'intervals is the clean\n  negative, and every row at ~0 with wide intervals just '
          f'means not enough games.')
    if tnd and tod:
        mde = Z95 * math.sqrt(0.25 / tnd + 0.25 / tod)
        need = math.ceil(2 * 0.25 * (Z95 / 0.05) ** 2)
        print(f'  At n={tnd}+{tod} decided the pooled difference resolves +/- {mde:.1%}; '
              f'a 5 pp edge needs\n  ~{need:,} decided games per arm.')

    if args.out_csv:
        with open(args.out_csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['archetype', 'new_wins', 'new_decided', 'old_wins', 'old_decided',
                        'diff', 'ci_lo', 'ci_hi'])
            w.writerows(rows)
        print(f'\n  wrote {args.out_csv}')


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description='Measure what A1 (unit-type embedding) and A3 (FiLM) actually bought.')
    parser.add_argument('--new', help='policy_factored_v2 checkpoint (default: newest in data/)')
    parser.add_argument('--old', help='policy_factored_v1 checkpoint (default: newest in data/)')
    parser.add_argument('--seed', type=int, default=0)
    sub = parser.add_subparsers(dest='cmd', required=True)

    w = sub.add_parser('weights', help='is FiLM switched on; did the embedding learn')
    w.add_argument('--states', type=int, default=256)
    w.set_defaults(func=cmd_weights)

    h = sub.add_parser('hand', help="A3: does the hand re-rank the board (v1 control = 0)")
    h.add_argument('--states', type=int, default=64)
    h.add_argument('--swaps', type=int, default=12)
    h.set_defaults(func=cmd_hand)

    c = sub.add_parser('comps', help='A1: head-to-head on forced attribute-extreme drafts')
    c.add_argument('--games', type=int, default=600, help='total, split across archetypes')
    c.add_argument('--archetype', default='all', choices=['all'] + list(ARCHETYPES))
    c.add_argument('--max-turns', type=int, default=2000)
    c.add_argument('--out-csv')
    c.set_defaults(func=cmd_comps)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
