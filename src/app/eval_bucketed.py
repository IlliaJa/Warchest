"""Bucketed evaluation + loss autopsy vs GreedyBot — docs/future_steps.md Step 0.

The training-loop eval (`PPOTrainer._maybe_eval`) reports one aggregate win
rate over 20 games. That hides the question that actually matters before any
further reward/capacity work: is the ~75% ceiling driven by a handful of
unwinnable random drafts, or by a real, fixable skill gap spread evenly
across compositions? This script runs many games against GreedyBot, breaks
the result down by which unit types were drafted on each side, and dumps a
full record of every loss (final base score, both compositions, whether the
policy ever initiated a tactic) so the two cases can be told apart.

Usage:
    python -m src.app.eval_bucketed --games 200
    python -m src.app.eval_bucketed --model-path data/warchest_ppo_20260702-1442.pth --games 500
"""
import argparse
import csv
import glob
import time
from collections import defaultdict

import numpy as np
import torch

from src.services.policy.policy import Policy
from src.services.environment.warchest_env import (
    WarChestEnv, VERB_OF_ACTION, V_TACTIC, V_BOLSTER, DECLINE_ACTION_ID,
)
from src.services.environment.roster import UNIT_BY_ID
from src.services.bots.greedy_bot import GreedyBot

# The only unit whose ability triggers via the 'extra_maneuver' pending state
# (chain maneuvers by paying its own stack) rather than TACTIC_VERB — so
# `used_tactic` below is structurally blind to it; tracked separately.
_STACK_CHAIN_UNIT_IDS = {u.id for u in UNIT_BY_ID.values() if u.extra_maneuvers_from_stack}


def _find_latest_model() -> str:
    candidates = sorted(glob.glob('data/warchest_ppo_*.pth'))
    if not candidates:
        raise FileNotFoundError('No models found in data/warchest_ppo_*.pth')
    return candidates[-1]


def _unit_names(ids):
    return tuple(sorted(UNIT_BY_ID[i].name for i in ids))


def play_one_game(env, policy, bot, main_pid, max_t):
    """Play one policy-vs-GreedyBot game; return a per-game diagnostic record."""
    state, _ = env.reset()
    used_tactic = False
    bolster_count = 0
    chain_offered = 0
    chain_used = 0
    acting_pid = main_pid
    terminated = truncated = False
    for _ in range(max_t):
        acting_pid = env.active_player
        pending = env.state.pending
        is_chain_offer = (
            acting_pid == main_pid and pending is not None
            and pending.kind == 'extra_maneuver'
        )
        with torch.no_grad():
            if acting_pid == main_pid:
                action, _, _ = policy.act(state)
            else:
                action, _, _ = bot.act(state)
        if is_chain_offer:
            chain_offered += 1
            chain_used += int(action != DECLINE_ACTION_ID)
        env_action = WarChestEnv.remap_action(action) if acting_pid == 2 else action
        if acting_pid == main_pid and VERB_OF_ACTION[env_action] == V_TACTIC:
            used_tactic = True
        if acting_pid == main_pid and VERB_OF_ACTION[env_action] == V_BOLSTER:
            bolster_count += 1
        state, _, terminated, truncated, step_info = env.step(env_action)
        if not step_info['action'].is_valid:
            state, _, terminated, truncated, step_info = env.make_random_step()
        if terminated or truncated:
            break

    opp_pid = 3 - main_pid
    my_bases = len(env.board.get_controlled_bases(main_pid))
    opp_bases = len(env.board.get_controlled_bases(opp_pid))
    if terminated:
        outcome = 'win' if acting_pid == main_pid else 'lose'
    else:
        outcome = 'truncated'
    return {
        'outcome': outcome,
        'my_composition': _unit_names(env.state.compositions[main_pid]),
        'opp_composition': _unit_names(env.state.compositions[opp_pid]),
        'has_stack_chain_unit': bool(_STACK_CHAIN_UNIT_IDS & set(env.state.compositions[main_pid])),
        'chain_offered': chain_offered,
        'chain_used': chain_used,
        'my_bases': my_bases,
        'opp_bases': opp_bases,
        'used_tactic': used_tactic,
        'bolster_count': bolster_count,
        'turns': env.action_count,
        'main_pid': main_pid,
    }


def _wr_and_se(wins, n):
    if n == 0:
        return float('nan'), float('nan')
    p = wins / n
    se = (p * (1 - p) / n) ** 0.5
    return p, se


def _print_unit_breakdown(records, title, composition_key):
    """WR when a unit type is / isn't present in `composition_key` ('my_composition'
    or 'opp_composition'), per type, sorted by the size of the WR swing."""
    print(f'\n=== {title} ===')
    rows = []
    for unit in UNIT_BY_ID.values():
        with_wins = with_n = without_wins = without_n = 0
        for r in records:
            if r['outcome'] == 'truncated':
                continue
            present = unit.name in r[composition_key]
            is_win = r['outcome'] == 'win'
            if present:
                with_n += 1
                with_wins += is_win
            else:
                without_n += 1
                without_wins += is_win
        wr_with, se_with = _wr_and_se(with_wins, with_n)
        wr_without, se_without = _wr_and_se(without_wins, without_n)
        swing = wr_with - wr_without if with_n and without_n else float('nan')
        rows.append((unit.name, wr_with, se_with, with_n, wr_without, se_without, without_n, swing))
    rows.sort(key=lambda row: (row[-1] if row[-1] == row[-1] else 0.0))  # NaN-safe sort
    print(f'{"unit":<16} {"WR w/ (n)":<16} {"WR w/o (n)":<16} {"swing":>7}')
    for name, wr_w, se_w, n_w, wr_wo, se_wo, n_wo, swing in rows:
        w_str = f'{wr_w:.2f}±{se_w:.2f}({n_w})' if n_w else 'n/a'
        wo_str = f'{wr_wo:.2f}±{se_wo:.2f}({n_wo})' if n_wo else 'n/a'
        swing_str = f'{swing:+.2f}' if swing == swing else 'n/a'
        print(f'{name:<16} {w_str:<16} {wo_str:<16} {swing_str:>7}')


def _print_exact_composition_breakdown(records, min_games=3):
    counts = defaultdict(lambda: [0, 0])  # composition -> [wins, games (excl. truncated)]
    for r in records:
        if r['outcome'] == 'truncated':
            continue
        c = counts[r['my_composition']]
        c[1] += 1
        c[0] += r['outcome'] == 'win'
    repeated = {k: v for k, v in counts.items() if v[1] >= min_games}
    print(f'\n=== Exact-composition WR (compositions seen >= {min_games} times) ===')
    if not repeated:
        print(f'(none — with a random 4-of-16 draft, {len(counts)} distinct compositions '
              f'appeared across {sum(v[1] for v in counts.values())} decisive games; '
              f'increase --games a lot to get repeats, or add a --force-composition option)')
        return
    for comp, (wins, n) in sorted(repeated.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        wr, se = _wr_and_se(wins, n)
        print(f'{comp}: {wr:.2f}±{se:.2f} ({wins}/{n})')


def main():
    parser = argparse.ArgumentParser(
        description='Bucketed evaluation + loss autopsy vs GreedyBot (future_steps.md Step 0).')
    parser.add_argument('--model-path', type=str, default=None,
                         help='Path to .pth file. Defaults to the latest data/warchest_ppo_*.pth.')
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--max-t', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--out-csv', type=str, default=None,
                         help='Optional path to dump every game record as CSV.')
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    model_path = args.model_path or _find_latest_model()
    print(f'Loading model: {model_path}')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    policy = Policy(device=device, hidden_dim=args.hidden_dim).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    bot = GreedyBot()
    env = WarChestEnv(save_game_history=False)

    records = []
    t0 = time.time()
    for i in range(args.games):
        main_pid = int(np.random.choice([1, 2]))
        records.append(play_one_game(env, policy, bot, main_pid, args.max_t))
        if (i + 1) % max(1, args.games // 10) == 0:
            print(f'  {i + 1}/{args.games} games played ({time.time() - t0:.0f}s elapsed)')

    decisive = [r for r in records if r['outcome'] != 'truncated']
    wins = sum(r['outcome'] == 'win' for r in decisive)
    truncated_n = len(records) - len(decisive)
    wr, se = _wr_and_se(wins, len(decisive))

    print(f'\n=== Overall ===')
    print(f'games={len(records)}  decisive={len(decisive)}  truncated={truncated_n}')
    print(f'WR vs greedy = {wr:.3f} ± {se:.3f} (95% CI ≈ ±{1.96 * se:.3f})')

    tactic_games = [r for r in decisive if r['used_tactic']]
    no_tactic_games = [r for r in decisive if not r['used_tactic']]
    wr_tactic, se_tactic = _wr_and_se(sum(r['outcome'] == 'win' for r in tactic_games), len(tactic_games))
    wr_no_tactic, se_no_tactic = _wr_and_se(
        sum(r['outcome'] == 'win' for r in no_tactic_games), len(no_tactic_games))
    print(f'\nUsed a tactic at least once: {len(tactic_games)}/{len(decisive)} games '
          f'({len(tactic_games) / len(decisive):.1%})')
    print(f'  WR when tactic used:     {wr_tactic:.3f} ± {se_tactic:.3f} (n={len(tactic_games)})')
    print(f'  WR when tactic NOT used: {wr_no_tactic:.3f} ± {se_no_tactic:.3f} (n={len(no_tactic_games)})')

    bolster_games = [r for r in decisive if r['bolster_count'] > 0]
    total_bolsters = sum(r['bolster_count'] for r in decisive)
    print(f'\nBolster used at least once: {len(bolster_games)}/{len(decisive)} games '
          f'({len(bolster_games) / len(decisive):.1%}), {total_bolsters} bolster actions total '
          f'({total_bolsters / len(decisive):.2f} per game)')

    # Stack-chain units (Berserker: extra_maneuvers_from_stack) never touch V_TACTIC —
    # their ability triggers via the 'extra_maneuver' pending state instead, so this
    # is tracked and reported separately from used_tactic above.
    has_chain_unit = [r for r in decisive if r['has_stack_chain_unit']]
    never_offered = [r for r in has_chain_unit if r['chain_offered'] == 0]
    offered = [r for r in has_chain_unit if r['chain_offered'] > 0]
    chained = [r for r in offered if r['chain_used'] > 0]
    offered_but_declined = [r for r in offered if r['chain_used'] == 0]
    total_offers = sum(r['chain_offered'] for r in has_chain_unit)
    total_accepts = sum(r['chain_used'] for r in has_chain_unit)
    print(f'\nDrafted a stack-chain unit (Berserker): {len(has_chain_unit)}/{len(decisive)} games')
    if has_chain_unit:
        print(f'  Never reached stack>=2 (chain never offered): {len(never_offered)}/{len(has_chain_unit)}')
        print(f'  Chain offered at least once:                  {len(offered)}/{len(has_chain_unit)} '
              f'({total_accepts}/{total_offers} individual offers accepted)')
        wr_never, se_never = _wr_and_se(sum(r['outcome'] == 'win' for r in never_offered), len(never_offered))
        wr_chained, se_chained = _wr_and_se(sum(r['outcome'] == 'win' for r in chained), len(chained))
        wr_declined, se_declined = _wr_and_se(
            sum(r['outcome'] == 'win' for r in offered_but_declined), len(offered_but_declined))
        print(f'    WR | never got to stack>=2:        {wr_never:.3f} ± {se_never:.3f} (n={len(never_offered)})')
        print(f'    WR | offered, always declined:     {wr_declined:.3f} ± {se_declined:.3f} (n={len(offered_but_declined)})')
        print(f'    WR | offered and chained >=1 time: {wr_chained:.3f} ± {se_chained:.3f} (n={len(chained)})')

    _print_unit_breakdown(decisive, 'WR by unit type in the POLICY\'s own composition', 'my_composition')
    _print_unit_breakdown(decisive, 'WR by unit type in the OPPONENT\'s composition', 'opp_composition')
    _print_exact_composition_breakdown(decisive)

    losses = [r for r in decisive if r['outcome'] == 'lose']
    print(f'\n=== Loss autopsy ({len(losses)} losses) ===')
    print(f'{"my_composition":<45} {"opp_composition":<45} {"score":<7} {"tactic":<7} {"turns":<6}')
    for r in losses:
        score = f'{r["my_bases"]}-{r["opp_bases"]}'
        print(f'{",".join(r["my_composition"]):<45} {",".join(r["opp_composition"]):<45} '
              f'{score:<7} {str(r["used_tactic"]):<7} {r["turns"]:<6}')

    if args.out_csv:
        with open(args.out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            for r in records:
                row = dict(r)
                row['my_composition'] = ','.join(row['my_composition'])
                row['opp_composition'] = ','.join(row['opp_composition'])
                writer.writerow(row)
        print(f'\nWrote {len(records)} game records to {args.out_csv}')


if __name__ == '__main__':
    main()
