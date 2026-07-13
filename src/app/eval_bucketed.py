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
from collections import Counter, defaultdict

import numpy as np
import torch
from rich.progress import track

from src.services.policy.policy import Policy
from src.services.environment.warchest_env import (
    WarChestEnv, VERB_OF_ACTION, V_TACTIC, V_BOLSTER, V_CLAIM, V_ATTACK, V_CONTROL,
    DECLINE_ACTION_ID,
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
    opp_pid = 3 - main_pid
    used_tactic = False
    # docs/IDEAS.md #9: base-lead (my_bases - opp_bases) at the moment each tactic is
    # initiated, to disambiguate reverse-causation (tactics reached for only when
    # already behind) from an execution/exploration gap (tactics attempted from a lead
    # too, just executed poorly).
    tactic_base_leads = []
    bolster_count = 0
    bolsters = Counter()             # bolstered unit id -> times its stack was grown
    bolsters_with_supply = Counter()  # ... restricted to bolsters made while >=1 coin
                                      # of that unit still sat in the player's supply
    supply_left_sum = Counter()       # sum of coins-left-in-supply over those bolsters
    bolsters_fully_available = Counter()  # ... further restricted to: board + supply
                                           # == the unit's total roster coin count, i.e.
                                           # none of that type is sitting in hand/bag/
                                           # discard and none has been lost to the box
    chain_offered = 0
    chain_used = 0
    # Initiative can only be claimed by the round's non-owner (the player who is
    # about to play the round's *last* coin under normal symmetric hands), so a
    # successful claim tends to hand that same player the next round's *first*
    # coin too — two consecutive actions with no opposing move in between. We
    # track: how often main_pid claims (`initiative_claims`), how often that
    # claim actually materializes as them opening the next round (`double_turns`,
    # detected via the env.state.round_number step-over-step boundary), and how
    # often that opening move is a productive one — attack or claim_base.
    initiative_claims = 0
    double_turns = 0
    productive_double_turns = 0
    claimed_initiative_this_round = False
    expect_double_turn_check = False
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

        if expect_double_turn_check:
            if acting_pid == main_pid:
                double_turns += 1
                if VERB_OF_ACTION[env_action] in (V_ATTACK, V_CONTROL):
                    productive_double_turns += 1
            expect_double_turn_check = False

        if acting_pid == main_pid and VERB_OF_ACTION[env_action] == V_TACTIC:
            used_tactic = True
            lead_now = (len(env.board.get_controlled_bases(main_pid))
                        - len(env.board.get_controlled_bases(opp_pid)))
            tactic_base_leads.append(lead_now)
        if acting_pid == main_pid and VERB_OF_ACTION[env_action] == V_BOLSTER:
            bolster_count += 1
            # Bolster is a spatial action: the bolstered unit is the one on the target
            # cell. Decode the (absolute-frame) env_action and read the unit there
            # before stepping (bolster does not touch supply, so the count is the same
            # before/after). We flag bolsters made while coins of that same unit still
            # sit in the player's supply — e.g. a Knight (4 total) bolstered to stack 2
            # with 2 still in supply, a Swordsman (5 total) with 3 still in supply.
            _, br, bq = WarChestEnv.decode_action(env_action)
            bunit = env.board.get_unit_at(br, bq)
            if bunit is not None:
                bolsters[bunit.id] += 1
                supply_left = env.state.supply[main_pid].get(bunit.id, 0)
                if supply_left > 0:
                    bolsters_with_supply[bunit.id] += 1
                    supply_left_sum[bunit.id] += supply_left
                    # Stricter still: board + supply == the type's full roster count.
                    # +1 because this check runs pre-step — the hand coin about to
                    # join bunit.stack isn't reflected in bunit.stack yet, but it is
                    # the one being committed, so it counts toward "board" here.
                    board_total = sum(
                        u.stack for u in env.board.units
                        if u.player_id == main_pid and u.id == bunit.id)
                    if board_total + 1 + supply_left == UNIT_BY_ID[bunit.id].total_coins:
                        bolsters_fully_available[bunit.id] += 1
        round_before = env.state.round_number
        state, _, terminated, truncated, step_info = env.step(env_action)
        action_valid = step_info['action'].is_valid
        if not action_valid:
            state, _, terminated, truncated, step_info = env.make_random_step()

        if acting_pid == main_pid and action_valid and VERB_OF_ACTION[env_action] == V_CLAIM:
            claimed_initiative_this_round = True
            initiative_claims += 1
        if env.state.round_number > round_before:
            if claimed_initiative_this_round:
                expect_double_turn_check = True
            claimed_initiative_this_round = False

        if terminated or truncated:
            break

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
        'initiative_claims': initiative_claims,
        'double_turns': double_turns,
        'productive_double_turns': productive_double_turns,
        'my_bases': my_bases,
        'opp_bases': opp_bases,
        'used_tactic': used_tactic,
        'tactic_base_leads': tactic_base_leads,  # base-lead at each tactic initiation
        'bolster_count': bolster_count,
        'bolsters': bolsters,                          # Counter: unit id -> bolsters
        'bolsters_with_supply': bolsters_with_supply,  # ... while supply still had coins
        'supply_left_sum': supply_left_sum,            # sum of supply-left over those
        'bolsters_fully_available': bolsters_fully_available,  # ... and board+supply==total
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


def _print_initiative_breakdown(records):
    """How often main_pid claims initiative (`claim_initiative`), and — the metric
    of interest — how often that claim actually buys a double turn (claiming is
    only legal for the round's non-owner, who under normal symmetric hands also
    plays the round's last coin, so a successful claim tends to hand them the next
    round's first coin too, i.e. two consecutive actions with no opposing move in
    between). Of those double turns, we further split out how many were spent
    productively — the opening move of the new round was an attack or claim_base —
    versus 'wasted' on a quieter move/deploy/bolster/recruit/pass."""
    n = len(records)
    total_claims = sum(r['initiative_claims'] for r in records)
    total_double_turns = sum(r['double_turns'] for r in records)
    total_productive = sum(r['productive_double_turns'] for r in records)
    claim_games = [r for r in records if r['initiative_claims'] > 0]
    print(f'\n=== Initiative seizing (claim_initiative -> double turn across the round boundary) ===')
    print(f'Claimed initiative at least once: {len(claim_games)}/{n} games '
          f'({len(claim_games) / n:.1%}), {total_claims} claims total ({total_claims / n:.2f} per game)')
    if total_claims == 0:
        print('  (policy never claims initiative)')
        return
    dt_share = total_double_turns / total_claims
    print(f'  Claim materialized into a double turn (also opened next round): '
          f'{total_double_turns}/{total_claims} ({dt_share:.1%})')
    if total_double_turns > 0:
        prod_share = total_productive / total_double_turns
        print(f'  Of those double turns, opening move was attack/claim_base: '
              f'{total_productive}/{total_double_turns} ({prod_share:.1%})')

    wr_claim, se_claim = _wr_and_se(sum(r['outcome'] == 'win' for r in claim_games), len(claim_games))
    no_claim = [r for r in records if r['initiative_claims'] == 0]
    wr_no, se_no = _wr_and_se(sum(r['outcome'] == 'win' for r in no_claim), len(no_claim))
    print(f'  WR when claimed initiative >=1: {wr_claim:.3f} ± {se_claim:.3f} (n={len(claim_games)})')
    print(f'  WR when never claimed:          {wr_no:.3f} ± {se_no:.3f} (n={len(no_claim)})')

    productive_games = [r for r in records if r['productive_double_turns'] > 0]
    non_productive_claim_games = [r for r in claim_games if r['productive_double_turns'] == 0]
    if productive_games and non_productive_claim_games:
        wr_prod, se_prod = _wr_and_se(sum(r['outcome'] == 'win' for r in productive_games), len(productive_games))
        wr_nonprod, se_nonprod = _wr_and_se(
            sum(r['outcome'] == 'win' for r in non_productive_claim_games), len(non_productive_claim_games))
        print(f'  WR when a double turn opened with attack/claim_base: '
              f'{wr_prod:.3f} ± {se_prod:.3f} (n={len(productive_games)})')
        print(f'  WR when claimed but never landed a productive double turn: '
              f'{wr_nonprod:.3f} ± {se_nonprod:.3f} (n={len(non_productive_claim_games)})')


def _print_tactic_lead_breakdown(records):
    """docs/IDEAS.md #9: disambiguate whether tactics correlate with losing because
    they're reached for only when already behind (reverse causation — no fix needed)
    or attempted from a lead too but executed poorly (an execution/exploration gap —
    docs/IDEAS.md #8 is on target). Buckets every tactic-initiation instance by
    base-lead (my_bases - opp_bases) at the moment it was initiated."""
    leads = [lead for r in records for lead in r['tactic_base_leads']]
    n = len(leads)
    print(f'\n=== Tactic-initiation base-lead (docs/IDEAS.md #9) ===')
    if n == 0:
        print('  (tactic never initiated — nothing to disambiguate)')
        return
    behind = sum(1 for l in leads if l < 0)
    even = sum(1 for l in leads if l == 0)
    ahead = sum(1 for l in leads if l > 0)
    games_with_tactic = sum(1 for r in records if r['tactic_base_leads'])
    print(f'{n} tactic initiations across {games_with_tactic} games')
    print(f'  behind (lead<0): {behind}/{n} ({behind / n:.1%})')
    print(f'  even   (lead=0): {even}/{n} ({even / n:.1%})')
    print(f'  ahead  (lead>0): {ahead}/{n} ({ahead / n:.1%})')
    print(f'  mean lead at initiation: {sum(leads) / n:+.2f}')
    if behind / n > 0.7:
        print('  -> clusters when already behind: consistent with reverse causation, #8 not needed')
    elif ahead / n > 0.3:
        print('  -> spread into lead states too: consistent with an execution/exploration gap, see #8')
    else:
        print('  -> mixed; inspect the distribution above before concluding either way')


def _print_bolster_breakdown(records):
    """How often the policy bolsters (plays a hand coin onto an on-board unit, growing
    its stack) and — the metric of interest — how often it does so while coins of that
    same unit still sit in its supply (e.g. a Knight bolstered with 2 still in supply,
    a Swordsman with 3). Bolstering commits a coin to the board while the unit is not
    yet fully drawn out of supply, so this flags stack investment made 'early'."""
    n = len(records)
    bolster_games = [r for r in records if r['bolster_count'] > 0]
    total_bolsters = sum(r['bolster_count'] for r in records)
    total_with_supply = sum(sum(r['bolsters_with_supply'].values()) for r in records)
    total_fully_available = sum(sum(r['bolsters_fully_available'].values()) for r in records)
    print(f'\n=== Bolstering (and bolsters made while supply still holds that unit) ===')
    print(f'Bolstered at least once: {len(bolster_games)}/{n} games '
          f'({len(bolster_games) / n:.1%}), {total_bolsters} bolster actions total '
          f'({total_bolsters / n:.2f} per game)')
    if total_bolsters == 0:
        print('  (policy never bolsters)')
        return
    share = total_with_supply / total_bolsters
    print(f'Of those, {total_with_supply}/{total_bolsters} ({share:.1%}) were made while '
          f'>=1 coin of the same unit was still in supply.')
    share_full = total_fully_available / total_bolsters
    print(f'  ...and {total_fully_available}/{total_bolsters} ({share_full:.1%}) were made while '
          f'board + supply == the unit\'s full roster count (none in hand/bag/discard, none boxed).')

    wr_b, se_b = _wr_and_se(sum(r['outcome'] == 'win' for r in bolster_games), len(bolster_games))
    no_bolster = [r for r in records if r['bolster_count'] == 0]
    wr_no, se_no = _wr_and_se(sum(r['outcome'] == 'win' for r in no_bolster), len(no_bolster))
    print(f'  WR when bolstered >=1:  {wr_b:.3f} ± {se_b:.3f} (n={len(bolster_games)})')
    print(f'  WR when never bolstered:{wr_no:.3f} ± {se_no:.3f} (n={len(no_bolster)})')

    # Per unit: total bolsters, how many while supply still had coins, and the average
    # number of coins that were left in supply at those moments. Sorted by total desc.
    per_unit_total = Counter()
    per_unit_supply = Counter()
    per_unit_supply_sum = Counter()
    per_unit_fully_available = Counter()
    for r in records:
        per_unit_total.update(r['bolsters'])
        per_unit_supply.update(r['bolsters_with_supply'])
        per_unit_supply_sum.update(r['supply_left_sum'])
        per_unit_fully_available.update(r['bolsters_fully_available'])
    print(f'\n  {"unit":<16} {"bolsters":>9} {"w/ supply left":>15} '
          f'{"avg supply left":>16} {"fully available":>16}')
    for uid, tot in per_unit_total.most_common():
        name = UNIT_BY_ID[uid].name
        with_sup = per_unit_supply[uid]
        avg = per_unit_supply_sum[uid] / with_sup if with_sup else float('nan')
        avg_str = f'{avg:.2f}' if avg == avg else '—'
        full_avail = per_unit_fully_available[uid]
        print(f'  {name:<16} {tot:>9} {with_sup:>15} {avg_str:>16} {full_avail:>16}')


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
    for _ in track(range(args.games), description='games'):
        main_pid = int(np.random.choice([1, 2]))
        records.append(play_one_game(env, policy, bot, main_pid, args.max_t))

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

    _print_tactic_lead_breakdown(decisive)
    _print_bolster_breakdown(decisive)
    _print_initiative_breakdown(decisive)

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
                row['tactic_base_leads'] = ','.join(str(l) for l in row['tactic_base_leads'])
                for col in ('bolsters', 'bolsters_with_supply', 'supply_left_sum',
                            'bolsters_fully_available'):
                    row[col] = ','.join(
                        f'{UNIT_BY_ID[uid].name}:{cnt}' for uid, cnt in row[col].items())
                writer.writerow(row)
        print(f'\nWrote {len(records)} game records to {args.out_csv}')


if __name__ == '__main__':
    main()
