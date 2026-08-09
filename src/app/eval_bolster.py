"""Evaluate BolsterBot vs a chosen opponent with a *forced* Berserker/Priest draft.

The bolster archetype is only defined when the bot holds a Berserker and/or a Warrior
Priest, and the env drafts randomly, so this harness pins those units into the bot's
composition via `reset(options={'force_units': ...})` — something the bot can't do for
itself (the draft happens before it acts). Plays `--games` games with balanced colors
(bot as P1 half, P2 half) and reports the bot's win rate plus a verb-usage breakdown, so
"does it actually bolster / use the Berserker chain?" is visible alongside the win rate.

**Drafts are an explicit, dumpable list** (docs/IDEAS.md L5). The gauntlet's antithetic
trick — replay a draft with the seats swapped so each side plays both compositions — does
not apply here: the bot's composition is pinned by construction, so it is the *treatment*,
not a nuisance draw, and averaging over it would destroy the thing being measured. The
control that does apply is common random numbers **across arms**: every arm faces the same
list of drafts. A shared `--seed` almost achieved that already, but only by accident — it
relies on every arm consuming the RNG in exactly the same order, which silently breaks the
moment a bot's constructor or the env's reset changes. Generating the full 4/4 draft up
front and pinning *both* sides makes it explicit and robust, and `--dump-drafts` /
`--drafts` pins it across code versions too.

Colour balancing is kept, though it is close to free: the base layout is exactly
180-degree-rotation symmetric and `set_init_state` draws the initiative owner independently
of player id, so there is no first-player advantage to cancel.

    python src/app/eval_bolster.py --games 60 --opponent lookahead
    python src/app/eval_bolster.py --games 60 --opponent lookahead --key-units 8      # berserker-only
    python src/app/eval_bolster.py --games 60 --opponent greedy_sim
    # identical drafts across two arms, provably:
    python src/app/eval_bolster.py --games 60 --opponent lookahead --dump-drafts d.json
    python src/app/eval_bolster.py --games 60 --opponent policy    --drafts d.json
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np

from src.services.environment.warchest_env import WarChestEnv
from src.services.environment.roster import UNIT_IDS
from src.services.environment.game_state import UNITS_PER_PLAYER
from src.services.bots.bolster_bot import BolsterBot, KEY_UNIT_IDS
import glob
import torch

from src.services.gauntlet import (
    lookahead_agent, greedy_sim_agent, greedy_fast_agent, random_agent, checkpoint_agent,
)


def _build_opponent(name, *, time_budget, policy_path=None):
    if name == 'lookahead':
        return lookahead_agent('lookahead', time_budget=time_budget)
    if name == 'greedy_sim':
        return greedy_sim_agent('greedy_sim')
    if name == 'greedy_fast':
        return greedy_fast_agent('greedy_fast')
    if name == 'random':
        return random_agent('random')
    if name == 'policy':
        path = policy_path or max(glob.glob('data/warchest_ppo_*.pth'))
        agent = checkpoint_agent(path, torch.device('cpu'))
        if agent is None:
            raise SystemExit(f'could not load policy checkpoint {path!r}')
        return agent
    raise SystemExit(f'unknown opponent {name!r}')


def build_draft_list(n_games, key_units, *, seed):
    """`n_games` full drafts as (bot_composition, opp_composition) tuples.

    Both sides are fully determined here rather than left to `set_init_state`'s RNG, so
    the same list can be replayed by any arm regardless of what else consumes randomness
    (see the module docstring). `key_units` are always in the bot's four; the remaining
    slots on both sides are drawn from the rest of the roster, disjoint as the real draft
    requires.
    """
    key = list(dict.fromkeys(int(u) for u in key_units))  # de-dup, keep order
    if len(key) > UNITS_PER_PLAYER:
        raise SystemExit(f'--key-units has {len(key)} units, more than the {UNITS_PER_PLAYER} '
                         f'a player drafts')
    rng = np.random.default_rng(seed)
    pool = [u for u in UNIT_IDS if u not in key]
    drafts = []
    for _ in range(n_games):
        fill = rng.choice(pool, size=2 * UNITS_PER_PLAYER - len(key), replace=False)
        fill = [int(u) for u in fill]
        bot_comp = tuple(key + fill[:UNITS_PER_PLAYER - len(key)])
        opp_comp = tuple(fill[UNITS_PER_PLAYER - len(key):])
        drafts.append((bot_comp, opp_comp))
    return drafts


def play_game(bot, opp, bot_pid, bot_comp, opp_comp, *, seed, max_turns=2000):
    """One game. `bot` plays as `bot_pid`; BOTH compositions are pinned.

    Returns (result, turns) where result is 'win'/'loss'/'draw' from the bot's view.
    Pinning both sides (not just the bot's) is what makes the draft identical across
    arms — otherwise the opponent's four are redrawn from whatever RNG state the arm
    happens to be in.
    """
    env = WarChestEnv(save_game_history=False)
    np.random.seed(seed)
    env.reset(options={'force_units': {bot_pid: list(bot_comp), 3 - bot_pid: list(opp_comp)}})
    agents = {bot_pid: bot, 3 - bot_pid: opp}

    for t in range(max_turns):
        pid = env.active_player
        action = agents[pid].act(env)
        _, _, terminated, truncated, info = env.step(action)
        if not info['action'].is_valid:
            _, _, terminated, truncated, info = env.make_random_step()
        if terminated:
            winner = pid
            return ('win' if winner == bot_pid else 'loss'), t + 1
        if truncated:
            return 'draw', t + 1
    return 'draw', max_turns


def main():
    ap = argparse.ArgumentParser(description='Evaluate BolsterBot vs an opponent (forced draft).')
    ap.add_argument('--games', type=int, default=60)
    ap.add_argument('--opponent', default='lookahead',
                    choices=['lookahead', 'greedy_sim', 'greedy_fast', 'random', 'policy'])
    ap.add_argument('--policy-path', default=None, help='Checkpoint for --opponent policy (default: newest).')
    ap.add_argument('--time-budget', type=float, default=0.1, help='Opponent (lookahead) per-move budget.')
    ap.add_argument('--bot-time-budget', type=float, default=0.1, help='BolsterBot per-move search budget.')
    ap.add_argument('--key-units', type=int, nargs='+', default=list(KEY_UNIT_IDS),
                    help='Unit ids forced into the bot draft (default: Berserker 8 + Warrior Priest 16).')
    ap.add_argument('--build-target', type=int, default=4, help='Berserker stack the controller builds to.')
    ap.add_argument('--max-branching', type=int, default=8)
    ap.add_argument('--pure', action='store_true',
                    help='Disable the archetype controller (pure LookaheadBot + forced draft baseline).')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--draft-seed', type=int, default=None,
                    help='Seed for the draft list (default: --seed). Two arms sharing this '
                         'face identical drafts; see the module docstring.')
    ap.add_argument('--drafts', default=None,
                    help='Replay a draft list dumped by --dump-drafts, instead of generating '
                         'one. Guarantees identical drafts across arms even if the code that '
                         'consumes randomness has changed in between.')
    ap.add_argument('--dump-drafts', default=None,
                    help='Write the generated draft list here as JSON, to replay with --drafts.')
    args = ap.parse_args()

    bot = BolsterBot(build_target=args.build_target, time_budget=args.bot_time_budget,
                     max_branching=args.max_branching, archetype=not args.pure)
    opp = _build_opponent(args.opponent, time_budget=args.time_budget, policy_path=args.policy_path)

    if args.drafts:
        with open(args.drafts) as fh:
            drafts = [(tuple(b), tuple(o)) for b, o in json.load(fh)]
        if len(drafts) < args.games:
            raise SystemExit(f'{args.drafts!r} holds {len(drafts)} drafts, fewer than '
                             f'--games {args.games}')
    else:
        drafts = build_draft_list(
            args.games, args.key_units,
            seed=args.seed if args.draft_seed is None else args.draft_seed)
    if args.dump_drafts:
        with open(args.dump_drafts, 'w') as fh:
            json.dump([[list(b), list(o)] for b, o in drafts], fh)
        print(f'wrote {len(drafts)} drafts to {args.dump_drafts}')

    results = Counter()
    turns = []
    t0 = time.perf_counter()
    for g in range(args.games):
        bot_pid = 1 if g % 2 == 0 else 2   # alternate colors
        bot_comp, opp_comp = drafts[g]
        res, n = play_game(bot, opp, bot_pid, bot_comp, opp_comp, seed=args.seed + g)
        results[res] += 1
        turns.append(n)
        if (g + 1) % 10 == 0:
            wr = results['win'] / (g + 1)
            print(f'  [{g + 1}/{args.games}] WR={wr:.3f}  ({dict(results)})')

    n = args.games
    wr = results['win'] / n
    print(f'\nBolsterBot(key={args.key_units}) vs {args.opponent}: '
          f'WR={wr:.3f}  W={results["win"]} L={results["loss"]} D={results["draw"]}  '
          f'(n={n}, {time.perf_counter() - t0:.1f}s, turns avg={np.mean(turns):.0f})')
    total = sum(bot.usage.values()) or 1
    usage = {k: f'{v / total:.2f}' for k, v in bot.usage.most_common()}
    print(f'bot verb usage: {usage}')


if __name__ == '__main__':
    main()
