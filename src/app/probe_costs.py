"""Reproduce the cost + input-sparsity tables behind `docs/IDEAS.md` § New directions (N.0).

Two questions, both of which decide what is affordable in the rollout hot path and how much
of the network's input surface is doing any work:

  * **Table A — per-decision cost.** What does one env step, one policy forward, one critic
    forward and one move from each bot actually cost? This is the table that says whether a
    proposed opponent can be sampled in training at all: `play_episode` calls the opponent
    once per opponent ply, and `logs/ppo_*.log` shows model inference at ~89 % of rollout
    core-time, so an opponent's per-move cost is very nearly the whole story.
  * **Table B — observation sparsity.** The draft is 4 of 16 unit types per side, disjoint,
    so most per-type planes and per-type global entries are identically zero — and *which*
    ones changes every game. Reported per decision (what one forward pass sees), which is the
    number that matters for the architecture, not per game.

Timings are single-threaded on purpose (`torch.set_num_threads(1)`): that is how the parallel
rollout workers run, and it is the only setting under which the numbers compose with the
per-worker budget in `docs/parallel_rollouts.md`.

    python src/app/probe_costs.py
    python src/app/probe_costs.py --search-games 6 --sparsity-games 20
    python src/app/probe_costs.py --skip-search        # fast: drops the two search-bot rows
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch

torch.set_num_threads(1)  # match the rollout workers; see module docstring

from src.services.environment.warchest_env import WarChestEnv
from src.services.bots import GreedyBot, RandomBot, ThreatAwareGreedyBot
from src.services.policy.policy import Policy, Critic


def _fresh_env():
    return WarChestEnv(save_game_history=False, debug_mode=False)


def _midgame_env(plies=40, seed=1):
    """An env advanced into a real midgame by random legal play."""
    rng = np.random.default_rng(seed)
    env = _fresh_env()
    env.reset()
    for _ in range(plies):
        legal = env.get_possible_actions()
        _, _, terminated, truncated, _ = env.step(legal[rng.integers(len(legal))])
        if terminated or truncated:
            env.reset()
    return env


def _timed(fn, reps):
    """Mean seconds per call over `reps` calls."""
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def _play_game(bot_p1, bot_p2, env_iface, max_t=1000):
    """One game; returns (plies, seconds). `env_iface[i]` says whether that bot takes `env`."""
    env = _fresh_env()
    obs, _ = env.reset()
    plies = 0
    t0 = time.perf_counter()
    for _ in range(max_t):
        pid = env.active_player
        idx = 0 if pid == 1 else 1
        bot = bot_p1 if idx == 0 else bot_p2
        if env_iface[idx]:
            action = bot.act(env)
        else:
            ego_action, _, _ = bot.act(obs)
            action = WarChestEnv.remap_action(ego_action) if pid == 2 else ego_action
        obs, _, terminated, truncated, info = env.step(action)
        if not info['action'].is_valid:
            obs, _, terminated, truncated, info = env.make_random_step()
        plies += 1
        if terminated or truncated:
            break
    return plies, time.perf_counter() - t0


def _per_move_cost(bot, bot_takes_env, games, opp_move_ms, env_ms):
    """Cost of one *own* move, from full games against the cheap obs GreedyBot.

    A single probe state is not representative — `legal_at_root` swings the node count of a
    search bot by an order of magnitude — so this measures whole games and backs the bot's own
    share out of the total. With both sides moving on about half the plies,

        ms_per_ply = 0.5 * cost_bot + 0.5 * cost_opp + cost_env

    so `cost_bot = 2 * (ms_per_ply - cost_env) - cost_opp`. Both subtractions matter: omitting
    `cost_env` inflates a cheap bot's row by the whole env step, and omitting `cost_opp`
    inflates it again by the opponent.
    """
    opp = GreedyBot()
    plies = 0
    secs = 0.0
    for _ in range(games):
        p, s = _play_game(bot, opp, (bot_takes_env, False))
        plies += p
        secs += s
    ms_per_ply = 1000 * secs / plies
    return 2 * (ms_per_ply - env_ms) - opp_move_ms, ms_per_ply, plies / games


def _obs_bot_cost(bot, games, seed=11):
    """Direct per-move cost of an obs-only bot, timed on real decision observations.

    An `act(obs)` bot needs no game-level subtraction — it is a pure function of one
    observation — so timing it on a sample of real states is both simpler and tighter than
    backing it out of a game total.
    """
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(games):
        env = _fresh_env()
        obs, _ = env.reset()
        for _ in range(1000):
            states.append(obs)
            legal = env.get_possible_actions()
            obs, _, terminated, truncated, _ = env.step(legal[rng.integers(len(legal))])
            if terminated or truncated:
                break
    t0 = time.perf_counter()
    for obs in states:
        bot.act(obs)
    return 1000 * (time.perf_counter() - t0) / len(states), len(states)


def table_a(args):
    print('\n=== Table A: per-decision cost (single-threaded) ===')
    env = _midgame_env()
    obs = env._encode_observation() if hasattr(env, '_encode_observation') else None

    rng = np.random.default_rng(0)

    def _step_once():
        legal = env.get_possible_actions()
        _, _, terminated, truncated, _ = env.step(legal[rng.integers(len(legal))])
        if terminated or truncated:
            env.reset()

    step_ms = 1000 * _timed(_step_once, 3000)
    legal_ms = 1000 * _timed(env.get_possible_actions, 20000)
    print(f'env.step + get_possible_actions      {step_ms:8.3f} ms   ({1/step_ms*1000:.0f} steps/s)')
    print(f'get_possible_actions alone           {legal_ms:8.3f} ms')

    from src.services.bots.lookahead_bot import _clone_state
    state = _midgame_env().state
    clone_ms = 1000 * _timed(lambda: _clone_state(state), 20000)
    print(f'_clone_state                         {clone_ms:8.3f} ms')

    probe_env = _fresh_env()
    obs, _ = probe_env.reset()
    for hidden in (64, 128):
        net = Policy(device='cpu', hidden_dim=hidden).eval()
        print(f'Policy.act  hidden_dim={hidden:<4}            '
              f'{1000 * _timed(lambda: net.act(obs), 300):8.3f} ms')

    critic = Critic(device='cpu', hidden_dim=192)
    onehot = torch.zeros(1, Critic.OPP_DIM)
    priv = torch.zeros(1, critic.priv_dim)
    with torch.no_grad():
        single_ms = 1000 * _timed(lambda: critic.value_single(obs, onehot, priv), 300)
        batch = {
            'board': torch.from_numpy(np.stack([obs['board']] * 64)),
            'global': torch.from_numpy(np.stack([obs['global']] * 64)),
            'opp_onehot': torch.zeros(64, Critic.OPP_DIM),
            'privileged': torch.zeros(64, critic.priv_dim),
        }
        batch_ms = 1000 * _timed(lambda: critic.value_batch(batch), 50)
    print(f'Critic.value_single hidden_dim=192   {single_ms:8.3f} ms')
    print(f'Critic.value_batch(64) hidden_dim=192{batch_ms:8.3f} ms -> {batch_ms / 64:.3f} ms/state')

    # Bot rows. The obs-only bots are timed directly; the env-taking bots are backed out of
    # full games, using the obs bots' direct cost as the opponent term.
    random_ms, _ = _obs_bot_cost(RandomBot(), args.fast_games)
    greedy_ms, n_states = _obs_bot_cost(GreedyBot(), args.fast_games)
    threat_ms, _ = _obs_bot_cost(ThreatAwareGreedyBot(), args.fast_games)
    print(f'RandomBot.act(obs)                   {random_ms:8.3f} ms   ({n_states} real states)')
    print(f'GreedyBot.act(obs)                   {greedy_ms:8.3f} ms   ({n_states} real states)')
    print(f'ThreatAwareGreedyBot.act(obs)        {threat_ms:8.3f} ms   ({n_states} real states)')

    from src.services.bots.greedy_sim_bot import SimGreedyBot
    sim_ms, _, _ = _per_move_cost(SimGreedyBot(), True, args.sim_games, greedy_ms, step_ms)
    print(f'SimGreedyBot.act(env)                {sim_ms:8.3f} ms   '
          f'(full-game avg over {args.sim_games} games)')

    if args.skip_search:
        print('LookaheadBot rows skipped (--skip-search)')
        return
    from src.services.bots.lookahead_bot import LookaheadBot
    for budget in (0.02, 0.1):
        bot = LookaheadBot(time_budget=budget)
        move_ms, _, _ = _per_move_cost(bot, True, args.search_games, greedy_ms, step_ms)
        stats = getattr(bot, 'last_stats', {}) or {}
        nodes = stats.get('nodes_visited')
        rate = f', last call {nodes} nodes -> {nodes / max(stats.get("elapsed", budget), 1e-9):.0f} nodes/s' \
            if nodes else ''
        print(f'LookaheadBot.act(env) @{budget:<5}         {move_ms:8.3f} ms   '
              f'(full-game avg over {args.search_games} games{rate})')


def table_b(args):
    """Fraction of the observation that is exactly zero, per decision.

    Two populations are reported because they answer different questions: `dead` is what one
    forward pass sees (the architecture question), while the structural floor — the planes and
    global entries that *cannot* be non-zero because the type was never drafted — is what no
    amount of play will ever fill in.
    """
    print('\n=== Table B: observation sparsity, per decision ===')
    rng = np.random.default_rng(4)
    dead_planes, dead_globals = [], []
    for game in range(args.sparsity_games):
        env = _fresh_env()
        obs, _ = env.reset()
        for _ in range(1000):
            planes = np.abs(obs['board']).reshape(obs['board'].shape[0], -1).max(axis=1)
            dead_planes.append(int((planes <= 1e-9).sum()))
            dead_globals.append(int((np.abs(obs['global']) <= 1e-9).sum()))
            legal = env.get_possible_actions()
            obs, _, terminated, truncated, _ = env.step(legal[rng.integers(len(legal))])
            if terminated or truncated:
                break
    n_planes = env.observation_space['board'].shape[0]
    n_globals = env.observation_space['global'].shape[0]
    mp, mg = float(np.mean(dead_planes)), float(np.mean(dead_globals))
    surface = n_planes * 49 + n_globals
    print(f'decisions sampled                    {len(dead_planes)}')
    print(f'board planes exactly zero            {mp:6.1f} / {n_planes}   ({100 * mp / n_planes:.0f} %)')
    print(f'global dims exactly zero             {mg:6.1f} / {n_globals}   ({100 * mg / n_globals:.0f} %)')
    print(f'whole input surface exactly zero     {100 * (mp * 49 + mg) / surface:6.0f} %   '
          f'({surface} slots)')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fast-games', type=int, default=20,
                        help='games for the obs-only GreedyBot baseline row')
    parser.add_argument('--sim-games', type=int, default=6, help='games for the SimGreedyBot row')
    parser.add_argument('--search-games', type=int, default=4,
                        help='games per LookaheadBot budget (each is slow)')
    parser.add_argument('--sparsity-games', type=int, default=12, help='games for Table B')
    parser.add_argument('--skip-search', action='store_true',
                        help='drop the two LookaheadBot rows (much faster)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    table_a(args)
    table_b(args)
    print('\nRecorded in docs/IDEAS.md § New directions, N.0 (Tables A and B).')


if __name__ == '__main__':
    main()
