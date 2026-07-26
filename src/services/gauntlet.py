"""Round-robin gauntlet (docs/next_steps.md Step 1).

A fixed set of agents plays all-pairs, K games per pair with alternating colors,
producing a pairwise win-rate matrix, a Bradley-Terry rating (Elo-scaled) anchored
to the fixed field, and a transitivity metric to detect rock-paper-scissors cycles.

The one stable contract every entrant implements:

    GauntletAgent.act(env) -> action id in the ABSOLUTE (unrotated) env frame.

Each agent encodes the shared authoritative game state with its own (versioned)
obs encoder and un-rotates its choice internally, so agents built for different obs
versions can play a single game through one WarChestEnv. Reconstructing a matching
(encoder + arch + weights) from a checkpoint is the caller's job (see
policy.checkpoint + obs_encoders); the gauntlet only needs callable agents.
"""
import itertools
import os

import numpy as np

from .environment.warchest_env import WarChestEnv
from .environment.obs_encoders import get_encoder, latest_encoder
from .bots.greedy_bot import GreedyBot
from .bots.greedy_sim_bot import SimGreedyBot
from .bots.random_bot import RandomBot
from .bots.lookahead_bot import LookaheadBot
from .bots.lookahead_critic_bot import LookaheadCriticBot
from .bots.policy_critic_bot import PolicyCriticBot
from .bots.round_critic_bot import RoundCriticBot
from .bots.puct_bot import PuctBot
from .policy.checkpoint import load_policy_checkpoint
from .policy.policy import Policy


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #
class GauntletAgent:
    """Uniform gauntlet interface. `act(env)` returns an absolute env action id."""

    def __init__(self, name):
        self.name = name

    def act(self, env):
        raise NotImplementedError


class PolicyAgent(GauntletAgent):
    """A trained policy paired with the obs encoder it was trained under."""

    def __init__(self, name, policy, encoder):
        super().__init__(name)
        self.policy = policy
        self.encoder = encoder

    def act(self, env):
        obs = self.encoder.encode(env)
        action, _, _ = self.policy.act(obs)
        # All registered encoders share the stable action space, so the current
        # remap is correct; a future action-space era would carry its own.
        return WarChestEnv.remap_action(action) if env.active_player == 2 else action


class HeuristicAgent(GauntletAgent):
    """Wraps a Bot (GreedyBot / RandomBot) that reads an ego-centric obs + mask."""

    def __init__(self, name, bot, encoder=None):
        super().__init__(name)
        self.bot = bot
        self.encoder = encoder or latest_encoder()

    def act(self, env):
        obs = self.encoder.encode(env)
        action, _, _ = self.bot.act(obs)
        return WarChestEnv.remap_action(action) if env.active_player == 2 else action


def greedy_sim_agent(name='greedy_sim', encoder=None):
    """The 1-ply forward-simulation greedy (`SimGreedyBot`) — the strong greedy
    yardstick. Speaks `act(env)` + `.name` directly (like LookaheadBot), so it
    drops in without a HeuristicAgent obs-encoding wrapper. `encoder` is accepted
    for a uniform factory signature but unused (this bot reads the env, not obs).
    """
    return SimGreedyBot(name=name)


def greedy_fast_agent(name='greedy_fast', encoder=None):
    """The legacy obs-only `GreedyBot` — cheap, hand-blind, no simulation. Kept as
    a separate entrant (and as the training-loop opponent, via opponent_pool) now
    that `greedy_sim` is the heavier simulation bot.
    """
    return HeuristicAgent(name, GreedyBot(), encoder)


def random_agent(name='random', encoder=None):
    return HeuristicAgent(name, RandomBot(), encoder)


def lookahead_agent(name='lookahead', **kwargs):
    """LookaheadBot already speaks `act(env)` + `.name` (docs/lookahead_bot_plan.md),
    so it drops into the gauntlet directly — no HeuristicAgent/obs-encoding wrapper.
    """
    return LookaheadBot(name=name, **kwargs)


def lookahead_critic_agent(name='lookahead_critic', **kwargs):
    """LookaheadCriticBot, same `act(env)` contract as LookaheadBot — drops in directly."""
    return LookaheadCriticBot(name=name, **kwargs)


def policy_critic_agent(name='policy_critic', **kwargs):
    """PolicyCriticBot (policy-prior candidates + critic scoring), same `act(env)`
    contract as LookaheadCriticBot — drops in directly.
    """
    return PolicyCriticBot(name=name, **kwargs)


def round_critic_agent(name='round_critic', **kwargs):
    """RoundCriticBot (PolicyCriticBot that searches to the end of the current
    round), same `act(env)` contract — drops in directly.
    """
    return RoundCriticBot(name=name, **kwargs)


def puct_agent(name='puct', **kwargs):
    """PuctBot (full PUCT/MCTS: policy priors + critic value over a visit-counted
    tree), same `act(env)` contract as the other search bots — drops in directly.
    """
    return PuctBot(name=name, **kwargs)


def checkpoint_agent(path, device):
    """Build a PolicyAgent from a checkpoint, or None if it can't be reconstructed.

    A bare legacy checkpoint from an older architecture / obs version (different
    layer names or dims) is not loadable into the current code — resurrecting those
    is the subprocess/worktree path (docs/next_steps.md), out of scope here. We
    skip them with a warning so the gauntlet still runs on the loadable field.
    """
    try:
        meta = load_policy_checkpoint(path, map_location=device)
        encoder = get_encoder(meta['obs_version'])
        policy = Policy(device=device, hidden_dim=meta['hidden_dim'], obs_encoder=encoder).to(device)
        policy.load_state_dict(meta['state_dict'])
    except Exception as e:  # unreadable file, or incompatible arch/obs/dims
        reason = str(e).splitlines()[0] if str(e).strip() else type(e).__name__
        print(f'  ! skipping {os.path.basename(path)}: {reason}')
        return None
    policy.eval()
    name = os.path.splitext(os.path.basename(path))[0].replace('warchest_ppo_', 'ckpt_')
    return PolicyAgent(f'{name}[v{meta["obs_version"]}]', policy, encoder)


def build_agent(spec, *, device):
    """Rebuild a `GauntletAgent` from a small picklable `spec` dict.

    Needed because live `LookaheadBot`/`LookaheadCriticBot` instances are
    unconditionally unpicklable (both monkeypatch `_sim_env._draw_one` with a
    bound method whose `__name__` doesn't match the attribute it's stored under,
    which breaks pickle's bound-method reduction) — so parallel gauntlet workers
    rebuild every agent from a spec rather than receiving live objects, uniformly
    across all agent kinds (see `gauntlet_parallel.py`).
    """
    kind = spec['kind']
    if kind == 'greedy_sim':
        return greedy_sim_agent(spec['name'])
    if kind == 'greedy_fast':
        return greedy_fast_agent(spec['name'])
    if kind == 'random':
        return random_agent(spec['name'])
    if kind == 'lookahead':
        return lookahead_agent(spec['name'], **spec.get('kwargs', {}))
    if kind == 'lookahead_critic':
        return lookahead_critic_agent(spec['name'], device=device, **spec.get('kwargs', {}))
    if kind == 'policy_critic':
        return policy_critic_agent(spec['name'], device=device, **spec.get('kwargs', {}))
    if kind == 'round_critic':
        return round_critic_agent(spec['name'], device=device, **spec.get('kwargs', {}))
    if kind == 'puct':
        return puct_agent(spec['name'], device=device, **spec.get('kwargs', {}))
    if kind == 'policy':
        agent = checkpoint_agent(spec['path'], device)
        if agent is None:
            raise ValueError(f"could not build policy agent from checkpoint {spec['path']!r}")
        return agent
    raise ValueError(f'unknown agent spec kind: {kind!r}')


# --------------------------------------------------------------------------- #
# Game driver
# --------------------------------------------------------------------------- #
def play_game(agent_p1, agent_p2, *, seed=None, max_turns=2000):
    """Play one game with agent_p1 as player 1, agent_p2 as player 2.

    Returns 1 (P1 wins), 2 (P2 wins), or 0 (draw / truncation). The env's own
    encoder is irrelevant — each agent encodes the state itself — but it must exist,
    so we give it the latest.
    """
    env = WarChestEnv(save_game_history=False)
    if seed is not None:
        np.random.seed(seed)
    env.reset()
    agents = {1: agent_p1, 2: agent_p2}

    for _ in range(max_turns):
        pid = env.active_player
        action = agents[pid].act(env)
        _, _, terminated, truncated, info = env.step(action)
        if not info['action'].is_valid:
            # An agent proposed an illegal move; fall back to a random legal one
            # (mirrors the training loops) so one bad agent can't hang the game.
            _, _, terminated, truncated, info = env.make_random_step()
        if terminated:
            return pid  # the player who just moved delivered the finishing blow
        if truncated:
            return 0
    return 0


# --------------------------------------------------------------------------- #
# Round-robin + ratings
# --------------------------------------------------------------------------- #
def build_task_list(n, *, k_games, seed):
    """Deterministic `(i, j, game_seed, p1_is_i)` tasks for an n-agent round-robin.

    Pair order (`itertools.combinations`) and per-pair game/seed/color order are
    fixed here so both the sequential and parallel round-robin paths hand out the
    exact same seed to the exact same pairing regardless of dispatch order —
    which is what lets a parallel run reproduce a sequential run's result matrix
    bit-for-bit at a given seed (see `gauntlet_parallel.py`).
    """
    tasks = []
    rng_seed = seed
    for i, j in itertools.combinations(range(n), 2):
        for g in range(k_games):
            # Alternate colors: even games i=P1, odd games j=P1.
            tasks.append((i, j, rng_seed, g % 2 == 0))
            rng_seed += 1
    return tasks


def record_result(wins, games, i, j, p1_is_i, res):
    """Fold one game's raw `play_game` result (0/1/2) into the `wins`/`games` matrices."""
    if res == 1:
        winner = i if p1_is_i else j
    elif res == 2:
        winner = j if p1_is_i else i
    else:
        winner = None
    games[i, j] += 1
    games[j, i] += 1
    if winner is None:      # draw
        wins[i, j] += 0.5
        wins[j, i] += 0.5
    elif winner == i:
        wins[i, j] += 1.0
    else:
        wins[j, i] += 1.0


def _finalize_report(names, wins, games):
    """Shared post-processing: win-rate matrix, BT ratings, intransitivity."""
    with np.errstate(invalid='ignore', divide='ignore'):
        win_rate = np.where(games > 0, wins / games, np.nan)

    ratings = _bradley_terry_elo(wins)
    transitivity = _intransitive_fraction(win_rate)

    return {
        'names': names,
        'wins': wins,
        'games': games,
        'win_rate': win_rate,
        'ratings': dict(zip(names, ratings)),
        'intransitive_fraction': transitivity,
    }


def round_robin(agents, *, k_games=20, seed=0):
    """Play every pair K games with balanced colors.

    Returns a dict with agent names, the wins/games/win-rate matrices (wins counts
    draws as 0.5 to each side), Bradley-Terry ratings (Elo-scaled), and the
    intransitive-triple fraction.
    """
    n = len(agents)
    names = [a.name for a in agents]
    wins = np.zeros((n, n), dtype=np.float64)   # wins[i,j] = i's score vs j (draw=0.5)
    games = np.zeros((n, n), dtype=np.float64)

    for i, j, game_seed, p1_is_i in build_task_list(n, k_games=k_games, seed=seed):
        if p1_is_i:
            res = play_game(agents[i], agents[j], seed=game_seed)
        else:
            res = play_game(agents[j], agents[i], seed=game_seed)
        record_result(wins, games, i, j, p1_is_i, res)

    return _finalize_report(names, wins, games)


def _bradley_terry_elo(wins, *, n_iter=10000, tol=1e-10, anchor=1000.0, reg=1.0):
    """Fit Bradley-Terry strengths via MM iteration, return Elo-scaled ratings.

    `wins[i,j]` is i's score against j (fractional for draws). Ratings are shifted
    so their mean equals `anchor`; only differences are meaningful.

    `reg` adds one virtual drawn game per contested pair (Bayesian smoothing) so an
    unbeaten or winless agent gets a bounded rating instead of BT's ±infinity limit.
    """
    n = wins.shape[0]
    if n == 0:
        return np.array([])
    contested = (wins + wins.T) > 0
    wins = wins + 0.5 * reg * contested   # +½ virtual win to each side of each pair
    pair = wins + wins.T           # total games per pair (symmetric)
    W = wins.sum(axis=1)           # total score per agent
    p = np.ones(n)
    for _ in range(n_iter):
        denom = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j and pair[i, j] > 0:
                    denom[i] += pair[i, j] / (p[i] + p[j])
        # Avoid division by zero for agents with no wins (keep tiny strength).
        p_new = np.where(denom > 0, W / np.where(denom > 0, denom, 1.0), p)
        p_new = np.where(p_new <= 0, 1e-12, p_new)
        p_new /= p_new.sum()
        if np.abs(p_new - p).max() < tol:
            p = p_new
            break
        p = p_new
    scale = 400.0 / np.log(10.0)
    elo = scale * np.log(p)
    return elo - elo.mean() + anchor


def _intransitive_fraction(win_rate):
    """Fraction of ordered-by-strength triples that form a cycle (i>j>k>i).

    Uses the sign of (win_rate - 0.5): i 'beats' j if it wins >50% of their games.
    Returns 0.0 when there are fewer than 3 agents or no decided triples.
    """
    n = win_rate.shape[0]
    if n < 3:
        return 0.0
    beats = win_rate > 0.5
    total = cycles = 0
    for i, j, k in itertools.combinations(range(n), 3):
        # A triangle is intransitive iff it is a 3-cycle in either orientation.
        total += 1
        ij, jk, ki = beats[i, j], beats[j, k], beats[k, i]
        ji, kj, ik = beats[j, i], beats[k, j], beats[i, k]
        if (ij and jk and ki) or (ji and ik and kj):
            cycles += 1
    return cycles / total if total else 0.0
