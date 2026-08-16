"""Round-robin gauntlet (docs/history.md — measurement infra).

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
from .bots.threat_greedy_bot import ThreatAwareGreedyBot
from .bots.greedy_sim_bot import SimGreedyBot
from .bots.bolster_bot import BolsterBot
from .bots.random_eval_bot import (
    RandomEvalBot, RandomEvalLookaheadBot, RandomEvalCriticBot,
)
from .bots.policy_theta_bot import PolicyThetaBot
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


def greedy_sim_agent(name='greedy_sim', encoder=None, **kwargs):
    """The 1-ply forward-simulation greedy (`SimGreedyBot`) — the strong greedy
    yardstick. Speaks `act(env)` + `.name` directly (like LookaheadBot), so it
    drops in without a HeuristicAgent obs-encoding wrapper. `encoder` is accepted
    for a uniform factory signature but unused (this bot reads the env, not obs).

    `kwargs` reach `SimGreedyBot` (e.g. `see_opponent_hand` — its 2-ply reply model
    consumes the opponent's hand, so the flag is live here despite the shallow
    search; `src/app/eval_info_value.py` builds both variants from this).
    """
    return SimGreedyBot(name=name, **kwargs)


def random_eval_agent(name=None, encoder=None, *, theta=None, seed=0, **kwargs):
    """`RandomEvalBot` — a `SimGreedyBot` with sampled leaf coefficients (docs/IDEAS.md B1).

    Speaks `act(env)` + `.name` directly, like the other simulation bots. θ is **pinned**
    here (`resample_each_episode` is left at its default False): the gauntlet's antithetic
    draft pairing only cancels if the same agent plays both games of a pair, and a column
    of the report is only interpretable if it names one playstyle. Per-episode resampling
    belongs to the training pool — see `OpponentPool`.
    """
    return RandomEvalBot(theta=theta, seed=seed, name=name, **kwargs)


def random_eval_lookahead_agent(name=None, encoder=None, *, theta=None, seed=0, **kwargs):
    """`RandomEvalLookaheadBot` — the same θ family on the alpha-beta search (IDEAS.md B1).

    ~6x the per-move cost of `random_eval` (IDEAS.md Table A), for a base bot that beats
    `SimGreedyBot` 0.79. θ is pinned here for the same reason as `random_eval_agent`.
    """
    return RandomEvalLookaheadBot(theta=theta, seed=seed, name=name, **kwargs)


def random_eval_critic_agent(name=None, encoder=None, *, theta=None, seed=0, **kwargs):
    """`RandomEvalCriticBot` — the θ family on the critic-guided beam search.

    The strong branch: θ re-weights the hand-written half of the leaf blend and
    `critic_weight` sets how much of the leaf that half is. Needs a critic checkpoint, so
    unlike the other two it is not policy-independent.
    """
    return RandomEvalCriticBot(theta=theta, seed=seed, name=name, **kwargs)


def policy_theta_agent(name=None, encoder=None, *, theta=None, seed=0, **kwargs):
    """`PolicyThetaBot` — policy proposes, θ-weighted simulation checks (docs/IDEAS.md B1).

    The fast branch: ~5 ms/move against `lookahead_critic`'s ~104. Needs a policy
    checkpoint, so like the critic family it is not policy-independent.
    """
    return PolicyThetaBot(theta=theta, seed=seed, name=name, **kwargs)


def bolster_agent(name='bolster', encoder=None, **kwargs):
    """BolsterBot — the Berserker/Priest bolster-archetype exploiter
    (docs/independent_opponents.md). Speaks `act(env)` + `.name` directly, like the
    other simulation bots. NOTE: its archetype is only in force when it actually drafts
    a Berserker/Priest; the standard gauntlet drafts randomly, so to see its intended
    strength force the draft — `src/app/eval_bolster.py` does this. Without the forced
    draft it degrades to its `SimGreedyBot` base behaviour.
    """
    return BolsterBot(name=name, **kwargs)


def greedy_fast_agent(name='greedy_fast', encoder=None):
    """The legacy obs-only `GreedyBot` — cheap, hand-blind, no simulation. Kept as
    a separate entrant (and as the training-loop opponent, via opponent_pool) now
    that `greedy_sim` is the heavier simulation bot.
    """
    return HeuristicAgent(name, GreedyBot(), encoder)


def threat_greedy_agent(name='threat_greedy', encoder=None):
    """`ThreatAwareGreedyBot` — obs-only like `greedy_fast`, but reading the threat
    planes (docs/IDEAS.md B5). Same cost class as GreedyBot, so it is a like-for-like
    comparison: the only thing that changed is what the bot looks at.
    """
    return HeuristicAgent(name, ThreatAwareGreedyBot(), encoder)


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
    is the subprocess/worktree path (docs/history.md — the gauntlet design contract), out of scope here. We
    skip them with a warning so the gauntlet still runs on the loadable field.
    """
    try:
        meta = load_policy_checkpoint(path, map_location=device)
        encoder = get_encoder(meta['obs_version'])
        policy = Policy(device=device, hidden_dim=meta['hidden_dim'], obs_encoder=encoder,
                        arch=meta['arch']).to(device)
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
        return greedy_sim_agent(spec['name'], **spec.get('kwargs', {}))
    if kind == 'bolster':
        return bolster_agent(spec['name'], **spec.get('kwargs', {}))
    if kind == 'random_eval':
        return random_eval_agent(spec.get('name'), **spec.get('kwargs', {}))
    if kind == 'random_eval_lookahead':
        return random_eval_lookahead_agent(spec.get('name'), **spec.get('kwargs', {}))
    if kind == 'policy_theta':
        return policy_theta_agent(spec.get('name'), device=device,
                                  **spec.get('kwargs', {}))
    if kind == 'random_eval_critic':
        return random_eval_critic_agent(spec.get('name'), device=device,
                                        **spec.get('kwargs', {}))
    if kind == 'greedy_fast':
        return greedy_fast_agent(spec['name'])
    if kind == 'threat_greedy':
        return threat_greedy_agent(spec['name'])
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
    # Episode-boundary hook for agents that carry per-episode state (`RandomEvalBot`
    # resamples θ here). Duck-typed — most agents don't define it. Called *after*
    # `env.reset()` so a hook that touches the global RNG cannot shift the draft this
    # game's seed just pinned.
    for agent in (agent_p1, agent_p2):
        hook = getattr(agent, 'new_episode', None)
        if hook is not None:
            hook()

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
def build_task_list(n, *, k_games, seed, paired=False):
    """Deterministic `(i, j, game_seed, p1_is_i)` tasks for an n-agent round-robin.

    Pair order (`itertools.combinations`) and per-pair game/seed/color order are
    fixed here so both the sequential and parallel round-robin paths hand out the
    exact same seed to the exact same pairing regardless of dispatch order —
    which is what lets a parallel run reproduce a sequential run's result matrix
    bit-for-bit at a given seed (see `gauntlet_parallel.py`).

    `paired` makes each color swap replay the previous game's draft: `play_game` seeds
    the global RNG before `env.reset()` and `set_init_state` draws the whole 4/4
    disjoint draft from it, so two games at the same seed open identically, and
    swapping which agent sits in seat 1 makes each agent play both compositions.

    **It defaults OFF, because it was measured and it does not work here**
    (docs/IDEAS.md L5). The idea was sound on paper — writing `p` for the true win rate
    and `D` for the draft advantage, two unpaired games give `Var = 2p(1-p)` while a
    pair that flips the sign of `D` gives `2p(1-p) - 2*Var(D)`. And the draft really
    does decide games: with the *same* deterministic bot on both sides and the
    compositions swapped, the same composition won both games **63.3 %** of the time
    (190/300, se 2.8 pp).

    But the variance reduction needs the two games of a pair to be **negatively
    correlated**, and they are not. Measured on 150 pairs each:
    `greedy_fast` vs `greedy_sim` **r = -0.003 +/- 0.082**, and two policy checkpoints
    **r = -0.005 +/- 0.082**. The mechanism never engages, because the 63.3 % was
    measured with one deterministic bot playing itself — in a real gauntlet the
    entrants differ, and policy agents *sample* their actions, so an identical opening
    diverges on the first ply and the shared draft stops propagating. Two direct
    variance checks (n=120 each) came out at ratio 1.29 and 0.77, i.e. ~1.4 sigma in
    opposite directions: noise, not an effect.

    So `paired=True` is kept as an opt-in for a field of deterministic entrants, where
    the argument may still hold, and is off by default so every previously recorded
    gauntlet number stays reproducible bit-for-bit at its seed.

    Note it could never have applied to a forced-draft archetype bot
    (`eval_bolster.py`) in any case: there the composition is pinned to the *agent*
    rather than the seat, so it is the treatment rather than a nuisance draw, and the
    control is common random numbers across arms instead.
    """
    tasks = []
    rng_seed = seed
    for i, j in itertools.combinations(range(n), 2):
        for g in range(k_games):
            # Alternate colors: even games i=P1, odd games j=P1.
            tasks.append((i, j, rng_seed, g % 2 == 0))
            # Paired: hold the seed across the odd game so it replays the even game's
            # draft with the seats swapped, then advance. Unpaired: advance every game.
            rng_seed += (g % 2) if paired else 1
        # An odd k_games leaves a trailing unpartnered game; it still consumed a seed,
        # so advance past it or the next pair would silently reuse that draft.
        if paired and k_games % 2:
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


def _finalize_report(names, wins, games, results=None):
    """Shared post-processing: win-rate matrix, BT ratings, intransitivity.

    `results` is the optional raw per-game log, `[(i, j, game_seed, p1_is_i, res), ...]`
    in completion order. The aggregate matrices are enough to rank a field, but a
    *paired* analysis needs to line games up by seed across pairs — see
    `src/app/eval_info_value.py`, which plays two arms on one seed block and compares
    them game by game.
    """
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
        'results': results if results is not None else [],
    }


def round_robin(agents, *, k_games=20, seed=0, tasks=None, paired=False):
    """Play every pair K games with balanced colors.

    Returns a dict with agent names, the wins/games/win-rate matrices (wins counts
    draws as 0.5 to each side), Bradley-Terry ratings (Elo-scaled), the
    intransitive-triple fraction, and the raw per-game log.

    `tasks` overrides the default all-pairs schedule with an explicit
    `[(i, j, game_seed, p1_is_i), ...]` list — for experiments that need a specific
    pairing/seed layout rather than a round-robin (`k_games`/`seed`/`paired` are then
    unused). `paired` is the antithetic-draft schedule; see `build_task_list`.
    """
    n = len(agents)
    names = [a.name for a in agents]
    wins = np.zeros((n, n), dtype=np.float64)   # wins[i,j] = i's score vs j (draw=0.5)
    games = np.zeros((n, n), dtype=np.float64)
    results = []

    if tasks is None:
        tasks = build_task_list(n, k_games=k_games, seed=seed, paired=paired)
    for i, j, game_seed, p1_is_i in tasks:
        if p1_is_i:
            res = play_game(agents[i], agents[j], seed=game_seed)
        else:
            res = play_game(agents[j], agents[i], seed=game_seed)
        record_result(wins, games, i, j, p1_is_i, res)
        results.append((i, j, game_seed, p1_is_i, res))

    return _finalize_report(names, wins, games, results)


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
