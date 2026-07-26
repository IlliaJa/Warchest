"""Critic-guided beam-search bot, LookaheadBot's cousin (docs/lookahead_bot_plan.md).

`LookaheadBot` prunes with a cheap, hand-crafted, pre-move ordering key
(`_ordering_key`) capped at `max_branching`, then alpha-beta-searches whatever
survives, scoring unexplored leaves with a hand-tuned material/base/positional
formula (`_leaf_potential`). This bot replaces both pieces with a trained
`Critic` network: at every node, every legal move is actually applied, the
resulting states are scored by the critic in one batched forward pass, and
only the `beam_width` best survive to be recursed into.

Pruning is scoped to each node's own children, not globally across the whole
search (an earlier version of this file did the latter — kept a single running
beam and pruned to the `beam_width` best/worst states *overall* each round.
That is a real bug, not a style choice: at an opponent reply node, "keep the
worst states overall" discards a root move whose worst reply is merely
mediocre — not because it's a bad root move, but because some *other* root
move's reply happened to be more extreme, so it wasn't in the global bottom-K.
That systematically favors root moves whose opponent replies are catastrophic
over ones whose replies are merely bad, i.e. exactly backwards. Recursing
per-node, mirroring `_minimax`'s scoping, is the fix.)

Turns don't strictly alternate (pending tactic continuations, empty-hand
skips), so which direction "best" means at a node depends on who's about to
move there: root_player's own nodes keep the `beam_width` highest-scoring
children (maximizing), the opponent's nodes keep the `beam_width`
lowest-scoring ones (minimizing — the opponent's best replies are whatever
hurts root_player most) — the same maximizing/minimizing split `_minimax`
uses. The critic's score is used twice at each node: once (cheap, 1-ply) to
decide which children are even worth recursing into, then replaced by the
deeper recursive value for whichever children survive.

Root-perspective scoring without perspective-spoofing: the critic is trained
to value whoever is about to act in the observation it's given (the obs
encoder always rotates the board ego-centrically around the current mover),
so asking it "what is this worth to root_player" when root_player isn't the
mover would require faking `state.active_player` before encoding — which
would also corrupt the legal-action mask and pending-context features the
encoder computes along the way. Cheaper and exactly correct instead: encode
every state from its *real* mover's perspective (always consistent, exactly
how the encoder is used during real play) and negate the value when the mover
isn't root_player — the same "value to the player to move" convention
negamax uses. This game's reward terms (base/material PBRS, win/loss) are
antisymmetric between players by construction (docs/rewards.md), so the
critic trained against them is expected to be antisymmetric too.

Reuses `LookaheadBot`'s forward-simulation harness verbatim: state cloning
(`_clone_state`), single-determinization future draws, the
`see_opponent_hand` visibility flag, and the real `Action.reward` + holding-
reward accounting (`_own_action_reward`/`_holding_reward`/`_truncation_value`)
— only "how many candidates survive" and "how is a non-terminal state scored"
change. The critic also sees the true hidden game state via
`WarChestEnv.get_privileged_features()` — the same privileged input it was
trained on, just read exactly (both hands, both bags) rather than only at the
main actor's own decision points during training rollout collection.

Per-node cost is dominated by the critic itself, not by cloning/applying:
profiling one node's ~15 legal actions found ~0.7ms/action for
`generate_observation()` (the full v10 encoder — threat grids, mask, etc.)
plus ~0.3-0.4ms/action for the `hidden_dim=192` critic's forward pass, vs.
~0.07ms/action for clone+apply. `LookaheadBot`'s `_leaf_potential` is a few
scalar lookups by comparison — its `time_budget=0.1` default was tuned for
that cost, not this one. At the same 0.1s budget this bot only ever
evaluated the root's own children (depth 0, `nodes_visited` stuck at 1-8) —
i.e. a pure 1-ply greedy-by-critic move, never actually considering the
opponent's reply — which is a materially worse agent than the fixed pruning
bug's docstring above implies, and is what a bad win rate here usually means.
Two mitigations, both applied: `max_branching` reuses `LookaheadBot`'s own
cheap pre-move ordering key (`_ordering_key`) to cut the raw legal-action
count before any of the expensive per-action work happens; `time_budget`
defaults higher (this bot is eval-only per docs/lookahead_bot_plan.md's scope
note, never in the rollout hot path, so there's no speed floor to hit).
"""
import glob
import logging
import os
import random
import re
import time
import types

import numpy as np
import torch

from .lookahead_bot import LookaheadBot, _TimeUp, _clone_state, _determinized_draw_one
from ..environment.warchest_env import WarChestEnv
from ..environment.obs_encoders import get_encoder
from ..environment.rollout_core import OPP_TYPE_IDX
from ..policy.checkpoint import load_critic_checkpoint
from ..policy.policy import Critic

CRITIC_GLOB = 'data/lookahead_critic/lookahead_critic_v*.pth'


def _latest_critic_path():
    """Highest-numbered `lookahead_critic_v{N}.pth` under `data/lookahead_critic/`,
    or None if no such checkpoint exists. Mirrors `gauntlet.py`'s own
    `_latest_critic_path` (a separate small copy rather than a shared import —
    services/ shouldn't depend on app/gauntlet.py) so this bot's own default
    stays correct the same way the gauntlet CLI's does, rather than pointing at
    one hardcoded, eventually-stale version.
    """
    candidates = glob.glob(CRITIC_GLOB)
    if not candidates:
        return None

    def version(path):
        m = re.search(r'_v(\d+)\.pth$', os.path.basename(path))
        return int(m.group(1)) if m else -1

    return max(candidates, key=version)


# Same logger name ppo.py's setup_run_logger configures (file handler at DEBUG,
# console at INFO). `act()` logs at two levels: DEBUG per real move (detailed,
# would spam a console shown at INFO — see `_record_agg_stats`'s docstring for
# the INFO-level rolling summary meant to be readable at a glance instead).
# Silent by default: neither gauntlet.py nor gauntlet_parallel.py configures
# any handler, so `logging.basicConfig(level=logging.INFO)` (DEBUG for the
# per-move trace too; or attach a handler to this 'warchest' logger
# specifically) is needed to actually see either.
logger = logging.getLogger('warchest')


class _Child:
    """One candidate move out of a node: the state it leads to (None if the
    game/round ended, since nothing further can be simulated from it) plus
    enough of the reward path to finish scoring it once `est` is filled in.
    """

    __slots__ = ('action_id', 'partial_value', 'state', 'queues', 'terminal', 'est', 'leaf')

    def __init__(self, action_id, partial_value, state, queues, terminal):
        self.action_id = action_id
        self.partial_value = partial_value
        self.state = state
        self.queues = queues
        self.terminal = terminal
        self.est = partial_value if terminal else None
        # `leaf`: this child has a real (non-terminal) state and IS critic-scored,
        # but the search must NOT recurse past it. The depth-bounded base search
        # never sets it; round-bounded subclasses (RoundCriticBot) set it at the
        # round boundary so a move that ends the round becomes a scored leaf.
        self.leaf = False


class LookaheadCriticBot(LookaheadBot):
    """Beam-limited search over `WarChestEnv` states, scored by a trained `Critic`.

    Args:
        critic_path: path to a checkpoint saved by
            `policy.checkpoint.save_critic_checkpoint`. Defaults to
            `_latest_critic_path()` — the highest-numbered
            `data/lookahead_critic/lookahead_critic_v{N}.pth` — so this always
            picks up the newest trained critic rather than a version pinned in
            code (mirrors `gauntlet.py`'s own default resolution). Raises
            `FileNotFoundError` if none exists and none was passed explicitly.
        beam_width: how many children survive (and get recursed into) at each
            node — see module docstring for why this is per-node, not global.
        max_branching: cap on raw legal actions considered per node, applied
            *before* cloning/applying/critic-scoring via `LookaheadBot`'s own
            cheap ordering key — see module docstring; this is what makes real
            recursion (not just a 1-ply critic-greedy move) fit in the budget.
        time_budget, see_opponent_hand, max_depth, gamma: as `LookaheadBot`.
            Iterative deepening tries `depth=0,1,2,...` (each unit is one full
            ply of "expand + critic-score + keep beam_width" recursed into)
            until `time_budget` runs out, same iterative-deepening contract
            as `LookaheadBot.act`.
        opp_type: which of the critic's trained opponent-identity one-hot
            slots (`rollout_core.OPP_TYPE_IDX`) to feed alongside every state
            — the critic was trained conditioned on this, and there's no
            "unknown opponent" slot to fall back on. `'pool'` (self-play
            snapshots) is the closest analogue to an arbitrary eval opponent.
        n_determinizations: independent `_act_once` searches per `act()` call,
            each under a fresh sampled future-draw order, weighted-vote
            combined (see `act()`) — hedges this bot's own single-
            determinization variance without changing the total per-move time
            budget. Defaults to 1 (off): measured *worse* than 1 at every
            split tried (even a lopsided 80/20 two-way split) at this bot's
            0.5s default budget — the search is already too depth-starved
            relative to `LookaheadBot`'s alpha-beta for taking any of that
            budget away from the primary search to be worth the hedge (see
            docs/bots.md's experiment log). Left available, not deleted, for
            a meaningfully larger `time_budget` where the primary search
            stops being the bottleneck.
        stats_log_every: log an aggregated-statistics summary (see `act()`)
            every this many real moves, rolling — 0 disables it.
        device: torch device for the critic's forward passes.
    """

    # When True, `_beam_value` treats a child that crosses into the next round as
    # a critic-scored leaf (no recursion past it) and flags any depth-cut that
    # stopped a still-in-round line. Off for this depth-bounded base search; set
    # True by RoundCriticBot, which searches to the end of the current round
    # rather than to a fixed ply depth. Every use is gated on this flag so the
    # base search pays nothing for the hook.
    _BOUNDS_BY_ROUND = False

    def __init__(self, critic_path=None, beam_width=5, max_branching=5,
                 time_budget=0.5, see_opponent_hand=True, max_depth=40, gamma=0.99,
                 opp_type='pool', n_determinizations=1, stats_log_every=20,
                 device='cpu', name='lookahead_critic'):
        if critic_path is None:
            critic_path = _latest_critic_path()
            if critic_path is None:
                raise FileNotFoundError(
                    f'No checkpoint matching {CRITIC_GLOB} — pass critic_path '
                    f'explicitly, or train and save one first.'
                )
        # rich_eval=False: this bot's value-scale calibration is moment-matched to
        # `_leaf_potential`'s exact distribution (see docs/bots.md), so its leaf
        # must stay the legacy base/material/pos/risk formula. The new
        # bolster/recruit/tempo terms would shift that distribution and invalidate
        # the calibration; they belong to the hand-tuned LookaheadBot, not here.
        super().__init__(time_budget=time_budget, max_branching=max_branching,
                          see_opponent_hand=see_opponent_hand, max_depth=max_depth,
                          gamma=gamma, shaping_anneal=1.0, rich_eval=False, name=name)
        self.beam_width = beam_width
        self.n_determinizations = n_determinizations
        self.stats_log_every = stats_log_every
        self._reset_agg_stats()
        self.device = device
        # Round-bounded search state (see `_BOUNDS_BY_ROUND`). Inert while that
        # flag is False: `_root_round` is the round number at the search root and
        # `_round_incomplete` records whether the budget cut a line off before the
        # round ended (used by RoundCriticBot's iterative deepening to stop).
        self._root_round = None
        self._round_incomplete = False

        meta = load_critic_checkpoint(critic_path, map_location=device)
        encoder = get_encoder(meta['obs_version'])
        self._critic = Critic(device=device, hidden_dim=meta['hidden_dim'], obs_encoder=encoder).to(device)
        self._critic.load_state_dict(meta['state_dict'])
        self._critic.eval()

        # LookaheadBot.__init__ built `_sim_env` against the *latest* encoder; the
        # critic's obs shapes are pinned to whatever version it was trained under
        # (recorded in the checkpoint), which may not be the same one.
        self._sim_env = WarChestEnv(save_game_history=False, obs_encoder=encoder)
        self._sim_env._draw_one = types.MethodType(_determinized_draw_one, self._sim_env)
        self._sim_env._sim_draw_queues = {1: [], 2: []}

        opp_onehot = np.zeros(len(OPP_TYPE_IDX), dtype=np.float32)
        opp_onehot[OPP_TYPE_IDX[opp_type]] = 1.0
        self._opp_onehot = torch.from_numpy(opp_onehot).to(device)

        self._value_scale, self._value_shift = 1.0, 0.0
        if meta.get('return_mean') is not None and meta.get('return_std') is not None:
            # Exact recovery: this checkpoint was saved with `save_critic_checkpoint`'s
            # optional return_mean/return_std (ppo.py's ReturnNormalizer EMA at save
            # time, see checkpoint.py's module docstring) — denormalize precisely
            # instead of approximating via `_calibrate_value_scale`'s moment-match.
            self._value_scale = float(meta['return_std'])
            self._value_shift = float(meta['return_mean'])
            logger.debug('lookahead_critic: using exact checkpoint return_mean=%.4f return_std=%.4f',
                         self._value_shift, self._value_scale)
        else:
            self._calibrate_value_scale()

    def _calibrate_value_scale(self, n_games=8, n_samples=160, seed=12345):
        """One-time affine fit *approximating* the critic's real reward-scale —
        the fallback for checkpoints saved before `save_critic_checkpoint` grew
        its optional `return_mean`/`return_std` (see `__init__`, which uses
        those directly, exactly, whenever a checkpoint has them; this method
        only runs for older checkpoints that don't).

        `Critic.value_batch` was trained against *normalised* returns (ppo.py's
        `ReturnNormalizer`: an EMA of return mean/std, used to keep the critic's
        loss scale stable — see its docstring). `ppo.py` always denormalises
        (`value * std + mean`) before treating the critic's output as a real
        value anywhere (rollout bootstrapping, GAE). Older checkpoints never
        recorded that EMA (`checkpoint.py` only saved
        `state_dict`/`obs_version`/`arch`/`hidden_dim`), so the exact
        denormalisation used when such a checkpoint was saved can't be
        recovered — feeding the network's raw output straight into
        `_beam_value`, which sums it with real reward-scale path returns, was
        giving the critic's contribution an arbitrary, depth-dependent weight
        relative to the real rewards it's added to (nodes reached via more/fewer
        real-reward-bearing steps ended up on incommensurable scales — confirmed to be the fix that
        matters: swapping this bot's scoring for `_leaf_potential` outright,
        same beam-search shape otherwise, beat GreedyBot 5/6 games where the raw
        critic scored ~25%).

        Matching the raw output's mean/std to `_leaf_potential`'s over a
        handful of quick self-play rollouts recovers a substitute affine
        correction: `_leaf_potential` is already reward-scale-correct (the
        exact quantity `_minimax` sums real path rewards against), so aligning
        the critic's first two moments to it makes the critic's *directional*
        signal usable at a compatible scale, without needing the lost EMA.
        """
        rng = random.Random(seed)
        states = []
        env = self._sim_env
        for g in range(n_games):
            if len(states) >= n_samples:
                break
            env.reset(seed=seed + g)
            done = False
            while not done and len(states) < n_samples:
                legal = env.get_possible_actions()
                # Mostly the cheap ordering-key's pick (docs/lookahead_bot_plan.md's
                # move-ordering heuristic) rather than pure uniform-random, so the
                # sampled states resemble ones a real game/search actually reaches
                # (random-vs-random wanders into board configurations neither this
                # bot nor a real opponent would ever produce) — with a random
                # fallback slice for state diversity.
                if legal and rng.random() < 0.8:
                    mover = env.active_player
                    dist_grid = self._dist_grid_to_targets(self._capturable_bases(mover))
                    melee_threat = self._melee_threatened_cells(mover)
                    action = min(legal, key=lambda a: self._ordering_key(a, dist_grid, melee_threat, mover))
                else:
                    action = rng.choice(legal)
                _, _, term, trunc, _ = env.step(action)
                done = term or trunc
                if not done:
                    states.append(_clone_state(env.state))
        if len(states) < 2:
            return
        raw = np.array(self._critic_values_raw(states))
        heur = np.array([self._leaf_potential(s, s.active_player) for s in states])
        raw_std = raw.std()
        if raw_std < 1e-6:
            return
        self._value_scale = float(heur.std() / raw_std)
        self._value_shift = float(heur.mean() - self._value_scale * raw.mean())

    def act(self, env) -> int:
        """Vote across `n_determinizations` independent searches instead of
        betting the whole decision on one sampled future-draw order.

        `_prepare_root` samples a fresh, unseeded determinization every time
        it's called (`LookaheadBot`'s single-determinization design — one
        sample per search, reused across that search's whole tree, cheaper
        than resampling per node). A single sample can make a node's estimated
        value swing on a guessed-future that never happens, which is pure
        noise in the decision, not signal — this bot has no control over
        `LookaheadBot`'s own single-determinization variance (not this bot's
        file to change), but its *own* variance from the same mechanism is
        fully fixable here: split `time_budget` across `n_determinizations`
        independent `_act_once` searches (same total wall-clock per move) and
        vote, rather than spend the whole budget on one draw of the dice. Each
        individual search gets a smaller sub-budget (shallower/narrower), but
        the resulting decision is hedged against any single one of them
        guessing an unlucky future — the classic determinization-averaging
        fix for imperfect-information game search (Perfect Information Monte
        Carlo), applied to the one piece of hidden information (draw order)
        this search has to guess at all.
        """
        root_player = env.active_player
        legal = env.get_possible_actions()
        if len(legal) <= 1:
            return legal[0]

        n = max(1, self.n_determinizations)
        # An *equal* split measured worse than n=1 (0.30s/0.30s beat 0.60s
        # solo by vote, but 0.25s/0.25s lost to 0.50s solo — see docs/bots.md):
        # this search is already depth-starved at 0.5s (LookaheadBot reaches
        # depth 4-6 here in a fifth of the time), so halving the budget costs
        # a whole ply more often than the vote recovers. Weighting it instead
        # — one primary search keeps most of the budget (nearly the full
        # single-search depth), the rest are cheap hedges — was the config
        # that actually held up: a second opinion on whether the primary's
        # single determinization got unlucky, without sacrificing the
        # primary's own depth to buy it.
        if n == 1:
            weights = [1.0]
        else:
            weights = [0.8] + [0.2 / (n - 1)] * (n - 1)
        votes, val_weighted, stats_list = {}, {}, []
        for w in weights:
            action, val, stats = self._act_once(env, root_player, legal, self.time_budget * w)
            votes[action] = votes.get(action, 0.0) + w
            val_weighted[action] = val_weighted.get(action, 0.0) + w * (val if val is not None else 0.0)
            stats_list.append(stats)
        # Weighted-plurality vote across determinizations; ties broken by
        # whichever tied action scored better on average where it did win.
        best_action = max(votes, key=lambda a: (votes[a], val_weighted[a] / votes[a]))
        self.last_stats = {
            'depth_reached': max(s['depth_reached'] for s in stats_list),
            'depths': [s['depth_reached'] for s in stats_list],
            'nodes_visited': sum(s['nodes_visited'] for s in stats_list),
            'elapsed': sum(s['elapsed'] for s in stats_list),
            'legal_at_root': len(legal),
            'best_value': val_weighted[best_action] / votes[best_action],
            'determinization_votes': votes,
        }
        # DEBUG (not INFO): fires once per real move, so this would spam a
        # console that shows INFO — see the `logger` module-docstring comment
        # for how to actually surface it.
        logger.debug(
            '%s act(): depth_reached=%d (per-search=%s) nodes_visited=%d '
            'elapsed=%.3fs/%.3fs budget legal_at_root=%d best_value=%.4f',
            self.name, self.last_stats['depth_reached'], self.last_stats['depths'],
            self.last_stats['nodes_visited'], self.last_stats['elapsed'], self.time_budget,
            self.last_stats['legal_at_root'], self.last_stats['best_value'],
        )
        self._record_agg_stats()
        return best_action

    def _reset_agg_stats(self):
        self._agg = {
            'n': 0, 'nodes_visited': 0, 'depth_sum': 0, 'depth_min': None, 'depth_max': None,
            'elapsed_sum': 0.0, 'legal_at_root_sum': 0,
        }

    def _record_agg_stats(self):
        """Roll `self.last_stats` into a running window and, every
        `stats_log_every` real moves, log it as one human-readable INFO
        summary — min/avg/max depth_reached (the number that answers "is the
        search actually looking ahead, or stuck at the leaf") alongside
        nodes_visited and elapsed vs. budget, instead of the per-move DEBUG
        trace above (still there for inspecting one specific decision, but
        too granular to read across a whole game at a glance).
        """
        if not self.stats_log_every:
            return
        a = self._agg
        s = self.last_stats
        a['n'] += 1
        a['nodes_visited'] += s['nodes_visited']
        a['depth_sum'] += s['depth_reached']
        a['depth_min'] = s['depth_reached'] if a['depth_min'] is None else min(a['depth_min'], s['depth_reached'])
        a['depth_max'] = s['depth_reached'] if a['depth_max'] is None else max(a['depth_max'], s['depth_reached'])
        a['elapsed_sum'] += s['elapsed']
        a['legal_at_root_sum'] += s['legal_at_root']
        if a['n'] < self.stats_log_every:
            return
        logger.info(
            '%s: last %d move(s) — depth_reached avg=%.2f min=%d max=%d, '
            'nodes_visited avg=%.1f, elapsed avg=%.3fs/%.3fs budget, legal_at_root avg=%.1f',
            self.name, a['n'], a['depth_sum'] / a['n'], a['depth_min'], a['depth_max'],
            a['nodes_visited'] / a['n'], a['elapsed_sum'] / a['n'], self.time_budget,
            a['legal_at_root_sum'] / a['n'],
        )
        self._reset_agg_stats()

    def _act_once(self, env, root_player, legal, time_budget):
        """One full iterative-deepening beam search under a fresh
        determinization and its own (possibly split) time budget — the body
        `act()` ran directly before `n_determinizations` voting was added.
        Returns `(action, value, stats)` instead of mutating `self.last_stats`
        directly, so `act()` can combine several of these.
        """
        root_state, root_queues = self._prepare_root(env, root_player)
        start = time.monotonic()
        deadline = start + time_budget
        best_action = legal[0]
        best_val = None
        depth = 0
        depth_reached = -1
        self._nodes_visited = 0
        # Iterative deepening re-enters the *same* tree (root_state/root_queues
        # are fixed for this whole act() call) at depth=0,1,2,... — every node
        # a shallower pass already fully expanded, critic-scored and pruned to
        # its beam survivors gets identically re-expanded from scratch by each
        # deeper pass, since nothing about it changed. Caching a node's
        # survivors, keyed by the path of action ids taken from root, turns
        # each new outer-loop iteration into "extend the previous one" instead
        # of "redo it plus one more ply" — the redundant work was small early
        # on but geometric in the beam width, so it was a real fraction of the
        # 0.5s budget by the time depth reached 2-3.
        self._survivor_cache = {}
        while depth <= self.max_depth:
            try:
                val, action = self._beam_value(root_state, root_queues, root_player,
                                                depth, deadline, ply=0, path=())
            except _TimeUp:
                break
            if action is not None:
                best_action, best_val = action, val
            depth_reached = depth
            if time.monotonic() >= deadline:
                break
            depth += 1
        stats = {
            'depth_reached': depth_reached,
            'nodes_visited': self._nodes_visited,
            'elapsed': time.monotonic() - start,
        }
        return best_action, best_val, stats

    # ------------------------------------------------------------------

    def _beam_width_at(self, ply):
        """Beam width narrows with ply: the root's own decision (`ply == 0`)
        is what `act()` actually returns, so it keeps the full configured
        width; deeper plies only exist to sanity-check that decision against a
        real reply, so a narrower beam there is the cheap way to buy depth
        instead of width. Per-node cost is critic-forward-dominated (module
        docstring profiling), and cost multiplies across recursion levels, so
        without this the search rarely got past depth 2-3 in the 0.5s budget
        (vs. `LookaheadBot`'s alpha-beta reaching depth 4-6 in a fifth of the
        time) — this bot only ever loses tactical races it can't see coming.
        """
        if ply <= 1:
            return self.beam_width
        return max(2, self.beam_width - (ply - 1))

    def _max_branching_at(self, ply):
        """Same rationale as `_beam_width_at`, applied to the raw-action cap
        before cloning/applying/critic-scoring even starts.
        """
        if not self.max_branching:
            return None
        if ply <= 1:
            return self.max_branching
        return max(3, self.max_branching - 2 * (ply - 1))

    def _prune_candidates(self, state, legal, mover, max_branching):
        """Cheap pre-move cut of `legal` to the `max_branching` most promising
        actions, *before* any of the per-action clone/apply/critic-score work.

        Ordering is `LookaheadBot`'s greedy pre-move key (`_ordering_key`) — the
        same heuristic its own move ordering uses. `PolicyCriticBot` overrides
        this to rank candidates by a trained policy's move prior instead, which
        is the one difference between the two bots (everything downstream — the
        critic scoring, beam, iterative deepening, determinization — is shared).
        `state` is the node being expanded; the sim env is already set to it by
        `_legal_from`, but it's passed explicitly so an override can re-encode it
        without depending on that side effect.
        """
        dist_grid = self._dist_grid_to_targets(self._capturable_bases(mover))
        melee_threat = self._melee_threatened_cells(mover)
        ordered = sorted(legal, key=lambda a: self._ordering_key(a, dist_grid, melee_threat, mover))
        return ordered[:max_branching]

    def _beam_value(self, state, queues, root_player, depth, deadline, ply, path):
        """Root-perspective value of `state` plus the action that achieves it,
        searching `depth` more levels of beam-limited recursion.

        Every legal child is applied and scored (terminal/truncated children
        exactly, others via the critic) *before* any pruning happens, matching
        the user-specified shape: make the possible moves, evaluate them, keep
        the top `beam_width`, then (if `depth` allows) do the same from each of
        those. `depth == 0` stops after the first evaluate-and-keep — the
        critic's 1-ply estimate for the survivor is the returned value, no
        recursion. `depth > 0` recurses into each survivor and replaces that
        shallow estimate with the deeper value, keeping the best/worst of those
        depending on `maximizing` — mirrors `_minimax`'s alpha-beta shape
        without the alpha-beta (pruning already happened via the critic).

        `path` (the action ids taken from root to get here) identifies this
        node stably across iterative-deepening passes within one `act()` call
        (root_state/root_queues/the determinized future draws are all fixed
        for the whole call, so the same path always reaches the same state) —
        see `act()`'s `_survivor_cache` docstring.
        """
        if time.monotonic() >= deadline:
            raise _TimeUp
        self._nodes_visited += 1

        cached = self._survivor_cache.get(path)
        if cached is not None:
            survivors, maximizing = cached
        else:
            mover = state.active_player
            legal = self._legal_from(state)
            maximizing = (mover == root_player)
            holding = self._holding_reward(state, root_player) if maximizing else 0.0
            discount = self.gamma ** ply
            max_branching = self._max_branching_at(ply)

            if max_branching and len(legal) > max_branching:
                legal = self._prune_candidates(state, legal, mover, max_branching)

            children = []
            for action_id in legal:
                child_state = _clone_state(state)
                child_queues = {1: list(queues[1]), 2: list(queues[2])}
                result = self._apply(child_state, child_queues, action_id)
                own_action = (result.player_id == root_player)
                if result.finishes_game:
                    step_reward = result.reward if own_action else -result.reward
                    children.append(_Child(action_id, discount * step_reward, None, None, True))
                    continue
                step_reward = (self._own_action_reward(result) if own_action else 0.0) + holding
                partial = discount * step_reward
                if child_state.round_number >= self._sim_env.max_rounds:
                    trunc = self._truncation_value(child_state, root_player) * self.gamma ** (ply + 1)
                    children.append(_Child(action_id, partial + trunc, None, None, True))
                else:
                    child = _Child(action_id, partial, child_state, child_queues, False)
                    # Round boundary: this move emptied both hands, so the child is
                    # the start of the next round. Score it, but don't search past it.
                    if self._BOUNDS_BY_ROUND and child_state.round_number > self._root_round:
                        child.leaf = True
                    children.append(child)

            pending = [c for c in children if not c.terminal]
            if pending:
                values = self._critic_root_values([c.state for c in pending], root_player)
                for c, v in zip(pending, values):
                    # Mostly critic, blended with `_leaf_potential`: the
                    # critic is only ever calibrated to a *moment-matched*
                    # scale (see `_calibrate_value_scale` — the checkpoint has
                    # no ground truth to denormalise against), so blending in
                    # the heuristic that `_minimax` already relies on
                    # successfully hedges against the critic's own directional
                    # accuracy being noisier than a fully-trained value
                    # function's would be (this checkpoint is a 1500-episode
                    # run — see module docstring). 0.7/0.3 measured best
                    # against LookaheadBot (swept 1.0/0.5/0.4/0.3/0.2/0.0
                    # critic weight; both a pure critic and a pure heuristic
                    # scored markedly worse than this blend).
                    heur = self._leaf_potential(c.state, root_player)
                    c.est = c.partial_value + self.gamma ** (ply + 1) * (0.7 * v + 0.3 * heur)

            children.sort(key=lambda c: c.est, reverse=maximizing)
            survivors = children[:self._beam_width_at(ply)]
            self._survivor_cache[path] = (survivors, maximizing)

        if depth <= 0:
            # Round-bounded search: if a survivor is still mid-round (not a scored
            # leaf, not terminal), the depth limit — not the round's end — stopped
            # us here, so this pass did not analyse the round to completion.
            if self._BOUNDS_BY_ROUND and any(not (c.terminal or c.leaf) for c in survivors):
                self._round_incomplete = True
            best = survivors[0]
            return best.est, best.action_id

        best_val, best_action = None, None
        for c in survivors:
            if c.terminal or c.leaf:
                val = c.est
            else:
                val = self._beam_value(
                    c.state, c.queues, root_player, depth - 1, deadline, ply + 1, path + (c.action_id,))[0]
            if best_val is None or (maximizing and val > best_val) or (not maximizing and val < best_val):
                best_val, best_action = val, c.action_id
        return best_val, best_action

    def _critic_values_raw(self, states):
        """Batched raw `Critic.value_batch` output — normalised scale, see
        `_calibrate_value_scale`; not yet corrected, not yet sign-flipped.
        """
        boards, globals_, privs = [], [], []
        for state in states:
            self._sim_env.set_state(state)
            obs = self._sim_env.generate_observation()
            boards.append(obs['board'])
            globals_.append(obs['global'])
            privs.append(self._sim_env.get_privileged_features())
        batch = {
            'board': torch.from_numpy(np.stack(boards)).to(self.device),
            'global': torch.from_numpy(np.stack(globals_)).to(self.device),
            'opp_onehot': self._opp_onehot.unsqueeze(0).expand(len(states), -1),
            'privileged': torch.from_numpy(np.stack(privs)).to(self.device),
        }
        with torch.inference_mode():
            return self._critic.value_batch(batch).cpu().numpy()

    def _critic_root_values(self, states, root_player):
        """Root-perspective critic value for each state — a single batched
        forward pass, rescaled onto real reward units (`_calibrate_value_scale`)
        then sign-flipped (see module docstring for the negamax convention).
        """
        raw = self._critic_values_raw(states)
        movers = [state.active_player for state in states]
        values = raw * self._value_scale + self._value_shift
        return [v if m == root_player else -v for v, m in zip(values, movers)]
