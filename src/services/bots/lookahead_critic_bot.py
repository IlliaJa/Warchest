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
import random
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

DEFAULT_CRITIC_PATH = 'data/lookahead_critic/lookahead_critic_v1.pth'


class _Child:
    """One candidate move out of a node: the state it leads to (None if the
    game/round ended, since nothing further can be simulated from it) plus
    enough of the reward path to finish scoring it once `est` is filled in.
    """

    __slots__ = ('action_id', 'partial_value', 'state', 'queues', 'terminal', 'est')

    def __init__(self, action_id, partial_value, state, queues, terminal):
        self.action_id = action_id
        self.partial_value = partial_value
        self.state = state
        self.queues = queues
        self.terminal = terminal
        self.est = partial_value if terminal else None


class LookaheadCriticBot(LookaheadBot):
    """Beam-limited search over `WarChestEnv` states, scored by a trained `Critic`.

    Args:
        critic_path: path to a checkpoint saved by
            `policy.checkpoint.save_critic_checkpoint`. Defaults to the critic
            saved alongside the 2026-07-07 1500-episode PPO run.
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
        device: torch device for the critic's forward passes.
    """

    def __init__(self, critic_path=DEFAULT_CRITIC_PATH, beam_width=5, max_branching=8,
                 time_budget=0.5, see_opponent_hand=True, max_depth=40, gamma=0.99,
                 opp_type='pool', device='cpu', name='lookahead_critic'):
        super().__init__(time_budget=time_budget, max_branching=max_branching,
                          see_opponent_hand=see_opponent_hand, max_depth=max_depth,
                          gamma=gamma, shaping_anneal=1.0, name=name)
        self.beam_width = beam_width
        self.device = device

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
        self._calibrate_value_scale()

    def _calibrate_value_scale(self, n_games=8, n_samples=160, seed=12345):
        """One-time affine fit recovering the critic's real reward-scale.

        `Critic.value_batch` was trained against *normalised* returns (ppo.py's
        `ReturnNormalizer`: an EMA of return mean/std, used to keep the critic's
        loss scale stable — see its docstring). `ppo.py` always denormalises
        (`value * std + mean`) before treating the critic's output as a real
        value anywhere (rollout bootstrapping, GAE). That EMA is training-loop
        state, never written to the checkpoint (`checkpoint.py` only saves
        `state_dict`/`obs_version`/`arch`/`hidden_dim`), so the exact
        denormalisation used when this checkpoint was saved can't be recovered
        — feeding the network's raw output straight into `_beam_value`, which
        sums it with real reward-scale path returns, was giving the critic's
        contribution an arbitrary, depth-dependent weight relative to the real
        rewards it's added to (nodes reached via more/fewer real-reward-bearing
        steps ended up on incommensurable scales — confirmed to be the fix that
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
        root_player = env.active_player
        legal = env.get_possible_actions()
        if len(legal) <= 1:
            return legal[0]

        root_state, root_queues = self._prepare_root(env, root_player)
        start = time.monotonic()
        deadline = start + self.time_budget
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
        self.last_stats = {
            'depth_reached': depth_reached,
            'nodes_visited': self._nodes_visited,
            'elapsed': time.monotonic() - start,
            'legal_at_root': len(legal),
            'best_value': best_val,
        }
        return best_action

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
                dist_grid = self._dist_grid_to_targets(self._capturable_bases(mover))
                melee_threat = self._melee_threatened_cells(mover)
                legal = sorted(legal, key=lambda a: self._ordering_key(a, dist_grid, melee_threat, mover))
                legal = legal[:max_branching]

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
                    children.append(_Child(action_id, partial, child_state, child_queues, False))

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
            best = survivors[0]
            return best.est, best.action_id

        best_val, best_action = None, None
        for c in survivors:
            val = c.est if c.terminal else self._beam_value(
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
