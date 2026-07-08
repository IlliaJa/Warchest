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

DEFAULT_CRITIC_PATH = 'data/warchest_critic_20260707-0026.pth'


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
        while depth <= self.max_depth:
            try:
                val, action = self._beam_value(root_state, root_queues, root_player,
                                                depth, deadline, ply=0)
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

    def _beam_value(self, state, queues, root_player, depth, deadline, ply):
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
        """
        if time.monotonic() >= deadline:
            raise _TimeUp
        self._nodes_visited += 1

        mover = state.active_player
        legal = self._legal_from(state)
        maximizing = (mover == root_player)
        holding = self._holding_reward(state, root_player) if maximizing else 0.0
        discount = self.gamma ** ply

        if self.max_branching and len(legal) > self.max_branching:
            dist_grid = self._dist_grid_to_targets(self._capturable_bases(mover))
            melee_threat = self._melee_threatened_cells(mover)
            legal = sorted(legal, key=lambda a: self._ordering_key(a, dist_grid, melee_threat, mover))
            legal = legal[:self.max_branching]

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
                c.est = c.partial_value + self.gamma ** (ply + 1) * v

        children.sort(key=lambda c: c.est, reverse=maximizing)
        survivors = children[:self.beam_width]

        if depth <= 0:
            best = survivors[0]
            return best.est, best.action_id

        best_val, best_action = None, None
        for c in survivors:
            val = c.est if c.terminal else self._beam_value(
                c.state, c.queues, root_player, depth - 1, deadline, ply + 1)[0]
            if best_val is None or (maximizing and val > best_val) or (not maximizing and val < best_val):
                best_val, best_action = val, c.action_id
        return best_val, best_action

    def _critic_root_values(self, states, root_player):
        """Root-perspective critic value for each state — a single batched
        forward pass (see module docstring for the negamax sign flip).
        """
        boards, globals_, privs, movers = [], [], [], []
        for state in states:
            self._sim_env.set_state(state)
            obs = self._sim_env.generate_observation()
            boards.append(obs['board'])
            globals_.append(obs['global'])
            privs.append(self._sim_env.get_privileged_features())
            movers.append(state.active_player)
        batch = {
            'board': torch.from_numpy(np.stack(boards)).to(self.device),
            'global': torch.from_numpy(np.stack(globals_)).to(self.device),
            'opp_onehot': self._opp_onehot.unsqueeze(0).expand(len(states), -1),
            'privileged': torch.from_numpy(np.stack(privs)).to(self.device),
        }
        with torch.inference_mode():
            values = self._critic.value_batch(batch).cpu().numpy()
        return [v if m == root_player else -v for v, m in zip(values, movers)]
