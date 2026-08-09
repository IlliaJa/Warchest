"""Alpha-beta search bot, 5-10 plies deep (docs/lookahead_bot_plan.md).

Unlike GreedyBot/RandomBot (`Bot.act(obs)` — ego-centric, hand-blind), this bot needs
the real `GameState` to forward-simulate, so it takes the live `env` directly, mirroring
`GauntletAgent.act(env)` in `gauntlet.py` rather than the `Bot` ABC in this package.
`Bot` and `GauntletAgent` are expected to merge into one interface later; this lives
here because it belongs with the other bots, not because it satisfies `Bot` today.

Design decisions (see the plan doc for the full rationale):
  - Ply = one action id (a tactic's continuation clicks each count separately).
  - Depth is not fixed: iterative deepening bounded by a wall-clock budget, since this
    bot is meant to be queried repeatedly during training rollout collection and must
    stay cheap per call.
  - Future draws: both players' bag composition is exact (not hidden), so the only
    remaining randomness is draw *order*. One full order is sampled once per `act()`
    call ("single determinization") and reused deterministically for every branch
    explored in that call — cheaper than resampling per node, and correct because
    sibling branches that haven't consumed a draw yet still share the same unconsumed
    future. This also transparently covers mid-round chance draws (e.g. Warrior
    Priest's bonus coin), since they go through the same patched draw call.
  - Opponent hand visibility is a constructor flag: the "fair" mode does not use the
    real hidden hand, it pools the opponent's hand + bag (both hidden) and re-splits
    them randomly, then determinizes future draws from the resulting bag exactly like
    the cheat mode — one mechanism for both, rather than a separate estimator.
  - Value function: the reward that matters is split across *two* places in this repo
    — the environment (`Action.reward`: win/loss, attack, invalid, per-move penalty)
    and the training loop (`rollout_core.py`: base + material PBRS, holding reward).
    A search that only replays the PBRS half (as an earlier version of this file did)
    throws away half of the signal the policy is actually trained on. This version
    accumulates the real `Action.reward` along the path plus the same holding reward,
    added to the same PBRS potential at the leaf, all gamma-discounted by ply — i.e.
    the same objective `rollout_core.play_episode` computes for the main actor,
    reused rather than re-derived. See docs/lookahead_bot_plan.md (2026-07-06 update).
  - Positional term: most turns in a game have no reward-bearing move within the
    search horizon at all (no attack/claim reachable yet), so the leaf potential above
    is flat and ties get broken by move-ordering alone. GreedyBot's one real edge over
    a plain material/base heuristic is that it always marches toward the nearest
    capturable base (BFS) even with no lookahead; without an equivalent, this search's
    tie-break has no spatial sense at all, which is strictly worse. `_nearest_dist`
    reproduces that same directional pull as a leaf-potential term. (`docs/rewards.md`
    rejected this exact idea for the *trained policy's reward* — BFS-per-training-step
    cost and a farming-exploit risk for a *learned* policy — neither applies to a
    per-decision search heuristic with no gradient to exploit.)
"""
import copy
import math
import random
import time
import types
from collections import Counter

import numpy as np

from ..environment.warchest_env import (
    WarChestEnv, SPATIAL_SIZE, LOSS_REWARD,
    CONTROL_VERB, BOLSTER_VERB, DEPLOY_VERB_BASE, TACTIC_VERB,
    RECRUIT_ACTION, CLAIM_INITIATIVE_ACTION, PASS_ACTION, DECLINE_ACTION,
)
from ..environment.game_state import Pending
from ..environment.board import Board
from ..environment.cell_ids import (
    CONTROLLED_BASE_PLAYER_1_CELL_ID, CONTROLLED_BASE_PLAYER_2_CELL_ID,
)
from ..environment.roster import UNIT_BY_ID
from .evaluation import HeuristicEvaluator, capturable_bases


def _fast_counter_copy(counter):
    """Shallow-copy a `Counter`, bypassing `Counter.__init__`/`update`'s generic
    argument-type dispatch (iterable vs. mapping vs. kwargs) — ~3.5x faster for the
    small (few keys) counters `GameState` uses, measured with `timeit`. Every input
    here is always already a `Counter`, so there's no dispatch to actually do.
    """
    new = Counter.__new__(Counter)
    dict.update(new, counter)
    return new


def _clone_state(state):
    """Hand-rolled GameState clone, replacing generic `deepcopy` on the search's hot
    path (`_minimax` calls this once per node — profiling showed it as the dominant
    per-node cost, ~0.3-0.5ms, capping the whole search to a few hundred nodes even at
    a 0.5s budget). `deepcopy`'s generic memoised walk pays object-identity bookkeeping
    for a structure that's actually shallow: a handful of int-keyed `Counter`s, a
    `Board` wrapping a numpy array + a flat list of `BaseUnit` (each holding only
    immutable/value fields, plus a back-reference to its `Board` that must be
    repointed — see `BaseUnit.board`, otherwise unused but kept correct here).
    `compositions` is fixed at game start and never mutated in place, so it's aliased
    rather than copied.
    """
    new_board = Board.__new__(Board)
    new_board.size = state.board.size
    new_board.board_size = state.board.board_size
    new_board.board = state.board.board.copy()
    new_units = []
    for u in state.board.units:
        nu = copy.copy(u)
        nu.board = new_board
        new_units.append(nu)
    new_board.units = new_units

    new_state = copy.copy(state)
    new_state.board = new_board
    fc = _fast_counter_copy
    new_state.bags = {1: fc(state.bags[1]), 2: fc(state.bags[2])}
    new_state.hands = {1: fc(state.hands[1]), 2: fc(state.hands[2])}
    new_state.discard_faceup = {1: fc(state.discard_faceup[1]), 2: fc(state.discard_faceup[2])}
    new_state.discard_facedown = {1: fc(state.discard_facedown[1]), 2: fc(state.discard_facedown[2])}
    new_state.supply = {1: fc(state.supply[1]), 2: fc(state.supply[2])}
    new_state.boxed = {1: fc(state.boxed[1]), 2: fc(state.boxed[2])}
    if state.pending is not None:
        p = state.pending
        new_state.pending = Pending(kind=p.kind, unit_loc=p.unit_loc, optional=p.optional,
                                     data=dict(p.data))
    return new_state


class _TimeUp(Exception):
    """Internal signal that the wall-clock search budget has been exhausted."""


def _determinized_draw_one(sim_env, player):
    """Replacement for `WarChestEnv._draw_one`, bound onto the search's sim env.

    Pops the next coin from the pre-shuffled per-node determinization queue instead
    of drawing fresh randomness, so every branch explored within one `act()` call sees
    the same future draw order. Falls back to a genuine random draw (matching the real
    rules engine) if the queue runs dry — only possible on searches deep enough to
    exhaust a player's whole current bag, which 5-10 plies essentially never reaches.
    """
    queue = sim_env._sim_draw_queues.get(player)
    if queue:
        coin = queue.pop()
    else:
        coins = list(sim_env.state.bags[player].elements())
        coin = int(np.random.choice(coins))
    sim_env.state.bags[player][coin] -= 1
    if sim_env.state.bags[player][coin] == 0:
        del sim_env.state.bags[player][coin]
    return coin


_FACEDOWN_PRIORITY = {
    RECRUIT_ACTION: 5,
    CLAIM_INITIATIVE_ACTION: 6,
    PASS_ACTION: 7,
    DECLINE_ACTION: 7,
}


class LookaheadBot:
    """GreedyBot's lookahead cousin: alpha-beta search bounded by a wall-clock budget.

    Args:
        time_budget: seconds allotted per `act()` call (iterative deepening stops here).
        max_branching: cap on candidate actions expanded per node, after move ordering.
        see_opponent_hand: if True (default — the stress-test / training-pressure mode),
            reads the opponent's real hand. If False, pools the opponent's hand and bag
            (both otherwise hidden) and re-splits them randomly before searching, so the
            bot never actually looks at hidden information.
        max_depth: hard ply cap regardless of remaining time budget (safety net).
        gamma: per-ply discount, matching ppo.py's training gamma (0.99 there).
        shaping_anneal: multiplier on the material-PBRS + holding terms (rollout_core.py
            anneals these down over training as the critic gets capable; this search has
            no critic to fall back on, so it defaults to 1.0 — full dense guidance).
        name: label used when this bot is dropped into gauntlet.py's round-robin.
    """

    # Distance used by move ordering when a side has no on-board unit or no
    # capturable base exists — larger than any real hex distance on the 7x7 board
    # so it always reads "far". (The leaf's own copy lives on HeuristicEvaluator.)
    _FAR_DIST = 12

    def __init__(self, time_budget=0.1, max_branching=8, see_opponent_hand=True,
                 max_depth=40, gamma=0.99, shaping_anneal=1.0, rich_eval=False,
                 name='lookahead', theta=None):
        self.time_budget = time_budget
        self.max_branching = max_branching
        self.see_opponent_hand = see_opponent_hand
        self.max_depth = max_depth
        self.gamma = gamma
        self.shaping_anneal = shaping_anneal
        self.name = name
        # Static leaf evaluation is delegated to the shared HeuristicEvaluator so
        # SimGreedyBot's shallow scoring and this search agree on "how good is a
        # state". `rich_eval` turns on the extra durability/economy/tempo/progress
        # terms; it defaults OFF because measurement showed them net-harmful (they
        # reward long-horizon things a search leaf can't cash in — a rich=True
        # lookahead scored only 20% vs the same bot with rich=False; see
        # docs/bots.md). Off also reproduces the exact old `_leaf_potential`
        # formula, which LookaheadCriticBot's value-scale calibration depends on.
        # `theta` (docs/IDEAS.md B1) re-weights all eight coefficients at once and subsumes
        # `rich_eval`; theta=None keeps the historical rich_eval behaviour exactly, and
        # passing both raises rather than silently letting one win.
        self._evaluator = HeuristicEvaluator(shaping_anneal=shaping_anneal,
                                             enable_new_terms=rich_eval, theta=theta)
        # A plain rules-engine instance reused across act() calls purely for forward
        # simulation — never exposed to or stepped by the caller.
        self._sim_env = WarChestEnv(save_game_history=False)
        self._sim_env._draw_one = types.MethodType(_determinized_draw_one, self._sim_env)
        self._sim_env._sim_draw_queues = {1: [], 2: []}
        # docs/rewards.md: holding_reward = rate * (my_bases - opp_bases), fired every
        # main-actor ply. Shared derivation with ppo.py, so the two cannot drift.
        self._holding_reward_rate = self._sim_env.default_holding_reward_rate()
        # Diagnostics from the most recent act() call (docs/lookahead_bot_plan.md).
        self.last_stats = {}
        # Rolling count of which verb class this bot actually committed to — the "does it
        # use the whole game?" check (docs/bots.md), and the behaviour profile the θ-family
        # measurement reads (src/app/eval_theta_family.py). Lives here rather than on
        # SimGreedyBot so both search depths report it the same way. Never read by the
        # engine; purely diagnostic. Forced moves (one legal action) are deliberately not
        # counted — the profile is what the bot *chose*, not what the rules left it.
        self.usage = Counter()

    def act(self, env) -> int:
        """Return an absolute-frame action id for `env.active_player`."""
        root_player = env.active_player
        legal = env.get_possible_actions()
        if len(legal) <= 1:
            return legal[0]

        root_state, root_queues = self._prepare_root(env, root_player)

        start = time.monotonic()
        deadline = start + self.time_budget
        best_action = legal[0]
        best_val = None
        depth = 1
        depth_reached = 0
        self._nodes_visited = 0
        while depth <= self.max_depth:
            try:
                val, action = self._minimax(root_state, root_queues, root_player,
                                             depth, -math.inf, math.inf, deadline, ply=0)
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
        self.usage[self._classify(best_action)] += 1
        return best_action

    @staticmethod
    def _classify(action_id):
        """Verb class of a committed action, for the `usage` counter."""
        if action_id >= SPATIAL_SIZE:
            kind, _ = WarChestEnv.decode_facedown(action_id)
            return kind  # 'recruit' / 'claim_initiative' / 'pass' / 'decline'
        verb, _, _ = WarChestEnv.decode_action(action_id)
        if 6 <= verb <= 11:
            return 'attack'
        if verb == CONTROL_VERB:
            return 'control'
        if verb == BOLSTER_VERB:
            return 'bolster'
        if DEPLOY_VERB_BASE <= verb < TACTIC_VERB:
            return 'deploy'
        if verb == TACTIC_VERB:
            return 'tactic'
        if verb <= 5:
            return 'move'
        return 'select'

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _prepare_root(self, env, root_player):
        """Clone the real state and sample one determinization of future draws.

        In fair mode, the opponent's hidden coins are pooled and re-split first, so the
        rest of the pipeline (draw-queue sampling) is identical in both modes.
        """
        state = _clone_state(env.state)
        opp = 3 - root_player
        if not self.see_opponent_hand:
            self._resplit_hidden(state, opp)
        queues = {
            root_player: self._shuffled(state.bags[root_player]),
            opp: self._shuffled(state.bags[opp]),
        }
        return state, queues

    @staticmethod
    def _resplit_hidden(state, opp):
        """Re-deal the opponent's hidden coins uniformly over their THREE hidden piles.

        Hand, bag and face-down discard are all hidden; only their sizes and their
        union are public (docs/search_under_uncertainty.md §1.1) — the bag's contents
        are *not* public, contrary to what this file assumed for a long time. The
        previous version pooled only `hands + bags`, which left `discard_facedown` at
        its true value: "fair" mode still knew exactly which coins the opponent had
        buried face down, and therefore re-split the correct pool. That understates
        how blind a genuinely blind bot is, and biases
        `src/app/eval_info_value.py` toward measuring no information gap at all.

        The union is preserved, so `_reshuffle` (which merges both discard piles back
        into the bag) is unaffected in aggregate.
        """
        hand_size = sum(state.hands[opp].values())
        fd_size = sum(state.discard_facedown[opp].values())
        pool = list((state.hands[opp] + state.bags[opp]
                     + state.discard_facedown[opp]).elements())
        random.shuffle(pool)
        state.hands[opp] = Counter(pool[:hand_size])
        state.discard_facedown[opp] = Counter(pool[hand_size:hand_size + fd_size])
        state.bags[opp] = Counter(pool[hand_size + fd_size:])

    @staticmethod
    def _shuffled(counter):
        pool = list(counter.elements())
        random.shuffle(pool)
        return pool

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _minimax(self, state, queues, root_player, depth, alpha, beta, deadline, ply):
        """Total gamma^ply-discounted, root_player-perspective return from `state`
        onward, searching `depth` more plies.

        Plain minimax, not negamax: turns don't strictly alternate (pending tactic
        continuations and empty-hand skips both keep the same active player across
        plies), so whose turn it is is read fresh from `state` at every node rather
        than assumed from the recursion depth's parity.

        The return value is `sum of discounted per-ply rewards along the path so far`
        + `discounted leaf potential` — i.e. every node returns a value already
        expressed in root-time units, so callers just add their own ply's reward and
        pass the sum up (see rollout_core.py's `shaped_reward`, which this mirrors).
        """
        if time.monotonic() >= deadline:
            raise _TimeUp
        self._nodes_visited += 1
        if depth == 0:
            return self._leaf_potential(state, root_player) * self.gamma ** ply, None

        legal = self._legal_from(state)
        mover = state.active_player
        dist_grid = self._dist_grid_to_targets(self._capturable_bases(mover))
        melee_threat = self._melee_threatened_cells(mover)
        ordered = sorted(legal, key=lambda a: self._ordering_key(a, dist_grid, melee_threat, mover))
        if self.max_branching and len(ordered) > self.max_branching:
            ordered = ordered[:self.max_branching]

        maximizing = (state.active_player == root_player)
        # Holding reward (docs/rewards.md): fires every root_player ply, a function of
        # `state` (before the move), so it's the same additive term for every child
        # considered here — still meaningful once summed over a whole path, since
        # different lines reach different base counts at different plies.
        holding = self._holding_reward(state, root_player) if maximizing else 0.0
        discount = self.gamma ** ply

        best_val = -math.inf if maximizing else math.inf
        best_action = None
        for action_id in ordered:
            child_state = _clone_state(state)
            child_queues = {1: list(queues[1]), 2: list(queues[2])}
            result = self._apply(child_state, child_queues, action_id)
            own_action = (result.player_id == root_player)
            # Non-terminal per-action reward (attack/invalid/move-penalty) is only
            # attributed on root_player's own plies — mirroring rollout_core.py, which
            # never tracks the opponent's per-action reward for the main actor either
            # (the opponent's impact instead shows up via the state-based potential).
            step_reward = (self._own_action_reward(result) if own_action else 0.0) + holding
            if result.finishes_game:
                # rollout_core.py folds LOSS_REWARD onto the main actor when the
                # opponent's move ends the game — mirrored (not silent) here, unlike
                # the non-terminal case above, since that's what it actually does.
                step_reward = result.reward if own_action else -result.reward
                future = 0.0
            elif child_state.round_number >= self._sim_env.max_rounds:
                future = self._truncation_value(child_state, root_player) * self.gamma ** (ply + 1)
            else:
                future, _ = self._minimax(child_state, child_queues, root_player,
                                           depth - 1, alpha, beta, deadline, ply + 1)
            val = discount * step_reward + future
            if maximizing:
                if val > best_val:
                    best_val, best_action = val, action_id
                alpha = max(alpha, best_val)
            else:
                if val < best_val:
                    best_val, best_action = val, action_id
                beta = min(beta, best_val)
            if beta <= alpha:
                break  # alpha-beta cutoff
        return best_val, best_action

    @staticmethod
    def _own_action_reward(result):
        """`result.reward`, minus the tiny per-turn tempo cost.

        `TURN_TEMPO_REWARD` only makes sense against a long horizon plus a
        bootstrapped value function that can see the eventual payoff of advancing
        (the trained policy has a critic for this). This search has neither and is
        depth-bounded to a handful of plies, so accumulating it verbatim makes
        "don't move" look better than "advance to fight" — the opposite of what a
        nudge against stalling is meant to do once the offsetting long-run reward
        falls outside the search horizon.

        Read off `tempo_cost` rather than compared against the constant: since the
        cost moved to the turn boundary it rides on top of every turn-ending reward
        instead of standing alone on plain moves, so equality no longer identifies it.
        """
        return result.reward - result.tempo_cost

    def _legal_from(self, state):
        self._sim_env.set_state(state)
        return self._sim_env.get_possible_actions()

    def _apply(self, state, queues, action_id):
        self._sim_env.set_state(state)
        self._sim_env._sim_draw_queues = queues
        return self._sim_env._apply_action(action_id)

    def _holding_reward(self, state, root_player):
        self._sim_env.set_state(state)
        opp = 3 - root_player
        base_diff = (len(self._sim_env.board.get_controlled_bases(root_player))
                     - len(self._sim_env.board.get_controlled_bases(opp)))
        return self.shaping_anneal * self._holding_reward_rate * base_diff

    def _leaf_potential(self, state, root_player):
        """Static leaf value — delegated to the shared `HeuristicEvaluator` (see
        `evaluation.py`). Set the sim env to `state` first, then evaluate against
        whatever `_sim_env` currently is (LookaheadCriticBot swaps it out for one
        built against the critic's obs version, so we read the attribute live rather
        than capturing it in the evaluator).
        """
        self._sim_env.set_state(state)
        return self._evaluator.evaluate(self._sim_env, root_player)

    def _capturable_bases(self, player):
        """Cells `player` could still usefully capture — delegates to the shared
        `capturable_bases` (used here for `_ordering_key`; also called by the critic
        bots). Same notion of "target" the leaf's `_nearest_dist` uses.
        """
        return capturable_bases(self._sim_env.board, player)

    def _dist_grid_to_targets(self, targets):
        """{cell: hex-distance to the nearest cell in `targets`}, for every cell
        reachable from them — single multi-source BFS instead of one BFS per
        candidate move, so ordering a node's children costs O(board size) once,
        not O(board size) per candidate.
        """
        if not targets:
            return {}
        dist = {t: 0 for t in targets}
        frontier = list(targets)
        d = 0
        board = self._sim_env.board
        while frontier:
            d += 1
            nxt = []
            for cell in frontier:
                for nb in board.get_adjacent_cells(*cell):
                    if nb not in dist:
                        dist[nb] = d
                        nxt.append(nb)
            frontier = nxt
        return dist

    def _melee_threatened_cells(self, mover):
        """Cells adjacent to an opponent unit that could make a normal attack this
        turn — a deliberately cheap (melee-only, no BFS/blocking) approximation of
        danger for move ordering. Full threat accounting (ranged/charge/berserker)
        still happens at the leaf via `_material_at_risk`; this only needs to catch
        the common case cheaply, so an obviously reckless "closer to the objective
        but walks into a free capture" move doesn't rank ahead of a safer one with
        similar distance and eat a branching-cap slot the search then has to spend
        real depth disproving, instead of just not offering it as a good option.
        """
        env = self._sim_env
        opp = 3 - mover
        hand = env.state.hands[opp]
        threatened = set()
        for u in env.board.units:
            if u.player_id != opp or not UNIT_BY_ID[u.id].can_normal_attack:
                continue
            if any(hand[c] >= 1 for c in env.attack_enabler_coins(u)):
                threatened.update(env.board.get_adjacent_cells(*u.loc))
        return threatened

    def _ordering_key(self, action_id, dist_grid, melee_threat, mover):
        """Move-ordering key: (bucket, tie-break...). Buckets mirror GreedyBot's
        priority (attack > control > tactic > move > deploy > bolster > face-down).
        Three buckets are tie-broken by something other than arbitrary iteration
        order:
          - "move": *first* by whether the destination is in `melee_threat` (walking
            into a free capture ranks behind every safe move, regardless of
            distance), *then* by `dist_grid` (distance to the nearest capturable
            base) — the gap profiling found: with >8 legal moves on ~79% of turns
            and a hard branching cap, an arbitrary tie-break inside the largest
            bucket meant the best-looking move often wasn't even a candidate.
          - "attack": kills (target's stack would drop to 0) before mere damage, then
            lower remaining stack first (focus fire on the closest-to-dead target).
          - "control": stealing an enemy-held base before claiming a neutral one — a
            steal swings `base_diff` by 2, a neutral claim only by 1.
        """
        if action_id >= SPATIAL_SIZE:
            kind, _ = WarChestEnv.decode_facedown(action_id)
            return (_FACEDOWN_PRIORITY.get(kind, 7), 0, 0)
        verb, r, q = WarChestEnv.decode_action(action_id)
        board = self._sim_env.board
        if 6 <= verb <= 11:
            dr, dq = board.offsets[verb - 6]
            target = board.get_unit_at(r + dr, q + dq)
            stack = target.stack if target is not None else self._FAR_DIST
            return (0, stack, 0)  # a kill (stack==1) sorts first, then lowest stack
        if verb == CONTROL_VERB:
            opp_cell_id = (CONTROLLED_BASE_PLAYER_2_CELL_ID if mover == 1
                           else CONTROLLED_BASE_PLAYER_1_CELL_ID)
            stealing = board.board[r, q] == opp_cell_id
            return (1, 0 if stealing else 1, 0)
        if verb == TACTIC_VERB:
            return (2, 0, 0)
        if verb <= 5:
            dr, dq = board.offsets[verb]
            dest = (r + dr, q + dq)
            danger = 1 if dest in melee_threat else 0
            return (3, danger, dist_grid.get(dest, self._FAR_DIST))
        if DEPLOY_VERB_BASE <= verb < TACTIC_VERB:
            return (4, 0, 0)
        if verb == BOLSTER_VERB:
            return (5, 0, 0)
        return (6, 0, 0)  # select (only legal mid-tactic)

    def _truncation_value(self, state, root_player):
        """Base-diff-proportional truncation reward (docs/rewards.md, "C17"), for the
        rare case a search branch runs into the round limit within its horizon.
        """
        self._sim_env.set_state(state)
        opp = 3 - root_player
        wbc = self._sim_env.winning_base_count
        diff = (len(self._sim_env.board.get_controlled_bases(root_player))
                - len(self._sim_env.board.get_controlled_bases(opp)))
        if diff > 0:
            return 0.0
        deficit_frac = min(-diff, wbc) / wbc
        return LOSS_REWARD * (0.5 + 0.5 * deficit_frac)
