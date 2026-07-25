"""Shallow (2-ply-in-turns) forward-simulation greedy bot.

`GreedyBot` (greedy_bot.py) is obs-only and hand-blind: it reads the ego-centric
observation, follows a fixed priority ladder (attack > control > move > deploy >
pass), takes the *first* action in each bucket, and — because it can't simulate —
never recruits, bolsters, claims initiative, or initiates a tactic. That leaves
half the roster (Archer/Lancer, whose only attack is a tactic; Cavalry, Ensign,
Marshall, …) unusable and, on flat positions, makes it pick the same move
`LookaheadBot`'s ordering does (see docs/bots.md).

`SimGreedyBot` stays deliberately shallow but plays the real game: it enumerates
the *real* legal action set (via the live env, like `LookaheadBot`), plays out
each candidate as a whole turn (action + any pending tactic continuation), then
lets the opponent play their single best whole turn against it, and scores the
result with the same `HeuristicEvaluator` the deeper search uses at its leaf. The
opponent-reply ply is what separates it from a pure 1-ply greedy — without it,
the bot walks into free recaptures and, because the leaf potential is a shaping
quantity rather than a bounded value, prefers stalling to actually winning. That
env-aware scoring:

  - covers every verb (recruit/bolster/claim_initiative/tactic) for free — they
    are just legal actions that now get scored on their simulated consequence;
  - picks the *best* attack/control/deploy, not the first, and gets Pikeman
    counter-coins, the Knight bolster-gate, recapture, etc. right automatically,
    because the resulting state already reflects them;
  - resolves pending tactic continuations greedily before scoring (Archer's
    initiate→select, Cavalry's move→attack), so initiating a tactic is valued by
    what it accomplishes rather than by the half-finished pending state — without
    which even a 2-ply bot would still shun tactics.

It is a subclass of `LookaheadBot` purely to reuse its rules-engine plumbing
(`_prepare_root`, `_apply`, `_legal_from`, reward accounting, the shared
evaluator); it overrides `act()` with a fixed 2-ply-in-turns argmax/minimax and
does not use the parent's iterative-deepening `_minimax` (so `greedy` stays a
shallow fixed-depth bot, distinct from the budget-driven, deep `lookahead`). It is
heavier than the obs-only `GreedyBot` (many clone+applies per move), which is why
the cheap obs-only one is kept as the training-loop opponent; this one is the
gauntlet's `greedy` yardstick.
"""
import math
from collections import Counter

from .lookahead_bot import LookaheadBot, _clone_state
from ..environment.warchest_env import (
    WarChestEnv, SPATIAL_SIZE, CONTROL_VERB, BOLSTER_VERB, DEPLOY_VERB_BASE, TACTIC_VERB,
)


class SimGreedyBot(LookaheadBot):
    """Shallow 2-ply-in-turns greedy over the true legal action set."""

    # A game-ending win/loss must dominate any positional heuristic. The leaf
    # potential (base PBRS especially) is a *shaping* quantity, not a bounded value
    # estimate — with a large base lead it can numerically exceed WIN_REWARD, so a
    # plain `max` over "win now (=1.0)" vs "hold a 5-base position (~1.16)" would
    # pick the stall. The 2-ply opponent reply mitigates this (a reply that
    # recaptures drops the stall's value) but doesn't fully cover it — when the
    # opponent has no punishing move, the stall's Φ would still edge out the win —
    # so terminal outcomes are made to strictly dominate. Sign: win (+) / loss (-).
    _TERMINAL_VALUE = 1e6

    def __init__(self, gamma=0.99, shaping_anneal=1.0, see_opponent_hand=True,
                 rich_eval=False, reply_branching=8, name='greedy'):
        # max_branching=0 (no cap) / time_budget=0.0 / max_depth=1 are inert here —
        # act() is overridden and drives its own fixed 2-ply-in-turns search, never
        # _minimax — but they keep the parent constructor (which builds _sim_env and
        # the shared evaluator) happy.
        # rich_eval defaults OFF: the extra durability/economy/tempo/progress terms
        # were measured net-harmful here too (economy in particular made the bot
        # spam recruit — ~1/3 of its moves — for a long-horizon payoff a shallow
        # search can't cash in, dropping its win rate vs the obs GreedyBot from ~48%
        # to ~40%). Features are still used, driven by simulated consequence, not by
        # standing eval bonuses. See docs/bots.md.
        super().__init__(time_budget=0.0, max_branching=0, see_opponent_hand=see_opponent_hand,
                         max_depth=1, gamma=gamma, shaping_anneal=shaping_anneal,
                         rich_eval=rich_eval, name=name)
        # Cap on the opponent's candidate replies considered at the 2nd ply, taken
        # after LookaheadBot's cheap ordering key (attacks/captures first). Only the
        # *reply* is capped — root's own actions are always considered in full, so no
        # feature (recruit/bolster/tactic/initiative) is ever pruned from the bot's
        # own choice. The reply only needs to surface the punishing move, which the
        # ordering key ranks first anyway.
        self.reply_branching = reply_branching
        # Rolling count of which verb class the bot actually committed to, for the
        # "does it use the whole game now?" validation (docs/bots.md). Never read by
        # the engine; purely diagnostic.
        self.usage = Counter()

    def act(self, env) -> int:
        """Return an absolute-frame action id for `env.active_player`.

        Fixed 2-ply-in-turns search: for every legal root action, play out the
        bot's whole turn (the action plus any pending tactic continuation) to a
        quiescent state, then let the opponent play *their* best whole turn against
        it, and score the result. Root maximizes; the opponent reply minimizes.
        """
        root_player = env.active_player
        legal = env.get_possible_actions()
        if len(legal) <= 1:
            return legal[0]

        root_state, root_queues = self._prepare_root(env, root_player)
        best_action = legal[0]
        best_val = -math.inf
        for action_id in legal:
            state = _clone_state(root_state)
            queues = {1: list(root_queues[1]), 2: list(root_queues[2])}
            val = self._value_after_my_turn(state, queues, action_id, root_player)
            if val > best_val:
                best_val, best_action = val, action_id

        self.usage[self._classify(best_action)] += 1
        self.last_stats = {
            'legal_at_root': len(legal),
            'best_value': best_val,
            'chosen': best_action,
        }
        return best_action

    def _value_after_my_turn(self, state, queues, action_id, root_player):
        """Apply `action_id`, resolve the bot's pending sub-turn, then subtract the
        opponent's best reply. Mutates (state, queues). Returns the root-perspective
        value of committing to `action_id`.
        """
        result = self._apply(state, queues, action_id)
        own = result.player_id == root_player
        if result.finishes_game:
            return math.copysign(self._TERMINAL_VALUE, result.reward if own else -result.reward)
        reward = self._own_action_reward(result) if own else 0.0

        acc, terminal = self._resolve_pending(state, queues, root_player)
        reward += acc
        if terminal is not None:
            return terminal  # bot won during its own tactic — dominating ±_TERMINAL_VALUE
        if state.round_number >= self._sim_env.max_rounds:
            return reward + self._truncation_value(state, root_player)

        holding = self._holding_reward(state, root_player)
        return reward + holding + self._opponent_best_reply(state, queues, root_player)

    def _opponent_best_reply(self, state, queues, root_player):
        """Value of the state after the mover (normally the opponent) plays their
        single best whole turn — the 2nd ply. The opponent minimizes the root's
        eval; on the rare node where it is still root's turn, root maximizes. Only
        the top `reply_branching` candidates (by ordering key) are examined.
        """
        mover = state.active_player
        legal = self._legal_from(state)
        if not legal:
            return self._leaf_potential(state, root_player)
        dist_grid = self._dist_grid_to_targets(self._capturable_bases(mover))
        melee = self._melee_threatened_cells(mover)
        ordered = sorted(legal, key=lambda a: self._ordering_key(a, dist_grid, melee, mover))
        if self.reply_branching:
            ordered = ordered[:self.reply_branching]

        maximizing = (mover == root_player)
        best = -math.inf if maximizing else math.inf
        for action_id in ordered:
            s2 = _clone_state(state)
            q2 = {1: list(queues[1]), 2: list(queues[2])}
            val = self._reply_value(s2, q2, action_id, root_player)
            best = max(best, val) if maximizing else min(best, val)
        return best

    def _reply_value(self, state, queues, action_id, root_player):
        """Apply one 2nd-ply action, resolve its pending sub-turn, score the leaf."""
        result = self._apply(state, queues, action_id)
        own = result.player_id == root_player
        if result.finishes_game:
            return math.copysign(self._TERMINAL_VALUE, result.reward if own else -result.reward)
        reward = self._own_action_reward(result) if own else 0.0
        acc, terminal = self._resolve_pending(state, queues, root_player)
        reward += acc
        if terminal is not None:
            return terminal
        if state.round_number >= self._sim_env.max_rounds:
            return reward + self._truncation_value(state, root_player)
        return reward + self._leaf_potential(state, root_player)

    def _resolve_pending(self, state, queues, root_player):
        """Play out any pending tactic continuation until it clears (or the game
        ends). At each step the mover picks the continuation best *for the mover*:
        root maximizes its own leaf, the opponent minimizes it. Mutates (state,
        queues).

        Returns (accumulated_own_reward, terminal_value); terminal_value is None
        unless a continuation ended the game. Greedy-per-step, not an optimal
        continuation search — fine for the short (1-2 click) tactic sub-turns.
        """
        acc = 0.0
        while state.pending is not None:
            maximize = (state.active_player == root_player)
            cont = self._legal_from(state)
            if not cont:
                break
            best_c = cont[0]
            best_score = -math.inf if maximize else math.inf
            for c in cont:
                s2 = _clone_state(state)
                q2 = {1: list(queues[1]), 2: list(queues[2])}
                r2 = self._apply(s2, q2, c)
                own2 = r2.player_id == root_player
                if r2.finishes_game:
                    score = math.copysign(self._TERMINAL_VALUE, r2.reward if own2 else -r2.reward)
                else:
                    score = (self._own_action_reward(r2) if own2 else 0.0) \
                        + self._leaf_potential(s2, root_player)
                if (maximize and score > best_score) or (not maximize and score < best_score):
                    best_score, best_c = score, c
            r = self._apply(state, queues, best_c)
            own = r.player_id == root_player
            if r.finishes_game:
                return acc, math.copysign(self._TERMINAL_VALUE, r.reward if own else -r.reward)
            acc += self._own_action_reward(r) if own else 0.0
        return acc, None

    @staticmethod
    def _classify(action_id):
        """Verb class of a committed action, for the usage counter."""
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
