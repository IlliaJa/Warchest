"""Round-bounded, policy-guided, critic-scored search — `PolicyCriticBot`'s cousin.

`PolicyCriticBot` (and `LookaheadCriticBot`) search to a fixed *ply* depth, grown
by iterative deepening until the time budget runs out — so how far ahead they see
has nothing to do with the game's structure, only with how much compute fits in
the budget. This bot instead searches to the end of the **current round**: it
keeps recursing while play stays in the same round and stops (critic-scores the
leaf) the moment a move crosses into the next round.

A Warchest round is both players emptying their `HAND_SIZE`-coin hands, one coin
per turn (see `WarChestEnv._advance_turn`/`_start_new_round`). So the natural
horizon shrinks as the round is played out: from the round's first decision the
bot looks over all the coins still to be played this round (its own remaining
coins plus the opponent's — several plies, more when a coin resolves as a
multi-ply tactic); one decision later there is one fewer coin to account for, and
so on down to the last coin of the round. The search leaf is a *whole-round
outcome* evaluated by the critic, not an arbitrary mid-round position.

The search is **unbounded by default** (no time budget): it deepens one ply per
pass and stops as soon as a pass reaches the round's end on every surviving line
(`_round_incomplete` stays False) — deeper passes would be identical. Since a
round is finite, that always happens well within `max_depth` (the only hard
backstop). Iterative deepening is kept only because the per-node survivor cache
makes the re-passes nearly free and it yields the clean `plies_to_round_end`
metric; a finite `time_budget` may still be passed to cap the search, in which
case a cut-off pass is reported as `round_complete=False`.

Candidate selection (the policy move prior) and leaf scoring (the critic, blended
with `_leaf_potential`) are inherited unchanged from `PolicyCriticBot` — the only
difference from it is the search horizon.
"""
import logging
import time

from .lookahead_bot import _TimeUp
from .policy_critic_bot import PolicyCriticBot

logger = logging.getLogger('warchest')


class RoundCriticBot(PolicyCriticBot):
    """`PolicyCriticBot` that searches to the end of the current round.

    Args:
        time_budget: per-move ceiling in seconds, or `None` (default) for no limit
            — the intended mode. With `None` the search always runs to the round's
            end (however long that takes); a finite value caps it, and a cut-off
            move then logs `round_complete=False`.
        max_depth: hard ply cap on the round search — a backstop only. A round is
            at most a few coins per player, so a completed round-analysis stops the
            deepening well before this; the cap just bounds pathological cases
            (e.g. a determinization where hands never empty).
        all other args: as `PolicyCriticBot` / `LookaheadCriticBot`.
    """

    _BOUNDS_BY_ROUND = True

    def __init__(self, policy_path=None, *, critic_path=None, beam_width=5, max_branching=3,
                 time_budget=None, see_opponent_hand=True, max_depth=40, gamma=0.99,
                 opp_type='pool', n_determinizations=1, stats_log_every=20,
                 device='cpu', name='round_critic'):
        # time_budget=None -> unbounded: deadline math in _act_once becomes
        # start + inf, so `_beam_value` never raises _TimeUp and only the round
        # boundary (or max_depth) stops the search.
        super().__init__(policy_path=policy_path, critic_path=critic_path, beam_width=beam_width,
                          max_branching=max_branching,
                          time_budget=(float('inf') if time_budget is None else time_budget),
                          see_opponent_hand=see_opponent_hand, max_depth=max_depth, gamma=gamma,
                          opp_type=opp_type, n_determinizations=n_determinizations,
                          stats_log_every=stats_log_every, device=device, name=name)
        self._last_round_info = None

    def _act_once(self, env, root_player, legal, time_budget):
        """Iterative deepening bounded by the round rather than a fixed depth.

        Identical in shape to `LookaheadCriticBot._act_once`, with two additions:
        `_root_round` is published so `_beam_value` knows where the round boundary
        is, and the deepening stops as soon as a pass analyses the whole round
        (`_round_incomplete` False) instead of only when the budget runs out.
        Records `self._last_round_info` (round-completion / actions-left / budget
        used) for the logs.
        """
        root_state, root_queues = self._prepare_root(env, root_player)
        self._root_round = root_state.round_number
        start = time.monotonic()
        deadline = start + time_budget
        best_action, best_val = legal[0], None
        depth = 0
        depth_reached = -1
        round_complete = False
        self._nodes_visited = 0
        self._survivor_cache = {}
        while depth <= self.max_depth:
            self._round_incomplete = False
            try:
                val, action = self._beam_value(root_state, root_queues, root_player,
                                                depth, deadline, ply=0, path=())
            except _TimeUp:
                break
            if action is not None:
                best_action, best_val = action, val
            depth_reached = depth
            if not self._round_incomplete:
                round_complete = True  # every line reached the round's end — done
                break
            if time.monotonic() >= deadline:
                break
            depth += 1

        elapsed = time.monotonic() - start
        self._last_round_info = {
            'round_complete': round_complete,
            'actions_left_at_root': sum(root_state.hands[root_player].values()),
        }
        stats = {
            'depth_reached': depth_reached,
            'nodes_visited': self._nodes_visited,
            'elapsed': elapsed,
        }
        return best_action, best_val, stats

    # ------------------------------------------------------------------
    # Round-aware aggregated logging (replaces the base depth-only summary).
    # ------------------------------------------------------------------

    def _reset_agg_stats(self):
        self._agg = {
            'n': 0, 'nodes_visited': 0, 'depth_sum': 0, 'depth_min': None, 'depth_max': None,
            'elapsed_sum': 0.0, 'elapsed_max': 0.0, 'legal_at_root_sum': 0,
            'round_complete': 0, 'actions_left_sum': 0,
        }

    def _record_agg_stats(self):
        """Roll `last_stats` + `_last_round_info` into a window and, every
        `stats_log_every` real moves, log one INFO summary. Beyond the base's
        depth/nodes it reports what this bot exists to expose: how often the round
        was analysed to its end (`round_complete`), how many actions the mover had
        left this round at the decision (`actions_left`, the shrinking 3→2→1
        horizon), and how much of the time budget each move actually spent.
        """
        if not self.stats_log_every:
            return
        a = self._agg
        s = self.last_stats
        info = self._last_round_info
        a['n'] += 1
        a['nodes_visited'] += s['nodes_visited']
        a['depth_sum'] += s['depth_reached']
        a['depth_min'] = s['depth_reached'] if a['depth_min'] is None else min(a['depth_min'], s['depth_reached'])
        a['depth_max'] = s['depth_reached'] if a['depth_max'] is None else max(a['depth_max'], s['depth_reached'])
        a['elapsed_sum'] += s['elapsed']
        a['elapsed_max'] = max(a['elapsed_max'], s['elapsed'])
        a['legal_at_root_sum'] += s['legal_at_root']
        a['round_complete'] += int(info['round_complete'])
        a['actions_left_sum'] += info['actions_left_at_root']
        if a['n'] < self.stats_log_every:
            return
        logger.info(
            '%s: last %d move(s) — round_complete=%d/%d, plies_to_round_end avg=%.2f min=%d max=%d, '
            'actions_left_at_root avg=%.2f, elapsed avg=%.3fs max=%.3fs/move, '
            'nodes_visited avg=%.1f, legal_at_root avg=%.1f',
            self.name, a['n'], a['round_complete'], a['n'],
            a['depth_sum'] / a['n'], a['depth_min'], a['depth_max'],
            a['actions_left_sum'] / a['n'], a['elapsed_sum'] / a['n'], a['elapsed_max'],
            a['nodes_visited'] / a['n'], a['legal_at_root_sum'] / a['n'],
        )
        self._reset_agg_stats()
