"""Shared static state evaluation for the search/greedy bots.

`HeuristicEvaluator` is the single source of truth for "how good is this state for
`root_player`", in the same reward-scale units `rollout_core.py` shapes training
reward with (base + material PBRS). It is used by:
  - `LookaheadBot._leaf_potential` (the alpha-beta search leaf),
  - `LookaheadCriticBot` (blended with the trained critic), inherited unchanged,
  - `GreedyBot`'s 1-ply forward-simulation scoring.

Two tiers of terms:

  * **Legacy** (always on) — the exact base/material/positional/risk formula that
    lived in `LookaheadBot._leaf_potential`. With `enable_new_terms=False` the
    output is byte-identical to the old method (see
    `tests/test_evaluation.py::test_legacy_equivalence`), which is what keeps
    `LookaheadCriticBot`'s value-scale calibration — moment-matched against this
    exact distribution — valid.

  * **New** (`enable_new_terms=True`) — extra terms meant to make the search/greedy
    value the game features earlier bots ignored:
      - `durability`: own stack height above 1 (survivability) → nudges `bolster`.
      - `economy`: supply drained into the active coin cycle → nudges `recruit`.
      - `tempo`: holding initiative → nudges `claim_initiative`.
      - `progress`: a convex bonus for standing one base from the win.

    **These default OFF, and are kept only as a documented negative result.**
    Measurement (docs/bots.md) showed them net-harmful in every bot that uses this
    evaluator: they reward long-horizon assets (a deeper deck, initiative, a
    bolstered stack) that a depth-bounded search leaf cannot cash in, so the bot
    trades away tempo for them. `economy` was the worst — it made SimGreedyBot spam
    `recruit`; a rich=True LookaheadBot scored only ~20% vs the same bot with
    rich=False. The features are already used *correctly* without these terms,
    because every consumer scores an already-simulated resulting state: a tactic's
    kill, a bolster that saves a hanging unit (via the risk term), Pikeman
    counter-coins, the Knight bolster-gate and recaptures all show up in
    `boxed`/stacks/at-risk on their own. The terms are retained (off) in case a
    trained-value or much deeper context can use them; do not turn them on without
    re-measuring.

The evaluator is stateless w.r.t. the env — every method takes the (already
`set_state`-d) `WarChestEnv` as an argument rather than capturing one, because
`LookaheadCriticBot` swaps its `_sim_env` out from under `LookaheadBot.__init__`
(it rebuilds it against the critic's obs version).

New-term coefficients are first-cut and kept well below `SHAPING_C` (bases win
the game); they are meant to be swept in the gauntlet, not treated as tuned.
"""
from collections import deque

import numpy as np

from ..environment.rollout_core import SHAPING_C, C_MAT
from ..environment.cell_ids import UNCONTROLLED_BASE_CELL_ID


def capturable_bases(board, player):
    """Cells `player` could still usefully capture: uncontrolled, or held by the
    other player. Shared by `_nearest_dist` (leaf) and `LookaheadBot._ordering_key`
    (move ordering) so both use the same notion of "target".
    """
    targets = set(board.get_controlled_bases(3 - player))
    rows, cols = np.where(board.board == UNCONTROLLED_BASE_CELL_ID)
    targets.update(zip(rows.tolist(), cols.tolist()))
    return targets


class HeuristicEvaluator:
    """Root-player-perspective static evaluation of a `WarChestEnv` state."""

    # --- legacy term coefficients (do not change without re-checking the critic
    # bot's calibration, which is moment-matched to this exact distribution) ---
    # Value of being one hex closer than the opponent to the nearest capturable
    # base — comparable to, but smaller than, a single attack (0.02), so it acts
    # as a directional tie-break rather than overriding a real tactical gain.
    POS_COEFF = 0.01
    # Weight on material-at-risk (stack coins a unit stands to lose next turn, not
    # yet actually lost) — half of C_MAT, since it's predictive, not realized.
    RISK_COEFF = 0.5 * C_MAT
    # Distance used when a side has no on-board unit or no capturable base exists —
    # larger than any real hex distance on the 7x7 board, so it always reads "far".
    _FAR_DIST = 12

    # --- new term coefficients (first-cut; sweep in the gauntlet) ---
    # Per bolstered coin (stack height above 1) held on the board. Below C_MAT so
    # a standing durability asset never outweighs a coin actually boxed.
    DUR_COEFF = 0.4 * C_MAT
    # Cap on stack-above-1 counted per unit, so one over-bolstered stack can't
    # dominate the term.
    _MAX_BOLSTER = 4
    # Per supply coin drained into the active cycle (recruiting lowers own supply).
    ECON_COEFF = 0.15 * C_MAT
    # Value of holding initiative this round (tempo).
    INIT_COEFF = 0.005
    # Convex bonus for standing exactly one base short of the win — the linear
    # base term treats the decisive 6th base like the 1st; this restores urgency.
    PROG_COEFF = 0.15

    def __init__(self, shaping_anneal=1.0, enable_new_terms=False):
        self.shaping_anneal = shaping_anneal
        self.enable_new_terms = enable_new_terms

    def evaluate(self, env, root_player):
        """Total root_player-perspective static value of `env`'s current state."""
        opp = 3 - root_player
        base_diff = (len(env.board.get_controlled_bases(root_player))
                     - len(env.board.get_controlled_bases(opp)))
        base_term = SHAPING_C * base_diff * env.winning_base_count
        mat_term = (self.shaping_anneal * C_MAT
                    * (env.boxed_total(opp) - env.boxed_total(root_player)))
        pos_term = self.POS_COEFF * (self._nearest_dist(env, opp)
                                     - self._nearest_dist(env, root_player))
        risk_term = self.RISK_COEFF * self._material_at_risk(env, root_player, opp)
        total = base_term + mat_term + pos_term + risk_term
        if self.enable_new_terms:
            total += (self._durability(env, root_player, opp)
                      + self._economy(env, root_player, opp)
                      + self._tempo(env, root_player)
                      + self._progress(env, root_player, opp))
        return total

    # ------------------------------------------------------------------
    # Legacy terms (moved verbatim from LookaheadBot; see its docstring)
    # ------------------------------------------------------------------

    def _material_at_risk(self, env, root_player, opp):
        """opp_at_risk - own_at_risk: the "material-at-risk" quantity the obs
        encoder exposes to the trained policy (docs/observation_improvement.md).
        Computed exactly (we know both hands, real or determinized) via the same
        `unit_threat_footprint`/`attack_enabler_coins` primitives, accumulated into
        a plain dict rather than allocating full 7x7 numpy threat grids.
        """
        own_units = [u for u in env.board.units if u.player_id == root_player]
        opp_units = [u for u in env.board.units if u.player_id == opp]
        own_at_risk = self._at_risk(env, opp, opp_units, own_units)
        opp_at_risk = self._at_risk(env, root_player, own_units, opp_units)
        return opp_at_risk - own_at_risk

    @staticmethod
    def _at_risk(env, attacker_side, attackers, targets):
        """Sum of min(incoming hits, stack) over `targets`, from `attackers`' threats."""
        if not attackers or not targets:
            return 0
        hand = env.state.hands[attacker_side]
        hits_by_cell = {}
        for u in attackers:
            footprint = env.unit_threat_footprint(u)
            if not footprint or not any(hand[c] >= 1 for c in env.attack_enabler_coins(u)):
                continue
            for cell, _kind, hits in footprint:
                hits_by_cell[cell] = hits_by_cell.get(cell, 0) + hits
        return sum(min(hits_by_cell.get(u.loc, 0), u.stack) for u in targets)

    def _nearest_dist(self, env, player):
        """Hex-distance from `player`'s nearest on-board unit to the nearest base
        they could still usefully capture — multi-source BFS over the current
        board, mirroring GreedyBot's own march-to-base heuristic.
        """
        board = env.board
        starts = [u.loc for u in board.units if u.player_id == player]
        if not starts:
            return self._FAR_DIST
        targets = capturable_bases(board, player)
        if not targets:
            return self._FAR_DIST
        if any(s in targets for s in starts):
            return 0
        visited = set(starts)
        frontier = deque(starts)
        dist = 0
        while frontier:
            dist += 1
            for _ in range(len(frontier)):
                cell = frontier.popleft()
                for nb in board.get_adjacent_cells(*cell):
                    if nb in targets:
                        return dist
                    if nb not in visited:
                        visited.add(nb)
                        frontier.append(nb)
        return self._FAR_DIST

    # ------------------------------------------------------------------
    # New terms (features the bots were written before — see module docstring)
    # ------------------------------------------------------------------

    def _durability(self, env, root_player, opp):
        """Own bolstered coins minus opponent's: stack height above 1 is
        survivability (each extra coin absorbs one hit), and — for Berserker —
        fuel for chained maneuvers. Makes `bolster` register as a gain.
        """
        own = other = 0
        for u in env.board.units:
            extra = min(max(u.stack - 1, 0), self._MAX_BOLSTER)
            if u.player_id == root_player:
                own += extra
            elif u.player_id == opp:
                other += extra
        return self.DUR_COEFF * (own - other)

    def _economy(self, env, root_player, opp):
        """Opponent supply minus own supply: recruiting drains a coin out of
        supply into the active cycle (bag/hand/board), so lower own supply means
        more force in circulation. Crude but directionally correct; makes
        `recruit` register as a gain instead of never being taken.
        """
        own_supply = sum(env.state.supply[root_player].values())
        opp_supply = sum(env.state.supply[opp].values())
        return self.ECON_COEFF * (opp_supply - own_supply)

    def _tempo(self, env, root_player):
        """Small bonus for holding initiative (acting first next round) — makes
        `claim_initiative` register as a gain.
        """
        return self.INIT_COEFF * (1.0 if env.state.initiative_owner == root_player else -1.0)

    def _progress(self, env, root_player, opp):
        """Convex bonus for standing exactly one base short of the win. The linear
        base term values the decisive 6th base the same as the 1st; this makes the
        near-win state markedly more attractive (and the mirror threat more scary).
        """
        wbc = env.winning_base_count
        my = len(env.board.get_controlled_bases(root_player))
        their = len(env.board.get_controlled_bases(opp))
        return self.PROG_COEFF * (int(my == wbc - 1) - int(their == wbc - 1))
