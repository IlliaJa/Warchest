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

**θ — the randomised-coefficient family (docs/IDEAS.md B1).** Every one of the
eight coefficients above (the two imported reward scales included) is multiplied
by an entry of `theta`, a dict over `THETA_KEYS`. `theta=LEGACY_THETA` (the
default) is bit-identical to the pre-θ evaluator and `theta=RICH_THETA` is
bit-identical to `enable_new_terms=True`, so the parametrisation costs the
existing consumers nothing — see `tests/test_evaluation.py`. Sampling θ instead
of fixing it (`sample_theta`) turns this one evaluator into a *family* of
policy-independent playstyles for the price of a constructor argument, which is
what `RandomEvalBot` uses to generate opponent-pool entrants.

The family's reach is bounded by what the evaluator actually controls: the search
also replays the env's own `Action.reward` (attack bonus, terminal outcome) and
the holding reward, neither of which θ scales. So no θ makes a bot fully blind to
captures or to winning — θ redistributes emphasis, it does not remove the game's
own incentives. Measured consequences are in `docs/bots.md`.
"""
from collections import deque

import numpy as np

from ..environment.rollout_core import SHAPING_C, C_MAT
from ..environment.cell_ids import UNCONTROLLED_BASE_CELL_ID


# Order is fixed: it is the vector layout for any θ search (docs/IDEAS.md B2) and the
# tie-break order of `theta_tag`. 'tempo' scales INIT_COEFF (initiative), named for the
# term rather than the constant.
THETA_KEYS = ('base', 'material', 'pos', 'risk',
              'durability', 'economy', 'tempo', 'progress')

# The pre-θ default: legacy terms at their tuned weight, new terms off.
LEGACY_THETA = {'base': 1.0, 'material': 1.0, 'pos': 1.0, 'risk': 1.0,
                'durability': 0.0, 'economy': 0.0, 'tempo': 0.0, 'progress': 0.0}
# The old `enable_new_terms=True` (documented negative result — see above).
RICH_THETA = {k: 1.0 for k in THETA_KEYS}


def normalize_theta(theta):
    """Validate `theta` and fill missing keys from `LEGACY_THETA`. -> plain float dict.

    Missing keys default rather than raise so a caller can write `{'economy': 8.0}` and
    mean "legacy, but recruit-hungry"; an *unknown* key does raise, because it is always
    a typo (a silently-ignored 'econ' would produce a bot that looks tuned and isn't).
    """
    if theta is None:
        return dict(LEGACY_THETA)
    unknown = set(theta) - set(THETA_KEYS)
    if unknown:
        raise ValueError(f'unknown theta keys {sorted(unknown)}; valid keys are {list(THETA_KEYS)}')
    out = dict(LEGACY_THETA)
    for k, v in theta.items():
        v = float(v)
        if v < 0.0:
            raise ValueError(f'theta[{k!r}] = {v}: coefficients are magnitudes, must be >= 0 '
                             f'(a negative weight makes the bot seek the thing it should avoid, '
                             f'which is a broken bot rather than a different playstyle)')
        out[k] = v
    return out


# --------------------------------------------------------------------------- #
# θ sampling — the B1 family
# --------------------------------------------------------------------------- #
# Log-uniform multiplier ranges, one per θ key, intended to span the archetype list
# docs/independent_opponents.md Phase 1 wants hand-written: base racer, material grinder,
# bolster brawler, recruit economy, initiative bot, closer.
#
# **These bounds are measured, not guessed** — `src/app/eval_theta_family.py --sweep KEY`
# moves one dial at a time and reads the resulting verb profile and win rate. The
# 2026-08-09 sweeps (vs `greedy_sim`, 16 games/rung, recorded in docs/bots.md) found the
# dials are nothing like interchangeable, so three bounds below are *narrower* than the
# term's coefficient would suggest:
#
#   economy    the one clean archetype generator: recruit 0.02 -> 0.19 by a weight of 0.5,
#              and flat in win rate out to 8. Kept wide.
#   durability does NOT make a bolster brawler. Bolster saturates at ~0.087 by weight 0.5
#              and then *falls*, while `pass` climbs 0.11 -> 0.65 and the win rate goes
#              0.75 -> 0.19 -> 0.00. The term rewards keeping units alive, and the surest
#              way to keep a unit alive is never to use it. Capped at 1.0: that keeps the
#              (real, and #R8-relevant) 17x bolster coverage and cuts the pure-turtle
#              region, which is not a playstyle — it is an opponent that hands the policy
#              free wins and a distorted advantage group.
#   pos        below 1.0 the bot wanders: move 0.39 -> 0.53, turns 100 -> 167, win rate
#              0.69 -> 0.28. Floored at 0.5 and never zeroed.
#   tempo      inert as measured — claim_initiative moved 0.126 -> 0.133 across a 0..20
#              sweep, because that verb's rate is set by the rules, not by the evaluation.
#              Kept (narrowed) rather than removed: one opponent is not the whole game.
#   progress   likewise inert; every column flat across 0..8. Kept for the same reason.
#
# `base` is the narrowest and never zeroed: bases are how the game is won, and a bot that
# does not want them is not a playstyle, it is `RandomBot` (already in the pool).
# `material` and `risk` are unswept and left at their first guess — do not read them as
# validated.
THETA_RANGES = {
    'base': (0.5, 2.0),
    'material': (0.2, 5.0),
    'pos': (0.5, 8.0),
    'risk': (0.2, 5.0),
    'durability': (0.25, 1.0),
    'economy': (0.5, 12.0),
    'tempo': (0.5, 8.0),
    'progress': (0.3, 6.0),
}

# Probability of drawing an exact 0 (term switched off) instead of a log-uniform weight.
# A zeroed term is a qualitatively different bot, not a small one — "ignores material
# entirely" is a state distribution self-play never produces — so the family needs some
# mass exactly at 0 rather than merely near it. Higher for the new terms because their
# *default* is off, and a family in which every member has all four turned on would not
# contain the current evaluator's own behaviour. `pos` is exempt for the reason above:
# pos=0 is not an archetype, it is a bot that walks in circles.
THETA_ZERO_PROB = {'base': 0.0, 'material': 0.15, 'pos': 0.0, 'risk': 0.15,
                   'durability': 0.35, 'economy': 0.35, 'tempo': 0.35, 'progress': 0.35}

# Short display tags, for gauntlet column headers (truncated to 6 chars) and log lines.
THETA_TAGS = {'base': 'bas', 'material': 'mat', 'pos': 'pos', 'risk': 'rsk',
              'durability': 'dur', 'economy': 'eco', 'tempo': 'ini', 'progress': 'prg'}


def sample_theta(rng, *, ranges=THETA_RANGES, zero_prob=THETA_ZERO_PROB):
    """Draw one θ (log-uniform per key, with a zero-inflation atom). -> float dict.

    `rng` must be a `numpy.random.Generator` — deliberately not the global RNG, which
    the gauntlet re-seeds per game to pin the draft (`gauntlet.play_game`); drawing θ
    from that stream would shift every subsequent draw and break the antithetic-draft
    pairing.
    """
    theta = {}
    for key in THETA_KEYS:
        if rng.random() < zero_prob.get(key, 0.0):
            theta[key] = 0.0
            continue
        lo, hi = ranges[key]
        theta[key] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return theta


def theta_tag(theta, *, ranges=THETA_RANGES):
    """3-char tag naming the term θ pushes hardest, ties broken by `THETA_KEYS` order.

    "Hardest" is measured as position within that key's own sampling range, in log space —
    not the raw multiplier. The ranges differ by an order of magnitude between keys (tempo
    reaches 20x, base only 2x), so ranking raw multipliers would label almost every draw
    `ini`. A zeroed term never wins the tag.

    A *label* for reports and column headers, not a claim about behaviour: the terms also
    have very different natural scales, so the most-stretched multiplier is not necessarily
    the most influential term. Read the measured verb profile for that
    (`src/app/eval_theta_family.py`).
    """
    theta = normalize_theta(theta)

    def stretch(key):
        v = theta[key]
        if v <= 0.0:
            return -np.inf
        lo, hi = ranges[key]
        return (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo))

    best = max(THETA_KEYS, key=lambda k: (stretch(k), -THETA_KEYS.index(k)))
    return THETA_TAGS[best]


def format_theta(theta):
    """One-line `key=value` rendering of θ, ordered by `THETA_KEYS`."""
    theta = normalize_theta(theta)
    return ' '.join(f'{THETA_TAGS[k]}={theta[k]:.2f}' for k in THETA_KEYS)


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

    def __init__(self, shaping_anneal=1.0, enable_new_terms=False, theta=None):
        """`theta` (docs/IDEAS.md B1) multiplies the eight coefficients; None picks
        `RICH_THETA` if `enable_new_terms` else `LEGACY_THETA`. Passing both an explicit
        `theta` and `enable_new_terms=True` is a contradiction (θ already says whether the
        new terms are on) and raises.
        """
        if theta is not None and enable_new_terms:
            raise ValueError('pass either theta or enable_new_terms=True, not both — '
                             'theta already carries the new terms\' weights')
        self.shaping_anneal = shaping_anneal
        if theta is None:
            theta = RICH_THETA if enable_new_terms else LEGACY_THETA
        self.set_theta(theta)

    def set_theta(self, theta):
        """Re-weight the eight terms in place (used by `RandomEvalBot` per episode, so a
        pool opponent changes playstyle without rebuilding its search env).
        """
        self.theta = normalize_theta(theta)
        t = self.theta
        # Effective coefficients, folded once here rather than per leaf evaluation — this
        # runs on the search hot path. Multiplication order is chosen to keep a θ of 1.0
        # bit-identical to the pre-θ expressions (x * 1.0 is exact in IEEE-754, but only if
        # the remaining products associate the same way).
        self._c_base = t['base'] * SHAPING_C
        self._c_mat = t['material'] * self.shaping_anneal * C_MAT
        self._c_pos = t['pos'] * self.POS_COEFF
        self._c_risk = t['risk'] * self.RISK_COEFF
        self._c_dur = t['durability'] * self.DUR_COEFF
        self._c_econ = t['economy'] * self.ECON_COEFF
        self._c_init = t['tempo'] * self.INIT_COEFF
        self._c_prog = t['progress'] * self.PROG_COEFF
        # Skipping the four new terms when every weight is 0 keeps the legacy path free of
        # their (BFS-free but not free) work, and keeps the sum's association identical.
        self.enable_new_terms = any(t[k] for k in ('durability', 'economy', 'tempo', 'progress'))

    def evaluate(self, env, root_player):
        """Total root_player-perspective static value of `env`'s current state."""
        opp = 3 - root_player
        base_diff = (len(env.board.get_controlled_bases(root_player))
                     - len(env.board.get_controlled_bases(opp)))
        base_term = self._c_base * base_diff * env.winning_base_count
        mat_term = self._c_mat * (env.boxed_total(opp) - env.boxed_total(root_player))
        pos_term = self._c_pos * (self._nearest_dist(env, opp)
                                  - self._nearest_dist(env, root_player))
        risk_term = self._c_risk * self._material_at_risk(env, root_player, opp)
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
        return self._c_dur * (own - other)

    def _economy(self, env, root_player, opp):
        """Opponent supply minus own supply: recruiting drains a coin out of
        supply into the active cycle (bag/hand/board), so lower own supply means
        more force in circulation. Crude but directionally correct; makes
        `recruit` register as a gain instead of never being taken.
        """
        own_supply = sum(env.state.supply[root_player].values())
        opp_supply = sum(env.state.supply[opp].values())
        return self._c_econ * (opp_supply - own_supply)

    def _tempo(self, env, root_player):
        """Small bonus for holding initiative (acting first next round) — makes
        `claim_initiative` register as a gain.
        """
        return self._c_init * (1.0 if env.state.initiative_owner == root_player else -1.0)

    def _progress(self, env, root_player, opp):
        """Convex bonus for standing exactly one base short of the win. The linear
        base term values the decisive 6th base the same as the 1st; this makes the
        near-win state markedly more attractive (and the mirror threat more scary).
        """
        wbc = env.winning_base_count
        my = len(env.board.get_controlled_bases(root_player))
        their = len(env.board.get_controlled_bases(opp))
        return self._c_prog * (int(my == wbc - 1) - int(their == wbc - 1))
