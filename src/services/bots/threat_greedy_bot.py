"""ThreatAwareGreedy — a sub-millisecond prophylaxis bot (docs/IDEAS.md B5).

The observation already carries graded threat maps that **no bot reads**: board
planes 38-40 are the hit-count *I* could land on each cell this turn, 41-43 the
same for the opponent (both gated by coin availability — exact for my hand,
worst-case over the opponent's hidden pool). They are recomputed by the encoder on
every step whether or not anyone consumes them, so a bot that reads them gets one
to two plies of tactical information for free, with **no simulation at all**.

`GreedyBot` predates those planes and reads only walls and bases; the only other
place the same quantity exists is `evaluation.py::_material_at_risk`, which
recomputes it from the env for a search leaf. This bot is the cheap obs-only
consumer of the planes that were always there.

The ladder, each rung a filter over the legal-action mask:

  1. **free capture** — the target stack is killable this turn *and* the attacker's
     own cell survives the reply (`_danger_after`);
  2. **claim** — take the base the unit is standing on;
  3. **un-hang** — any unit whose incoming hits >= its stack runs to a cell that is
     not covered, or bolsters if one more coin is enough to survive; never off a base
     it holds, which it parks on instead;
  4. **attack** — the same safety test, without the requirement that the whole stack
     die this turn;
  5. **march** — one multi-source BFS toward claimable bases: park, then distance,
     then avoid landing on a covered cell;
  6. **deploy** — onto an uncovered cell (a fresh unit has stack 1, so *any* hit
     kills it), nearest to a claimable base;
  7. select-continuation, then face-down / decline as a last resort.

Rungs 2 and 3 are **swapped relative to the item's text**, which lists un-hang above
claim. Retreating ahead of claiming measured 0.26 against `greedy_fast` (80 games,
colours balanced): the unit standing on a claimable base is exactly the unit the
opponent covers, so "un-hang first" spends the coin walking off the win condition.

Do not expect this bot to be *strong*. Against `greedy_fast` over 1600 games it is
**0.515 ± 0.024** — a tie — and dropping the un-hang rung entirely moves that by
0.000. What replicated is the cost claim, not the prophylaxis one; `docs/IDEAS.md`
B5 § Measured has the ablations.

Like `GreedyBot` it never *initiates* a tactic, so charge/ranged hits enter its own
threat sum without being realizable by its own action set — the "killable" test is
the encoder's worst-case semantics, deliberately, since the same optimism is what
the policy sees. Mid-tactic continuation clicks reuse the move/attack verbs and are
handled by the ladder unchanged.

Costs one pass over 48x7x7 planes plus one 49-cell BFS per decision (0.11 ms). The
BFS is `board_geometry.distance_to`, which came out of this bot and was then applied
to `GreedyBot` — taking *it* from 0.90 ms to 0.04 ms, so this bot no longer holds a
cost advantage over the baseline either (docs/IDEAS.md Table A).
"""
import numpy as np

from .base import Bot
from .board_geometry import N_CELLS, STEP, UNREACHABLE, distance_to
from ..environment.roster import UNIT_BY_ID, UNIT_IDS, NUM_UNIT_TYPES
from ..environment.warchest_env import (
    BOARD_DIM, SPATIAL_SIZE, N_COIN_TYPES, TACTIC_VERB, SELECT_VERB,
    CONTROL_VERB, BOLSTER_VERB, DEPLOY_VERB_BASE, ROYAL_COIN_IDX, DECLINE_ACTION_ID,
)
from ..environment.obs_encoders.v11 import (
    BOARD_CHANNELS, OWN_UNIT_PLANE_BASE, OPP_UNIT_PLANE_BASE, OWN_THREAT_PLANE_BASE,
    ENEMY_THREAT_PLANE_BASE, N_THREAT_KINDS, THREAT_NORM, STACK_NORM,
)

# Verb ranges in the spatial action scheme (mirrors greedy_bot.py).
_VERB_MOVE_END = 5
_VERB_ATTACK_START = 6
_VERB_ATTACK_END = 11
_VERB_DEPLOY_START = DEPLOY_VERB_BASE
_VERB_DEPLOY_END = TACTIC_VERB - 1

# Base planes (stable across obs v10/v11).
_INVALID_PLANE = 0
_UNCONTROLLED_BASE_PLANE = 2
_OWN_BASE_PLANE = 3
_OPP_BASE_PLANE = 4

# Face-down block layout (offsets from SPATIAL_SIZE): [0:C) claim, [C:2C) pass.
_PASS_OFFSET = N_COIN_TYPES

_UNIT_ID_BY_PLANE = np.array(UNIT_IDS, dtype=np.int64)
_COUNTERS_BY_ID = np.zeros(max(UNIT_IDS) + 1, dtype=bool)
for _u in UNIT_IDS:
    _COUNTERS_BY_ID[_u] = UNIT_BY_ID[_u].counter_when_attacked


class ThreatAwareGreedyBot(Bot):
    """Threat-plane-driven greedy. Reads an ego-centric obs + mask, no simulation."""

    RANDOM_ACTION_PROB = 0.0

    def act(self, obs: dict) -> tuple[int, None, None]:
        valid = np.flatnonzero(obs['valid_action_mask'])
        if self.RANDOM_ACTION_PROB and np.random.random() < self.RANDOM_ACTION_PROB:
            return int(np.random.choice(valid)), None, None

        board = obs['board']
        if board.shape[0] != BOARD_CHANNELS:
            raise ValueError(
                f'ThreatAwareGreedyBot reads the v10/v11 plane layout ({BOARD_CHANNELS} '
                f'channels); got {board.shape[0]}')
        s = _State(board)

        spatial = valid[valid < SPATIAL_SIZE]
        verbs = spatial // N_CELLS
        cells = spatial % N_CELLS

        for rung in (self._free_capture, self._claim, self._unhang,
                     self._attack, self._march, self._deploy, self._select):
            action = rung(s, spatial, verbs, cells)
            if action is not None:
                return int(action), None, None

        # Nothing on the ladder applies: prefer burning the Royal coin, then ending an
        # optional continuation, over a random click that may walk into a threat.
        royal_pass = SPATIAL_SIZE + _PASS_OFFSET + ROYAL_COIN_IDX
        for fallback in (royal_pass, DECLINE_ACTION_ID):
            if obs['valid_action_mask'][fallback]:
                return int(fallback), None, None
        return int(np.random.choice(valid)), None, None

    # ------------------------------------------------------------------ #
    # Ladder rungs. Each returns an action id or None.
    # ------------------------------------------------------------------ #
    def _free_capture(self, s, spatial, verbs, cells):
        """Kill the whole stack this turn without losing the attacker."""
        best, best_key = None, None
        for action, src, dst in _attacks(spatial, verbs, cells):
            if _suicidal(s, src, dst):
                continue
            if s.own_hits[dst] < s.opp_stack[dst]:  # cannot finish the stack this turn
                continue
            if _danger_after(s, src, dst) >= s.own_stack[src]:  # the attacker is answered
                continue
            # Lethal now (one hit removes the unit) beats merely killable; then the
            # bigger stack, then the quieter cell.
            key = (s.opp_stack[dst] > 1, -s.opp_stack[dst], s.enemy_hits[src])
            if best_key is None or key < best_key:
                best, best_key = action, key
        return best

    def _unhang(self, s, spatial, verbs, cells):
        """Rescue any unit whose incoming hits already cover its stack."""
        hanging = (s.own_stack > 0) & (s.enemy_hits >= s.own_stack)
        if not hanging.any():
            return None

        best, best_key = None, None
        for action, verb, src in zip(spatial, verbs, cells):
            if not hanging[src]:
                continue
            if verb <= _VERB_MOVE_END:
                if s.own_base[src] > 0:
                    continue  # park: a unit on a base I hold never runs, it bolsters
                dst = STEP[verb, src]
                if dst < 0 or s.enemy_hits[dst] >= s.own_stack[src]:
                    continue  # running into another covered cell is not a rescue
                key = (-s.own_stack[src], s.enemy_hits[dst], s.dist[dst])
            elif verb == BOLSTER_VERB and s.enemy_hits[src] == s.own_stack[src]:
                # One more coin puts the stack out of reach of everything aimed at it.
                key = (-s.own_stack[src], 0, s.dist[src])
            else:
                continue
            if best_key is None or key < best_key:
                best, best_key = action, key
        return best

    def _claim(self, s, spatial, verbs, cells):
        controls = [(a, c) for a, v, c in zip(spatial, verbs, cells) if v == CONTROL_VERB]
        if not controls:
            return None
        # The claimer parks on the base afterwards, so take the quietest one first.
        return min(controls, key=lambda ac: s.enemy_hits[ac[1]])[0]

    def _attack(self, s, spatial, verbs, cells):
        """Trade a coin off the board without losing the attacker to the reply.

        An attacker that is *already* covered may still strike if the blow is lethal:
        rung 2 ran first and found it no escape square, so trading it for a kill beats
        marching it into the same reply.
        """
        best, best_key = None, None
        for action, src, dst in _attacks(spatial, verbs, cells):
            if _suicidal(s, src, dst):
                continue
            if _danger_after(s, src, dst) >= s.own_stack[src]:
                continue
            key = (s.opp_stack[dst] > 1, -s.opp_stack[dst], s.enemy_hits[src])
            if best_key is None or key < best_key:
                best, best_key = action, key
        return best

    def _march(self, s, spatial, verbs, cells):
        best, best_key = None, None
        for action, verb, src in zip(spatial, verbs, cells):
            if verb > _VERB_MOVE_END:
                continue
            dst = STEP[verb, src]
            if dst < 0 or s.dist[dst] >= UNREACHABLE:
                continue
            hangs = s.enemy_hits[dst] >= s.own_stack[src]
            leaves_base = s.own_base[src] > 0
            # Park first (a unit on a base I hold moves only if nothing else can),
            # then distance, and safety only as a tie-break between equal steps.
            # Putting `hangs` ahead of `dist` costs the race outright: the opponent
            # threat model is worst-case over its whole hidden pool, so nearly every
            # cell near a base reads as covered, and a safety-first march walks away
            # from the win condition — measured, claim_base fell 12.7 % -> 9.3 % of
            # decisions and the bot lost 0.26 to greedy_fast.
            key = (leaves_base, s.dist[dst], hangs)
            if best_key is None or key < best_key:
                best, best_key = action, key
        return best

    def _deploy(self, s, spatial, verbs, cells):
        best, best_key = None, None
        for action, verb, cell in zip(spatial, verbs, cells):
            if not _VERB_DEPLOY_START <= verb <= _VERB_DEPLOY_END:
                continue
            # A deployed unit arrives with one coin, so a single covering hit kills it.
            key = (s.enemy_hits[cell] > 0, s.dist[cell])
            if best_key is None or key < best_key:
                best, best_key = action, key
        return best

    def _select(self, s, spatial, verbs, cells):
        """Non-directional target click (ranged attack, or a granted maneuver).

        Which of the two is live is readable off the board: a ranged target is an
        enemy-occupied cell, a grant recipient a friendly one.
        """
        best, best_key = None, None
        for action, verb, cell in zip(spatial, verbs, cells):
            if verb != SELECT_VERB:
                continue
            if s.opp_stack[cell] > 0:  # ranged target — no counter reaches the shooter
                key = (0, s.opp_stack[cell] > 1, -s.opp_stack[cell])
            elif s.own_stack[cell] > 0:  # grant recipient — push the leading unit
                key = (1, s.dist[cell], -s.own_stack[cell])
            else:
                continue
            if best_key is None or key < best_key:
                best, best_key = action, key
        return best


class _State:
    """Per-decision integer views of the planes the ladder reads."""

    def __init__(self, board):
        flat = board.reshape(board.shape[0], N_CELLS)
        # Planes are stack/hit counts divided by 5 (and threats clipped at 1.0), so
        # multiply back and round — float32 thirds would otherwise break `>=` ties.
        own_units = flat[OWN_UNIT_PLANE_BASE:OWN_UNIT_PLANE_BASE + NUM_UNIT_TYPES]
        opp_units = flat[OPP_UNIT_PLANE_BASE:OPP_UNIT_PLANE_BASE + NUM_UNIT_TYPES]
        self.own_stack = _counts(own_units.sum(0) * STACK_NORM)
        self.opp_stack = _counts(opp_units.sum(0) * STACK_NORM)
        self.own_hits = _counts(
            flat[OWN_THREAT_PLANE_BASE:OWN_THREAT_PLANE_BASE + N_THREAT_KINDS].sum(0) * THREAT_NORM)
        self.enemy_hits = _counts(
            flat[ENEMY_THREAT_PLANE_BASE:ENEMY_THREAT_PLANE_BASE + N_THREAT_KINDS].sum(0) * THREAT_NORM)
        # At most one unit per cell, so the occupied plane identifies the type.
        self.opp_id = _UNIT_ID_BY_PLANE[opp_units.argmax(0)]
        self.own_base = flat[_OWN_BASE_PLANE]

        targets = (flat[_UNCONTROLLED_BASE_PLANE] + flat[_OPP_BASE_PLANE]) > 0
        passable = flat[_INVALID_PLANE] == 0
        self.dist = distance_to(targets, passable)


def _counts(x):
    return np.rint(x).astype(np.int32)


def _attacks(spatial, verbs, cells):
    """(action, source cell, target cell) for every legal attack; attacks never move
    the attacker, so the source cell is also the cell that eats the reply."""
    out = []
    for action, verb, src in zip(spatial, verbs, cells):
        if not _VERB_ATTACK_START <= verb <= _VERB_ATTACK_END:
            continue
        dst = STEP[verb - _VERB_ATTACK_START, src]
        if dst >= 0:
            out.append((action, src, int(dst)))
    return out


def _danger_after(s, src, dst):
    """Hits the safety filter should hold against the attacker's cell.

    A blow that empties the stack is always taken, whatever the planes say: the coin
    it boxes is gone for the rest of the game, the unit at `dst` is one of the summands
    covering `src` in the first place, and the reply — if it even comes — costs the
    opponent a coin of their own. Only an attack that leaves the target standing has to
    prove the attacker survives.

    This is a measured setting, not a derivation. Filtering lethal blows too (the
    planes read literally) loses 3.7 pp against `greedy_fast` over 400 games
    (0.495 vs 0.532): the opponent threat model is worst-case over the whole hidden
    pool, so "safe" almost never holds for a stack-1 unit, and the filter degenerates
    into a refusal to trade at all.
    """
    return 0 if s.opp_stack[dst] <= 1 else s.enemy_hits[src]


def _suicidal(s, src, dst):
    """Pikeman's counter removes a coin from an adjacent attacker even when the
    Pikeman itself dies, so a lone attacker trades itself away — the one hanging
    move the threat planes do not cover (they model attacks, not counters)."""
    return _COUNTERS_BY_ID[s.opp_id[dst]] and s.own_stack[src] <= 1
