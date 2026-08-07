"""BolsterBot — Berserker/Warrior-Priest bolster archetype (fully scripted per the brief).

Phase-1 exploiter (docs/independent_opponents.md): a policy-independent opponent that
bolsters and uses the key units' tactics. Draft forced to hold the key units at reset
(callers arrange it — `eval_bolster.py`, `play_bolster.py`).

Deterministic priority pipeline (`_normal`). A 2-ply forward-sim (`_score`, inherited
from `SimGreedyBot`) is used ONLY to pick the best target/destination inside a scripted
priority, never to pick the priority itself — so the bot's *choices* are the brief's, and
search only chooses *where*. Rules, highest first:

KEY UNITS — the only units that fight/maneuver (points 4/5/6), tried only when a key
coin is in hand:
  1 attack (KILL first — point 4/5), best target by `_score`   `_key_attack`
  2 deploy on flank home (Berserker RIGHT, Priest LEFT — point 4/5/6)  `_deploy_home`
  3 claim a base — Berserker NEVER claims/captures (combat only); Priest may  `_key_claim`
  4 bolster to walk stack, OFF a home base (point 3/5; never on a home — keeps it free)
                                                              `_bolster_key`
  5 move, best destination by `_score`                        `_key_move`
FILLERS — pure coin-sinks (point 6): deploy + bolster to stack 2, then frozen (never
maneuvered, never recruited):
  6 deploy a drawn PAIR (2 coins of one filler) — always (point 2)  `_deploy_filler_pair`
  7 bolster a deployed filler to stack 2                       `_bolster_filler`
  8 deploy the most-important filler if BOTH home bases are free (point 2 answer)
                                                              `_deploy_filler_single`
  9 recruit a key unit with a spare single coin (point 2)     `_recruit_key`
 9b recruit a filler paying the Royal coin, once hand+bag coins in circulation
    (all types) drop to <=4 — spends the free coin instead of letting it idle
                                                              `_recruit_filler_with_royal`
 10 Priest initiative trick (point 5)                          `_claim_initiative`
 11 pass (only if nothing above applies)                       `_pass`

Berserker never claims/captures bases (explicit ban — combat-only unit; a second
key unit or the initial deploy still gets it on/off its home base, just never via
the CONTROL_VERB claim action). Enforced in both `_key_claim` and `_continuation`.

Flanks: the render's horizontal axis is board row r, so LEFT = smaller-r home base.
"""
import math
from collections import Counter

from .greedy_sim_bot import SimGreedyBot
from .lookahead_bot import _clone_state
from ..environment.warchest_env import (
    WarChestEnv, SPATIAL_SIZE, CONTROL_VERB, BOLSTER_VERB, TACTIC_VERB, DEPLOY_VERBS,
)
from ..environment.roster import UNIT_BY_ID, ROYAL_ID

BERSERKER_ID = 8
WARRIOR_PRIEST_ID = 16
KEY_UNIT_IDS = (BERSERKER_ID, WARRIOR_PRIEST_ID)

# Deploy-importance order (most important first), by coin id.
IMPORTANCE = (BERSERKER_ID, WARRIOR_PRIEST_ID, 10, 3, 4, 13, 5, 6, 2, 15, 7, 1, 9, 11, 12)
_IMPORTANCE_RANK = {u: i for i, u in enumerate(IMPORTANCE)}
DEPLOY_VERB_BY_UNIT = {uid: verb for verb, uid in DEPLOY_VERBS.items()}
START_BASES = {1: [(1, 0), (4, 1)], 2: [(2, 5), (5, 6)]}


class BolsterBot(SimGreedyBot):
    """Deterministic scripted Berserker/Priest archetype (2-ply eval for targets only)."""

    def __init__(self, *, build_target=2, priest_target=2, name='bolster', **_ignored):
        super().__init__(name=name, see_opponent_hand=True)
        self.build_target = build_target
        self.priest_target = priest_target
        self.usage = Counter()

    # ------------------------------------------------------------------
    def act(self, env) -> int:
        legal = env.get_possible_actions()
        if len(legal) == 1:
            self.usage[self._classify(legal[0])] += 1
            return legal[0]
        me = env.active_player
        if env.state.pending is not None:
            a = self._continuation(env, me, legal)
        else:
            a = self._normal(env, me, legal)
        self.usage[self._classify(a)] += 1
        return a

    def _assign(self, me, keys):
        left, right = self._homes(me)
        if BERSERKER_ID in keys and WARRIOR_PRIEST_ID in keys:      # point 6
            return {WARRIOR_PRIEST_ID: left, BERSERKER_ID: right}
        if BERSERKER_ID in keys:                                    # point 4
            return {BERSERKER_ID: left}
        if WARRIOR_PRIEST_ID in keys:                               # point 5
            return {WARRIOR_PRIEST_ID: left}
        return {}

    def _normal(self, env, me, legal):
        legal_set = set(legal)
        hand = env.state.hands[me]
        keys = [u for u in KEY_UNIT_IDS if u in env.state.compositions[me]]
        assign = self._assign(me, keys)
        starts = set(START_BASES[me])
        rs, rq = self._prepare_root(env, me)

        # --- KEY UNITS (fight; only when a key coin is in hand) ---
        a = self._key_attack(env, me, assign, hand, rs, rq, legal_set)          # 1
        if a is not None:
            return a
        for uid in (BERSERKER_ID, WARRIOR_PRIEST_ID):                           # 2
            if uid in assign and hand.get(uid, 0) >= 1 and self._unit(env, me, uid) is None:
                d = self._deploy_home(env, me, uid, assign[uid], legal_set)
                if d is not None:
                    return d
        a = self._key_claim(env, me, assign, hand, legal_set)                   # 3
        if a is not None:
            return a
        a = self._bolster_key(env, me, assign, hand, starts, legal_set)         # 4
        if a is not None:
            return a
        a = self._key_move(env, me, assign, hand, rs, rq, legal_set)            # 5
        if a is not None:
            return a

        # --- FILLERS (coin-sinks; never maneuvered, never recruited) ---
        a = self._deploy_filler_pair(env, me, hand, assign, legal_set)          # 6
        if a is not None:
            return a
        a = self._bolster_filler(env, me, hand, legal_set)                      # 7
        if a is not None:
            return a
        a = self._deploy_filler_single(env, me, hand, keys, assign, legal_set)   # 8
        if a is not None:
            return a
        a = self._recruit_key(env, me, hand, legal_set)                         # 9
        if a is not None:
            return a
        a = self._recruit_filler_with_royal(env, me, hand, legal_set)          # 9b
        if a is not None:
            return a
        if WARRIOR_PRIEST_ID in keys:                                           # 10
            a = self._claim_initiative(env, me, hand, legal_set)
            if a is not None:
                return a
        return self._pass(env, me, legal)                                       # 11

    # ------------------------------------------------------------------
    # Key-unit rules
    # ------------------------------------------------------------------
    def _key_attack(self, env, me, assign, hand, rs, rq, legal_set):
        cands = []
        for uid in assign:
            if hand.get(uid, 0) < 1:
                continue
            u = self._unit(env, me, uid)
            if u is None:
                continue
            for d in range(6):
                a = WarChestEnv.encode_action(6 + d, *u.loc)
                if a in legal_set:
                    cands.append(a)
        if not cands:
            return None
        return max(cands, key=lambda a: self._score(rs, rq, a, me))

    def _key_claim(self, env, me, assign, hand, legal_set):
        for uid in assign:
            if uid == BERSERKER_ID:
                continue  # Berserker never claims/captures bases — combat only
            if hand.get(uid, 0) < 1:
                continue
            u = self._unit(env, me, uid)
            if u is None:
                continue
            a = WarChestEnv.encode_action(CONTROL_VERB, *u.loc)
            if a in legal_set:
                return a
        return None

    def _bolster_key(self, env, me, assign, hand, starts, legal_set):
        best, best_stack = None, math.inf
        for uid in assign:
            if hand.get(uid, 0) < 1:
                continue
            u = self._unit(env, me, uid)
            if u is None or u.loc in starts:
                continue  # never bolster on a home base (keep it free — the user's exception)
            target = self.build_target if uid == BERSERKER_ID else self.priest_target
            if u.stack >= target:
                continue
            a = WarChestEnv.encode_action(BOLSTER_VERB, *u.loc)
            if a in legal_set and u.stack < best_stack:
                best, best_stack = a, u.stack
        return best

    def _key_move(self, env, me, assign, hand, rs, rq, legal_set):
        best, best_v = None, -math.inf
        for uid in assign:
            if hand.get(uid, 0) < 1:
                continue
            u = self._unit(env, me, uid)
            if u is None:
                continue
            for d in range(6):
                a = WarChestEnv.encode_action(d, *u.loc)
                if a in legal_set:
                    v = self._score(rs, rq, a, me)
                    if v > best_v:
                        best, best_v = a, v
        return best

    # ------------------------------------------------------------------
    # Filler rules (coin-sinks)
    # ------------------------------------------------------------------
    def _deploy_filler_pair(self, env, me, hand, assign, legal_set):
        """Point 2: 2 coins of one (filler) unit → always deploy it. Prefer a free base
        that isn't a still-needed key home, but deploy on a reserved home if it's the only
        free base (a drawn pair is always committed)."""
        pairs = [u for u in hand if u not in KEY_UNIT_IDS and u != ROYAL_ID
                 and hand[u] >= 2 and self._room(env, me, u)]
        if not pairs:
            return None
        uid = min(pairs, key=lambda u: _IMPORTANCE_RANK.get(u, 99))
        return self._deploy_filler(env, me, uid, assign, legal_set)

    def _deploy_filler_single(self, env, me, hand, keys, assign, legal_set):
        """Deploy a single filler when it makes sense to sink a coin: BOTH home bases free
        (point 2, opening), OR both key units already deployed (then a filler may deploy
        even with a single base free). Prefer a filler whose SECOND coin is still in the
        bag — drawable next round so the deployed unit can be played immediately."""
        homes = START_BASES[me]
        both_homes_free = sum(1 for c in homes if env.board.get_unit_at(*c) is None) == 2
        both_keys_down = bool(keys) and all(self._unit(env, me, u) is not None for u in keys)
        if not (both_homes_free or both_keys_down):
            return None
        fillers = [u for u in hand if u not in KEY_UNIT_IDS and u != ROYAL_ID
                   and hand[u] >= 1 and self._room(env, me, u)]
        if not fillers:
            return None
        bag = env.state.bags[me]
        uid = min(fillers, key=lambda u: (bag.get(u, 0) < 1, _IMPORTANCE_RANK.get(u, 99)))
        return self._deploy_filler(env, me, uid, assign, legal_set)

    def _deploy_filler(self, env, me, uid, assign, legal_set):
        verb = DEPLOY_VERB_BY_UNIT[uid]
        reserved = {assign[k] for k in assign if self._unit(env, me, k) is None}
        free = [c for c in env.board.get_controlled_bases(me) if env.board.get_unit_at(*c) is None]
        for cell in sorted(free, key=lambda c: c in reserved):  # non-reserved first
            a = WarChestEnv.encode_action(verb, *cell)
            if a in legal_set:
                return a
        for a in legal_set:                                     # any legal deploy of it
            if a < SPATIAL_SIZE and WarChestEnv.decode_action(a)[0] == verb:
                return a
        return None

    def _bolster_filler(self, env, me, hand, legal_set):
        for u in env.board.units:
            if u.player_id == me and u.id not in KEY_UNIT_IDS and u.stack < 2 \
                    and hand.get(u.id, 0) >= 1:
                a = WarChestEnv.encode_action(BOLSTER_VERB, *u.loc)
                if a in legal_set:
                    return a
        return None

    def _room(self, env, me, uid):
        on_board = sum(1 for u in env.board.units if u.player_id == me and u.id == uid)
        return on_board < UNIT_BY_ID[uid].max_on_board

    # ------------------------------------------------------------------
    # Economy rules
    # ------------------------------------------------------------------
    def _recruit_key(self, env, me, hand, legal_set):
        supply = env.state.supply[me]
        for take in (BERSERKER_ID, WARRIOR_PRIEST_ID):
            if supply.get(take, 0) <= 0:
                continue
            for pay in self._pay_order(hand):
                if pay == take:
                    continue
                a = WarChestEnv.encode_recruit(take, pay)
                if a in legal_set:
                    return a
        return None

    def _recruit_filler_with_royal(self, env, me, hand, legal_set):
        """Top up the filler economy: pay the Royal coin (never a scarce unit coin) to
        recruit a filler, but only once coins in circulation (hand + bag, all types) have
        run low — i.e. most of the deck is sitting in discard, about to reshuffle, and the
        Royal coin would otherwise sit idle."""
        if hand.get(ROYAL_ID, 0) < 1:
            return None
        total_circulating = sum(hand.values()) + sum(env.state.bags[me].values())
        if total_circulating > 4:
            return None
        supply = env.state.supply[me]
        for uid in IMPORTANCE:
            if uid in KEY_UNIT_IDS or supply.get(uid, 0) <= 0:
                continue
            a = WarChestEnv.encode_recruit(uid, ROYAL_ID)
            if a in legal_set:
                return a
        return None

    def _claim_initiative(self, env, me, hand, legal_set):
        if me == env.state.initiative_owner or env.state.initiative_transferred_this_round:
            return None
        for coin in self._pay_order(hand):
            a = WarChestEnv.encode_facedown(0, coin)
            if a in legal_set:
                return a
        return None

    def _pass(self, env, me, legal):
        legal_set = set(legal)
        for coin in self._pay_order(env.state.hands[me]):
            a = WarChestEnv.encode_facedown(1, coin)
            if a in legal_set:
                return a
        return legal[0]

    def _pay_order(self, hand):
        return sorted(hand.keys(), key=lambda c: (c != ROYAL_ID, -_IMPORTANCE_RANK.get(c, -1)))

    # ------------------------------------------------------------------
    # Pending continuation — key units only, keep hunting kills
    # ------------------------------------------------------------------
    def _continuation(self, env, me, legal):
        legal_set = set(legal)
        kill = attack = claim = None
        for a in legal:
            if a >= SPATIAL_SIZE:
                continue
            verb, r, q = WarChestEnv.decode_action(a)
            actor = env.board.get_unit_at(r, q)
            if actor is None or actor.id not in KEY_UNIT_IDS:
                continue                                   # never maneuver a filler (point 6)
            if 6 <= verb <= 11:
                dr, dq = env.board.offsets[verb - 6]
                tgt = env.board.get_unit_at(r + dr, q + dq)
                if tgt is not None and tgt.stack <= 1:
                    kill = kill or a
                else:
                    attack = attack or a
            elif verb == CONTROL_VERB and actor.id != BERSERKER_ID:
                claim = claim or a                          # Berserker never claims/captures bases
        for a in (kill, attack, claim):
            if a is not None:
                return a
        r = self._recruit_key(env, me, env.state.hands[me], legal_set)
        if r is not None:
            return r
        from ..environment.warchest_env import DECLINE_ACTION_ID
        if DECLINE_ACTION_ID in legal_set:
            return DECLINE_ACTION_ID
        return self._pass(env, me, legal)

    # ------------------------------------------------------------------
    def _homes(self, me):
        left, right = sorted(START_BASES[me], key=lambda c: c[0])
        return left, right

    def _unit(self, env, me, uid):
        return next((u for u in env.board.units if u.player_id == me and u.id == uid), None)

    def _deploy_home(self, env, me, uid, home, legal_set):
        verb = DEPLOY_VERB_BY_UNIT[uid]
        for cell in [home] + [h for h in START_BASES[me] if h != home]:
            if env.board.get_unit_at(*cell) is None:
                a = WarChestEnv.encode_action(verb, *cell)
                if a in legal_set:
                    return a
        for a in legal_set:
            if a < SPATIAL_SIZE and WarChestEnv.decode_action(a)[0] == verb:
                return a
        return None

    def _score(self, rs, rq, action, me):
        state = _clone_state(rs)
        queues = {1: list(rq[1]), 2: list(rq[2])}
        return self._value_after_my_turn(state, queues, action, me)

    # ------------------------------------------------------------------
    @staticmethod
    def _classify(action_id):
        if action_id >= SPATIAL_SIZE:
            kind, _ = WarChestEnv.decode_facedown(action_id)
            return kind
        verb, _, _ = WarChestEnv.decode_action(action_id)
        if 6 <= verb <= 11:
            return 'attack'
        if verb == CONTROL_VERB:
            return 'control'
        if verb == BOLSTER_VERB:
            return 'bolster'
        if verb in DEPLOY_VERBS:
            return 'deploy'
        if verb == TACTIC_VERB:
            return 'tactic'
        if verb <= 5:
            return 'move'
        return 'select'
