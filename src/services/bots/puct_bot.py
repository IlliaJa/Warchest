"""Full PUCT/MCTS bot — the AlphaZero decomposition done properly (docs/bots.md).

`PolicyCriticBot`/`LookaheadCriticBot` are *not* MCTS: they run an explicit,
alpha-beta-shaped beam search. The policy prior there only cuts each node's raw
legal moves down to `max_branching` candidates, and among the survivors the beam
ranks purely by the critic — the prior is used once and thrown away, and it never
enters a selection formula. This bot is the real thing: a Monte-Carlo search tree
where every node keeps per-edge visit counts `N`, accumulated value `W`, and the
policy prior `P`, and each simulation descends by the PUCT rule

    argmax_a  [ ±Q(s,a) + c_puct * P(s,a) * sqrt(ΣN) / (1 + N(s,a)) ]

so the prior keeps steering exploration for the whole search, not just the first
cut. "policy proposes, value evaluates" — the AlphaZero split — but over an
actual visit-counted tree instead of a fixed-ply beam.

Why a subclass of `PolicyCriticBot`: that class already loads *both* nets this
search needs (the policy for priors, the critic for leaf values), plus the obs
encoders, the value-scale calibration, the single-determinization forward-sim
harness (`_prepare_root`/`_apply`/`_clone_state`/`_legal_from`), the real reward
plumbing (`_own_action_reward`/`_holding_reward`/`_truncation_value`), and the
`act()` wrapper that votes across `n_determinizations` independent searches. All
of that is inherited verbatim; the *only* thing this bot replaces is the search
itself — `_act_once` builds and runs a PUCT tree instead of a beam.

Design decisions specific to Warchest:

  - **Root-perspective min/max, not textbook negamax.** Turns don't strictly
    alternate (pending tactic continuations and empty-hand skips keep the same
    active player across plies — see `LookaheadBot._minimax`), and this repo's
    whole reward accounting is written in *root_player* perspective (own-action
    reward attributed only on root's plies, holding reward only on root's plies,
    terminal reward sign-flipped for the opponent). So Q/W/rewards are all stored
    in root perspective and the perspective split lives only in *selection*: at a
    node where the mover is root_player the exploitation term is `+Q` (maximize),
    at an opponent node it is `-Q` (the opponent picks whatever hurts root most).
    The exploration bonus is always added — it is about visit counts, not sides.
    This reuses `LookaheadCriticBot`'s reward decomposition unchanged rather than
    re-deriving it under a mover-perspective convention.

  - **One net eval per *node*, not per *child*.** `LookaheadCriticBot` critic-
    scores every child at every node (a forward pass per legal move, every node).
    AlphaZero-style expansion evaluates a node once: one policy forward for the
    priors over all its children, one critic forward for its own leaf value. So a
    PUCT expansion is ~two forwards regardless of branching, and simulations that
    revisit an already-expanded node pay nothing — this is what lets a real tree
    fit in a 0.1s budget where the per-child beam is depth-starved.

  - **Deterministic tree under one determinization.** Like the other search bots,
    one future-draw order is sampled per `act()` call (`_prepare_root`) and reused
    for the whole tree, so applying an action from a node always reaches the same
    child state. Each node therefore stores its own remaining draw queues and the
    tree is expanded once and descended thereafter — no re-application. Hidden-
    information variance (the draw order guessed wrong) is hedged, as elsewhere,
    by `act()`'s `n_determinizations` voting (PIMC).

  - **Leaf value = critic + heuristic blend**, the same `critic_weight` mix
    `LookaheadCriticBot` uses (0.7/0.3 by default) and for the same reason: the
    critic is only moment-matched-calibrated here, so blending in the reward-scale-
    correct `_leaf_potential` hedges its directional noise.

  - **Final move by visit count** (AlphaZero's choice): the most-visited root
    action, tie-broken by mean value in root's favour. More stable than argmax-Q
    at the low simulation counts a 0.1s budget allows, and it directly yields the
    visit distribution that expert-iteration would later use as a policy target
    (docs/history.md — "search moves become new training targets"). Optional
    root Dirichlet noise (`dirichlet_alpha`, off by default) is provided for that
    self-play-target use; it only hurts as a deterministic eval/opponent, so it is
    disabled unless asked for.

Speaks the gauntlet's `act(env)` contract and, sharing the `pool` opponent-onehot
slot like `lookahead_critic`, also drops into `opponent_pool.py` as a training
opponent (`rollout_core._opponent_env_action` routes it through `act(env)`).
"""
import logging
import math
import time

import numpy as np
import torch

from .lookahead_bot import _clone_state
from .policy_critic_bot import PolicyCriticBot
from ..environment.warchest_env import WarChestEnv

logger = logging.getLogger('warchest')


class _Node:
    """One game state in the search tree.

    `value` is only meaningful for terminal nodes (the root-perspective return
    the game/round ends on); non-terminal nodes carry `leaf_value` instead, the
    critic+heuristic blend computed once at expansion and used both as the leaf
    return and as first-play urgency for their unvisited children.
    """

    __slots__ = ('state', 'queues', 'mover', 'terminal', 'value',
                 'leaf_value', 'expanded', 'children', 'N')

    def __init__(self, state, queues, mover, terminal=False, value=0.0):
        self.state = state
        self.queues = queues
        self.mover = mover
        self.terminal = terminal
        self.value = value
        self.leaf_value = 0.0
        self.expanded = False
        self.children = {}  # action_id -> _Edge
        self.N = 0


class _Edge:
    """One candidate move out of a node. `reward` (root-perspective, undiscounted
    immediate step reward) and `child` are filled lazily the first time the edge
    is traversed, so moves the search never selects cost no clone/apply at all.
    """

    __slots__ = ('prior', 'reward', 'child', 'N', 'W')

    def __init__(self, prior):
        self.prior = prior
        self.reward = None
        self.child = None
        self.N = 0
        self.W = 0.0


class PuctBot(PolicyCriticBot):
    """MCTS with PUCT selection, policy priors, and a critic value net.

    Args:
        policy_path, critic_path, see_opponent_hand, max_depth, gamma, opp_type,
            n_determinizations, stats_log_every, device: as `PolicyCriticBot` /
            `LookaheadCriticBot`.
        c_puct: exploration constant in the PUCT bonus. Higher = trust the prior
            (explore breadth) longer before the accumulated value takes over.
        critic_weight: weight of the critic vs `_leaf_potential` in a leaf's
            value (the rest goes to the heuristic), mirroring the beam bots' 0.7/0.3.
        max_branching: keep only this many highest-prior moves as children per
            node (0/None = all legal). A soft focus so a 0.1s budget spends its
            simulations on the moves the policy rates, not on a 15-wide fan-out;
            priors are renormalised over the kept set.
        time_budget: seconds per `act()` call (before `n_determinizations`
            splitting), same contract as the other search bots. Default 0.1.
        max_simulations: hard cap on simulations per search, or None for
            time-only. A backstop for an unbounded `time_budget`.
        dirichlet_alpha: if > 0, mix Dirichlet(alpha) noise into the root priors
            (fraction `dirichlet_frac`) for self-play target generation. Off by
            default — pure exploitation for eval/opponent use.
        dirichlet_frac: mixing weight for the root noise when it is enabled.
        value_mode: `'shaped'` (default) — the standard bot: leaf blends critic +
            `_leaf_potential` and the tree accumulates real shaped edge rewards, for a
            critic on the PPO shaped-return scale. `'outcome'` — pure AlphaZero: leaf =
            critic only, no intermediate shaped rewards, for a critic distilled to
            predict the game outcome z (expert iteration). Pair `'outcome'` only with a
            z-scale critic (saved with `return_mean=0`/`return_std=1`).
        name: gauntlet/opponent label.
    """

    def __init__(self, policy_path=None, *, critic_path=None, c_puct=1.5,
                 critic_weight=0.7, max_branching=8, time_budget=0.1,
                 max_simulations=None, dirichlet_alpha=0.0, dirichlet_frac=0.25,
                 value_mode='shaped', outcome_heuristic_frac=0.2, see_opponent_hand=True,
                 max_depth=40, gamma=0.99,
                 opp_type='pool', n_determinizations=1, stats_log_every=20,
                 device='cpu', name='puct'):
        # beam_width is inherited but unused by MCTS (no per-node beam); pass 1.
        super().__init__(policy_path=policy_path, critic_path=critic_path, beam_width=1,
                          max_branching=max_branching, time_budget=time_budget,
                          see_opponent_hand=see_opponent_hand, max_depth=max_depth, gamma=gamma,
                          opp_type=opp_type, n_determinizations=n_determinizations,
                          stats_log_every=stats_log_every, device=device, name=name)
        if value_mode not in ('shaped', 'outcome'):
            raise ValueError(f"value_mode must be 'shaped' or 'outcome', got {value_mode!r}")
        self.c_puct = c_puct
        self.critic_weight = critic_weight
        self.max_simulations = max_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        # 'shaped' (default): leaf = critic_weight·critic + (1-w)·heuristic, plus the
        # real shaped edge rewards — the gauntlet/opponent bot, critic on the PPO
        # shaped-return scale. 'outcome': pure AlphaZero — leaf = critic only, no
        # intermediate shaped rewards (terminals are already ±1 outcome-scale), for a
        # critic distilled to predict the game outcome z (expert iteration). See the
        # value_mode gates in `_expand`/`_create_child`.
        self.value_mode = value_mode
        # In outcome mode the leaf is mostly the z-outcome critic, hedged with a small
        # clipped heuristic slice (`outcome_heuristic_frac`): a freshly distilled z-critic
        # is trained on very little data, so its directional noise shouldn't fully drive
        # early-round self-play. `_leaf_potential` is O(1) (base_term alone ~0.3/base), so
        # it's roughly commensurate with z ∈ [-1,1]; it's clipped to [-1,1] before the
        # blend to stay on scale. 0.0 recovers the pure-critic AlphaZero leaf.
        self.outcome_heuristic_frac = outcome_heuristic_frac
        # z is an *undiscounted* game outcome, so the backup must be undiscounted too:
        # with gamma<1 a win d plies deep backs up as gamma**d instead of 1, which the
        # critic's undiscounted z target never sees. AlphaZero uses gamma=1 for exactly
        # this. Shaped mode keeps the passed gamma (its returns are PBRS-scale).
        if value_mode == 'outcome':
            self.gamma = 1.0
        # Per-search root visit distributions and raw (pre-noise) policy priors collected
        # during one act() call (reset there): visits -> last_stats['visit_counts'] (the
        # expert-iteration policy target), priors -> last_stats['policy_argmax'] (for the
        # self-play policy/search agreement stat).
        self._search_visits = []
        self._search_priors = []

    def act(self, env) -> int:
        """As `LookaheadCriticBot.act` (iterative determinization voting), plus it
        exposes the combined root visit distribution in `last_stats['visit_counts']`.

        The inherited `act()` runs each determinization's search via `_act_once` and
        rebuilds `last_stats` from their scalar stats, but drops the per-search visit
        trees. Each `_act_once` stashes its own root visits in `self._search_visits`
        (reset here); after voting we fold them into one normalised distribution
        (absolute frame) for expert-iteration data-gen to read.
        """
        self._search_visits = []
        self._search_priors = []
        action = super().act(env)
        self.last_stats['visit_counts'] = self._combine_visit_counts(self._search_visits)
        # The policy's own top move (argmax of the raw, pre-noise priors) — self-play
        # data-gen compares it to the most-visited action to log move-level agreement,
        # the direct read on whether the search actually diverges from its prior.
        priors = self._combine_visit_counts(self._search_priors)
        self.last_stats['policy_argmax'] = max(priors, key=priors.get) if priors else None
        return action

    @staticmethod
    def _combine_visit_counts(search_visits):
        """Fold per-search root visit counts into one normalised distribution.

        `search_visits` is a list of `(n_sims, {absolute_action_id: visit_count})`.
        Determinizations run different sub-budgets (weights `[0.8, 0.2/(n-1), ...]`),
        so raw counts aren't comparable across searches; normalise each to a
        distribution, then weight by its own simulation count (the primary 0.8-budget
        search naturally dominates) and sum. With the default `n_determinizations=1`
        this is exactly that single search's normalised visit distribution.
        """
        combined, total_w = {}, 0.0
        for n_sims, visits in search_visits:
            s = sum(visits.values())
            if s <= 0:
                continue
            w = float(n_sims)
            total_w += w
            for a, n in visits.items():
                combined[a] = combined.get(a, 0.0) + w * (n / s)
        if total_w > 0:
            for a in combined:
                combined[a] /= total_w
        return combined

    # ------------------------------------------------------------------
    # Search — replaces LookaheadCriticBot's beam `_act_once`
    # ------------------------------------------------------------------

    def _act_once(self, env, root_player, legal, time_budget):
        """One PUCT search under a fresh determinization and its own time budget.

        Returns `(action, value, stats)` — the same tuple `LookaheadCriticBot.act`
        expects, so the inherited `n_determinizations` voting and stats logging
        work unchanged. `value` is the root-perspective mean value of the chosen
        move; `stats` reports expansions as `nodes_visited` and the deepest ply
        any simulation reached as `depth_reached`.
        """
        root_state, root_queues = self._prepare_root(env, root_player)
        root = _Node(root_state, root_queues, root_state.active_player)

        self._n_expansions = 0
        self._max_depth_reached = 0
        self._expand(root, root_player)
        # Snapshot the raw policy priors before any root noise is mixed in — folded into
        # last_stats['policy_argmax'] by act() for the self-play agreement stat.
        root_priors = {a: e.prior for a, e in root.children.items()}
        if self.dirichlet_alpha > 0.0 and len(root.children) > 1:
            self._add_dirichlet_noise(root)

        start = time.monotonic()
        deadline = start + time_budget
        sims = 0
        # Each simulation descends to a single new leaf (one expansion) and backs
        # its value up. A simulation is cheap (a few clone/applies + at most one
        # policy+critic forward), so we let an in-flight one finish and just stop
        # launching new ones past the deadline rather than interrupting mid-descent.
        while time.monotonic() < deadline:
            if self.max_simulations is not None and sims >= self.max_simulations:
                break
            self._simulate(root, root_player, depth=0)
            sims += 1

        best_action = self._select_final(root, root_player)
        best_edge = root.children[best_action]
        best_val = (best_edge.W / best_edge.N) if best_edge.N > 0 else None
        visit_counts = {a: e.N for a, e in root.children.items()}  # absolute frame
        self._search_visits.append((sims, visit_counts))
        self._search_priors.append((sims, root_priors))
        stats = {
            'depth_reached': self._max_depth_reached,
            'nodes_visited': self._n_expansions,
            'elapsed': time.monotonic() - start,
            'simulations': sims,
            'visit_counts': visit_counts,
        }
        return best_action, best_val, stats

    def _simulate(self, node, root_player, depth):
        """Run one PUCT simulation from `node`, returning the root-perspective
        return of the line taken (immediate edge reward + discounted child value).

        Descends by `_select` until it reaches an unexpanded node (expand it, its
        blended leaf value is the return) or a terminal node (its stored value),
        then unwinds, adding each edge's `reward + gamma * child_value` to that
        edge's running `W`/`N`. The per-level `gamma` factor reproduces the
        `gamma**ply` discounting `LookaheadCriticBot._beam_value` applies absolutely.
        """
        node.N += 1
        if node.terminal:
            return node.value
        if not node.expanded:
            self._expand(node, root_player)
            return node.leaf_value
        if depth > self._max_depth_reached:
            self._max_depth_reached = depth
        if depth >= self.max_depth:
            return node.leaf_value

        action = self._select(node, root_player)
        edge = node.children[action]
        if edge.child is None:
            self._create_child(node, edge, action, root_player)
        child = edge.child
        if child.terminal:
            child_value = child.value
        else:
            child_value = self._simulate(child, root_player, depth + 1)
        g = edge.reward + self.gamma * child_value
        edge.N += 1
        edge.W += g
        return g

    def _select(self, node, root_player):
        """PUCT-select a child action id. Exploitation is `+Q` at root_player's
        own nodes and `-Q` at the opponent's (the opponent minimises root value);
        the exploration bonus is added the same way at both. Unvisited edges take
        the node's own leaf value as first-play urgency, so an unexplored move is
        assumed roughly as good as the position it comes from — symmetric between
        the two node types, unlike a fixed `Q=0` init which would bias min nodes.
        """
        sign = 1.0 if node.mover == root_player else -1.0
        sqrt_total = math.sqrt(max(1, node.N))
        fpu = node.leaf_value
        best_score, best_action = -math.inf, None
        for action, edge in node.children.items():
            q = (edge.W / edge.N) if edge.N > 0 else fpu
            u = self.c_puct * edge.prior * sqrt_total / (1 + edge.N)
            score = sign * q + u
            if score > best_score:
                best_score, best_action = score, action
        return best_action

    def _select_final(self, node, root_player):
        """The move `act()` returns: most-visited root child, tie-broken by mean
        value in root's favour (higher Q for a root node, lower for an opponent
        one — the latter never happens at the real root, which is always
        root_player's, but kept correct for symmetry).
        """
        sign = 1.0 if node.mover == root_player else -1.0

        def key(item):
            _, edge = item
            q = (edge.W / edge.N) if edge.N > 0 else 0.0
            return (edge.N, sign * q)

        return max(node.children.items(), key=key)[0]

    # ------------------------------------------------------------------
    # Expansion / child creation
    # ------------------------------------------------------------------

    def _expand(self, node, root_player):
        """Populate `node`'s children (with policy priors) and its leaf value.

        One policy forward gives the prior over all legal moves; the top
        `max_branching` are kept as edges with priors renormalised over that set.
        One critic forward (blended with `_leaf_potential`) gives the root-
        perspective leaf value used as this node's return and its children's FPU.
        """
        self._n_expansions += 1
        legal = self._legal_from(node.state)
        priors = self._policy_priors(node.state, node.mover, legal)

        if self.max_branching and len(legal) > self.max_branching:
            kept = sorted(legal, key=lambda a: priors[a], reverse=True)[:self.max_branching]
        else:
            kept = legal
        z = sum(priors[a] for a in kept) or 1.0
        node.children = {a: _Edge(priors[a] / z) for a in kept}

        v = self._critic_root_values([node.state], root_player)[0]
        if self.value_mode == 'outcome':
            # Mostly the z-outcome critic, hedged with a small clipped heuristic slice
            # (`outcome_heuristic_frac`) so a cold-start z-critic's directional noise
            # doesn't fully drive early-round self-play. `_leaf_potential` is clipped to
            # [-1,1] to stay commensurate with the z scale; frac=0 => pure-critic leaf.
            f = self.outcome_heuristic_frac
            if f > 0.0:
                heur = float(np.clip(self._leaf_potential(node.state, root_player), -1.0, 1.0))
                node.leaf_value = (1.0 - f) * v + f * heur
            else:
                node.leaf_value = v
        else:
            heur = self._leaf_potential(node.state, root_player)
            node.leaf_value = self.critic_weight * v + (1.0 - self.critic_weight) * heur
        node.expanded = True

    def _policy_priors(self, state, mover, legal):
        """{absolute action id: policy probability} over `legal`.

        The policy is ego-centric (board rotated 180° when the mover is player 2),
        so each absolute legal id is mapped to its ego index via `remap_action`
        before its prior is read — the same mapping `PolicyCriticBot._prune_candidates`
        uses. `_obs_logits` returns masked joint *log-probs* (illegal ids at -inf),
        so `exp` gives a distribution already zero on illegal moves.
        """
        self._sim_env.set_state(state)
        obs = self._policy_encoder.encode(self._sim_env)
        with torch.inference_mode():
            logp = self._policy._obs_logits(obs)[0].cpu().numpy()
        probs = np.exp(logp)
        return {a: float(probs[WarChestEnv.remap_action(a) if mover == 2 else a]) for a in legal}

    def _create_child(self, node, edge, action, root_player):
        """Apply `action` from `node` (once, deterministic under the fixed
        determinization) and fill `edge.reward` + `edge.child`.

        Mirrors `LookaheadCriticBot._beam_value`'s per-child accounting exactly,
        expressed as an undiscounted immediate reward (the per-level `gamma` in
        `_simulate` supplies the discount): a game-ending move carries only its
        sign-flipped terminal reward; any other move carries own-action reward
        plus holding reward, and a move that hits the round limit gets a terminal
        child holding the truncation value (backed up one `gamma` deeper).
        """
        mover = node.mover
        child_state = _clone_state(node.state)
        child_queues = {1: list(node.queues[1]), 2: list(node.queues[2])}
        result = self._apply(child_state, child_queues, action)
        own_action = (result.player_id == root_player)

        if result.finishes_game:
            edge.reward = result.reward if own_action else -result.reward
            edge.child = _Node(None, None, None, terminal=True, value=0.0)
            return

        if self.value_mode == 'outcome':
            # No intermediate shaped rewards: the z-critic leaf carries the whole future
            # outcome; only terminals (±1) and the truncation value (-1, set below)
            # contribute directly.
            edge.reward = 0.0
        else:
            holding = self._holding_reward(node.state, root_player) if mover == root_player else 0.0
            edge.reward = (self._own_action_reward(result) if own_action else 0.0) + holding
        if child_state.round_number >= self._sim_env.max_rounds:
            if self.value_mode == 'outcome':
                # Truncation = neither side forced a win before the round limit (the bots
                # circled). Score it -1 for root, matching the z=-1 label self-play now
                # gives truncated games (SelfPlayDataset.label_last). Caveat: in one
                # root-perspective tree this lets the opponent (a root-value minimiser)
                # steer *toward* truncation — a deliberate conservative bias that pushes
                # root to actually close the game out rather than stall.
                trunc = -1.0
            else:
                trunc = self._truncation_value(child_state, root_player)
            edge.child = _Node(None, None, None, terminal=True, value=trunc)
        else:
            edge.child = _Node(child_state, child_queues, child_state.active_player)

    def _add_dirichlet_noise(self, root):
        """Mix Dirichlet(alpha) noise into the root priors (self-play only)."""
        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for action, n in zip(actions, noise):
            edge = root.children[action]
            edge.prior = (1.0 - self.dirichlet_frac) * edge.prior + self.dirichlet_frac * float(n)
