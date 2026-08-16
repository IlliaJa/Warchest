"""`PolicyThetaBot` — the fast, strong member of the θ family (docs/IDEAS.md B1).

Built from three measurements, each of which killed a simpler idea:

  1. **θ is a behaviour dial, not a strength dial.** A 48-candidate search over θ on
     `LookaheadBot` returned the *default* θ as the winner (0.391 vs `lookahead_critic`,
     n=64); every apparent leader at n=8 regressed to the mean. The same held on
     `SimGreedyBot` in per-dial sweeps. So no coefficient vector is going to out-fight a
     search bot on its own.
  2. **The base search bot is the binding constraint**, and raising it has a ceiling too:
     `SimGreedyBot` → ~0.0, `LookaheadBot` → 0.39, and tuning `LookaheadCriticBot`'s own
     knobs (blend weight, width) tops out near 0.56 — real, but short of a decisive win,
     and it costs a critic forward pass per node.
  3. **The trained policy is already competitive at ~1/120th of the cost**: 0.53 +-0.06
     against `lookahead_critic` over 64 games, at ~0.86 ms/move against ~104 ms
     (IDEAS.md Table A). Its weakness is not evaluation, it is that it is a single
     reflex with no forward check — the documented blind spots (never bolsters, ignores
     unit tactics) are exactly the kind a one-ply simulation catches.

So this bot takes the policy as the *proposer* and a θ-weighted `HeuristicEvaluator` as
the *checker*: one policy forward ranks the legal moves, the top `top_k` get played out
as whole turns (with the opponent's best reply, inherited from `SimGreedyBot`), and the
final score adds the policy's own log-prior back in with weight `policy_weight`. That
last term is what keeps it anchored: at a large weight it degenerates to the raw policy,
at zero it is a policy-pruned `SimGreedyBot`, and in between the simulation only overrules
the policy when it sees something concrete.

The θ family then rides on the checker, not the proposer. Every member shares the policy's
strength while its heuristic half values different things — which is the point, since a
pool of opponents that all play the policy's own moves is exactly the self-play collapse
`docs/independent_opponents.md` diagnoses. Note the honest caveat that follows: with a
policy in the loop these are **not policy-independent** the way the `LookaheadBot` family
is. They are fast, strong and *behaviourally* varied; use the `RandomEvalLookaheadBot`
family when independence from the learner is what matters.

Cost is dominated by the single policy forward plus `top_k` shallow playouts — a few ms
per move, an order of magnitude under `SimGreedyBot` at full width and ~30x under
`lookahead_critic`.
"""
import glob
import logging
from collections import namedtuple

import numpy as np
import torch

from .greedy_sim_bot import SimGreedyBot
from .random_eval_bot import ThetaSampling
from .evaluation import normalize_theta
from .lookahead_bot import _clone_state
from ..environment.warchest_env import WarChestEnv
from ..environment.obs_encoders import get_encoder
from ..policy.checkpoint import load_policy_checkpoint
from ..policy.policy import Policy

POLICY_GLOB = 'data/warchest_ppo_*.pth'

logger = logging.getLogger('warchest')

# Config the family is measured at, and the reason it is not the strongest one found.
# `top_k`/`policy_weight` trade strength against variety, and the trade is *sharp*
# (all numbers vs `lookahead_critic`, 24 games/arm, 8 sampled θ):
#
#   policy_weight  top_k   mean WR   behaviour spread ratio
#        0.15        6       0.67          0.88x   <- strongest, but no family at all
#        0.05       10       0.67          0.65x
#        0.00       12       0.39          4.12x   <- real variety, some members too weak
#
# At any policy_weight above ~0, the prior swamps θ: log-prior gaps between the 1st and
# 10th candidate run to ~7 nats, so even a 0.05 weight contributes ~0.35 to the score
# against heuristic differences of ~0.05. There is no middle setting that buys both, which
# is why the family is *selected* rather than sampled: run the search at the diverse end
# and keep the θ that clear the strength bar (`src/app/search_theta.py`).
FAMILY_POLICY_WEIGHT = 0.0
FAMILY_TOP_K = 12

# The shipped family. Selected by `search_theta.py --base policy_theta` (40 candidates,
# 16/32/48 successive halving) and then **re-measured on a disjoint seed block** so these
# win rates are not the ones that selected them: seeds 9000+ chose, seeds 77000+ verified.
#
# Verification (2026-08-09, 32 games each vs `lookahead_critic` at equal 0.1 s/move):
# every member wins, the spread ratio is 3.03x the re-seeded control, and 0/6 are
# degenerate. `durability`/`economy` are ~0 in five of six — not a choice, an outcome:
# both were measured to wreck the bot (docs/bots.md), so the search eliminated them itself.
#
# **θ is recorded to full precision and every key that was measured is written out**, even
# when its value is 0. Both matter, and the second one is not pedantry: `normalize_theta`
# fills a *missing* key from `LEGACY_THETA`, where the four legacy terms are 1.0 — so
# omitting `'risk': 0.0` from the brawler below silently shipped a different bot than the
# one these win rates were measured on. `tests/test_policy_theta_bot.py` pins this.
#
# `role` is a short, stable label for reports and gauntlet column headers. It says what the
# member *does*, which `evaluation.theta_tag` deliberately does not: that function ranks a
# coefficient against its own sampling range, which is informative for a random draw and
# useless for a fixed family (it labelled four of these six `bas`).
#
# Roles name **behaviour, never rank**. An earlier revision called member 0 `str` for
# "strongest", which is wrong twice over: `str` reads as the Python type, and "strongest"
# is a property of *this* measurement — against this opponent, with this policy supplying
# the candidate prior. Re-run the search on a newer checkpoint and the ordering can move,
# leaving a name that lies. Where a member's behaviour was measured directly the role comes
# from that (`bol` bolsters 18.5 % of its moves); otherwise it comes from the θ profile.
FamilyMember = namedtuple('FamilyMember', 'role wr theta')

POLICY_THETA_FAMILY_MEMBERS = (
    # The balanced member: six live terms and no dominant one — its largest coefficient
    # sits at 0.67 of that key's own log-range, the flattest profile of any non-default
    # member (`cls` 0.99, `rac` 0.91, `bas` 0.89). It also happens to be the strongest of
    # the six here, which is why the role is *not* named for that: strength is measured
    # against one opponent on one policy prior, the shape of θ is not.
    FamilyMember('bal', 0.781, {'base': 1.265482, 'material': 1.48835, 'pos': 2.528133,
                                'risk': 0.729618, 'durability': 0.0, 'economy': 0.0,
                                'tempo': 2.16052, 'progress': 0.307532}),
    # The closer: a 5.9x weight on standing exactly one base from the win.
    FamilyMember('cls', 0.656, {'base': 1.603812, 'material': 1.567043, 'pos': 0.665571,
                                'risk': 1.019514, 'durability': 0.0, 'economy': 0.0,
                                'tempo': 1.849286, 'progress': 5.869621}),
    # Base-hungry and nearly material-blind.
    FamilyMember('bas', 0.594, {'base': 1.712699, 'material': 0.223274, 'pos': 1.281297,
                                'risk': 0.894802, 'durability': 0.0, 'economy': 0.0,
                                'tempo': 0.0, 'progress': 0.0}),
    # The default θ, kept deliberately: a family without the incumbent has no control in
    # it, and it is a perfectly good member on its own.
    FamilyMember('def', 0.562, {'base': 1.0, 'material': 1.0, 'pos': 1.0, 'risk': 1.0,
                                'durability': 0.0, 'economy': 0.0, 'tempo': 0.0,
                                'progress': 0.0}),
    # The racer: 6.3x positional, barely looks at material or risk.
    FamilyMember('rac', 0.562, {'base': 0.769936, 'material': 0.303526, 'pos': 6.259481,
                                'risk': 0.347608, 'durability': 0.0, 'economy': 0.0,
                                'tempo': 1.293789, 'progress': 0.0}),
    # The brawler, and the reason the family is worth having: it bolsters on **18.5 %** of
    # its moves against the others' ~1 %, and still beats lookahead_critic. `docs/IDEAS.md`
    # #R8 records bolster underuse as a standing blind spot; this is the first opponent in
    # the repo that both bolsters and wins. Note `risk` is 0.0 — that is the measured
    # value, and it is exactly the key whose omission produced a wrong bot once already.
    FamilyMember('bol', 0.531, {'base': 1.166047, 'material': 0.489911, 'pos': 1.389847,
                                'risk': 0.0, 'durability': 0.302504, 'economy': 0.0,
                                'tempo': 0.503599, 'progress': 1.126227}),
)

# Plain θ tuple — the form every consumer wants.
POLICY_THETA_FAMILY = tuple(m.theta for m in POLICY_THETA_FAMILY_MEMBERS)


def _latest_policy_path():
    candidates = glob.glob(POLICY_GLOB)
    return max(candidates) if candidates else None


class PolicyThetaBot(ThetaSampling, SimGreedyBot):
    """Policy proposes, θ-weighted simulation checks. See the module docstring."""

    _NAME_PREFIX = 'ptheta'

    def __init__(self, policy_path=None, policy_weight=FAMILY_POLICY_WEIGHT,
                 top_k=FAMILY_TOP_K, device='cpu', reply_branching=2, theta=None, seed=0,
                 resample_each_episode=False, family=None, name=None, **kwargs):
        """
        Args:
            policy_path: policy checkpoint; defaults to the newest `data/warchest_ppo_*.pth`.
            policy_weight: weight on the policy's log-prior in the final score. The
                heuristic side of that sum lives on the reward scale (a base is
                `SHAPING_C * winning_base_count` = 0.3, an attack 0.02) while log-priors
                run roughly -0.5 to -8 over the candidates, so this is the knob that says
                how much concrete simulated gain it takes to overrule the policy. 0 makes
                it a policy-pruned SimGreedyBot; large makes it the raw policy.
            top_k: legal moves kept by prior for simulation. The cost driver.
            reply_branching: opponent replies examined at the 2nd ply (SimGreedyBot's
                knob). Defaults to 2 rather than 8 — this bot exists to be fast, and the
                reply ply only has to surface the punishing move.
            family: pool of θ to draw from on each `new_episode()` when
                `resample_each_episode` — defaults to the verified `POLICY_THETA_FAMILY`.
                Drawing from the *verified* six rather than from the raw prior is what
                keeps every episode's opponent above the strength bar; the prior contains
                plenty of θ that lose (0.08 at the low end), and a training pool has no
                opportunity to re-measure.
        """
        self.family = tuple(POLICY_THETA_FAMILY if family is None else family)
        super().__init__(theta=theta, seed=seed, resample_each_episode=resample_each_episode,
                         name=name, reply_branching=reply_branching, **kwargs)
        self.policy_weight = float(policy_weight)
        self.top_k = int(top_k)

        if policy_path is None:
            policy_path = _latest_policy_path()
            if policy_path is None:
                raise FileNotFoundError(
                    f'No checkpoint matching {POLICY_GLOB} — pass policy_path explicitly, '
                    f'or train and save one first.')
        self.policy_path = policy_path
        meta = load_policy_checkpoint(policy_path, map_location=device)
        self._policy_encoder = get_encoder(meta['obs_version'])
        self._policy = Policy(device=device, hidden_dim=meta['hidden_dim'],
                              obs_encoder=self._policy_encoder, arch=meta['arch']).to(device)
        self._policy.load_state_dict(meta['state_dict'])
        self._policy.eval()
        self.device = device
        logger.debug('policy_theta: policy from %s (obs v%s, hidden_dim=%d)',
                     policy_path, meta['obs_version'], meta['hidden_dim'])

    def _draw_theta(self):
        """Uniform over the verified `family`, not the raw θ prior — see `__init__`."""
        return normalize_theta(self.family[int(self._rng.integers(len(self.family)))])

    def _priors(self, env, root_player):
        """Ego-frame joint log-probs over the whole action space, for `env`'s state.

        Read off the **live** env rather than the search's `_sim_env`: this is the root
        decision, the real state is already there, and encoding it directly avoids a
        `set_state` round-trip. The policy's distribution is ego-centric (the board is
        rotated when the mover is player 2), so callers map absolute ids through
        `_ego_index` before indexing this.
        """
        obs = self._policy_encoder.encode(env)
        with torch.inference_mode():
            return self._policy._obs_logits(obs)[0].cpu().numpy()

    @staticmethod
    def _ego_index(action_id, root_player):
        """Absolute action id -> the policy's ego-frame index (`remap_action` is self-inverse)."""
        return WarChestEnv.remap_action(action_id) if root_player == 2 else action_id

    def act(self, env) -> int:
        root_player = env.active_player
        legal = env.get_possible_actions()
        if len(legal) <= 1:
            return legal[0]

        logp = self._priors(env, root_player)
        prior = {a: float(logp[self._ego_index(a, root_player)]) for a in legal}
        # Sorting by prior and truncating is where the policy's strength enters: the
        # simulation never even considers a move the policy rates outside its top_k, so a
        # weak heuristic cannot drag the bot into a move the policy would never play.
        candidates = sorted(legal, key=lambda a: prior[a], reverse=True)[:self.top_k]

        root_state, root_queues = self._prepare_root(env, root_player)
        best_action, best_val = candidates[0], -np.inf
        for action_id in candidates:
            state = _clone_state(root_state)
            queues = {1: list(root_queues[1]), 2: list(root_queues[2])}
            val = self._value_after_my_turn(state, queues, action_id, root_player)
            val += self.policy_weight * prior[action_id]
            if val > best_val:
                best_val, best_action = val, action_id

        self.usage[self._classify(best_action)] += 1
        self.last_stats = {
            'legal_at_root': len(legal),
            'candidates': len(candidates),
            'best_value': best_val,
            'chosen': best_action,
        }
        return best_action
