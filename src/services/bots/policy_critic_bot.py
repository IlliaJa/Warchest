"""Policy-guided, critic-scored beam search — `LookaheadCriticBot`'s sibling.

`LookaheadCriticBot` cuts each node's raw legal actions down to `max_branching`
candidates with a cheap hand-crafted ordering key (`_ordering_key`) before the
expensive per-action clone/apply/critic-score work. This bot replaces *that one
step* with a trained policy: the actor's move prior ranks the candidates, so the
survivors that actually get applied and critic-scored are the moves the policy
considers most promising, not the ones a positional heuristic prefers.

Everything downstream is inherited verbatim from `LookaheadCriticBot`: the same
critic scores the resulting states (blended with `_leaf_potential`), the same
per-node beam keeps the `beam_width` best, and the same iterative deepening +
single-determinization forward-sim harness drives the search. The split mirrors
AlphaZero's decomposition — policy proposes, value evaluates — but grafted onto
this codebase's explicit alpha-beta-shaped beam rather than MCTS.

The policy carries its own (versioned) obs encoder from its checkpoint, which
need not match the critic's. Both read the *same* simulated `state`: the policy
encodes it with its encoder to get the move prior, the critic encodes the child
states with its own. The policy's action distribution is ego-centric (the board
is rotated 180° when the mover is player 2), while the search works in the
absolute action frame, so each absolute candidate is mapped to its ego index via
`WarChestEnv.remap_action` (self-inverse) before its prior is read.
"""
import glob
import logging
import os

import torch

from .lookahead_critic_bot import LookaheadCriticBot
from ..environment.warchest_env import WarChestEnv
from ..environment.obs_encoders import get_encoder
from ..policy.checkpoint import load_policy_checkpoint
from ..policy.policy import Policy

POLICY_GLOB = 'data/warchest_ppo_*.pth'

logger = logging.getLogger('warchest')


def _latest_policy_path():
    """Newest `data/warchest_ppo_*.pth`, or None if none exist. The timestamped
    `warchest_ppo_YYYYMMDD-HHMM` names sort chronologically as plain strings, so
    the lexicographic max is the most recent run — mirrors `_latest_critic_path`.
    """
    candidates = glob.glob(POLICY_GLOB)
    return max(candidates) if candidates else None


class PolicyCriticBot(LookaheadCriticBot):
    """`LookaheadCriticBot` with policy-prior candidate selection.

    Args:
        policy_path: path to a policy checkpoint saved by
            `policy.checkpoint.save_policy_checkpoint`. Defaults to
            `_latest_policy_path()` (newest `data/warchest_ppo_*.pth`). Raises
            `FileNotFoundError` if none exists and none was passed.
        critic_path, beam_width, max_branching, time_budget, see_opponent_hand,
            max_depth, gamma, opp_type, n_determinizations, stats_log_every,
            device: all as `LookaheadCriticBot` — this bot only changes how the
            `max_branching` candidates are chosen, not how they are scored.
    """

    def __init__(self, policy_path=None, *, critic_path=None, beam_width=5, max_branching=5,
                 time_budget=0.5, see_opponent_hand=True, max_depth=40, gamma=0.99,
                 opp_type='pool', n_determinizations=1, stats_log_every=20,
                 device='cpu', name='policy_critic'):
        if policy_path is None:
            policy_path = _latest_policy_path()
            if policy_path is None:
                raise FileNotFoundError(
                    f'No checkpoint matching {POLICY_GLOB} — pass policy_path '
                    f'explicitly, or train and save a policy first.'
                )
        super().__init__(critic_path=critic_path, beam_width=beam_width, max_branching=max_branching,
                          time_budget=time_budget, see_opponent_hand=see_opponent_hand,
                          max_depth=max_depth, gamma=gamma, opp_type=opp_type,
                          n_determinizations=n_determinizations, stats_log_every=stats_log_every,
                          device=device, name=name)

        meta = load_policy_checkpoint(policy_path, map_location=device)
        self._policy_encoder = get_encoder(meta['obs_version'])
        self._policy = Policy(device=device, hidden_dim=meta['hidden_dim'],
                              obs_encoder=self._policy_encoder, arch=meta['arch']).to(device)
        self._policy.load_state_dict(meta['state_dict'])
        self._policy.eval()
        logger.debug('policy_critic: policy from %s (obs v%s, hidden_dim=%d)',
                     policy_path, meta['obs_version'], meta['hidden_dim'])

    def _prune_candidates(self, state, legal, mover, max_branching):
        """Rank `legal` by the policy's move prior and keep the top `max_branching`.

        The policy is encoded once for this node (its own encoder, which may
        differ from the critic's) to get ego-frame log-probs over the whole
        action space; each absolute legal id is mapped to its ego index to read
        its prior. Ties and any ids the policy assigns equal mass are broken by
        the raw legal order, which is deterministic per state.
        """
        self._sim_env.set_state(state)
        obs = self._policy_encoder.encode(self._sim_env)
        with torch.inference_mode():
            logp = self._policy._obs_logits(obs)[0].cpu().numpy()

        def ego_idx(a):
            return WarChestEnv.remap_action(a) if mover == 2 else a

        ordered = sorted(legal, key=lambda a: logp[ego_idx(a)], reverse=True)
        return ordered[:max_branching]
