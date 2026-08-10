import numpy as np
from collections import deque

from .bots import GreedyBot, RandomBot


class OpponentPool:
    """Training opponent selector combining random, greedy, and frozen policy snapshots.

    Weights do not need to sum to 1 — they are normalised automatically on each call.
    When no snapshots exist yet, 'pool' is excluded and the remaining weights
    are renormalised between random and greedy.
    """

    def __init__(self, max_size=20, snapshot_every=1, *, p_random, p_greedy, p_pool,
                 p_lookahead_critic=0.0, lookahead_critic_time_budget=0.1,
                 lookahead_critic_device='cpu', p_puct=0.0, puct_time_budget=0.1,
                 puct_device='cpu', p_random_eval=0.0, random_eval_seed=0,
                 random_eval_reply_branching=2, p_policy_theta=0.0):
        self._snapshots = deque(maxlen=max_size)
        self._snapshot_every = snapshot_every
        self._batch_count = 0
        # Monotonic count of all snapshots ever appended (never reset by maxlen eviction).
        # Lets the parallel collector broadcast only snapshots added since the last batch
        # (see new_snapshots_since) instead of shipping the whole pool every batch.
        self._append_count = 0
        self._greedy_bot = GreedyBot()
        self._weights = {'random': p_random, 'greedy': p_greedy, 'pool': p_pool,
                         'lookahead_critic': p_lookahead_critic, 'puct': p_puct,
                         'random_eval': p_random_eval, 'policy_theta': p_policy_theta}
        # RandomEvalBot: the B1 randomised-coefficient family (docs/IDEAS.md B1). One
        # instance, resampling its 8-dim leaf-evaluator θ on every `sample()` — so the
        # slice is not one opponent but a *continuum* of policy-independent playstyles,
        # which is the coverage self-play cannot generate (docs/independent_opponents.md).
        # Its cost is SimGreedyBot's (~18 ms/move, IDEAS.md Table A), an order of magnitude
        # under lookahead_critic's, so `reply_branching` is trimmed to 2 by default rather
        # than the bot's own 8: the reply ply only has to surface the punishing move, which
        # the ordering key ranks first anyway.
        self._random_eval_seed = random_eval_seed
        self._random_eval_reply_branching = random_eval_reply_branching
        self._random_eval_bot = None
        # PolicyThetaBot: the strong, fast branch of the same family — measured 0.53-0.78
        # per member against `lookahead_critic` at ~1/3 its per-move cost (docs/bots.md).
        self._policy_theta_bot = None
        # LookaheadCriticBot is a search opponent: it is eval-scoped by design
        # (docs/bots.md), so as a training opponent it runs at a much smaller
        # per-move time_budget than its own default to keep rollout throughput
        # viable. Built lazily on first sample (loads a Critic checkpoint +
        # calibrates) and reused across episodes like the greedy/pool opponents,
        # since a pool opponent is only ever active within one sequential episode.
        self._lookahead_time_budget = lookahead_critic_time_budget
        self._lookahead_device = lookahead_critic_device
        self._lookahead_bot = None
        # PuctBot: full PUCT/MCTS search opponent. Like lookahead_critic it is
        # search-scoped and runs at a small per-move budget in the rollout hot path;
        # unlike it, it also needs a *policy* checkpoint (for priors) on top of the
        # critic, so it can only be sampled once at least one `data/warchest_ppo_*.pth`
        # exists (a fresh run with no snapshot yet must keep p_puct=0). Built lazily
        # and reused the same way.
        self._puct_time_budget = puct_time_budget
        self._puct_device = puct_device
        self._puct_bot = None
        # Reused across sample() calls: a pool opponent is only ever active within a
        # single (sequential) episode, so one instance can be recycled — we swap its
        # weights via load_state_dict instead of reconstructing a net + re-transferring
        # to device every episode (was a per-episode cost during rollout collection).
        self._cached_opp = None

    def set_weights(self, *, p_random, p_greedy, p_pool, p_lookahead_critic=0.0, p_puct=0.0,
                    p_random_eval=0.0, p_policy_theta=0.0):
        """Replace sampling weights. Values are normalised automatically."""
        self._weights = {'random': p_random, 'greedy': p_greedy, 'pool': p_pool,
                         'lookahead_critic': p_lookahead_critic, 'puct': p_puct,
                         'random_eval': p_random_eval, 'policy_theta': p_policy_theta}

    @property
    def weights(self):
        """Current sampling weights as set_weights(**) kwargs (p_-prefixed)."""
        return {'p_random': self._weights['random'],
                'p_greedy': self._weights['greedy'],
                'p_pool': self._weights['pool'],
                'p_lookahead_critic': self._weights['lookahead_critic'],
                'p_puct': self._weights['puct'],
                'p_random_eval': self._weights['random_eval'],
                'p_policy_theta': self._weights['policy_theta']}

    def maybe_snapshot(self, policy):
        """Copy current policy weights into the pool (called after each batch update).

        Stored as detached CPU tensors so the snapshot is cheap to broadcast to CPU-only
        rollout workers and never pins GPU memory.
        """
        self._batch_count += 1
        if self._batch_count % self._snapshot_every == 0:
            sd = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            self._snapshots.append(sd)
            self._append_count += 1

    def append_snapshot(self, state_dict):
        """Append an externally-produced snapshot (used by workers to mirror the pool)."""
        self._snapshots.append(state_dict)
        self._append_count += 1

    def new_snapshots_since(self, seen_count):
        """Return (list_of_new_state_dicts, current_append_count) for incremental broadcast.

        `maybe_snapshot` adds at most one snapshot per batch and the collector queries every
        batch, so the "new" snapshots are always still resident in the maxlen deque (they are
        its most recent entries) — no eviction race.
        """
        n_new = self._append_count - seen_count
        if n_new <= 0:
            return [], self._append_count
        n_new = min(n_new, len(self._snapshots))
        return list(self._snapshots)[-n_new:], self._append_count

    def _get_lookahead_bot(self):
        """Lazily build (once) and return the shared LookaheadCriticBot. Imported
        here, not at module top, so pools that never sample it (every parallel
        worker before its weight is set, most eval paths) never pay the torch /
        Critic-checkpoint import + calibration cost.
        """
        if self._lookahead_bot is None:
            from .bots.lookahead_critic_bot import LookaheadCriticBot
            self._lookahead_bot = LookaheadCriticBot(
                time_budget=self._lookahead_time_budget,
                device=self._lookahead_device,
                stats_log_every=0,  # silent in the rollout hot path
            )
        return self._lookahead_bot

    def _get_puct_bot(self):
        """Lazily build (once) and return the shared PuctBot. Imported here, not at
        module top, for the same reason as `_get_lookahead_bot`: pools that never
        sample it pay none of the torch / policy+critic-checkpoint import cost.
        Loads the newest policy + critic checkpoints (raises if either is missing).
        """
        if self._puct_bot is None:
            from .bots.puct_bot import PuctBot
            self._puct_bot = PuctBot(
                time_budget=self._puct_time_budget,
                device=self._puct_device,
                stats_log_every=0,  # silent in the rollout hot path
            )
        return self._puct_bot

    def _get_policy_theta_bot(self):
        """Lazily build (once) the shared PolicyThetaBot. Unlike `random_eval` it needs a
        policy checkpoint, so a fresh run with none on disk must keep `p_policy_theta=0`.
        θ is redrawn per episode from the *verified* six (`POLICY_THETA_FAMILY`), not from
        the raw prior — a training pool cannot re-measure, and the prior contains θ that
        lose outright.
        """
        if self._policy_theta_bot is None:
            from .bots.policy_theta_bot import PolicyThetaBot
            self._policy_theta_bot = PolicyThetaBot(
                seed=self._random_eval_seed,
                resample_each_episode=True,
                device=self._lookahead_device,
            )
        return self._policy_theta_bot

    def _get_random_eval_bot(self):
        """Lazily build (once) and return the shared RandomEvalBot. Its θ is resampled per
        episode by `sample`, so one instance covers the whole family.
        """
        if self._random_eval_bot is None:
            from .bots.random_eval_bot import RandomEvalBot
            self._random_eval_bot = RandomEvalBot(
                seed=self._random_eval_seed,
                resample_each_episode=True,
                reply_branching=self._random_eval_reply_branching,
            )
        return self._random_eval_bot

    def sample(self, policy_constructor, device):
        """Return (bot, opponent_type_str) sampled according to internal weights.

        Reused bot instances get a `new_episode()` call before being handed out — this is
        the episode boundary as far as an opponent is concerned, and it is where
        `RandomEvalBot` draws the θ that defines its playstyle for this episode.
        """
        types = ['random', 'greedy']
        if self._snapshots:
            types.append('pool')
        for optional in ('lookahead_critic', 'puct', 'random_eval', 'policy_theta'):
            if self._weights.get(optional, 0.0) > 0.0:
                types.append(optional)
        weights = np.array([self._weights[t] for t in types], dtype=float)
        weights /= weights.sum()
        choice = np.random.choice(types, p=weights)

        if choice == 'random':
            return RandomBot(), 'random'
        if choice == 'greedy':
            return self._greedy_bot, 'greedy'
        if choice == 'lookahead_critic':
            return self._get_lookahead_bot(), 'lookahead_critic'
        if choice == 'puct':
            return self._get_puct_bot(), 'puct'
        if choice == 'random_eval':
            bot = self._get_random_eval_bot()
            bot.new_episode()
            return bot, 'random_eval'
        if choice == 'policy_theta':
            bot = self._get_policy_theta_bot()
            bot.new_episode()
            return bot, 'policy_theta'
        idx = np.random.randint(len(self._snapshots))
        if self._cached_opp is None:
            self._cached_opp = policy_constructor()
        self._cached_opp.to(device)  # no-op if already on `device`
        self._cached_opp.load_state_dict(self._snapshots[idx])
        self._cached_opp.eval()
        return self._cached_opp, 'pool'

    def __len__(self):
        return len(self._snapshots)
