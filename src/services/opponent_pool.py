import numpy as np
from collections import deque

from .bots import GreedyBot, RandomBot


class OpponentPool:
    """Training opponent selector combining random, greedy, and frozen policy snapshots.

    Weights do not need to sum to 1 — they are normalised automatically on each call.
    When no snapshots exist yet, 'pool' is excluded and the remaining weights
    are renormalised between random and greedy.
    """

    def __init__(self, max_size=20, snapshot_every=1, *, p_random, p_greedy, p_pool):
        self._snapshots = deque(maxlen=max_size)
        self._snapshot_every = snapshot_every
        self._batch_count = 0
        # Monotonic count of all snapshots ever appended (never reset by maxlen eviction).
        # Lets the parallel collector broadcast only snapshots added since the last batch
        # (see new_snapshots_since) instead of shipping the whole pool every batch.
        self._append_count = 0
        self._greedy_bot = GreedyBot()
        self._weights = {'random': p_random, 'greedy': p_greedy, 'pool': p_pool}
        # Reused across sample() calls: a pool opponent is only ever active within a
        # single (sequential) episode, so one instance can be recycled — we swap its
        # weights via load_state_dict instead of reconstructing a net + re-transferring
        # to device every episode (was a per-episode cost during rollout collection).
        self._cached_opp = None

    def set_weights(self, *, p_random, p_greedy, p_pool):
        """Replace sampling weights. Values are normalised automatically."""
        self._weights = {'random': p_random, 'greedy': p_greedy, 'pool': p_pool}

    @property
    def weights(self):
        """Current sampling weights as set_weights(**) kwargs (p_-prefixed)."""
        return {'p_random': self._weights['random'],
                'p_greedy': self._weights['greedy'],
                'p_pool': self._weights['pool']}

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

    def sample(self, policy_constructor, device):
        """Return (bot, opponent_type_str) sampled according to internal weights."""
        types = ['random', 'greedy'] if not self._snapshots else ['random', 'greedy', 'pool']
        weights = np.array([self._weights[t] for t in types], dtype=float)
        weights /= weights.sum()
        choice = np.random.choice(types, p=weights)

        if choice == 'random':
            return RandomBot(), 'random'
        if choice == 'greedy':
            return self._greedy_bot, 'greedy'
        idx = np.random.randint(len(self._snapshots))
        if self._cached_opp is None:
            self._cached_opp = policy_constructor()
        self._cached_opp.to(device)  # no-op if already on `device`
        self._cached_opp.load_state_dict(self._snapshots[idx])
        self._cached_opp.eval()
        return self._cached_opp, 'pool'

    def __len__(self):
        return len(self._snapshots)
