import copy
import numpy as np
from collections import deque

from src.bots import GreedyBot, RandomBot


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
        self._greedy_bot = GreedyBot()
        self._weights = {'random': p_random, 'greedy': p_greedy, 'pool': p_pool}

    def set_weights(self, *, p_random, p_greedy, p_pool):
        """Replace sampling weights. Values are normalised automatically."""
        self._weights = {'random': p_random, 'greedy': p_greedy, 'pool': p_pool}

    def maybe_snapshot(self, policy):
        """Copy current policy weights into the pool (called after each batch update)."""
        self._batch_count += 1
        if self._batch_count % self._snapshot_every == 0:
            self._snapshots.append(copy.deepcopy(policy.state_dict()))

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
        opp = policy_constructor().to(device)
        opp.load_state_dict(self._snapshots[idx])
        opp.eval()
        return opp, 'pool'

    def __len__(self):
        return len(self._snapshots)
