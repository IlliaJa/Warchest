import copy
import numpy as np
from collections import deque


class OpponentPool:
    """Rolling window of frozen policy snapshots used as training opponents.

    Stores state_dict copies (not live Policy objects) to keep memory low.
    Each snapshot is taken once per PPO batch by default.
    """

    def __init__(self, max_size=20, snapshot_every=1):
        self._snapshots = deque(maxlen=max_size)
        self._snapshot_every = snapshot_every
        self._batch_count = 0

    def maybe_snapshot(self, policy):
        """Copy current policy weights into the pool (called after each batch update)."""
        self._batch_count += 1
        if self._batch_count % self._snapshot_every == 0:
            self._snapshots.append(copy.deepcopy(policy.state_dict()))

    def sample(self, policy_constructor, device, p_random=0.4):
        """Return (frozen_policy | None, opponent_type_str).

        opponent_type_str is 'random' when frozen_policy is None.
        Falls back to 'random' when the pool is empty.
        """
        if not self._snapshots or np.random.random() < p_random:
            return None, 'random'
        idx = np.random.randint(len(self._snapshots))
        opp = policy_constructor().to(device)
        opp.load_state_dict(self._snapshots[idx])
        opp.eval()
        return opp, 'pool'

    def __len__(self):
        return len(self._snapshots)
