import numpy as np

from .base import Bot


class RandomBot(Bot):
    """Selects uniformly at random from the valid-action mask."""

    def act(self, obs: dict) -> tuple[int, None, None]:
        valid = np.where(obs['valid_action_mask'] == 1)[0]
        return int(np.random.choice(valid)), None, None
