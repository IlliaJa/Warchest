from abc import ABC, abstractmethod


class Bot(ABC):
    """Common interface for all game-playing agents."""

    @abstractmethod
    def act(self, obs: dict) -> tuple[int, object, object]:
        """Choose an action given the current observation.

        Args:
            obs: observation dict from the environment.

        Returns:
            action: integer action index.
            log_prob: log-probability tensor, or None for non-learnable bots.
            value: critic value tensor, or None for non-learnable bots.
        """
