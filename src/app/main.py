import sys as _sys
from pathlib import Path as _Path

# Ensure project root is on sys.path when the script is run directly
_root = str(_Path(__file__).resolve().parent.parent.parent)
if _root not in _sys.path:
    _sys.path.insert(0, _root)

from src.services.environment.warchest_env import WarChestEnv
import numpy as np


# Example usage
if __name__ == '__main__':
    env = WarChestEnv()
    obs, _ = env.reset()
    env.render()

    for i in range(200):
        possible_actions = env.get_possible_actions()
        action_id = np.random.choice(possible_actions)
        obs, reward, terminated, truncated, info = env.step(action_id)
        if terminated or truncated:
            print('Game is finished')
            break
    env.render_game()
