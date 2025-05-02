from environment.warchest_env import WarChestEnv
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
        # env.render()
        if terminated or truncated:
            print('Game is finished')
            break
    env.render_game()