import numpy as np

import torch
from policy import Policy
from environment.warchest_env import WarChestEnv

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    env = WarChestEnv(save_game_history=True, debug_mode=True)
    state, _ = env.reset()
    env.render()

    training_hyperparameters = {
        'hidden_dim': 128
    }

    warchest_policy = Policy(
        action_dim= env.action_space.n,
        device=device,
        hidden_dim=training_hyperparameters["hidden_dim"]).to(device)
    warchest_policy.load_state_dict(torch.load('data/warchest_policy_20250502-13:00.pth'))
    warchest_policy.eval()

    last_step = 150
    cur_step = 0
    while True:
        action, log_prob = warchest_policy.act(state)
        state, reward, terminated, truncated, info = env.step(action)
        if info['action'].is_valid:
            # env.render()
            cur_step += 1

        if terminated or truncated or cur_step >= last_step:
            print('Game is finished')
            break
    env.render_game()