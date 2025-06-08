import numpy as np

import torch
from policy import Policy
from environment.warchest_env import WarChestEnv

def evaluate_agent(_env, n_eval_episodes, policy):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    ai_win_cnt = 0
    random_bot_win_cnt = 0
    draw_cnt = 0
    for episode in range(n_eval_episodes):
        _state, _ = _env.reset()
        ai_rewards_ep = 0
        random_bot_rewards_ep = 0

        turn_num = 0
        while True:
            # AI turn
            action, log_prob, value, _ = policy.act(_state)
            _state, reward, terminated, truncated, info = env.step(action)
            ai_rewards_ep += reward
            if not info['action'].is_valid:
                raise ValueError('Invalid action taken by the agent')
            if terminated:
                ai_win_cnt += 1
                break
            if truncated:
                draw_cnt += 1
                break

            # Random bot turn
            possible_actions = env.get_possible_actions()
            action_id = np.random.choice(possible_actions)
            _state, reward, terminated, truncated, info = env.step(action_id)
            random_bot_rewards_ep += reward
            if terminated:
                random_bot_win_cnt += 1
                break
            if truncated:
                draw_cnt += 1
                break
            turn_num += 1
        print(f'Game {episode} is finished, last turn number: {turn_num}, AI reward: {ai_rewards_ep:.1f}, Random bot reward: {random_bot_rewards_ep:.1f}')
        episode_rewards.append(ai_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f'Total games: {n_eval_episodes}, AI wins: {ai_win_cnt}, Draws: {draw_cnt}, Random bot wins: {random_bot_win_cnt}, Mean reward: {mean_reward}, Std reward: {std_reward}')

    return ai_win_cnt, draw_cnt, random_bot_win_cnt, mean_reward, std_reward

def play_ai_vs_ai(_env, policy):
    _state, _ = _env.reset()
    rewards = []
    while True:
        action, log_prob, value, _ = policy.act(_state)
        _state, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if not info['action'].is_valid:
            raise ValueError('Invalid action taken by the agent')

        if terminated or truncated:
            print('AI vs AI game is finished')
            break
    return _env, rewards

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    env = WarChestEnv(save_game_history=True, debug_mode=False)
    state, _ = env.reset()

    training_hyperparameters = {
        'hidden_dim': 64
    }

    warchest_policy = Policy(
        action_dim= env.action_space.n,
        device=device,
        hidden_dim=training_hyperparameters["hidden_dim"]).to(device)
    warchest_policy.load_state_dict(torch.load('data/warchest_policy_20250515-10:50.pth'))
    warchest_policy.eval()

    # Evaluate the agent
    total_games = 10
    ai_wins, draws, random_wins, mean_rwd, std_rwd = evaluate_agent(env, total_games, warchest_policy)

    # AI vs AI
    result_env, rewards = play_ai_vs_ai(env, warchest_policy)
    print("AI vs AI game rewards:", rewards)
    result_env.render_game()