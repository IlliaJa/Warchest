import argparse
import glob
import os
import numpy as np

import torch
from src.services.policy.policy import Policy
from src.services.environment.warchest_env import WarChestEnv
from src.services.bots.greedy_bot import GreedyBot


def evaluate_agent(_env, n_eval_episodes, policy):
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
            action, _, _ = policy.act(_state)
            env_action = WarChestEnv.remap_action(action) if _env.active_player == 2 else action
            _state, reward, terminated, truncated, info = _env.step(env_action)
            ai_rewards_ep += reward
            if not info['action'].is_valid:
                raise ValueError('Invalid action taken by the agent')
            if terminated:
                ai_win_cnt += 1
                break
            if truncated:
                draw_cnt += 1
                break

            possible_actions = _env.get_possible_actions()
            action_id = np.random.choice(possible_actions)
            _state, reward, terminated, truncated, info = _env.step(action_id)
            random_bot_rewards_ep += reward
            if terminated:
                random_bot_win_cnt += 1
                break
            if truncated:
                draw_cnt += 1
                break
            turn_num += 1
        print(f'Game {episode} finished, turn {turn_num}, AI reward: {ai_rewards_ep:.1f}, Random reward: {random_bot_rewards_ep:.1f}')
        episode_rewards.append(ai_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f'Total: {n_eval_episodes}, AI wins: {ai_win_cnt}, Draws: {draw_cnt}, Random wins: {random_bot_win_cnt}, Mean reward: {mean_reward:.3f} +/- {std_reward:.3f}')
    return ai_win_cnt, draw_cnt, random_bot_win_cnt, mean_reward, std_reward


def play_ai_vs_ai(_env, policy):
    _state, _ = _env.reset()
    rewards = []
    while True:
        action, _, _ = policy.act(_state)
        env_action = WarChestEnv.remap_action(action) if _env.active_player == 2 else action
        _state, reward, terminated, truncated, info = _env.step(env_action)
        rewards.append(reward)
        if not info['action'].is_valid:
            _state, reward, terminated, truncated, info = _env.make_random_step()
        if terminated or truncated:
            print('AI vs AI game finished')
            break
    return _env, rewards


def play_ai_vs_greedy(_env, policy, ai_pid=1):
    greedy_pid = 3 - ai_pid
    print(f'AI vs Greedy: AI=P{ai_pid}, Greedy=P{greedy_pid}')
    bot = GreedyBot()
    _state, _ = _env.reset()
    while True:
        pid = _env.active_player
        if pid == ai_pid:
            action, _, _ = policy.act(_state)
        else:
            action, _, _ = bot.act(_state)
        env_action = WarChestEnv.remap_action(action) if pid == 2 else action
        _state, _, terminated, truncated, info = _env.step(env_action)
        if not info['action'].is_valid:
            _state, _, terminated, truncated, info = _env.make_random_step()
        if terminated:
            winner = 'AI' if _env.active_player != ai_pid else 'Greedy'
            print(f'AI vs Greedy finished — {winner} wins (turn {_env.action_count})')
            break
        if truncated:
            print(f'AI vs Greedy truncated after {_env.action_count} turns')
            break
    return _env


def _find_latest_model() -> str:
    candidates = sorted(glob.glob('data/warchest_ppo_*.pth'))
    if not candidates:
        raise FileNotFoundError('No models found in data/warchest_ppo_*.pth')
    return candidates[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a saved Warchest policy.')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to .pth file. Defaults to the latest data/warchest_ppo_*.pth.')
    parser.add_argument('--opponent', type=str, default='random', choices=['random', 'greedy'],
                        help='Opponent for the rendered game (default: random).')
    parser.add_argument('--hidden-dim', type=int, default=64)
    args = parser.parse_args()

    model_path = args.model_path or _find_latest_model()
    print(f'Loading model: {model_path}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    policy = Policy(device=device, hidden_dim=args.hidden_dim).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    # Evaluate vs random
    env = WarChestEnv(save_game_history=False)
    evaluate_agent(env, n_eval_episodes=10, policy=policy)

    # Rendered game
    env_render = WarChestEnv(save_game_history=True)
    if args.opponent == 'greedy':
        play_ai_vs_greedy(env_render, policy)
    else:
        play_ai_vs_ai(env_render, policy)
    env_render.render_game()
