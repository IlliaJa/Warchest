import numpy as np

from collections import deque

import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import wandb

from policy import Policy
from environment.warchest_env import WarChestEnv, NUM_PLAYERS, WIN_REWARD, CLAIM_BASE_REWARD, LOSS_REWARD

def train_with_gae(env, policy, optimizer, n_training_episodes, max_t, gamma, lam, print_every):
    info = {1: {}, 2: {}}
    bot1_wins_deque = deque(maxlen=100)
    wins_against_random_deque = deque(maxlen=100)
    for p_info in info.values():
        p_info['scores'] = []
        p_info['scores_deque'] = deque(maxlen=print_every)

    for i_episode in range(1, n_training_episodes + 1):
        episode_start_time = time.time()
        for p_info in info.values():
            p_info['log_probs'] = []
            p_info['values'] = []
            p_info['rewards'] = []
            p_info['entropies'] = []

        state, _ = env.reset()
        player_1_is_random = np.random.random() < 0.3
        player_2_is_random = np.random.random() < 0.3
        policy_control_both_bots = (not player_1_is_random) and (not player_2_is_random)
        for turn_num in range(max_t):
            for pid in info:
                if (player_1_is_random and pid == 1) or (player_2_is_random and pid == 2):
                    action = np.random.choice(env.get_possible_actions())
                    log_prob = torch.tensor(-1e-6).to(device)
                    value = torch.tensor(0.0).to(device)
                    entropy = torch.tensor(0.0).to(device)
                else:
                    action, log_prob, value, entropy = policy.act(state)

                info[pid]['log_probs'].append(log_prob)
                info[pid]['values'].append(value)
                info[pid]['entropies'].append(entropy)

                state, reward, terminated, truncated, step_info = env.step(action)
                info[pid]['rewards'].append(reward)

                if not step_info['action'].is_valid:
                    print(f'Invalid action taken by the agent {pid}')
                    state, reward, terminated, truncated, step_info = env.make_random_step()

                if reward in (WIN_REWARD, CLAIM_BASE_REWARD):
                    opponent_pid = 1 if pid == 2 else 2
                    info[opponent_pid]['rewards'][-1] -= reward

                if terminated:
                    if policy_control_both_bots:
                        bot1_wins_deque.append(int(pid == 1))
                    if (not player_1_is_random) and player_2_is_random:
                        wins_against_random_deque.append(int(pid == 1))
                    if (not player_2_is_random) and player_1_is_random:
                        wins_against_random_deque.append(int(pid == 2))

                if truncated:
                    if policy_control_both_bots:
                        bot1_wins_deque.append(0)
                    if ((not player_1_is_random) and player_2_is_random) or ((not player_2_is_random) and player_1_is_random):
                        wins_against_random_deque.append(0)

                    # Both players lost because game is ended
                    info[1]['rewards'][-1] += LOSS_REWARD
                    info[2]['rewards'][-1] += LOSS_REWARD

                if terminated or truncated:
                    break
            if terminated or truncated:
                break

        for pid, p_info in info.items():
            rewards = p_info['rewards']
            values = p_info['values'] + [torch.tensor(0.0).to(device)]  # bootstrap with 0

            gae = 0
            advantages = []
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + gamma * values[t + 1] - values[t]
                gae = delta + gamma * lam * gae
                advantages.insert(0, gae)

            returns = [adv + val for adv, val in zip(advantages, values[:-1])]

            advantages = torch.tensor(advantages).to(device)
            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = torch.tensor(returns).to(device)

            log_probs = torch.stack(p_info['log_probs'])
            values = torch.stack(p_info['values'])
            entropies = torch.stack(p_info['entropies'])
            entropy_bonus = entropies.mean()
            entropy_coeff = 0.1 if i_episode < (n_training_episodes * 0.75) else 0.01

            actor_loss = -torch.mean(log_probs * advantages.detach())
            critic_loss = F.mse_loss(values.squeeze(), returns.detach())

            p_info['loss'] = actor_loss + critic_loss - entropy_coeff * entropy_bonus
            p_info['scores'].append(sum(rewards))
            p_info['scores_deque'].append(sum(rewards))
            p_info['entropy_bonus'] = entropy_bonus

        loss = None
        if (not player_1_is_random) and (not player_2_is_random):
            loss = (info[1]['loss'] / 2 + info[2]['loss'] / 2)
        elif (not player_1_is_random) and player_2_is_random:
            loss = info[1]['loss']
        elif player_1_is_random and not player_2_is_random:
            loss = info[2]['loss']
        else:
            # both players are random, so we do not update the model
            pass
        if loss is not None:
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            loss.backward()

            # Calculate the sum of all gradients
            total_grad_norm = 0
            for param in policy.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5

            # Log the gradient sum (norm) to wandb
            if use_wandb:
                wandb.log({'grad_norm': total_grad_norm})
            optimizer.step()

        if i_episode % print_every == 0:
            print(f"Episode {i_episode}\tAverage Score: {[round(np.mean(v['scores_deque']), 1) for v in info.values()]}")

        if use_wandb:
            wandb.log({
                'episode_time': time.time() - episode_start_time,
                'winrate_bot1': np.mean(bot1_wins_deque),
                'winrate_agaist_random': np.mean(wins_against_random_deque),
                # 'avg_loss': info[1]['loss'].item() / len(info[1]['log_probs']),
                'loss_bot1': info[1]['loss'].item(),
                'score_bot1': np.mean(info[1]['scores_deque']),
                'entropy_bonus': info[1]['entropy_bonus'].item(),
                'score_bot2': np.mean(info[2]['scores_deque']),
                'avg_log_prob_bot1': torch.mean(torch.stack(info[1]['log_probs'])).item(),
                'last_turn': turn_num
            })

    return [v['scores'] for v in info.values()]

if __name__ == '__main__':
    use_wandb = bool(1)
    save_game_history = False
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    environment = WarChestEnv(save_game_history=save_game_history, debug_mode=False)
    obs, _ = environment.reset()

    training_hyperparameters = {
        'device': device,
        "n_training_episodes": 3000,
        "max_t": 1000,
        "gamma": 0.9,
        "lr": 5e-3,
        "action_space": environment.action_space.n,
        'hidden_dim': 64,
        'lambda': 0.95
    }
    if use_wandb:
        run = wandb.init(
            project="warchest",
            config={
            "epochs": training_hyperparameters["n_training_episodes"],
            "learning_rate": training_hyperparameters["lr"]
            }
        )

    warchest_policy = Policy(
        action_dim=training_hyperparameters["action_space"],
        device=training_hyperparameters["device"],
        hidden_dim=training_hyperparameters["hidden_dim"]).to(device)
    warchest_optimizer = optim.Adam(warchest_policy.parameters(), lr=training_hyperparameters["lr"])
    exception_for_raising = None
    try:
        scores = train_with_gae(environment,
                           warchest_policy,
                           warchest_optimizer,
                           training_hyperparameters["n_training_episodes"],
                           training_hyperparameters["max_t"],
                           training_hyperparameters["gamma"],
                           training_hyperparameters["lambda"],
                           3)
    except KeyboardInterrupt:
        print('Interrupted')
    except Exception as e:
        exception_for_raising = e
    finally:
        if exception_for_raising is not None:
            raise exception_for_raising
        else:
            save_results = input('Save results? (y/n)')
            if save_results == 'y' or True:
                timestamp = time.strftime("%Y%m%d-%H:%M")
                filename = f'warchest_policy_{timestamp}.pth'

                torch.save(warchest_policy.state_dict(), f'data/{filename}')
                print(f"Model saved as {filename}")

# pd.DataFrame({'rewards': p_info['rewards'],
#               'log_probs': [l.cpu().detach().numpy() for l in p_info['saved_log_probs']],
#               'returns': returns
#               })

# pd.DataFrame({
#     'rewards': p_info['rewards'],
#     'values': values.squeeze().cpu().detach().numpy(),
#     'advantages': advantages.cpu().detach().numpy(),
#     'log_probs': log_probs.squeeze().cpu().detach().numpy(),
#     'returns': returns.cpu().detach().numpy()
# })
