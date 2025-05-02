import numpy as np

from collections import deque

import torch
import torch.optim as optim
import time
import wandb
import pandas as pd

from policy import Policy
from environment.warchest_env import WarChestEnv, NUM_PLAYERS, LOSS_REWARD

def reinforce_old(env, policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state, add_info = env.reset()
        # Line 4 of pseudocode
        t_iter = 1
        for _ in range(min(t_iter, max_t)):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            t_iter += 1
            if terminated or truncated:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        # Compute the discounted returns at each timestep,
        # as
        #      the sum of the gamma-discounted return at time t (G_t) + the reward at time t
        #
        # In O(N) time, where N is the number of time steps
        # (this definition of the discounted return G_t follows the definition of this quantity
        # shown at page 44 of Sutton&Barto 2017 2nd draft)
        # G_t = r_(t+1) + r_(t+2) + ...

        # Given this formulation, the returns at each timestep t can be computed
        # by re-using the computed future returns G_(t+1) to compute the current return G_t
        # G_t = r_(t+1) + gamma*G_(t+1)
        # G_(t-1) = r_t + gamma* G_t
        # (this follows a dynamic programming approach, with which we memorize solutions in order
        # to avoid computing them multiple times)

        # This is correct since the above is equivalent to (see also page 46 of Sutton&Barto 2017 2nd draft)
        # G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...


        ## Given the above, we calculate the returns at timestep t as:
        #               gamma[t] * return[t] + reward[t]
        #
        ## We compute this starting from the last timestep to the first, in order
        ## to employ the formula presented above and avoid redundant computations that would be needed
        ## if we were to do it from first to last.

        ## Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
        ## thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
        ## a normal python list would instead require O(N) to do this.
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft( gamma*disc_return_t + rewards[t]   )

            ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        ## eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # Line 8: PyTorch prefers gradient descent
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores

def reinforce(env, policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    info = {1: {}, 2: {}}
    for p_info in info.values():
        p_info['scores'] = []
        p_info['scores_deque'] = deque(maxlen=print_every)

    for i_episode in range(1, n_training_episodes+1):
        for p_info in info.values():
            p_info['saved_log_probs'] = []
            p_info['rewards'] = []
        state, add_info = env.reset()

        for turn_num in range(max_t):
            for pid in info:
                action_id, log_prob = policy.act(state)
                info[pid]['saved_log_probs'].append(log_prob)
                state, reward, terminated, truncated, step_info = env.step(action_id)
                info[pid]['rewards'].append(reward)

                if not step_info['action'].is_valid:
                    # to ensure that step is not skipped
                    state, reward, terminated, truncated, step_info = env.make_random_step()

                if terminated:
                    # All previous actions led to the loss, so we add negative reward in the end
                    opponent_pid = 1 if pid == 2 else 2
                    info[opponent_pid]['rewards'][-1] += LOSS_REWARD
                if terminated or truncated:
                    break # exit from inner (pid) loop if the game is ended
            if terminated or truncated:
                break # exit from outer (t) loop if the game is ended

        for pid, p_info in info.items():
            p_info['scores_deque'].append(sum(p_info['rewards']))
            p_info['scores'].append(sum(p_info['rewards']))
            p_info['log_prob_numpy'] = [l.cpu().detach().numpy() for l in p_info['saved_log_probs']]
            p_info['log_prob_for_invalid_turn'] = [l for l, r in zip(p_info['log_prob_numpy'], p_info['rewards']) if r < 0]

            returns = deque(maxlen=max_t)
            n_steps = len(p_info['rewards'])
            for t in range(n_steps)[::-1]:
                disc_return_t = (returns[0] if len(returns)>0 else 0)
                returns.appendleft(gamma*disc_return_t + p_info['rewards'][t])

            ## standardization of the returns is employed to make training more stable
            eps = np.finfo(np.float32).eps.item()
            ## eps is the smallest representable float, which is
            # added to the standard deviation of the returns to avoid numerical instabilities
            returns = torch.tensor(list(returns))
            returns = (returns - returns.mean()) / (returns.std() + eps)

            policy_loss = []
            for log_prob, disc_return in zip(p_info['saved_log_probs'], returns):
                policy_loss.append(-log_prob * disc_return)
            policy_loss = torch.cat(policy_loss).sum()
            p_info['policy_loss'] = policy_loss

        optimizer.zero_grad()
        for p_id, p_info in info.items():
            retain_graph = p_id == 1
            p_info['policy_loss'].backward(retain_graph=retain_graph)
        # info[1]['policy_loss'].backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print(f'Episode {i_episode}\tAverage Score: {[round(np.mean(v["scores_deque"]), 1) for v in info.values()]}')

        if use_wandb:
            wandb.log({
                "loss": np.mean([v["policy_loss"].item() for v in info.values()]),
                "score": np.mean([np.mean(v["scores_deque"]) for v in info.values()]),
                'avg_log_prob': np.mean([torch.mean(torch.stack(v["saved_log_probs"])).item() for v in info.values()]),
                'avg_reward': np.mean([np.mean(v["rewards"]) for v in info.values()]),
                'last_turn': turn_num
                # 'avg_log_prob_for_invalid_turn': np.mean([np.mean(v["log_prob_for_invalid_turn"]) for v in info.values()]),
            })
    return [v["scores"] for v in info.values()]


if __name__ == '__main__':
    use_wandb = True
    save_game_history = False
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    environment = WarChestEnv(save_game_history=save_game_history, debug_mode=False)
    obs, _ = environment.reset()

    training_hyperparameters = {
        'device': device,
        "h_size": 1,
        "n_training_episodes": 1500,
        "n_evaluation_episodes": 10,
        "max_t": 1000,
        "gamma": 0.9,
        "lr": 1e-2,
        "action_space": environment.action_space.n,
        'hidden_dim': 128,
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
    try:
        scores = reinforce(environment,
                           warchest_policy,
                           warchest_optimizer,
                           training_hyperparameters["n_training_episodes"],
                           training_hyperparameters["max_t"],
                           training_hyperparameters["gamma"],
                           3)
    except KeyboardInterrupt:
        print('Interrupted')
    finally:
        save_results = input('Save results? (y/n)')
        if save_results == 'y' or True:
            timestamp = time.strftime("%Y%m%d-%H:%M")
            filename = f'warchest_policy_{timestamp}.pth'

            torch.save(warchest_policy.state_dict(), f'data/{filename}')
            print(f"Model saved as {filename}")
            if use_wandb:
                run.log_artifact(filename)
                run.log({"model": wandb.Artifact(filename, type="model")})
                run.finish(0)
                print(f"Finished wandb run")


# pd.DataFrame({'rewards': p_info['rewards'],
#               'log_probs': [l.cpu().detach().numpy() for l in p_info['saved_log_probs']],
#               'returns': returns
#               })