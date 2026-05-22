import logging
import os
import numpy as np

from collections import deque

import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import wandb

from policy import Policy
from environment.warchest_env import WarChestEnv, NUM_PLAYERS, WIN_REWARD, CLAIM_BASE_REWARD, LOSS_REWARD

logger = logging.getLogger('warchest')


def setup_run_logger(run_id: str) -> None:
    os.makedirs('logs', exist_ok=True)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(f'logs/run_{run_id}.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)


def train_with_gae(env, policy, optimizer, n_training_episodes, max_t, gamma, lam, print_every):
    info = {1: {}, 2: {}}
    bot1_wins_deque = deque(maxlen=100)
    wins_against_random_deque = deque(maxlen=100)
    for p_info in info.values():
        p_info['scores'] = []
        p_info['scores_deque'] = deque(maxlen=print_every)

    for i_episode in range(1, n_training_episodes + 1):
        episode_start_time = time.time()
        invalid_action_count = 0
        for p_info in info.values():
            p_info['log_probs'] = []
            p_info['values'] = []
            p_info['rewards'] = []
            p_info['entropies'] = []

        state, _ = env.reset()
        player_1_is_random = np.random.random() < 0.3
        player_2_is_random = np.random.random() < 0.3
        policy_control_both_bots = (not player_1_is_random) and (not player_2_is_random)
        is_policy_vs_random = (not player_1_is_random and player_2_is_random) or (not player_2_is_random and player_1_is_random)
        policy_pid = 1 if player_2_is_random else 2  # which player is policy-controlled (only used when is_policy_vs_random)

        winner = None
        outcome = 'truncated'

        t_inference = 0.0
        t_env_step = 0.0
        rollout_start = time.time()

        for turn_num in range(max_t):
            for pid in info:
                is_random = (player_1_is_random and pid == 1) or (player_2_is_random and pid == 2)
                if is_random:
                    action = np.random.choice(env.get_possible_actions())
                    log_prob = torch.tensor(-1e-6).to(device)
                    value = torch.tensor(0.0).to(device)
                    entropy = torch.tensor(0.0).to(device)
                else:
                    _t = time.time()
                    action, log_prob, value, entropy = policy.act(state)
                    t_inference += time.time() - _t

                info[pid]['log_probs'].append(log_prob)
                info[pid]['values'].append(value)
                info[pid]['entropies'].append(entropy)

                _t = time.time()
                state, reward, terminated, truncated, step_info = env.step(action)
                t_env_step += time.time() - _t
                info[pid]['rewards'].append(reward)

                if not step_info['action'].is_valid:
                    invalid_action_count += 1
                    role = 'random' if is_random else 'policy'
                    logger.warning(
                        f'ep={i_episode} turn={turn_num} player={pid}({role}) invalid_action={action}'
                    )
                    state, reward, terminated, truncated, step_info = env.make_random_step()

                if terminated:
                    winner = pid
                    outcome = 'win'
                    opponent_pid = 1 if pid == 2 else 2
                    info[opponent_pid]['rewards'].append(LOSS_REWARD)
                    opp_lp = info[opponent_pid]['log_probs']
                    opp_val = info[opponent_pid]['values']
                    opp_ent = info[opponent_pid]['entropies']
                    info[opponent_pid]['log_probs'].append(
                        torch.full_like(opp_lp[0], -1e-6) if opp_lp else torch.tensor(-1e-6).to(device)
                    )
                    info[opponent_pid]['values'].append(
                        torch.full_like(opp_val[0], 0.0) if opp_val else torch.tensor(0.0).to(device)
                    )
                    info[opponent_pid]['entropies'].append(
                        torch.full_like(opp_ent[0], 0.0) if opp_ent else torch.tensor(0.0).to(device)
                    )
                    if policy_control_both_bots:
                        bot1_wins_deque.append(int(pid == 1))
                    if is_policy_vs_random:
                        wins_against_random_deque.append(int(pid == policy_pid))

                if truncated:
                    if policy_control_both_bots:
                        bot1_wins_deque.append(0)
                    if is_policy_vs_random:
                        wins_against_random_deque.append(0)

                    info[1]['rewards'][-1] += LOSS_REWARD
                    info[2]['rewards'][-1] += LOSS_REWARD

                if terminated or truncated:
                    break
            if terminated or truncated:
                break

        t_rollout = time.time() - rollout_start

        gae_start = time.time()
        for pid, p_info in info.items():
            rewards = p_info['rewards']
            values = p_info['values'] + [torch.tensor(0.0).to(device)]

            gae = 0
            advantages = []
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + gamma * values[t + 1] - values[t]
                gae = delta + gamma * lam * gae
                advantages.insert(0, gae)

            returns = [adv + val for adv, val in zip(advantages, values[:-1])]

            advantages = torch.tensor(advantages).to(device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = torch.tensor(returns).to(device)

            log_probs = torch.stack(p_info['log_probs'])
            values = torch.stack(p_info['values'])
            entropies = torch.stack(p_info['entropies'])
            entropy_bonus = entropies.mean()
            entropy_coeff = 0.005 if i_episode < (n_training_episodes * 0.75) else 0.001

            actor_loss = -torch.mean(log_probs * advantages.detach())
            critic_loss = F.mse_loss(values.squeeze(), returns.detach())

            p_info['loss'] = actor_loss + critic_loss - entropy_coeff * entropy_bonus
            p_info['scores'].append(sum(rewards))
            p_info['scores_deque'].append(sum(rewards))
            p_info['entropy_bonus'] = entropy_bonus

            logger.debug(
                f'ep={i_episode} pid={pid} '
                f'adv_mean={advantages.mean().item():.3f} adv_std={advantages.std().item():.3f} '
                f'actor_loss={actor_loss.item():.4f} critic_loss={critic_loss.item():.4f} '
                f'entropy={entropy_bonus.item():.4f}'
            )

        t_gae = time.time() - gae_start

        total_grad_norm = 0.0
        loss = None
        if (not player_1_is_random) and (not player_2_is_random):
            loss = (info[1]['loss'] / 2 + info[2]['loss'] / 2)
        elif (not player_1_is_random) and player_2_is_random:
            loss = info[1]['loss']
        elif player_1_is_random and not player_2_is_random:
            loss = info[2]['loss']

        backward_start = time.time()
        if loss is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)

            for param in policy.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5

            if torch.isnan(loss):
                logger.error(f'ep={i_episode} NaN loss detected, skipping optimizer step')
            else:
                optimizer.step()

            if use_wandb:
                wandb.log({'grad_norm': total_grad_norm})
        t_backward = time.time() - backward_start

        episode_time = time.time() - episode_start_time
        loss_str = f'{loss.item():.4f}' if loss is not None else 'n/a'
        p1_role = 'rng' if player_1_is_random else 'pol'
        p2_role = 'rng' if player_2_is_random else 'pol'
        wr_self = float(np.mean(bot1_wins_deque)) if bot1_wins_deque else 0.0
        wr_rng = float(np.mean(wins_against_random_deque)) if wins_against_random_deque else 0.0
        logger.info(
            f'ep={i_episode}/{n_training_episodes} '
            f'turns={turn_num} outcome={outcome} winner=p{winner} '
            f'p1={p1_role} p2={p2_role} '
            f'score_p1={info[1]["scores"][-1]:.1f} score_p2={info[2]["scores"][-1]:.1f} '
            f'loss={loss_str} '
            f'ent={info[1]["entropy_bonus"].item():.3f} '
            f'wr_self={wr_self:.3f} wr_rng={wr_rng:.3f} '
            f'grad={total_grad_norm:.3f} invalid={invalid_action_count} '
            f't={episode_time:.2f}s '
            f'[rollout={t_rollout:.2f}s inference={t_inference:.2f}s env={t_env_step:.2f}s gae={t_gae:.3f}s bwd={t_backward:.3f}s]'
        )

        if i_episode % print_every == 0:
            logger.info(
                f'--- ep={i_episode} avg_score='
                f'{[round(np.mean(v["scores_deque"]), 1) for v in info.values()]} ---'
            )

        if use_wandb:
            wandb.log({
                'episode_time': episode_time,
                'winrate_bot1': wr_self,
                'winrate_against_random': wr_rng,
                'loss_bot1': info[1]['loss'].item(),
                'score_bot1': np.mean(info[1]['scores_deque']),
                'entropy_bonus': info[1]['entropy_bonus'].item(),
                'score_bot2': np.mean(info[2]['scores_deque']),
                'avg_log_prob_bot1': torch.mean(torch.stack(info[1]['log_probs'])).item(),
                'last_turn': turn_num,
                'invalid_actions': invalid_action_count,
            })

    return [v['scores'] for v in info.values()]


if __name__ == '__main__':
    use_wandb = bool(1)
    save_game_history = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_id = time.strftime('%Y%m%d-%H%M%S')
    setup_run_logger(run_id)
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f'run_id={run_id} device=cuda ({gpu_name})')
    else:
        logger.info(f'run_id={run_id} device=cpu')

    environment = WarChestEnv(save_game_history=save_game_history, debug_mode=False)
    obs, _ = environment.reset()

    training_hyperparameters = {
        'device': device,
        'n_training_episodes': 3000,
        'max_t': 1000,
        'gamma': 0.9,
        'lr_actor': 1e-4,
        'lr_critic': 5e-4,
        'action_space': environment.action_space.n,
        'hidden_dim': 64,
        'lambda': 0.95,
    }
    logger.info(f'hyperparameters={training_hyperparameters}')

    if use_wandb:
        run = wandb.init(
            project='warchest',
            config={
                'epochs': training_hyperparameters['n_training_episodes'],
                'learning_rate': training_hyperparameters['lr_actor'],
            }
        )
        logger.info(f'wandb_run={run.url}')

    warchest_policy = Policy(
        action_dim=training_hyperparameters['action_space'],
        device=training_hyperparameters['device'],
        hidden_dim=training_hyperparameters['hidden_dim']).to(device)
    warchest_optimizer = optim.Adam([
        {'params': warchest_policy.board_encoder.parameters(), 'lr': training_hyperparameters['lr_actor']},
        {'params': warchest_policy.unit_encoder.parameters(), 'lr': training_hyperparameters['lr_actor']},
        {'params': warchest_policy.actor_head.parameters(), 'lr': training_hyperparameters['lr_actor']},
        {'params': warchest_policy.critic_head.parameters(), 'lr': training_hyperparameters['lr_critic']},
    ])

    exception_for_raising = None
    try:
        scores = train_with_gae(
            environment,
            warchest_policy,
            warchest_optimizer,
            training_hyperparameters['n_training_episodes'],
            training_hyperparameters['max_t'],
            training_hyperparameters['gamma'],
            training_hyperparameters['lambda'],
            3,
        )
    except KeyboardInterrupt:
        logger.info('Training interrupted by user')
    except Exception as e:
        exception_for_raising = e
        logger.exception(f'Training failed: {e}')
    finally:
        if exception_for_raising is not None:
            raise exception_for_raising
        else:
            save_results = input('Save results? (y/n)')
            if save_results == 'y':
                timestamp = time.strftime('%Y%m%d-%H%M')
                filename = f'warchest_policy_{timestamp}.pth'
                os.makedirs('data', exist_ok=True)
                torch.save(warchest_policy.state_dict(), f'data/{filename}')
                logger.info(f'Model saved to data/{filename}')
