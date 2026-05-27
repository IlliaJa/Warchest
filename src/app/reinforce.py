import sys as _sys
from pathlib import Path as _Path

# Ensure project root is on sys.path when the script is run directly
_root = str(_Path(__file__).resolve().parent.parent.parent)
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import logging
import os
import sys
import numpy as np

from collections import deque

import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import wandb

from src.services.policy.policy import Policy, Critic
from src.services.environment.warchest_env import (
    WarChestEnv, NUM_PLAYERS, WIN_REWARD, CLAIM_BASE_REWARD, LOSS_REWARD, CLAIM_BASE_ACTION
)

SHAPING_C = 0.05  # potential-based shaping scale; see docs/rewards.md idea 3

logger = logging.getLogger('warchest')


class RunningMeanStd:
    """Welford online algorithm for tracking running mean and variance."""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x: np.ndarray):
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = len(x)
        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / total_count) / total_count
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var ** 0.5 + 1e-8)


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


def train_with_gae(env, policy, critic, optimizer, n_training_episodes, max_t, gamma, lam, print_every):
    info = {1: {}, 2: {}}
    bot1_wins_deque = deque(maxlen=100)
    wins_against_random_deque = deque(maxlen=100)
    for p_info in info.values():
        p_info['scores'] = []
        p_info['scores_deque'] = deque(maxlen=print_every)

    returns_rms = RunningMeanStd()
    advantages_rms = RunningMeanStd()

    for i_episode in range(1, n_training_episodes + 1):
        episode_start_time = time.time()
        invalid_action_count = 0
        claims = {1: 0, 2: 0}
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
                    action, log_prob, entropy = policy.act(state)
                    with torch.no_grad():
                        value = critic.value_single(state)
                    t_inference += time.time() - _t

                info[pid]['log_probs'].append(log_prob)
                info[pid]['values'].append(value)
                info[pid]['entropies'].append(entropy)

                phi_before = SHAPING_C * (len(env.board.get_controlled_bases(pid)) - len(env.board.get_controlled_bases(3 - pid)))
                _t = time.time()
                state, reward, terminated, truncated, step_info = env.step(action)
                t_env_step += time.time() - _t
                info[pid]['rewards'].append(reward)

                if step_info['action'].type == CLAIM_BASE_ACTION and step_info['action'].is_valid:
                    claims[pid] += 1

                if not step_info['action'].is_valid:
                    invalid_action_count += 1
                    role = 'random' if is_random else 'policy'
                    logger.warning(
                        f'ep={i_episode} turn={turn_num} player={pid}({role}) invalid_action={action}'
                    )
                    state, reward, terminated, truncated, step_info = env.make_random_step()
                    # The policy's log_prob/value/entropy were recorded for the invalid action.
                    # Zero them out so this step produces no gradient — the reward stays as
                    # INVALID_ACTION_REWARD but nothing is attributed to any policy decision.
                    info[pid]['log_probs'][-1] = torch.tensor(0.0).to(device)
                    info[pid]['values'][-1] = torch.tensor(0.0).to(device)
                    info[pid]['entropies'][-1] = torch.tensor(0.0).to(device)

                phi_after = SHAPING_C * (len(env.board.get_controlled_bases(pid)) - len(env.board.get_controlled_bases(3 - pid)))
                info[pid]['rewards'][-1] += gamma * phi_after - phi_before

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
            is_random = (player_1_is_random and pid == 1) or (player_2_is_random and pid == 2)
            p_info['scores'].append(sum(p_info['rewards']))
            p_info['scores_deque'].append(sum(p_info['rewards']))
            if is_random:
                p_info['loss'] = torch.tensor(0.0)
                p_info['entropy_bonus'] = torch.tensor(0.0)
                continue

            rewards = p_info['rewards']
            values = p_info['values'] + [torch.tensor(0.0).to(device)]

            gae = 0
            advantages = []
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + gamma * values[t + 1] - values[t]
                gae = delta + gamma * lam * gae
                advantages.insert(0, gae)

            returns = [adv + val for adv, val in zip(advantages, values[:-1])]

            raw_adv = torch.tensor(advantages)
            raw_adv_std = raw_adv.std().item()
            raw_adv_mean = raw_adv.mean().item()

            advantages = torch.tensor(advantages)
            advantages_rms.update(advantages.numpy())
            advantages = advantages_rms.normalize(advantages).to(device)
            returns = torch.tensor(returns).to(device)
            returns_rms.update(returns.detach().cpu().numpy())
            returns = returns_rms.normalize(returns)

            log_probs = torch.stack(p_info['log_probs'])
            values_stacked = torch.stack(p_info['values'])
            entropies = torch.stack(p_info['entropies'])
            entropy_bonus = entropies.mean()
            entropy_coeff = 0.005 if i_episode < (n_training_episodes * 0.75) else 0.001

            actor_loss = -torch.mean(log_probs * advantages.detach())
            critic_loss = F.mse_loss(values_stacked.squeeze(), returns.detach())

            p_info['loss'] = actor_loss + critic_loss - entropy_coeff * entropy_bonus
            p_info['entropy_bonus'] = entropy_bonus

            logger.debug(
                f'ep={i_episode} pid={pid} '
                f'raw_adv mean={raw_adv_mean:.4f} std={raw_adv_std:.4f} '
                f'norm_adv mean={advantages.mean().item():.4f} sum={advantages.sum().item():.4f} '
                f'val mean={values_stacked.mean().item():.4f} std={values_stacked.std().item():.4f} '
                f'actor_loss={actor_loss.item():.3e} critic_loss={critic_loss.item():.4f} '
                f'entropy={entropy_bonus.item():.4f}'
            )

            values = values_stacked

        t_gae = time.time() - gae_start

        total_grad_norm = 0.0
        actor_clip_norm = 0.0
        critic_clip_norm = 0.0
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

            def _grad_norm(params):
                return sum(p.grad.data.norm(2).item() ** 2 for p in params if p.grad is not None) ** 0.5
            actor_head_grad = _grad_norm(policy.actor_head.parameters())
            critic_head_grad = _grad_norm(critic.parameters())
            board_enc_grad = _grad_norm(policy.board_encoder.parameters())

            actor_side_params = (
                list(policy.board_encoder.parameters())
                + list(policy.unit_encoder.parameters())
                + list(policy.actor_head.parameters())
            )
            actor_clip_norm = torch.nn.utils.clip_grad_norm_(actor_side_params, max_norm=1.0).item()
            critic_clip_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0).item()
            logger.debug(
                f'ep={i_episode} '
                f'actor_side_preclip={actor_clip_norm:.4f} critic_side_preclip={critic_clip_norm:.4f} '
                f'actor_head={actor_head_grad:.4f} critic_head={critic_head_grad:.4f} '
                f'board_enc={board_enc_grad:.4f}'
            )

            for param in policy.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5

            if torch.isnan(loss):
                logger.error(f'ep={i_episode} NaN loss detected, skipping optimizer step')
            else:
                optimizer.step()

            if use_wandb:
                wandb.log({
                    'grad_norm_actor': actor_clip_norm,
                    'grad_norm_critic': critic_clip_norm,
                })
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
            f'claims_p1={claims[1]} claims_p2={claims[2]} '
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
        'gamma': 0.99,
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
    warchest_critic = Critic(
        device=device,
        hidden_dim=training_hyperparameters['hidden_dim']).to(device)
    warchest_optimizer = optim.Adam([
        {'params': warchest_policy.board_encoder.parameters(), 'lr': training_hyperparameters['lr_actor']},
        {'params': warchest_policy.unit_encoder.parameters(), 'lr': training_hyperparameters['lr_actor']},
        {'params': warchest_policy.actor_head.parameters(), 'lr': training_hyperparameters['lr_actor']},
        {'params': warchest_critic.parameters(), 'lr': training_hyperparameters['lr_critic']},
    ])

    exception_for_raising = None
    try:
        scores = train_with_gae(
            environment,
            warchest_policy,
            warchest_critic,
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
            sys.stdout.write('Save results? (y/n)')
            sys.stdout.flush()
            save_results = sys.stdin.buffer.readline().decode('utf-8', errors='replace').strip()
            if save_results == 'y':
                timestamp = time.strftime('%Y%m%d-%H%M')
                filename = f'warchest_policy_{timestamp}.pth'
                os.makedirs('data', exist_ok=True)
                torch.save(warchest_policy.state_dict(), f'data/{filename}')
                logger.info(f'Model saved to data/{filename}')
