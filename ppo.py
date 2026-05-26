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

from policy import Policy
from environment.warchest_env import WarChestEnv, WIN_REWARD, LOSS_REWARD, CLAIM_BASE_ACTION
from opponent_pool import OpponentPool
from rollout_buffer import RolloutBuffer

SHAPING_C = 0.05
use_wandb = False

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

    fh = logging.FileHandler(f'logs/ppo_{run_id}.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)


def copy_obs(obs):
    """Shallow-copy observation dict, copying numpy arrays to prevent aliasing."""
    return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs.items()}


def collect_episode(env, policy, opp_policy, main_pid, max_t, gamma, device, buffer):
    """Run one episode; append main-actor steps to buffer. Return episode info dict."""
    state, _ = env.reset()
    outcome = 'truncated'
    invalid_count = 0
    claims = 0
    main_score = 0.0
    turns = 0

    for turn in range(max_t):
        acting_pid = env.active_player
        turns = turn

        if acting_pid == main_pid:
            obs_before = copy_obs(state)
            action, log_prob, value, _ = policy.act(obs_before)

            phi_before = SHAPING_C * (
                len(env.board.get_controlled_bases(main_pid))
                - len(env.board.get_controlled_bases(3 - main_pid))
            )
            state, reward, terminated, truncated, step_info = env.step(action)

            if not step_info['action'].is_valid:
                invalid_count += 1
                logger.warning(f'turn={turn} main_pid={main_pid} invalid_action={action}')
                state, reward, terminated, truncated, step_info = env.make_random_step()
                log_prob = torch.tensor(0.0).to(device)
                value = torch.tensor(0.0).to(device)

            phi_after = SHAPING_C * (
                len(env.board.get_controlled_bases(main_pid))
                - len(env.board.get_controlled_bases(3 - main_pid))
            )
            shaped_reward = reward + gamma * phi_after - phi_before
            main_score += shaped_reward

            if step_info['action'].type == CLAIM_BASE_ACTION and step_info['action'].is_valid:
                claims += 1

            buffer.add_step(obs_before, action, log_prob, shaped_reward, value)
        else:
            if opp_policy is not None:
                with torch.no_grad():
                    action, _, _, _ = opp_policy.act(state)
            else:
                action = np.random.choice(env.get_possible_actions())

            state, _, terminated, truncated, step_info = env.step(action)
            if not step_info['action'].is_valid:
                state, _, terminated, truncated, step_info = env.make_random_step()

        if terminated:
            outcome = 'win' if acting_pid == main_pid else 'lose'
            if acting_pid != main_pid:
                buffer.append_terminal_reward(LOSS_REWARD)
                main_score += LOSS_REWARD
            break

        if truncated:
            buffer.append_terminal_reward(LOSS_REWARD)
            main_score += LOSS_REWARD
            break

    buffer.end_episode()
    return {
        'outcome': outcome,
        'turns': turns,
        'invalid_count': invalid_count,
        'claims': claims,
        'main_score': main_score,
        'main_pid': main_pid,
    }


def train_ppo(
    env,
    policy,
    optimizer,
    policy_constructor,
    n_batches,
    collect_episodes,
    max_t,
    gamma,
    lam,
    ppo_epochs,
    ppo_eps,
    print_every,
    device,
):
    returns_rms = RunningMeanStd()
    advantages_rms = RunningMeanStd()
    pool = OpponentPool(max_size=20, snapshot_every=1)
    buffer = RolloutBuffer()

    score_deque = deque(maxlen=print_every * collect_episodes)
    wr_vs_random = deque(maxlen=100)
    wr_vs_pool = deque(maxlen=100)
    outcome_win_deque = deque(maxlen=100)
    outcome_lose_deque = deque(maxlen=100)

    for batch_num in range(1, n_batches + 1):
        batch_start = time.time()
        buffer.clear()
        batch_eps = []

        policy.train()
        for _ in range(collect_episodes):
            main_pid = np.random.choice([1, 2])
            opp_policy, opp_type = pool.sample(policy_constructor, device)
            ep = collect_episode(env, policy, opp_policy, main_pid, max_t, gamma, device, buffer)
            ep['opp_type'] = opp_type
            batch_eps.append(ep)

        buffer.compute_gae(gamma, lam, returns_rms, advantages_rms, device)

        entropy_coeff = 0.005 if batch_num < (n_batches * 0.75) else 0.001
        last_grad_norm = 0.0
        kl_accum = 0.0
        actor_accum = 0.0
        critic_accum = 0.0

        for epoch in range(ppo_epochs):
            optimizer.zero_grad()
            n_steps = len(buffer)
            ep_kl = 0.0
            ep_actor = 0.0
            ep_critic = 0.0

            for obs, action, lp_old, adv, ret in buffer.iterate():
                action_t = torch.tensor(action).to(device)
                lp_new, ent, val = policy.evaluate_actions(obs, action_t)
                ratio = (lp_new - lp_old.to(device)).exp()
                clipped_ratio = ratio.clamp(1 - ppo_eps, 1 + ppo_eps)
                actor_loss = -torch.min(ratio * adv, clipped_ratio * adv)
                critic_loss = F.mse_loss(val.squeeze(), ret)
                step_loss = (actor_loss + critic_loss - entropy_coeff * ent) / n_steps
                step_loss.backward()
                ep_kl += (lp_old.to(device) - lp_new).detach().item()
                ep_actor += actor_loss.item()
                ep_critic += critic_loss.item()

            last_grad_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in policy.parameters() if p.grad is not None
            ) ** 0.5
            has_nan = any(
                torch.isnan(p.grad).any()
                for p in policy.parameters() if p.grad is not None
            )
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            if not has_nan:
                optimizer.step()
            else:
                logger.error(f'batch={batch_num} epoch={epoch} NaN gradient, skipping step')

            kl_accum += ep_kl / n_steps
            actor_accum += ep_actor / n_steps
            critic_accum += ep_critic / n_steps

        pool.maybe_snapshot(policy)

        for ep in batch_eps:
            score_deque.append(ep['main_score'])
            outcome_win_deque.append(int(ep['outcome'] == 'win'))
            outcome_lose_deque.append(int(ep['outcome'] == 'lose'))
            if ep['opp_type'] == 'random':
                wr_vs_random.append(int(ep['outcome'] == 'win'))
            else:
                wr_vs_pool.append(int(ep['outcome'] == 'win'))

        wr_rnd = float(np.mean(wr_vs_random)) if wr_vs_random else 0.0
        wr_pool_val = float(np.mean(wr_vs_pool)) if wr_vs_pool else 0.0
        win_rate = float(np.mean(outcome_win_deque)) if outcome_win_deque else 0.0
        lose_rate = float(np.mean(outcome_lose_deque)) if outcome_lose_deque else 0.0
        avg_kl = kl_accum / ppo_epochs
        avg_actor = actor_accum / ppo_epochs
        avg_critic = critic_accum / ppo_epochs
        avg_turns = float(np.mean([ep['turns'] for ep in batch_eps]))
        total_invalid = sum(ep['invalid_count'] for ep in batch_eps)
        outcomes_str = ' '.join(
            f"{ep['outcome'][0]}({ep['opp_type'][0]})" for ep in batch_eps
        )

        logger.info(
            f'batch={batch_num}/{n_batches} [{outcomes_str}] '
            f'score={np.mean(score_deque):.2f} '
            f'wr_rnd={wr_rnd:.3f} wr_pool={wr_pool_val:.3f} '
            f'win={win_rate:.3f} lose={lose_rate:.3f} '
            f'actor={avg_actor:.3e} critic={avg_critic:.4f} kl={avg_kl:.4f} '
            f'grad={last_grad_norm:.3f} pool={len(pool)} '
            f'turns={avg_turns:.0f} invalid={total_invalid} '
            f't={time.time() - batch_start:.2f}s'
        )

        if use_wandb:
            wandb.log({
                'score_main': float(np.mean(score_deque)),
                'winrate_vs_random': wr_rnd,
                'winrate_vs_pool': wr_pool_val,
                'win_rate': win_rate,
                'lose_rate': lose_rate,
                'actor_loss': avg_actor,
                'critic_loss': avg_critic,
                'ppo_kl': avg_kl,
                'grad_norm': last_grad_norm,
                'pool_size': len(pool),
                'avg_turns': avg_turns,
            })


if __name__ == '__main__':
    use_wandb = bool(1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    run_id = time.strftime('%Y%m%d-%H%M%S')
    setup_run_logger(run_id)
    if device.type == 'cuda':
        logger.info(f'run_id={run_id} device=cuda ({torch.cuda.get_device_name(0)})')
    else:
        logger.info(f'run_id={run_id} device=cpu')

    environment = WarChestEnv(save_game_history=False, debug_mode=False)

    hp = {
        'n_batches': 100,
        'collect_episodes': 8,
        'max_t': 1000,
        'gamma': 0.99,
        'lam': 0.95,
        'ppo_epochs': 4,
        'ppo_eps': 0.2,
        'lr_actor': 1e-4,
        'lr_critic': 5e-4,
        'action_space': environment.action_space.n,
        'hidden_dim': 64,
        'print_every': 10,
    }
    logger.info(f'hyperparameters={hp}')

    if use_wandb:
        run = wandb.init(
            project='warchest',
            config={
                'algorithm': 'ppo',
                'n_batches': hp['n_batches'],
                'collect_episodes': hp['collect_episodes'],
                'ppo_epochs': hp['ppo_epochs'],
                'ppo_eps': hp['ppo_eps'],
                'learning_rate': hp['lr_actor'],
                'gamma': hp['gamma'],
                'lam': hp['lam'],
            }
        )
        logger.info(f'wandb_run={run.url}')

    def policy_constructor():
        return Policy(
            action_dim=hp['action_space'],
            device=device,
            hidden_dim=hp['hidden_dim'],
        )

    warchest_policy = policy_constructor().to(device)
    warchest_optimizer = optim.Adam([
        {'params': warchest_policy.board_encoder.parameters(), 'lr': hp['lr_actor']},
        {'params': warchest_policy.unit_encoder.parameters(), 'lr': hp['lr_actor']},
        {'params': warchest_policy.actor_head.parameters(), 'lr': hp['lr_actor']},
        {'params': warchest_policy.critic_head.parameters(), 'lr': hp['lr_critic']},
    ])

    exception_for_raising = None
    try:
        train_ppo(
            environment,
            warchest_policy,
            warchest_optimizer,
            policy_constructor,
            hp['n_batches'],
            hp['collect_episodes'],
            hp['max_t'],
            hp['gamma'],
            hp['lam'],
            hp['ppo_epochs'],
            hp['ppo_eps'],
            hp['print_every'],
            device,
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
                filename = f'warchest_ppo_{timestamp}.pth'
                os.makedirs('data', exist_ok=True)
                torch.save(warchest_policy.state_dict(), f'data/{filename}')
                logger.info(f'Model saved to data/{filename}')
