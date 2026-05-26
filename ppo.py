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

from policy import Policy, Critic
from environment.warchest_env import WarChestEnv, WIN_REWARD, LOSS_REWARD, CLAIM_BASE_ACTION
from opponent_pool import OpponentPool
from rollout_buffer import RolloutBuffer
from greedy_bot import GreedyBot
from elo import EloTracker

SHAPING_C = 0.05
use_wandb = False

logger = logging.getLogger('warchest')



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


def collect_episode(env, policy, critic, opp_policy, main_pid, max_t, gamma, device, buffer):
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
            action, log_prob, _ = policy.act(obs_before)
            with torch.no_grad():
                value = critic.value_single(obs_before)

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
                    action, _, _ = opp_policy.act(state)
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
            main_bases = len(env.board.get_controlled_bases(main_pid))
            opp_bases = len(env.board.get_controlled_bases(3 - main_pid))
            diff = main_bases - opp_bases
            if diff > 0:
                trunc_reward = 0.0
            elif diff == 0:
                trunc_reward = LOSS_REWARD * 0.5
            else:
                trunc_reward = LOSS_REWARD
            buffer.append_terminal_reward(trunc_reward)
            main_score += trunc_reward
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


def eval_episode(env, policy, opp, main_pid, max_t, device):
    """Play one game for evaluation only (no gradient updates). Returns 'win'/'lose'/'truncated'."""
    state, _ = env.reset()
    for _ in range(max_t):
        acting_pid = env.active_player
        with torch.no_grad():
            if acting_pid == main_pid:
                action, _, _ = policy.act(state)
            elif opp is not None:
                action, _, _ = opp.act(state)
            else:
                action = np.random.choice(env.get_possible_actions())

        state, _, terminated, truncated, step_info = env.step(action)
        if not step_info['action'].is_valid:
            state, _, terminated, truncated, step_info = env.make_random_step()

        if terminated:
            return 'win' if acting_pid == main_pid else 'lose'
        if truncated:
            return 'truncated'
    return 'truncated'


def train_ppo(
    env,
    policy,
    critic,
    optimizer,
    policy_constructor,
    n_batches,
    collect_episodes,
    max_t,
    gamma,
    lam,
    ppo_epochs,
    ppo_eps,
    minibatch_size,
    print_every,
    device,
    eval_every=10,
    eval_episodes=20,
):
    pool = OpponentPool(max_size=20, snapshot_every=1)
    buffer = RolloutBuffer()
    greedy_bot = GreedyBot()
    elo = EloTracker()

    score_deque = deque(maxlen=print_every * collect_episodes)
    wr_vs_random = deque(maxlen=100)
    wr_vs_pool = deque(maxlen=100)

    for batch_num in range(1, n_batches + 1):
        batch_start = time.time()
        buffer.clear()
        batch_eps = []

        policy.train()
        critic.train()
        for _ in range(collect_episodes):
            main_pid = np.random.choice([1, 2])
            opp_policy, opp_type = pool.sample(policy_constructor, device)
            ep = collect_episode(env, policy, critic, opp_policy, main_pid, max_t, gamma, device, buffer)
            ep['opp_type'] = opp_type
            batch_eps.append(ep)

        buffer.compute_gae(gamma, lam, device)

        entropy_coeff = 0.001
        last_actor_grad = 0.0
        last_critic_grad = 0.0
        kl_accum = 0.0
        actor_accum = 0.0
        critic_accum = 0.0
        entropy_accum = 0.0
        clip_frac_accum = 0.0
        val_mae_accum = 0.0
        val_mean_accum = 0.0
        val_std_accum = 0.0

        actor_side_params = (
            list(policy.board_encoder.parameters())
            + list(policy.unit_encoder.parameters())
            + list(policy.actor_head.parameters())
        )

        kl_target = 0.015  # stop epoch early if approx KL exceeds this
        n_updates = 0
        early_stopped = False
        for epoch in range(ppo_epochs):
            if early_stopped:
                break
            for batch in buffer.iter_minibatches(minibatch_size, device):
                encoded = Policy.encode_board_batch(
                    batch['boards'], batch['exploration_maps'], batch['active_players']
                )
                batch['board'] = torch.tensor(encoded, dtype=torch.float32).to(device)

                lp_new, ent = policy.evaluate_actions_batch(batch)
                val = critic.value_batch(batch)

                lp_old = batch['log_probs_old']
                adv = batch['advantages']
                ret = batch['returns']

                ratio = (lp_new - lp_old).exp()
                approx_kl = ((ratio - 1) - (lp_new - lp_old)).detach().mean().item()
                if approx_kl > kl_target:
                    logger.debug(f'batch={batch_num} epoch={epoch} early stop approx_kl={approx_kl:.4f}')
                    early_stopped = True
                    break

                clipped_ratio = ratio.clamp(1 - ppo_eps, 1 + ppo_eps)
                actor_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()
                critic_loss = F.mse_loss(val, ret)
                loss = actor_loss + critic_loss - entropy_coeff * ent.mean()

                optimizer.zero_grad()
                loss.backward()

                has_nan = any(
                    torch.isnan(p.grad).any()
                    for p in policy.parameters() if p.grad is not None
                )
                last_actor_grad = torch.nn.utils.clip_grad_norm_(actor_side_params, max_norm=1.0).item()
                last_critic_grad = torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0).item()
                if not has_nan:
                    optimizer.step()
                else:
                    logger.error(f'batch={batch_num} epoch={epoch} NaN gradient, skipping step')

                kl_accum += (lp_old - lp_new).detach().mean().item()
                actor_accum += actor_loss.item()
                critic_accum += critic_loss.item()
                entropy_accum += ent.detach().mean().item()
                clip_frac_accum += ((ratio - 1.0).abs() > ppo_eps).float().mean().item()
                val_det = val.detach()
                val_mae_accum += (val_det - ret).abs().mean().item()
                val_mean_accum += val_det.mean().item()
                val_std_accum += val_det.std().item()
                n_updates += 1

        denom = max(n_updates, 1)
        avg_clip_frac = clip_frac_accum / denom
        avg_val_mae = val_mae_accum / denom
        avg_val_mean = val_mean_accum / denom
        avg_val_std = val_std_accum / denom
        logger.debug(
            f'batch={batch_num} n_updates={n_updates} '
            f'adv mean={buffer.raw_adv_mean:.4f} std={buffer.raw_adv_std:.4f} '
            f'ret mean={buffer.raw_ret_mean:.4f} std={buffer.raw_ret_std:.4f} '
            f'val mean={avg_val_mean:.4f} std={avg_val_std:.4f} '
            f'clip_frac={avg_clip_frac:.3f} val_mae={avg_val_mae:.4f}'
        )

        pool.maybe_snapshot(policy)

        if batch_num % eval_every == 0:
            policy.eval()
            critic.eval()
            greedy_wins = 0
            random_eval_wins = 0
            for _ in range(eval_episodes):
                main_pid = np.random.choice([1, 2])

                outcome = eval_episode(env, policy, greedy_bot, main_pid, max_t, device)
                if outcome == 'win':
                    elo.win('policy', 'greedy')
                    greedy_wins += 1
                elif outcome == 'lose':
                    elo.win('greedy', 'policy')
                else:
                    elo.draw('policy', 'greedy')

                outcome = eval_episode(env, policy, None, main_pid, max_t, device)
                if outcome == 'win':
                    elo.win('policy', 'random')
                    random_eval_wins += 1
                elif outcome == 'lose':
                    elo.win('random', 'policy')
                else:
                    elo.draw('policy', 'random')

            policy.train()
            critic.train()
            elo_pol = elo.rating('policy')
            elo_grdy = elo.rating('greedy')
            elo_rnd = elo.rating('random')
            logger.info(
                f'[eval] batch={batch_num} '
                f'wr_greedy={greedy_wins / eval_episodes:.3f} '
                f'wr_random={random_eval_wins / eval_episodes:.3f} '
                f'elo_policy={elo_pol:.0f} elo_greedy={elo_grdy:.0f} elo_random={elo_rnd:.0f}'
            )
            if use_wandb:
                wandb.log({
                    'elo_policy': elo_pol,
                    'elo_greedy': elo_grdy,
                    'elo_random': elo_rnd,
                    'wr_vs_greedy': greedy_wins / eval_episodes,
                    'wr_vs_random_eval': random_eval_wins / eval_episodes,
                })

        for ep in batch_eps:
            score_deque.append(ep['main_score'])
            if ep['opp_type'] == 'random':
                wr_vs_random.append(int(ep['outcome'] == 'win'))
            else:
                wr_vs_pool.append(int(ep['outcome'] == 'win'))

        wr_rnd = float(np.mean(wr_vs_random)) if wr_vs_random else 0.0
        wr_pool_val = float(np.mean(wr_vs_pool)) if wr_vs_pool else 0.0
        avg_kl = kl_accum / denom
        avg_actor = actor_accum / denom
        avg_critic = critic_accum / denom
        avg_entropy = entropy_accum / denom
        avg_turns = float(np.mean([ep['turns'] for ep in batch_eps]))
        total_invalid = sum(ep['invalid_count'] for ep in batch_eps)
        outcomes_str = ' '.join(
            f"{ep['outcome'][0]}({ep['opp_type'][0]})" for ep in batch_eps
        )

        logger.info(
            f'batch={batch_num}/{n_batches} [{outcomes_str}] '
            f'score={np.mean(score_deque):.2f} '
            f'wr_rnd={wr_rnd:.3f} wr_pool={wr_pool_val:.3f} '
            f'actor={avg_actor:.3e} critic={avg_critic:.4f} kl={avg_kl:.4f} ent={avg_entropy:.3f} '
            f'grad_a={last_actor_grad:.3f} grad_c={last_critic_grad:.3f} pool={len(pool)} '
            f'turns={avg_turns:.0f} invalid={total_invalid} '
            f't={time.time() - batch_start:.2f}s'
        )

        if use_wandb:
            wandb.log({
                'score_main': float(np.mean(score_deque)),
                'winrate_vs_random': wr_rnd,
                'winrate_vs_pool': wr_pool_val,
                'actor_loss': avg_actor,
                'critic_loss': avg_critic,
                'ppo_kl': avg_kl,
                'entropy': avg_entropy,
                'grad_norm_actor': last_actor_grad,
                'grad_norm_critic': last_critic_grad,
                'clip_frac': avg_clip_frac,
                'val_mae': avg_val_mae,
                'val_mean': avg_val_mean,
                'val_std': avg_val_std,
                'adv_std': buffer.raw_adv_std,
                'ret_mean': buffer.raw_ret_mean,
                'ret_std': buffer.raw_ret_std,
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
        'collect_episodes': 16,
        'max_t': 1000,
        'gamma': 0.99,
        'lam': 0.95,
        'ppo_epochs': 1,
        'ppo_eps': 0.2,
        'minibatch_size': 64,
        'lr_actor': 1e-4,
        'lr_critic': 1e-4,
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
                'minibatch_size': hp['minibatch_size'],
                'lr_critic': hp['lr_critic'],
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
    warchest_critic = Critic(device=device, hidden_dim=hp['hidden_dim']).to(device)
    warchest_optimizer = optim.Adam([
        {'params': warchest_policy.board_encoder.parameters(), 'lr': hp['lr_actor']},
        {'params': warchest_policy.unit_encoder.parameters(), 'lr': hp['lr_actor']},
        {'params': warchest_policy.actor_head.parameters(), 'lr': hp['lr_actor']},
        {'params': warchest_critic.parameters(), 'lr': hp['lr_critic']},
    ])

    exception_for_raising = None
    try:
        train_ppo(
            environment,
            warchest_policy,
            warchest_critic,
            warchest_optimizer,
            policy_constructor,
            hp['n_batches'],
            hp['collect_episodes'],
            hp['max_t'],
            hp['gamma'],
            hp['lam'],
            hp['ppo_epochs'],
            hp['ppo_eps'],
            hp['minibatch_size'],
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
