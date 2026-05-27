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
from src.bots import GreedyBot, RandomBot
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


class PPOTrainer:
    """PPO training loop for Warchest."""

    KL_TARGET = 0.015  # stop PPO epoch early if per-minibatch approx-KL exceeds this
    ENTROPY_COEFF = 0.001

    def __init__(self, env, policy, critic, optimizer, policy_constructor, hp, device):
        # environment and models
        self._env = env
        self._policy = policy
        self._critic = critic
        self._optimizer = optimizer
        self._policy_constructor = policy_constructor
        self._device = device

        # hyperparameters
        self._n_batches = hp['n_batches']
        self._collect_episodes = hp['collect_episodes']
        self._max_t = hp['max_t']
        self._gamma = hp['gamma']
        self._lam = hp['lam']
        self._ppo_epochs = hp['ppo_epochs']
        self._ppo_eps = hp['ppo_eps']
        self._minibatch_size = hp['minibatch_size']
        self._print_every = hp['print_every']
        self._eval_every = hp.get('eval_every', 10)
        self._eval_episodes = hp.get('eval_episodes', 20)
        self._wr_finetune_threshold = hp['wr_random_finetune_threshold']
        self._opp_weights_finetune = {
            'p_random': hp['p_random_finetune'],
            'p_greedy': hp['p_greedy_finetune'],
            'p_pool': hp['p_pool_finetune'],
        }

        # training-lifetime state (persists across batches)
        self._pool = OpponentPool(
            max_size=20,
            snapshot_every=1,
            p_random=hp['p_random_initial'],
            p_greedy=hp['p_greedy_initial'],
            p_pool=hp['p_pool_initial'],
        )
        self._buffer = RolloutBuffer()
        self._greedy_bot = GreedyBot()
        self._elo = EloTracker()
        self._finetune_active = False
        self._score_deque = deque(maxlen=self._print_every * self._collect_episodes)
        self._wr_vs_random = deque(maxlen=100)
        self._wr_vs_pool = deque(maxlen=100)
        self._wr_vs_greedy = deque(maxlen=100)

        # pre-computed once; actor-side params are needed for separate gradient clipping
        self._actor_side_params = (
            list(self._policy.board_encoder.parameters())
            + list(self._policy.unit_encoder.parameters())
            + list(self._policy.actor_head.parameters())
        )

        # batch-temporary; written by _collect_batch, read by _log_batch
        self._batch_eps: list = []
        self._batch_start: float = 0.0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self):
        for batch_num in range(1, self._n_batches + 1):
            self._batch_start = time.time()
            self._collect_batch()
            update_stats = self._run_ppo_update(batch_num)
            self._pool.maybe_snapshot(self._policy)
            self._maybe_eval(batch_num)
            self._log_batch(batch_num, update_stats)

    # ------------------------------------------------------------------
    # Episode collection
    # ------------------------------------------------------------------

    def _collect_batch(self):
        """Fill the buffer with collect_episodes episodes, then compute GAE."""
        self._buffer.clear()
        self._batch_eps = []
        self._policy.train()
        self._critic.train()
        for _ in range(self._collect_episodes):
            main_pid = np.random.choice([1, 2])
            opp, opp_type = self._pool.sample(self._policy_constructor, self._device)
            ep = self._collect_episode(opp, main_pid)
            ep['opp_type'] = opp_type
            self._batch_eps.append(ep)
        self._buffer.compute_gae(self._gamma, self._lam, self._device)

    def _collect_episode(self, opp, main_pid) -> dict:
        """Run one episode; append main-actor steps to the buffer."""
        state, _ = self._env.reset()
        outcome = 'truncated'
        invalid_count = 0
        claims = 0
        main_score = 0.0
        turns = 0

        for turn in range(self._max_t):
            acting_pid = self._env.active_player
            turns = turn

            if acting_pid == main_pid:
                obs_before = copy_obs(state)
                action, log_prob, _ = self._policy.act(obs_before)
                with torch.no_grad():
                    value = self._critic.value_single(obs_before)

                phi_before = SHAPING_C * (
                    len(self._env.board.get_controlled_bases(main_pid))
                    - len(self._env.board.get_controlled_bases(3 - main_pid))
                )
                state, reward, terminated, truncated, step_info = self._env.step(action)

                if not step_info['action'].is_valid:
                    invalid_count += 1
                    logger.warning(f'turn={turn} main_pid={main_pid} invalid_action={action}')
                    state, reward, terminated, truncated, step_info = self._env.make_random_step()
                    log_prob = torch.tensor(0.0).to(self._device)
                    value = torch.tensor(0.0).to(self._device)

                phi_after = SHAPING_C * (
                    len(self._env.board.get_controlled_bases(main_pid))
                    - len(self._env.board.get_controlled_bases(3 - main_pid))
                )
                shaped_reward = reward + self._gamma * phi_after - phi_before
                main_score += shaped_reward

                if step_info['action'].type == CLAIM_BASE_ACTION and step_info['action'].is_valid:
                    claims += 1

                self._buffer.add_step(obs_before, action, log_prob, shaped_reward, value)
            else:
                with torch.no_grad():
                    action, _, _ = opp.act(state)
                state, _, terminated, truncated, step_info = self._env.step(action)
                if not step_info['action'].is_valid:
                    state, _, terminated, truncated, step_info = self._env.make_random_step()

            if terminated:
                outcome = 'win' if acting_pid == main_pid else 'lose'
                if acting_pid != main_pid:
                    self._buffer.append_terminal_reward(LOSS_REWARD)
                    main_score += LOSS_REWARD
                break

            if truncated:
                main_bases = len(self._env.board.get_controlled_bases(main_pid))
                opp_bases = len(self._env.board.get_controlled_bases(3 - main_pid))
                diff = main_bases - opp_bases
                if diff > 0:
                    trunc_reward = 0.0
                elif diff == 0:
                    trunc_reward = LOSS_REWARD * 0.5
                else:
                    trunc_reward = LOSS_REWARD
                self._buffer.append_terminal_reward(trunc_reward)
                main_score += trunc_reward
                break

        self._buffer.end_episode()
        return {
            'outcome': outcome,
            'turns': turns,
            'invalid_count': invalid_count,
            'claims': claims,
            'main_score': main_score,
            'main_pid': main_pid,
        }

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _run_ppo_update(self, batch_num: int) -> dict:
        """Run PPO inner epochs over the current buffer. Returns averaged update stats."""
        kl_accum = 0.0
        actor_accum = 0.0
        critic_accum = 0.0
        entropy_accum = 0.0
        clip_frac_accum = 0.0
        critic_mae_accum = 0.0
        critic_mean_accum = 0.0
        critic_std_accum = 0.0
        last_actor_grad = 0.0
        last_critic_grad = 0.0
        n_updates = 0
        early_stopped = False

        for epoch in range(self._ppo_epochs):
            if early_stopped:
                break
            for batch in self._buffer.iter_minibatches(self._minibatch_size, self._device):
                encoded = Policy.encode_board_batch(
                    batch['boards'], batch['exploration_maps'], batch['active_players']
                )
                batch['board'] = torch.tensor(encoded, dtype=torch.float32).to(self._device)

                lp_new, ent = self._policy.evaluate_actions_batch(batch)
                val = self._critic.value_batch(batch)

                lp_old = batch['log_probs_old']
                adv = batch['advantages']
                ret = batch['returns']

                ratio = (lp_new - lp_old).exp()
                approx_kl = ((ratio - 1) - (lp_new - lp_old)).detach().mean().item()
                if approx_kl > self.KL_TARGET:
                    logger.debug(
                        f'batch={batch_num} epoch={epoch} '
                        f'early stop approx_kl={approx_kl:.4f}'
                    )
                    early_stopped = True
                    break

                clipped_ratio = ratio.clamp(1 - self._ppo_eps, 1 + self._ppo_eps)
                actor_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()
                critic_loss = F.mse_loss(val, ret)
                loss = actor_loss + critic_loss - self.ENTROPY_COEFF * ent.mean()

                self._optimizer.zero_grad()
                loss.backward()

                has_nan = any(
                    torch.isnan(p.grad).any()
                    for p in self._policy.parameters() if p.grad is not None
                )
                last_actor_grad = torch.nn.utils.clip_grad_norm_(
                    self._actor_side_params, max_norm=1.0
                ).item()
                last_critic_grad = torch.nn.utils.clip_grad_norm_(
                    self._critic.parameters(), max_norm=1.0
                ).item()
                if not has_nan:
                    self._optimizer.step()
                else:
                    logger.error(
                        f'batch={batch_num} epoch={epoch} NaN gradient, skipping step'
                    )

                kl_accum += (lp_old - lp_new).detach().mean().item()
                actor_accum += actor_loss.item()
                critic_accum += critic_loss.item()
                entropy_accum += ent.detach().mean().item()
                clip_frac_accum += ((ratio - 1.0).abs() > self._ppo_eps).float().mean().item()
                val_det = val.detach()
                critic_mae_accum += (val_det - ret).abs().mean().item()
                critic_mean_accum += val_det.mean().item()
                critic_std_accum += val_det.std().item()
                n_updates += 1

        denom = max(n_updates, 1)
        return {
            'avg_kl': kl_accum / denom,
            'avg_actor': actor_accum / denom,
            'avg_critic': critic_accum / denom,
            'avg_entropy': entropy_accum / denom,
            'avg_clip_frac': clip_frac_accum / denom,
            'avg_critic_mae': critic_mae_accum / denom,
            'avg_critic_mean': critic_mean_accum / denom,
            'avg_critic_std': critic_std_accum / denom,
            'last_actor_grad': last_actor_grad,
            'last_critic_grad': last_critic_grad,
            'n_updates': n_updates,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _maybe_eval(self, batch_num: int):
        if batch_num % self._eval_every != 0:
            return

        self._policy.eval()
        self._critic.eval()
        greedy_wins = 0
        random_eval_wins = 0

        for _ in range(self._eval_episodes):
            main_pid = np.random.choice([1, 2])

            outcome = self._eval_episode(self._greedy_bot, main_pid)
            if outcome == 'win':
                self._elo.win('policy', 'greedy')
                greedy_wins += 1
            elif outcome == 'lose':
                self._elo.win('greedy', 'policy')
            else:
                self._elo.draw('policy', 'greedy')

            outcome = self._eval_episode(RandomBot(), main_pid)
            if outcome == 'win':
                self._elo.win('policy', 'random')
                random_eval_wins += 1
            elif outcome == 'lose':
                self._elo.win('random', 'policy')
            else:
                self._elo.draw('policy', 'random')

        self._policy.train()
        self._critic.train()

        elo_pol = self._elo.rating('policy')
        elo_grdy = self._elo.rating('greedy')
        elo_rnd = self._elo.rating('random')
        logger.info(
            f'[eval] batch={batch_num} '
            f'wr_greedy={greedy_wins / self._eval_episodes:.3f} '
            f'wr_random={random_eval_wins / self._eval_episodes:.3f} '
            f'elo_policy={elo_pol:.0f} elo_greedy={elo_grdy:.0f} elo_random={elo_rnd:.0f}'
        )
        if use_wandb:
            wandb.log({
                'elo_policy': elo_pol,
                'elo_greedy': elo_grdy,
                'elo_random': elo_rnd,
                'wr_vs_greedy': greedy_wins / self._eval_episodes,
                'wr_vs_random_eval': random_eval_wins / self._eval_episodes,
            })

    def _eval_episode(self, opp, main_pid) -> str:
        """Play one game for evaluation only. Returns 'win' / 'lose' / 'truncated'."""
        state, _ = self._env.reset()
        for _ in range(self._max_t):
            acting_pid = self._env.active_player
            with torch.no_grad():
                if acting_pid == main_pid:
                    action, _, _ = self._policy.act(state)
                else:
                    action, _, _ = opp.act(state)
            state, _, terminated, truncated, step_info = self._env.step(action)
            if not step_info['action'].is_valid:
                state, _, terminated, truncated, step_info = self._env.make_random_step()
            if terminated:
                return 'win' if acting_pid == main_pid else 'lose'
            if truncated:
                return 'truncated'
        return 'truncated'

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_batch(self, batch_num: int, update_stats: dict):
        for ep in self._batch_eps:
            self._score_deque.append(ep['main_score'])
            if ep['opp_type'] == 'random':
                self._wr_vs_random.append(int(ep['outcome'] == 'win'))
            elif ep['opp_type'] == 'greedy':
                self._wr_vs_greedy.append(int(ep['outcome'] == 'win'))
            else:
                self._wr_vs_pool.append(int(ep['outcome'] == 'win'))

        wr_rnd = float(np.mean(self._wr_vs_random)) if self._wr_vs_random else 0.0
        wr_pool = float(np.mean(self._wr_vs_pool)) if self._wr_vs_pool else 0.0
        wr_greedy = float(np.mean(self._wr_vs_greedy)) if self._wr_vs_greedy else 0.0

        if (
            not self._finetune_active
            and len(self._wr_vs_random) >= 20
            and wr_rnd >= self._wr_finetune_threshold
        ):
            self._pool.set_weights(**self._opp_weights_finetune)
            self._finetune_active = True
            logger.info(
                f'batch={batch_num} wr_vs_random={wr_rnd:.3f} >= {self._wr_finetune_threshold} '
                f'— switching to fine-tune opponent weights {self._opp_weights_finetune}'
            )

        s = update_stats
        avg_turns = float(np.mean([ep['turns'] for ep in self._batch_eps]))
        total_invalid = sum(ep['invalid_count'] for ep in self._batch_eps)
        outcomes_str = ' '.join(
            f"{ep['outcome'][0]}({ep['opp_type'][0]})" for ep in self._batch_eps
        )

        logger.debug(
            f'batch={batch_num} n_updates={s["n_updates"]} '
            f'adv mean={self._buffer.raw_adv_mean:.4f} std={self._buffer.raw_adv_std:.4f} '
            f'ret mean={self._buffer.raw_ret_mean:.4f} std={self._buffer.raw_ret_std:.4f} '
            f'critic mean={s["avg_critic_mean"]:.4f} std={s["avg_critic_std"]:.4f} '
            f'clip_frac={s["avg_clip_frac"]:.3f} critic_mae={s["avg_critic_mae"]:.4f}'
        )
        logger.info(
            f'batch={batch_num}/{self._n_batches} [{outcomes_str}] '
            f'score={np.mean(self._score_deque):.2f} '
            f'wr_rnd={wr_rnd:.3f} wr_pool={wr_pool:.3f} wr_greedy={wr_greedy:.3f} '
            f'actor={s["avg_actor"]:.3e} critic={s["avg_critic"]:.4f} '
            f'kl={s["avg_kl"]:.4f} ent={s["avg_entropy"]:.3f} '
            f'grad_a={s["last_actor_grad"]:.3f} grad_c={s["last_critic_grad"]:.3f} '
            f'pool={len(self._pool)} turns={avg_turns:.0f} invalid={total_invalid} '
            f't={time.time() - self._batch_start:.2f}s'
        )

        if use_wandb:
            wandb.log({
                'score_main': float(np.mean(self._score_deque)),
                'winrate_vs_random': wr_rnd,
                'winrate_vs_pool': wr_pool,
                'winrate_vs_greedy_train': wr_greedy,
                'actor_loss': s['avg_actor'],
                'critic_loss': s['avg_critic'],
                'ppo_kl': s['avg_kl'],
                'entropy': s['avg_entropy'],
                'grad_norm_actor': s['last_actor_grad'],
                'grad_norm_critic': s['last_critic_grad'],
                'clip_frac': s['avg_clip_frac'],
                'critic_mae': s['avg_critic_mae'],
                'critic_mean': s['avg_critic_mean'],
                'critic_std': s['avg_critic_std'],
                'adv_std': self._buffer.raw_adv_std,
                'ret_mean': self._buffer.raw_ret_mean,
                'ret_std': self._buffer.raw_ret_std,
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
        'n_batches': 300,
        'collect_episodes': 16,
        'max_t': 1000,
        'gamma': 0.99,
        'lam': 0.95,
        'ppo_epochs': 1,
        'ppo_eps': 0.2,
        'minibatch_size': 64,
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'action_space': environment.action_space.n,
        'hidden_dim': 64,
        'print_every': 10,
        # opponent sampling weights — initial phase (random opponent included)
        'p_random_initial': 0.40,
        'p_greedy_initial': 0.20,
        'p_pool_initial': 0.40,
        # opponent sampling weights — fine-tune phase (random removed from training)
        'p_random_finetune': 0.00,
        'p_greedy_finetune': 0.40,
        'p_pool_finetune': 0.60,
        # win-rate vs random that triggers the phase switch
        'wr_random_finetune_threshold': 0.90,
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

    trainer = PPOTrainer(
        environment,
        warchest_policy,
        warchest_critic,
        warchest_optimizer,
        policy_constructor,
        hp,
        device,
    )

    exception_for_raising = None
    try:
        trainer.train()
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
