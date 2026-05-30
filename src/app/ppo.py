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
from src.services.environment.warchest_env import WarChestEnv, WIN_REWARD, LOSS_REWARD, CLAIM_BASE_ACTION, DEPLOY_ACTION
from src.services.environment.game_state import DECK
from src.services.opponent_pool import OpponentPool
from src.utils.rollout_buffer import RolloutBuffer
from src.services.bots import GreedyBot, RandomBot
from src.utils.elo import EloTracker

SHAPING_C = 0.05
use_wandb = False

OPP_TYPE_IDX = {'random': 0, 'greedy': 1, 'pool': 2}

logger = logging.getLogger('warchest')


class ReturnNormalizer:
    """Exponential moving average of return mean/std for critic target whitening (A2).

    The critic is trained on normalised returns so its loss scale stays stable as the
    return distribution shifts. At rollout time the critic output is denormalised before
    being stored as V in the buffer, keeping GAE in the original reward scale.
    """

    def __init__(self, alpha=0.1):
        self._alpha = alpha
        self._mean = 0.0
        self._std = 1.0
        self._initialised = False

    def update(self, returns_tensor):
        m = returns_tensor.mean().item()
        s = max(returns_tensor.std().item(), 1e-6)
        if not self._initialised:
            self._mean = m
            self._std = s
            self._initialised = True
        else:
            self._mean = (1 - self._alpha) * self._mean + self._alpha * m
            self._std = max((1 - self._alpha) * self._std + self._alpha * s, 1e-6)

    def normalize(self, x):
        return (x - self._mean) / self._std

    def denormalize(self, x):
        return x * self._std + self._mean


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

    def __init__(self, env, policy, critic, actor_optimizer, critic_optimizer, policy_constructor, hp, device):
        # environment and models
        self._env = env
        self._policy = policy
        self._critic = critic
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
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
        self._entropy_coeff = hp['entropy_coeff']
        self._holding_reward_rate = hp['holding_reward_rate']
        self._minibatch_size = hp['minibatch_size']
        self._print_every = hp['print_every']
        self._eval_every = hp.get('eval_every', 10)
        self._eval_episodes = hp.get('eval_episodes', 20)
        self._wr_finetune_threshold = hp['wr_random_finetune_threshold']
        self._opp_weights_initial = {
            'p_random': hp['p_random_initial'],
            'p_greedy': hp['p_greedy_initial'],
            'p_pool': hp['p_pool_initial'],
        }
        self._opp_weights_finetune = {
            'p_random': hp['p_random_finetune'],
            'p_greedy': hp['p_greedy_finetune'],
            'p_pool': hp['p_pool_finetune'],
        }

        # training-lifetime state (persists across batches)
        self._pool = OpponentPool(
            max_size=20,
            snapshot_every=3,
            p_random=hp['p_random_initial'],
            p_greedy=hp['p_greedy_initial'],
            p_pool=hp['p_pool_initial'],
        )
        self._buffer = RolloutBuffer()
        self._greedy_bot = GreedyBot()
        self._elo = EloTracker()
        self._score_deque = deque(maxlen=self._print_every * self._collect_episodes)
        self._wr_vs_pool = deque(maxlen=100)
        self._wr_vs_greedy = deque(maxlen=100)

        # pre-computed once; actor-side params are needed for separate gradient clipping
        self._actor_side_params = list(self._policy.parameters())

        self._ret_normalizer = ReturnNormalizer()

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
            ep = self._collect_episode(opp, main_pid, opp_type)
            self._batch_eps.append(ep)
        self._buffer.compute_gae(self._gamma, self._lam, self._device)
        self._ret_normalizer.update(self._buffer.returns)

    def _collect_episode(self, opp, main_pid, opp_type) -> dict:
        """Run one episode; append main-actor steps to the buffer."""
        state, _ = self._env.reset()
        outcome = 'truncated'
        invalid_count = 0
        claims = 0
        deploys = 0
        deploy_turns = []
        main_score = 0.0
        turns = 0

        opp_onehot = np.zeros(len(OPP_TYPE_IDX), dtype=np.float32)
        opp_onehot[OPP_TYPE_IDX[opp_type]] = 1.0
        opp_onehot_t = torch.tensor(opp_onehot, dtype=torch.float32).unsqueeze(0).to(self._device)

        for turn in range(self._max_t):
            acting_pid = self._env.active_player
            turns = turn

            if acting_pid == main_pid:
                obs_before = copy_obs(state)
                action, log_prob, _ = self._policy.act(obs_before)
                with torch.no_grad():
                    value_norm = self._critic.value_single(obs_before, opp_onehot_t)
                    value = torch.tensor(
                        self._ret_normalizer.denormalize(value_norm.item()),
                        dtype=torch.float32,
                    ).to(self._device)

                my_bases = len(self._env.board.get_controlled_bases(main_pid))
                opp_bases = len(self._env.board.get_controlled_bases(3 - main_pid))
                base_diff = my_bases - opp_bases
                phi_before = SHAPING_C * base_diff
                holding_reward = self._holding_reward_rate * base_diff
                env_action = WarChestEnv.remap_action(action) if acting_pid == 2 else action
                logger.debug(
                    f'turn={turn} main_pid={main_pid} acting_pid={acting_pid} '
                    f'action={action} env_action={env_action}'
                )
                state, reward, terminated, truncated, step_info = self._env.step(env_action)

                if not step_info['action'].is_valid:
                    invalid_count += 1
                    logger.warning(f'turn={turn} main_pid={main_pid} invalid_action={action} env_action={env_action}')
                    state, reward, terminated, truncated, step_info = self._env.make_random_step()
                    log_prob = torch.tensor(0.0).to(self._device)
                    value = torch.tensor(0.0).to(self._device)

                phi_after = SHAPING_C * (
                    len(self._env.board.get_controlled_bases(main_pid))
                    - len(self._env.board.get_controlled_bases(3 - main_pid))
                )
                shaped_reward = reward + self._gamma * phi_after - phi_before + holding_reward
                main_score += shaped_reward

                if step_info['action'].type == CLAIM_BASE_ACTION and step_info['action'].is_valid:
                    claims += 1
                if step_info['action'].type == DEPLOY_ACTION and step_info['action'].is_valid:
                    deploys += 1
                    deploy_turns.append(turn)

                self._buffer.add_step(obs_before, action, log_prob, shaped_reward, value, opp_onehot)
            else:
                with torch.no_grad():
                    action, _, _ = opp.act(state)
                env_action = WarChestEnv.remap_action(action) if acting_pid == 2 else action
                logger.debug(
                    f'turn={turn} opp acting_pid={acting_pid} '
                    f'action={action} env_action={env_action}'
                )
                state, _, terminated, truncated, step_info = self._env.step(env_action)
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
            'deploys': deploys,
            'deploy_turns': deploy_turns,
            'main_score': main_score,
            'main_pid': main_pid,
            'opp_type': opp_type,
        }

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _run_ppo_update(self, batch_num: int) -> dict:
        """Run actor and critic updates independently over the current buffer."""
        actor_stats = self._update_actor(batch_num)
        critic_stats = self._update_critic(batch_num)
        return {**actor_stats, **critic_stats}

    def _update_actor(self, batch_num: int) -> dict:
        kl_accum = 0.0
        actor_accum = 0.0
        entropy_accum = 0.0
        clip_frac_accum = 0.0
        last_actor_grad = 0.0
        n_actor_updates = 0
        done = False

        for epoch in range(self._ppo_epochs):
            if done:
                break
            for batch in self._buffer.iter_minibatches(self._minibatch_size, self._device):
                lp_new, ent = self._policy.evaluate_actions_batch(batch)
                lp_old = batch['log_probs_old']
                ratio = (lp_new - lp_old).exp()
                approx_kl = ((ratio - 1) - (lp_new - lp_old)).detach().mean().item()
                if approx_kl > self.KL_TARGET:
                    logger.debug(
                        f'batch={batch_num} epoch={epoch} '
                        f'actor early stop approx_kl={approx_kl:.4f}'
                    )
                    done = True
                    break

                adv = batch['advantages']
                clipped_ratio = ratio.clamp(1 - self._ppo_eps, 1 + self._ppo_eps)
                actor_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()
                loss = actor_loss - self._entropy_coeff * ent.mean()

                self._actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()

                has_nan = any(
                    torch.isnan(p.grad).any()
                    for p in self._policy.parameters() if p.grad is not None
                )
                last_actor_grad = torch.nn.utils.clip_grad_norm_(
                    self._actor_side_params, max_norm=1.0
                ).item()
                if not has_nan:
                    self._actor_optimizer.step()
                else:
                    logger.error(
                        f'batch={batch_num} epoch={epoch} actor NaN gradient, skipping step'
                    )

                kl_accum += (lp_old - lp_new).detach().mean().item()
                actor_accum += actor_loss.item()
                entropy_accum += ent.detach().mean().item()
                clip_frac_accum += ((ratio - 1.0).abs() > self._ppo_eps).float().mean().item()
                n_actor_updates += 1

        denom = max(n_actor_updates, 1)
        return {
            'avg_kl': kl_accum / denom,
            'avg_actor': actor_accum / denom,
            'avg_entropy': entropy_accum / denom,
            'avg_clip_frac': clip_frac_accum / denom,
            'last_actor_grad': last_actor_grad,
            'n_actor_updates': n_actor_updates,
        }

    def _update_critic(self, batch_num: int) -> dict:
        critic_accum = 0.0
        critic_mae_accum = 0.0
        critic_mean_accum = 0.0
        critic_std_accum = 0.0
        last_critic_grad = 0.0
        n_critic_updates = 0
        done = False

        for epoch in range(self._ppo_epochs):
            if done:
                break
            for batch in self._buffer.iter_minibatches(self._minibatch_size, self._device):
                ret = batch['returns']
                ret_n = self._ret_normalizer.normalize(ret)
                v_old_n = self._ret_normalizer.normalize(batch['values_old'])

                val_n = self._critic.value_batch(batch)
                v_clipped_n = v_old_n + (val_n - v_old_n).clamp(-self._ppo_eps, self._ppo_eps)
                critic_loss = 0.5 * torch.max(
                    (val_n - ret_n) ** 2,
                    (v_clipped_n - ret_n) ** 2,
                ).mean()

                self._critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()

                has_nan = any(
                    torch.isnan(p.grad).any()
                    for p in self._critic.parameters() if p.grad is not None
                )
                last_critic_grad = torch.nn.utils.clip_grad_norm_(
                    self._critic.parameters(), max_norm=1.0
                ).item()
                if not has_nan:
                    self._critic_optimizer.step()
                else:
                    logger.error(
                        f'batch={batch_num} epoch={epoch} critic NaN gradient, skipping step'
                    )

                critic_accum += critic_loss.item()
                # MAE and stats logged in raw return scale for comparability across runs
                val_raw = self._ret_normalizer.denormalize(val_n.detach())
                critic_mae_accum += (val_raw - ret).abs().mean().item()
                critic_mean_accum += val_raw.mean().item()
                critic_std_accum += val_raw.std(correction=0).item()
                n_critic_updates += 1

        denom = max(n_critic_updates, 1)
        return {
            'avg_critic': critic_accum / denom,
            'avg_critic_mae': critic_mae_accum / denom,
            'avg_critic_mean': critic_mean_accum / denom,
            'avg_critic_std': critic_std_accum / denom,
            'last_critic_grad': last_critic_grad,
            'n_critic_updates': n_critic_updates,
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
        wr_random_eval = random_eval_wins / self._eval_episodes

        if wr_random_eval >= self._wr_finetune_threshold:
            self._pool.set_weights(**self._opp_weights_finetune)
        else:
            self._pool.set_weights(**self._opp_weights_initial)

        logger.info(
            f'[eval] batch={batch_num} '
            f'wr_greedy={greedy_wins / self._eval_episodes:.3f} '
            f'wr_random={wr_random_eval:.3f} '
            f'elo_policy={elo_pol:.0f} elo_greedy={elo_grdy:.0f}'
        )
        if use_wandb:
            wandb.log({
                'elo_policy': elo_pol,
                'wr_vs_greedy_eval': greedy_wins / self._eval_episodes,
                'wr_vs_random_eval': wr_random_eval,
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
            env_action = WarChestEnv.remap_action(action) if acting_pid == 2 else action
            state, _, terminated, truncated, step_info = self._env.step(env_action)
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
            if ep['opp_type'] == 'greedy':
                self._wr_vs_greedy.append(int(ep['outcome'] == 'win'))
            elif ep['opp_type'] == 'pool':
                self._wr_vs_pool.append(int(ep['outcome'] == 'win'))

        wr_pool = float(np.mean(self._wr_vs_pool)) if self._wr_vs_pool else 0.0
        wr_greedy = float(np.mean(self._wr_vs_greedy)) if self._wr_vs_greedy else 0.0

        s = update_stats
        avg_turns = float(np.mean([ep['turns'] for ep in self._batch_eps]))
        total_invalid = sum(ep['invalid_count'] for ep in self._batch_eps)
        outcomes_str = ' '.join(
            f"{ep['outcome'][0]}({ep['opp_type'][0]})" for ep in self._batch_eps
        )

        all_deploy_turns = [t for ep in self._batch_eps for t in ep['deploy_turns']]
        avg_deploys = float(np.mean([ep['deploys'] for ep in self._batch_eps]))
        deploy_turn_mean = float(np.mean(all_deploy_turns)) if all_deploy_turns else 0.0

        logger.debug(
            f'batch={batch_num} n_actor={s["n_actor_updates"]} n_critic={s["n_critic_updates"]} '
            f'adv mean={self._buffer.raw_adv_mean:.4f} std={self._buffer.raw_adv_std:.4f} '
            f'ret mean={self._buffer.raw_ret_mean:.4f} std={self._buffer.raw_ret_std:.4f} '
            f'critic mean={s["avg_critic_mean"]:.4f} std={s["avg_critic_std"]:.4f} '
            f'clip_frac={s["avg_clip_frac"]:.3f} critic_mae={s["avg_critic_mae"]:.4f}'
        )
        logger.info(
            f'batch={batch_num}/{self._n_batches} [{outcomes_str}] '
            f'score={np.mean(self._score_deque):.2f} '
            f'wr_pool={wr_pool:.3f} wr_greedy={wr_greedy:.3f} '
            f'actor={s["avg_actor"]:.3e} critic={s["avg_critic"]:.4f} '
            f'kl={s["avg_kl"]:.4f} ent={s["avg_entropy"]:.3f} '
            f'grad_a={s["last_actor_grad"]:.3f} grad_c={s["last_critic_grad"]:.3f} '
            f'pool={len(self._pool)} turns={avg_turns:.0f} invalid={total_invalid} '
            f'deploys={avg_deploys:.2f} deploy_turn={deploy_turn_mean:.0f} '
            f't={time.time() - self._batch_start:.2f}s'
        )

        if use_wandb:
            wandb.log({
                'score_main': float(np.mean(self._score_deque)),
                'wr_vs_pool_train': wr_pool,
                'wr_vs_greedy_train': wr_greedy,
                'actor_loss': s['avg_actor'],
                'critic_loss': s['avg_critic'],
                'approx_kl': s['avg_kl'],
                'entropy': s['avg_entropy'],
                'grad_norm_actor': s['last_actor_grad'],
                'grad_norm_critic': s['last_critic_grad'],
                'clip_frac': s['avg_clip_frac'],
                'critic_mae': s['avg_critic_mae'],
                'advantage_std': self._buffer.raw_adv_std,
                'avg_turns': avg_turns,
                'avg_deploys_per_ep': avg_deploys,
                'deploy_turn_mean': deploy_turn_mean,
            })


if __name__ == '__main__':
    use_wandb = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    run_id = time.strftime('%Y%m%d-%H%M%S')
    setup_run_logger(run_id)
    if device.type == 'cuda':
        logger.info(f'run_id={run_id} device=cuda ({torch.cuda.get_device_name(0)})')
    else:
        logger.info(f'run_id={run_id} device=cpu')

    environment = WarChestEnv(save_game_history=False, debug_mode=False)

    # The main player empties its full hand each round, so its lifetime action count
    # is about max_rounds * coins-per-round.
    holding_reward_rate = (
        WIN_REWARD
        / ((environment.winning_base_count - 1) * (environment.max_rounds * len(DECK)))
        * 0.8  # 0.8 is a safety margin so worst-case holding never exceeds WIN_REWARD
    )

    hp = {
        'n_batches': 600,
        'collect_episodes': 16,
        'max_t': 1000,
        'gamma': 0.99,
        'lam': 0.95,
        'ppo_epochs': 4,
        'ppo_eps': 0.2,
        'entropy_coeff': 0.025,
        'holding_reward_rate': holding_reward_rate,
        'minibatch_size': 64,
        'lr_actor': 1e-4,
        'lr_critic': 3e-4,
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
        return Policy(device=device, hidden_dim=hp['hidden_dim'])

    warchest_policy = policy_constructor().to(device)
    warchest_critic = Critic(device=device, hidden_dim=hp['hidden_dim']).to(device)
    actor_optimizer = optim.Adam(warchest_policy.parameters(), lr=hp['lr_actor'])
    critic_optimizer = optim.Adam(warchest_critic.parameters(), lr=hp['lr_critic'])

    trainer = PPOTrainer(
        environment,
        warchest_policy,
        warchest_critic,
        actor_optimizer,
        critic_optimizer,
        policy_constructor,
        hp,
        device,
    )

    exception_for_raising = None
    save_model = True
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info('Training interrupted by user')
        sys.stdout.write('Save results? (y/n)')
        sys.stdout.flush()
        save_results = sys.stdin.buffer.readline().decode('utf-8', errors='replace').strip()
        save_model = save_results == 'y'
    except Exception as e:
        exception_for_raising = e
        logger.exception(f'Training failed: {e}')
    finally:
        if exception_for_raising is not None:
            raise exception_for_raising
        else:
            if save_model:
                timestamp = time.strftime('%Y%m%d-%H%M')
                filename = f'warchest_ppo_{timestamp}.pth'
                os.makedirs('data', exist_ok=True)
                torch.save(warchest_policy.state_dict(), f'data/{filename}')
                logger.info(f'Model saved to data/{filename}')
