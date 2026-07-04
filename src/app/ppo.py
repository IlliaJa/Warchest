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
    WarChestEnv, WIN_REWARD, LOSS_REWARD, CLAIM_BASE_ACTION, ATTACK_ACTION,
)
from src.services.environment.game_state import HAND_SIZE
from src.services.opponent_pool import OpponentPool
from src.utils.rollout_buffer import RolloutBuffer
from src.services.bots import GreedyBot, RandomBot
from src.utils.elo import EloTracker

SHAPING_C = 0.05
# Material PBRS coefficient (rewards.md §9): potential over the boxed-coin differential.
# Kept well below SHAPING_C — bases win the game, material is only a means.
C_MAT = 0.015
# The holding reward and the material PBRS term are linearly annealed from
# SHAPING_ANNEAL_INIT down to SHAPING_ANNEAL_FINAL over the first
# SHAPING_ANNEAL_HALF_FRAC of the run, then held at the floor. This keeps the dense
# guidance early (weak critic, high entropy) and hands the final policy back toward
# the true terminal objective — the over-shaping antidote (see docs/decision.md,
# 2026-07-03). Base-diff PBRS (SHAPING_C) is intentionally left constant.
SHAPING_ANNEAL_INIT = 1.0
SHAPING_ANNEAL_FINAL = 0.1
SHAPING_ANNEAL_HALF_FRAC = 0.5
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
        # entropy coefficient is linearly annealed init -> final over training so the
        # policy is free to explore early and commits to a plan late.
        self._entropy_coeff_init = hp['entropy_coeff']
        self._entropy_coeff_final = hp.get('entropy_coeff_final', hp['entropy_coeff'])
        self._entropy_coeff = self._entropy_coeff_init
        # learning rates are linearly decayed init -> init*lr_final_frac (0 => decay to 0)
        self._lr_actor_init = hp['lr_actor']
        self._lr_critic_init = hp['lr_critic']
        self._lr_final_frac = hp.get('lr_final_frac', 0.0)
        self._holding_reward_rate = hp['holding_reward_rate']
        # anneal multiplier applied to holding + material shaping; set per batch.
        self._shaping_anneal = SHAPING_ANNEAL_INIT
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

        # training-lifetime state (persists across batches).
        # Snapshotting rarely (vs every batch) makes the fixed-size pool span a wide skill
        # range — old/weak to recent/strong — instead of 20 near-identical recent copies.
        # The current policy then beats the weak snapshots (positive advantage) and ties the
        # strong ones, so self-play games carry a real learning signal instead of ~0-advantage
        # mirror matches. Pool spans roughly max_size * snapshot_every batches.
        self._pool = OpponentPool(
            max_size=hp.get('pool_max_size', 20),
            snapshot_every=hp.get('pool_snapshot_every', 15),
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
        self._t_env: float = 0.0
        self._t_model_play: float = 0.0
        self._t_gradient: float = 0.0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def _update_schedules(self, batch_num: int):
        """Linearly anneal the entropy coefficient and both learning rates.

        ``frac`` runs 0.0 (first batch) -> 1.0 (last batch).
        """
        frac = (batch_num - 1) / max(self._n_batches - 1, 1)
        self._entropy_coeff = (
            self._entropy_coeff_init
            + frac * (self._entropy_coeff_final - self._entropy_coeff_init)
        )
        lr_scale = 1.0 - frac * (1.0 - self._lr_final_frac)
        for group in self._actor_optimizer.param_groups:
            group['lr'] = self._lr_actor_init * lr_scale
        for group in self._critic_optimizer.param_groups:
            group['lr'] = self._lr_critic_init * lr_scale

        # Holding + material shaping anneal: 1.0 -> SHAPING_ANNEAL_FINAL over the first
        # SHAPING_ANNEAL_HALF_FRAC of the run (half-point derived from n_batches so it
        # tracks a changed schedule length), then held at the floor.
        half = max(self._n_batches * SHAPING_ANNEAL_HALF_FRAC, 1.0)
        anneal_frac = min((batch_num - 1) / half, 1.0)
        self._shaping_anneal = (
            SHAPING_ANNEAL_INIT + anneal_frac * (SHAPING_ANNEAL_FINAL - SHAPING_ANNEAL_INIT)
        )

    def train(self):
        for batch_num in range(1, self._n_batches + 1):
            self._update_schedules(batch_num)
            self._batch_start = time.time()
            self._policy.to('cpu')
            self._critic.to('cpu')
            self._collect_batch()
            self._policy.to(self._device)
            self._critic.to(self._device)
            t0 = time.perf_counter()
            update_stats = self._run_ppo_update(batch_num)
            self._t_gradient = time.perf_counter() - t0
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
        self._t_env = 0.0
        self._t_model_play = 0.0
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
        _pt = time.perf_counter

        t0 = _pt()
        state, _ = self._env.reset()
        self._t_env += _pt() - t0

        outcome = 'truncated'
        invalid_count = 0
        claims = 0
        main_score = 0.0
        turns = 0
        opp_pid = 3 - main_pid  # absolute id of the main actor's opponent
        # score decomposition (sums to main_score) + entropy-ceiling tracking
        r_attack = r_shaping = r_holding = r_material = r_terminal = r_other = 0.0
        sum_log_nlegal = 0.0
        n_decisions = 0

        collect_device = self._policy.device
        opp_onehot = np.zeros(len(OPP_TYPE_IDX), dtype=np.float32)
        opp_onehot[OPP_TYPE_IDX[opp_type]] = 1.0
        opp_onehot_t = torch.tensor(opp_onehot, dtype=torch.float32).unsqueeze(0).to(collect_device)

        for turn in range(self._max_t):
            acting_pid = self._env.active_player
            turns = turn

            if acting_pid == main_pid:
                obs_before = state
                # Privileged critic input: the opponent's true hidden coin split,
                # captured at the main player's decision point. Never seen by the policy.
                t0 = _pt()
                privileged = self._env.get_privileged_features()
                self._t_env += _pt() - t0
                privileged_t = torch.from_numpy(privileged).unsqueeze(0)

                t0 = _pt()
                action, log_prob, _ = self._policy.act(obs_before)
                self._t_model_play += _pt() - t0

                with torch.no_grad():
                    t0 = _pt()
                    value_norm = self._critic.value_single(obs_before, opp_onehot_t, privileged_t)
                    self._t_model_play += _pt() - t0
                    value = torch.tensor(
                        self._ret_normalizer.denormalize(value_norm.item()),
                        dtype=torch.float32,
                    )

                # obs_before is ego-centric from main_pid (currently acting), so global[1]=my_bases/wbc
                _wbc = self._env.winning_base_count
                base_diff = (obs_before['global'][1] - obs_before['global'][2]) * _wbc
                phi_before = SHAPING_C * base_diff
                holding_reward = self._holding_reward_rate * base_diff
                # Material PBRS potential (rewards.md §9): boxed differential, opp minus me.
                # boxed_total is keyed by absolute pid, so no perspective flip is needed.
                phi_mat_before = C_MAT * (
                    self._env.boxed_total(opp_pid) - self._env.boxed_total(main_pid)
                )
                env_action = WarChestEnv.remap_action(action) if acting_pid == 2 else action

                t0 = _pt()
                state, reward, terminated, truncated, step_info = self._env.step(env_action)
                self._t_env += _pt() - t0

                if not step_info['action'].is_valid:
                    invalid_count += 1
                    logger.warning(f'turn={turn} main_pid={main_pid} invalid_action={action} env_action={env_action}')
                    t0 = _pt()
                    state, reward, terminated, truncated, step_info = self._env.make_random_step()
                    self._t_env += _pt() - t0
                    log_prob = torch.tensor(0.0)
                    value = torch.tensor(0.0)

                # state obs is ego-centric from whoever is now active; flip indices if it flipped to opponent
                if self._env.active_player == main_pid:
                    phi_after = SHAPING_C * (state['global'][1] - state['global'][2]) * _wbc
                else:
                    phi_after = SHAPING_C * (state['global'][2] - state['global'][1]) * _wbc
                phi_mat_after = C_MAT * (
                    self._env.boxed_total(opp_pid) - self._env.boxed_total(main_pid)
                )
                # Base-diff PBRS is constant; holding + material shaping are annealed together.
                base_shaping = self._gamma * phi_after - phi_before
                material_shaping = self._gamma * phi_mat_after - phi_mat_before
                annealed_holding = self._shaping_anneal * holding_reward
                annealed_material = self._shaping_anneal * material_shaping
                shaped_reward = reward + base_shaping + annealed_holding + annealed_material
                main_score += shaped_reward

                # decompose the reward so score/win decoupling is visible in the logs
                r_shaping += base_shaping
                r_holding += annealed_holding
                r_material += annealed_material
                if terminated:
                    r_terminal += reward  # dominated by WIN_REWARD on a winning move
                elif step_info['action'].type == ATTACK_ACTION:
                    r_attack += reward
                else:
                    r_other += reward
                n_legal = int(obs_before['valid_action_mask'].sum())
                sum_log_nlegal += float(np.log(max(n_legal, 1)))
                n_decisions += 1

                if step_info['action'].type == CLAIM_BASE_ACTION and step_info['action'].is_valid:
                    claims += 1

                self._buffer.add_step(obs_before, action, log_prob, shaped_reward, value, opp_onehot, privileged)
            else:
                with torch.no_grad():
                    t0 = _pt()
                    action, _, _ = opp.act(state)
                    self._t_model_play += _pt() - t0
                env_action = WarChestEnv.remap_action(action) if acting_pid == 2 else action
                t0 = _pt()
                state, _, terminated, truncated, step_info = self._env.step(env_action)
                self._t_env += _pt() - t0
                if not step_info['action'].is_valid:
                    t0 = _pt()
                    state, _, terminated, truncated, step_info = self._env.make_random_step()
                    self._t_env += _pt() - t0

            if terminated:
                outcome = 'win' if acting_pid == main_pid else 'lose'
                if acting_pid != main_pid:
                    self._buffer.append_terminal_reward(LOSS_REWARD)
                    main_score += LOSS_REWARD
                    r_terminal += LOSS_REWARD
                break

            if truncated:
                _wbc = self._env.winning_base_count
                if self._env.active_player == main_pid:
                    diff = (state['global'][1] - state['global'][2]) * _wbc
                else:
                    diff = (state['global'][2] - state['global'][1]) * _wbc
                # Base-diff-proportional truncation reward (C17): a smoother critic
                # target than the old 0 / -0.5 / -1.0 step function, so the critic sees
                # lower target variance at the states the agent spends most time in.
                # A draw from a winning position is still 0; ties and deficits scale
                # linearly from -0.5 (tie) toward LOSS_REWARD (full-deficit rout),
                # preserving the two anchor values of the old step function.
                if diff > 0:
                    trunc_reward = 0.0
                else:
                    deficit_frac = min(-diff, _wbc) / _wbc  # 0 at a tie ... 1 at max deficit
                    trunc_reward = LOSS_REWARD * (0.5 + 0.5 * deficit_frac)
                self._buffer.append_terminal_reward(trunc_reward)
                main_score += trunc_reward
                r_terminal += trunc_reward
                break

        self._buffer.end_episode()
        return {
            'outcome': outcome,
            'turns': turns,
            'invalid_count': invalid_count,
            'claims': claims,
            'main_score': main_score,
            'main_pid': main_pid,
            'opp_type': opp_type,
            'r_attack': r_attack,
            'r_shaping': r_shaping,
            'r_holding': r_holding,
            'r_material': r_material,
            'r_terminal': r_terminal,
            'r_other': r_other,
            'sum_log_nlegal': sum_log_nlegal,
            'n_decisions': n_decisions,
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

        # per-episode mean of each score component (sums to score)
        r_attack = float(np.mean([ep['r_attack'] for ep in self._batch_eps]))
        r_shaping = float(np.mean([ep['r_shaping'] for ep in self._batch_eps]))
        r_holding = float(np.mean([ep['r_holding'] for ep in self._batch_eps]))
        r_material = float(np.mean([ep['r_material'] for ep in self._batch_eps]))
        r_terminal = float(np.mean([ep['r_terminal'] for ep in self._batch_eps]))
        r_other = float(np.mean([ep['r_other'] for ep in self._batch_eps]))
        # entropy ceiling: mean over all main-player decisions of log(n_legal).
        # ent_frac = entropy / max_entropy makes "how decisive" readable without mental math.
        tot_dec = sum(ep['n_decisions'] for ep in self._batch_eps)
        max_entropy = (
            sum(ep['sum_log_nlegal'] for ep in self._batch_eps) / tot_dec if tot_dec else 0.0
        )
        entropy_frac = s['avg_entropy'] / max_entropy if max_entropy > 0 else 0.0

        total_t = self._t_env + self._t_model_play + self._t_gradient
        logger.debug(
            f'batch={batch_num} n_actor={s["n_actor_updates"]} n_critic={s["n_critic_updates"]} '
            f'adv mean={self._buffer.raw_adv_mean:.4f} std={self._buffer.raw_adv_std:.4f} '
            f'ret mean={self._buffer.raw_ret_mean:.4f} std={self._buffer.raw_ret_std:.4f} '
            f'critic mean={s["avg_critic_mean"]:.4f} std={s["avg_critic_std"]:.4f} '
            f'clip_frac={s["avg_clip_frac"]:.3f} critic_mae={s["avg_critic_mae"]:.4f}'
        )
        logger.info(
            f'batch={batch_num} timing: '
            f'env={self._t_env:.2f}s ({100*self._t_env/total_t:.0f}%) '
            f'model_play={self._t_model_play:.2f}s ({100*self._t_model_play/total_t:.0f}%) '
            f'gradient={self._t_gradient:.2f}s ({100*self._t_gradient/total_t:.0f}%) '
            f'total_accounted={total_t:.2f}s'
        )
        logger.info(
            f'batch={batch_num}/{self._n_batches} '
            f'score={np.mean(self._score_deque):.2f} '
            f'wr_pool={wr_pool:.3f} wr_greedy={wr_greedy:.3f} '
            f'actor={s["avg_actor"]:.3e} critic={s["avg_critic"]:.4f} '
            f'kl={s["avg_kl"]:.4f} ent={s["avg_entropy"]:.3f} '
            f'ent_max={max_entropy:.3f} ent_frac={entropy_frac:.2f} '
            f'ent_c={self._entropy_coeff:.4f} lr={self._actor_optimizer.param_groups[0]["lr"]:.2e} '
            f'grad_a={s["last_actor_grad"]:.3f} grad_c={s["last_critic_grad"]:.3f} '
            f'pool={len(self._pool)} turns={avg_turns:.0f} invalid={total_invalid} '
            f't={time.time() - self._batch_start:.2f}s'
        )
        logger.info(
            f'batch={batch_num} score_parts (per-ep mean): '
            f'attack={r_attack:.3f} shaping={r_shaping:.3f} holding={r_holding:.3f} '
            f'material={r_material:.3f} terminal={r_terminal:.3f} other={r_other:.3f} '
            f'anneal={self._shaping_anneal:.3f}'
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
                'entropy_coeff': self._entropy_coeff,
                'lr': self._actor_optimizer.param_groups[0]['lr'],
                'max_entropy': max_entropy,
                'entropy_frac': entropy_frac,
                'score_attack': r_attack,
                'score_shaping': r_shaping,
                'score_holding': r_holding,
                'score_material': r_material,
                'score_terminal': r_terminal,
                'score_other': r_other,
                'shaping_anneal': self._shaping_anneal,
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
        / ((environment.winning_base_count - 1) * (environment.max_rounds * HAND_SIZE))
        * 0.8  # 0.8 is a safety margin so worst-case holding never exceeds WIN_REWARD
    )

    hp = {
        'n_batches': 600,
        'collect_episodes': 64,
        'max_t': 1000,
        'gamma': 0.99,
        'lam': 0.95,
        'ppo_epochs': 4,
        'ppo_eps': 0.2,
        'entropy_coeff': 0.025,
        'entropy_coeff_final': 0.003,  # linearly annealed from entropy_coeff over the run
        'holding_reward_rate': holding_reward_rate,
        'minibatch_size': 64,
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'lr_final_frac': 0.0,  # LR decays linearly to lr_*_init * this (0.0 => to zero)
        'hidden_dim': 64,
        # Step 5 (docs/rewards_improvements.md): strengthen the *densifier*. The critic
        # is what turns the terminal reward into a per-step signal, so widen it alone
        # (policy left at hidden_dim) to keep the capacity A/B attributable. Safe because
        # the critic's board encoder is independent of the policy's during PPO rollout.
        'critic_hidden_dim': 128,
        'print_every': 10,
        # opponent sampling weights — initial phase (random opponent included)
        'p_random_initial': 0.40,
        'p_greedy_initial': 0.20,
        'p_pool_initial': 0.40,
        # opponent sampling weights — fine-tune phase (random removed from training).
        # Greedy is a small fixed anchor (0.1); the rest is self-play against the wide-skill pool.
        'p_random_finetune': 0.00,
        'p_greedy_finetune': 0.10,
        'p_pool_finetune': 0.90,
        # win-rate vs random that triggers the phase switch
        'wr_random_finetune_threshold': 0.90,
        # self-play pool cadence: snapshot rarely so the max_size-slot pool spans a wide
        # skill range (~pool_max_size * pool_snapshot_every batches) rather than near-copies.
        'pool_max_size': 20,
        'pool_snapshot_every': 15,
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
    warchest_critic = Critic(device=device, hidden_dim=hp['critic_hidden_dim']).to(device)
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
