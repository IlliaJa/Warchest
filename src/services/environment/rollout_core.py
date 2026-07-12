"""Pure, buffer-agnostic single-episode rollout.

Extracted from PPOTrainer._collect_episode so the exact same episode logic drives both
the single-process path (src/app/ppo.py) and the future parallel workers
(docs/parallel_rollouts.md). This module must stay free of any trainer / buffer / critic
state — it takes an env + policy + opponent and returns plain data.
"""

import logging
import time

import numpy as np
import torch

from .warchest_env import (
    WarChestEnv, LOSS_REWARD, CLAIM_BASE_ACTION, ATTACK_ACTION,
)

logger = logging.getLogger('warchest')

# Reward-shaping scales (owned here because they belong to the per-step reward, which lives
# in play_episode). ppo.py imports them from this module so there is a single definition.
SHAPING_C = 0.05
# Kept well below SHAPING_C — bases win the game, material is only a means.
C_MAT = 0.015

OPP_TYPE_IDX = {'random': 0, 'greedy': 1, 'pool': 2}

# Which OPP_TYPE_IDX conditioning slot each opponent label maps to. The critic's
# opponent one-hot is fixed at OPP_DIM=3 (policy.Critic) and existing checkpoints
# are pinned to it, so a new training opponent cannot claim its own slot without
# breaking arch compatibility. `lookahead_critic` therefore shares the `pool` slot
# — a strong, self-play-derived opponent is its closest existing analogue (and the
# same reason LookaheadCriticBot itself conditions on opp_type='pool' by default).
# The label stays distinct everywhere else (sampling weights, win-rate metrics);
# only the one-hot collapses it onto `pool`.
OPP_ONEHOT_SLOT = {**OPP_TYPE_IDX, 'lookahead_critic': OPP_TYPE_IDX['pool']}


def _opponent_env_action(opp, opp_type, env, obs, acting_pid):
    """Absolute-frame action from a pool opponent, uniform across bot families.

    Reactive bots (random/greedy/pool) read the ego-centric obs the previous
    env.step already produced (free — no re-encode) and return an ego action we
    un-rotate here. Search bots (lookahead_critic) read the live env instead: they
    clone + forward-simulate the full unrotated state, which the lossy ego obs
    cannot reconstruct, and already return an absolute action. One call site, the
    only branch is which projection of the same state each family consumes.
    """
    if opp_type == 'lookahead_critic':
        return opp.act(env)  # already absolute, un-rotated internally
    action, _, _ = opp.act(obs)
    return WarChestEnv.remap_action(action) if acting_pid == 2 else action


def play_episode(env, policy, opp, main_pid, opp_type, *,
                 gamma, shaping_anneal, holding_reward_rate, max_t):
    """Run one episode; return (steps, episode_dict).

    ``steps`` is a dict of parallel per-decision lists for the main actor's transitions
    (arg order matches RolloutBuffer.add_step, minus the deferred value):
        obs, actions, log_probs, rewards, opp_onehots, privileged
    The terminal / truncation reward is folded into ``rewards[-1]`` (equivalent to the old
    RolloutBuffer.append_terminal_reward, but scoped to this episode's own last step rather
    than the buffer's global last step — the old cross-episode edge only mattered for a
    zero-main-step episode, which never occurs in practice).

    ``episode_dict`` carries the score decomposition + metadata, plus ``t_env`` /
    ``t_model_play`` so the caller can accumulate timing (this function owns no trainer
    state). The critic is never called here — V(s) is computed in one batched pass later.
    """
    _pt = time.perf_counter
    t_env = 0.0
    t_model_play = 0.0

    obs_l, act_l, lp_l, rew_l, opp_l, priv_l = [], [], [], [], [], []

    t0 = _pt()
    state, _ = env.reset()
    t_env += _pt() - t0

    outcome = 'truncated'
    invalid_count = 0
    claims = 0
    main_score = 0.0
    turns = 0
    opp_pid = 3 - main_pid  # absolute id of the main actor's opponent
    r_attack = r_shaping = r_holding = r_material = r_terminal = r_other = 0.0
    sum_log_nlegal = 0.0
    n_decisions = 0

    opp_onehot = np.zeros(len(OPP_TYPE_IDX), dtype=np.float32)
    opp_onehot[OPP_ONEHOT_SLOT[opp_type]] = 1.0

    def _fold_terminal_reward(r):
        """Add a terminal/truncation reward onto this episode's last main-actor step."""
        if rew_l:
            rew_l[-1] += r

    for turn in range(max_t):
        acting_pid = env.active_player
        turns = turn

        if acting_pid == main_pid:
            obs_before = state
            # Privileged critic input: the opponent's true hidden coin split, captured at
            # the main player's decision point. Never seen by the policy. Stored per step;
            # V(s) is computed later in one batched pass, not here.
            t0 = _pt()
            privileged = env.get_privileged_features()
            t_env += _pt() - t0

            t0 = _pt()
            action, log_prob, _ = policy.act(obs_before)
            t_model_play += _pt() - t0

            # obs_before is ego-centric from main_pid (currently acting), so global[1]=my_bases/wbc
            _wbc = env.winning_base_count
            base_diff = (obs_before['global'][1] - obs_before['global'][2]) * _wbc
            phi_before = SHAPING_C * base_diff
            holding_reward = holding_reward_rate * base_diff
            # Material PBRS potential (rewards.md §9): boxed differential, opp minus me.
            # boxed_total is keyed by absolute pid, so no perspective flip is needed.
            phi_mat_before = C_MAT * (env.boxed_total(opp_pid) - env.boxed_total(main_pid))
            env_action = WarChestEnv.remap_action(action) if acting_pid == 2 else action

            t0 = _pt()
            state, reward, terminated, truncated, step_info = env.step(env_action)
            t_env += _pt() - t0

            if not step_info['action'].is_valid:
                invalid_count += 1
                logger.warning(f'turn={turn} main_pid={main_pid} invalid_action={action} env_action={env_action}')
                t0 = _pt()
                state, reward, terminated, truncated, step_info = env.make_random_step()
                t_env += _pt() - t0
                log_prob = torch.tensor(0.0)

            # state obs is ego-centric from whoever is now active; flip indices if it flipped to opponent
            if env.active_player == main_pid:
                phi_after = SHAPING_C * (state['global'][1] - state['global'][2]) * _wbc
            else:
                phi_after = SHAPING_C * (state['global'][2] - state['global'][1]) * _wbc
            phi_mat_after = C_MAT * (env.boxed_total(opp_pid) - env.boxed_total(main_pid))
            # Base-diff PBRS is constant; holding + material shaping are annealed together.
            base_shaping = gamma * phi_after - phi_before
            material_shaping = gamma * phi_mat_after - phi_mat_before
            annealed_holding = shaping_anneal * holding_reward
            annealed_material = shaping_anneal * material_shaping
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

            obs_l.append(obs_before)
            act_l.append(action)
            lp_l.append(log_prob)
            rew_l.append(shaped_reward)
            opp_l.append(opp_onehot)
            priv_l.append(privileged)
        else:
            with torch.no_grad():
                t0 = _pt()
                env_action = _opponent_env_action(opp, opp_type, env, state, acting_pid)
                t_model_play += _pt() - t0
            t0 = _pt()
            state, _, terminated, truncated, step_info = env.step(env_action)
            t_env += _pt() - t0
            if not step_info['action'].is_valid:
                t0 = _pt()
                state, _, terminated, truncated, step_info = env.make_random_step()
                t_env += _pt() - t0

        if terminated:
            outcome = 'win' if acting_pid == main_pid else 'lose'
            if acting_pid != main_pid:
                _fold_terminal_reward(LOSS_REWARD)
                main_score += LOSS_REWARD
                r_terminal += LOSS_REWARD
            break

        if truncated:
            _wbc = env.winning_base_count
            if env.active_player == main_pid:
                diff = (state['global'][1] - state['global'][2]) * _wbc
            else:
                diff = (state['global'][2] - state['global'][1]) * _wbc
            # Base-diff-proportional truncation reward (C17): a smoother critic target than
            # the old 0 / -0.5 / -1.0 step function. A draw from a winning position is still
            # 0; ties and deficits scale linearly from -0.5 (tie) toward LOSS_REWARD.
            if diff > 0:
                trunc_reward = 0.0
            else:
                deficit_frac = min(-diff, _wbc) / _wbc  # 0 at a tie ... 1 at max deficit
                trunc_reward = LOSS_REWARD * (0.5 + 0.5 * deficit_frac)
            _fold_terminal_reward(trunc_reward)
            main_score += trunc_reward
            r_terminal += trunc_reward
            break

    steps = {
        'obs': obs_l,
        'actions': act_l,
        'log_probs': lp_l,
        'rewards': rew_l,
        'opp_onehots': opp_l,
        'privileged': priv_l,
    }
    episode_dict = {
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
        't_env': t_env,
        't_model_play': t_model_play,
    }
    return steps, episode_dict
