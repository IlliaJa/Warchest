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

from .roster import UNIT_BY_ID
from .warchest_env import (
    WarChestEnv, LOSS_REWARD, CLAIM_BASE_ACTION, ATTACK_ACTION, BOLSTER_ACTION,
    VERB_OF_ACTION, V_BOLSTER,
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
# breaking arch compatibility. `lookahead_critic` (and `puct`) therefore share the
# `pool` slot — a strong, self-play-derived opponent is their closest existing
# analogue (and the same reason those bots condition on opp_type='pool' by default).
# The label stays distinct everywhere else (sampling weights, win-rate metrics);
# only the one-hot collapses it onto `pool`.
# `random_eval` (docs/IDEAS.md B1) shares the `greedy` slot instead: it is a SimGreedyBot
# with sampled leaf coefficients, i.e. a hand-written heuristic bot of roughly greedy
# strength, and nothing about it is policy-derived.
OPP_ONEHOT_SLOT = {**OPP_TYPE_IDX, 'lookahead_critic': OPP_TYPE_IDX['pool'],
                   'puct': OPP_TYPE_IDX['pool'],
                   # `puct_live` is the same bot searching on the *current* policy's
                   # snapshot rather than a frozen checkpoint, so it shares puct's slot.
                   'puct_live': OPP_TYPE_IDX['pool'],
                   'random_eval': OPP_TYPE_IDX['greedy'],
                   # policy_theta is policy-derived (its candidate set comes from a frozen
                   # checkpoint), so `pool` is its honest analogue, not `greedy`.
                   'policy_theta': OPP_TYPE_IDX['pool']}

# Fine-grained opponent id, used ONLY to group advantage normalisation
# (`RolloutBuffer.compute_gae(adv_norm='per_opponent')`, docs/next_iteration.md §5 row 6).
# Deliberately NOT `OPP_ONEHOT_SLOT`: that one collapses the search bots onto `pool`
# because the critic's one-hot is pinned at 3 for v1/v2 checkpoint compatibility, and
# collapsing them here would defeat the purpose — the finetune schedule is 75 % `pool` /
# 25 % `lookahead_critic`, two opponents of genuinely different strength, so grouping them
# together would leave in the advantage exactly the per-opponent offset this is removing.
# Not fed to any network, so it is free to grow: adding an entry breaks no checkpoint.
OPP_GROUP_IDX = {'random': 0, 'greedy': 1, 'pool': 2, 'lookahead_critic': 3, 'puct': 4,
                 'random_eval': 5, 'policy_theta': 6, 'puct_live': 7}
# Shared bucket for an opponent label nobody registered above. Warned about once per
# process rather than raised, so adding a bot to the pool cannot kill a running job — but
# it does mean that bot's advantages are centred together with any other stranger.
OPP_GROUP_OTHER = len(OPP_GROUP_IDX)
_warned_opp_groups = set()


def opp_group_id(opp_type):
    """Fine-grained opponent group id for advantage centring. See `OPP_GROUP_IDX`."""
    idx = OPP_GROUP_IDX.get(opp_type)
    if idx is None:
        if opp_type not in _warned_opp_groups:
            _warned_opp_groups.add(opp_type)
            logger.warning(
                f'opponent {opp_type!r} is not in OPP_GROUP_IDX; its advantages will be '
                f'centred together with every other unregistered opponent. Register it to '
                f'give it its own group.'
            )
        return OPP_GROUP_OTHER
    return idx

# Opponent families that forward-simulate the full game state and therefore read
# the live env (`act(env)`, absolute frame) instead of the ego-centric obs the
# reactive bots consume — see `_opponent_env_action`. `random_eval` is a SimGreedyBot
# subclass, so it belongs here despite being a *heuristic* bot rather than a deep search:
# what puts a bot in this set is the interface it speaks, not how hard it thinks.
_SEARCH_OPP_TYPES = frozenset({'lookahead_critic', 'puct', 'puct_live', 'random_eval',
                               'policy_theta'})


def _opponent_env_action(opp, opp_type, env, obs, acting_pid):
    """Absolute-frame action from a pool opponent, uniform across bot families.

    Reactive bots (random/greedy/pool) read the ego-centric obs the previous
    env.step already produced (free — no re-encode) and return an ego action we
    un-rotate here. Search bots (lookahead_critic, puct) read the live env instead:
    they clone + forward-simulate the full unrotated state, which the lossy ego obs
    cannot reconstruct, and already return an absolute action. One call site, the
    only branch is which projection of the same state each family consumes.
    """
    if opp_type in _SEARCH_OPP_TYPES:
        return opp.act(env)  # already absolute, un-rotated internally
    action, _, _ = opp.act(obs)
    return WarChestEnv.remap_action(action) if acting_pid == 2 else action


def play_episode(env, policy, opp, main_pid, opp_type, *,
                 gamma, shaping_anneal, holding_reward_rate, max_t,
                 base_shaping_anneal=1.0, collect_dense=False):
    """Run one episode; return (steps, episode_dict).

    ``steps`` is a dict of parallel per-decision lists for the main actor's transitions
    (arg order matches RolloutBuffer.add_step, minus the deferred value):
        obs, actions, log_probs, rewards, opp_onehots, privileged, opp_ids

    ``opp_ids`` is the fine-grained opponent group (`opp_group_id`), constant within an
    episode and used only to centre advantages per opponent — see `OPP_GROUP_IDX`. It is
    separate from ``opp_onehots``, which is the 3-wide critic input that `critic_v3` drops.
    The terminal / truncation reward is folded into ``rewards[-1]`` (equivalent to the old
    RolloutBuffer.append_terminal_reward, but scoped to this episode's own last step rather
    than the buffer's global last step — the old cross-episode edge only mattered for a
    zero-main-step episode, which never occurs in practice).

    ``episode_dict`` carries the score decomposition + metadata, plus ``t_env`` /
    ``t_model_play`` so the caller can accumulate timing (this function owns no trainer
    state). The critic is never called here — V(s) is computed in one batched pass later.

    ``base_shaping_anneal`` multiplies the base-differential PBRS term, the way
    ``shaping_anneal`` multiplies holding + material PBRS. Separate knobs because the base
    term used to be deliberately un-annealed, and that is what `docs/IDEAS.md` R.0.3
    measured as the failure: at the 0.1 floor one base paid 0.050 against 0.0015 per boxed
    coin (33 : 1) and the realised per-episode base payout (0.205) exceeded the realised
    terminal (0.125). Annealing both by the same multiplier holds the base : material ratio
    at its initial 3.3 : 1 for the whole run and puts the dense payout well under the
    terminal late on. Defaults to 1.0 (the pre-2026-08-18 behaviour) so a caller that does
    not know about the split gets the old reward; `ppo.py` passes ``shaping_anneal`` here.

    ``collect_dense`` (docs/IDEAS.md #12 — dense critic targets): also emit auxiliary
    value-regression samples at every *opponent* decision node, so the critic gets
    supervision on the half of the game tree the main stream never covers. Each such node
    is scored from its (opponent) mover's perspective, conditioned on the `pool` opponent
    slot (the passive side there is the learning policy ≈ a self-play snapshot — see
    docs/bots.md; there is no "self" slot and none is needed). Targets are Monte-Carlo
    return-to-go over a main-perspective per-ply reward stream, negated for the opponent's
    perspective (rewards are antisymmetric, docs/rewards.md). The opponent's own small
    per-move rewards are approximated as 0 (only terminal/truncation outcomes are folded
    in); the main policy path (`obs_l`/`rew_l`/GAE) is left byte-for-byte unchanged, and
    all dense work is gated on this flag so it costs nothing when off. The extra samples
    are returned under ``steps['aux_*']`` and consumed only as an auxiliary critic loss.
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
    bolster_count = 0
    # Stricter subset of bolster_count: board + supply == the bolstered unit's full
    # roster coin count (none of that type sits in hand/bag/discard, none boxed) —
    # a "genuinely available, nothing lost" bolster (docs/IDEAS.md #R8 bolster underuse).
    bolster_fully_available_count = 0
    main_score = 0.0
    turns = 0
    opp_pid = 3 - main_pid  # absolute id of the main actor's opponent
    r_attack = r_shaping = r_holding = r_material = r_terminal = r_other = r_tempo = 0.0
    sum_log_nlegal = 0.0
    n_decisions = 0

    opp_onehot = np.zeros(len(OPP_TYPE_IDX), dtype=np.float32)
    opp_onehot[OPP_ONEHOT_SLOT[opp_type]] = 1.0

    # Dense-target accumulators (collect_dense only): one main-perspective reward per ply
    # (main or opponent) for the Monte-Carlo return, plus the opponent-node samples.
    ply_rewards = []
    aux_boards, aux_globals, aux_priv, aux_ply_idx = [], [], [], []
    dense_pool_onehot = None
    if collect_dense:
        dense_pool_onehot = np.zeros(len(OPP_TYPE_IDX), dtype=np.float32)
        dense_pool_onehot[OPP_TYPE_IDX['pool']] = 1.0

    def _fold_terminal_reward(r):
        """Add a terminal/truncation reward onto this episode's last main-actor step."""
        if rew_l:
            rew_l[-1] += r

    def _fold_dense_terminal(r):
        """Mirror of _fold_terminal_reward for the dense per-ply reward stream: the terminal
        outcome lands on whichever ply actually ended the game, not on the last main step."""
        if collect_dense and ply_rewards:
            ply_rewards[-1] += r

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

            # Bolster targets the unit already on the cell; decode+look it up before
            # stepping (the mask guarantees a legal target, so this is only voided
            # below if the action turns out invalid and gets replaced).
            bolster_unit_id = None
            if VERB_OF_ACTION[env_action] == V_BOLSTER:
                _, br, bq = WarChestEnv.decode_action(env_action)
                bolster_target = env.board.get_unit_at(br, bq)
                if bolster_target is not None:
                    bolster_unit_id = bolster_target.id

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

            if (bolster_unit_id is not None and step_info['action'].is_valid
                    and step_info['action'].type == BOLSTER_ACTION):
                bolster_count += 1
                supply_left = env.state.supply[main_pid].get(bolster_unit_id, 0)
                board_total = sum(
                    u.stack for u in env.board.units
                    if u.player_id == main_pid and u.id == bolster_unit_id)
                if board_total + supply_left == UNIT_BY_ID[bolster_unit_id].total_coins:
                    bolster_fully_available_count += 1

            # state obs is ego-centric from whoever is now active; flip indices if it flipped to opponent
            if env.active_player == main_pid:
                phi_after = SHAPING_C * (state['global'][1] - state['global'][2]) * _wbc
            else:
                phi_after = SHAPING_C * (state['global'][2] - state['global'][1]) * _wbc
            phi_mat_after = C_MAT * (env.boxed_total(opp_pid) - env.boxed_total(main_pid))
            # All three dense terms are annealed by their own multiplier. Scaling a PBRS
            # term is still valid PBRS: `c * (gamma*Phi' - Phi)` telescopes exactly like the
            # potential `c*Phi`, and c is constant within an episode (it is set once per
            # batch by the caller), so nothing leaks across the telescope.
            base_shaping = gamma * phi_after - phi_before
            material_shaping = gamma * phi_mat_after - phi_mat_before
            annealed_base = base_shaping_anneal * base_shaping
            annealed_holding = shaping_anneal * holding_reward
            annealed_material = shaping_anneal * material_shaping
            shaped_reward = reward + annealed_base + annealed_holding + annealed_material
            main_score += shaped_reward

            # decompose the reward so score/win decoupling is visible in the logs
            r_shaping += annealed_base
            r_holding += annealed_holding
            r_material += annealed_material
            # The per-turn tempo cost rides on every turn-ending reward, so peel it into
            # its own bucket first — otherwise it would be spread across attack/other and
            # r_attack would report tempo rather than attacks (ATTACK_REWARD is now 0).
            tempo = step_info['action'].tempo_cost
            r_tempo += tempo
            env_reward = reward - tempo
            if terminated:
                r_terminal += env_reward  # dominated by WIN_REWARD on a winning move
            elif step_info['action'].type == ATTACK_ACTION:
                r_attack += env_reward
            else:
                r_other += env_reward
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
            if collect_dense:
                ply_rewards.append(shaped_reward)
        else:
            if collect_dense:
                # `state` is the opponent's own ego-centric decision-point obs (produced by
                # the previous step, when the turn flipped to them); privileged features are
                # read at this same node. Captured before the opponent moves so the sample is
                # the value of the position with the opponent to move.
                aux_boards.append(state['board'])
                aux_globals.append(state['global'])
                aux_priv.append(env.get_privileged_features())
                aux_ply_idx.append(len(ply_rewards))
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
            if collect_dense:
                # Opponent's own move earns the main player no direct reward; the terminal/
                # truncation outcome is folded onto this ply below if the game ends here.
                ply_rewards.append(0.0)

        if terminated:
            outcome = 'win' if acting_pid == main_pid else 'lose'
            if acting_pid != main_pid:
                _fold_terminal_reward(LOSS_REWARD)
                main_score += LOSS_REWARD
                r_terminal += LOSS_REWARD
                # Opponent's winning move ended on an opponent ply; the main-perspective
                # loss belongs there. (A main win already carries WIN in its own ply reward.)
                _fold_dense_terminal(LOSS_REWARD)
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
            _fold_dense_terminal(trunc_reward)
            break

    steps = {
        'obs': obs_l,
        'actions': act_l,
        'log_probs': lp_l,
        'rewards': rew_l,
        'opp_onehots': opp_l,
        'privileged': priv_l,
        'opp_ids': [opp_group_id(opp_type)] * len(obs_l),
    }
    if collect_dense and aux_boards:
        # Monte-Carlo return-to-go over the main-perspective per-ply reward stream, then
        # negated per opponent node (that node's mover is the opponent, whose value is the
        # antisymmetric negative of the main player's return from the same ply).
        n_ply = len(ply_rewards)
        returns_to_go = np.empty(n_ply, dtype=np.float32)
        running = 0.0
        for t in range(n_ply - 1, -1, -1):
            running = ply_rewards[t] + gamma * running
            returns_to_go[t] = running
        steps['aux_boards'] = np.stack(aux_boards)
        steps['aux_globals'] = np.stack(aux_globals)
        steps['aux_privileged'] = np.stack(aux_priv)
        steps['aux_opp_onehots'] = np.tile(dense_pool_onehot, (len(aux_boards), 1))
        steps['aux_targets'] = np.array([-returns_to_go[i] for i in aux_ply_idx],
                                        dtype=np.float32)
    episode_dict = {
        'outcome': outcome,
        'turns': turns,
        'invalid_count': invalid_count,
        'claims': claims,
        'bolster_count': bolster_count,
        'bolster_fully_available_count': bolster_fully_available_count,
        'main_score': main_score,
        'main_pid': main_pid,
        'opp_type': opp_type,
        'r_attack': r_attack,
        'r_shaping': r_shaping,
        'r_holding': r_holding,
        'r_material': r_material,
        'r_terminal': r_terminal,
        'r_other': r_other,
        'r_tempo': r_tempo,
        'sum_log_nlegal': sum_log_nlegal,
        'n_decisions': n_decisions,
        't_env': t_env,
        't_model_play': t_model_play,
    }
    return steps, episode_dict
