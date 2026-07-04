"""Parallel rollout collection across persistent worker processes.

See docs/parallel_rollouts.md. Workers are CPU-only and run ONLY the policy (act) + the
opponent move + env.step — the critic never leaves the main process (its values are computed
in one batched GPU pass after collection). Each batch the main process broadcasts the small
policy state_dict plus, incrementally, any new pool snapshot; workers return pre-stacked numpy
transitions which the main process concatenates into the RolloutBuffer.

Kept import-light at module top so `spawn` children re-import it cheaply; torch/env/policy are
imported inside the worker function.
"""

import logging
import multiprocessing as mp

logger = logging.getLogger('warchest')


def _worker_loop(worker_id, task_q, result_q, cfg):
    """Persistent worker: load weights, play its share of episodes, return numpy transitions."""
    import os
    import sys
    # Under spawn the child starts fresh; make the project root importable.
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if root not in sys.path:
        sys.path.insert(0, root)

    import matplotlib
    try:
        matplotlib.use('Agg', force=True)  # headless worker — never renders
    except Exception:
        pass

    import numpy as np
    import torch
    torch.set_num_threads(1)  # CRITICAL: else N workers × intra-op threads oversubscribe cores

    from src.services.environment.warchest_env import WarChestEnv
    from src.services.environment.rollout_core import play_episode
    from src.services.policy.policy import Policy
    from src.services.opponent_pool import OpponentPool

    seed = cfg['seed_base'] + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cpu')
    hidden_dim = cfg['policy_hidden_dim']

    def policy_constructor():
        return Policy(device=device, hidden_dim=hidden_dim)

    env = WarChestEnv(save_game_history=False, debug_mode=False)
    policy = policy_constructor()
    policy.eval()
    # snapshot_every is irrelevant here — the main process owns snapshotting; the worker only
    # mirrors the pool contents via append_snapshot. Weights are set per task.
    pool = OpponentPool(max_size=cfg['pool_max_size'], snapshot_every=10 ** 9,
                        p_random=1.0, p_greedy=0.0, p_pool=0.0)

    while True:
        task = task_q.get()
        if task is None:
            break
        try:
            policy.load_state_dict(task['policy_sd'])
            for sd in task['new_snapshots']:
                pool.append_snapshot(sd)
            pool.set_weights(**task['weights'])

            obs_l, act_l, lp_l, rew_l, opp_l, priv_l = [], [], [], [], [], []
            ends, episode_dicts = [], []
            t_env = t_model = 0.0

            for _ in range(task['n_episodes']):
                main_pid = int(np.random.choice([1, 2]))
                opp, opp_type = pool.sample(policy_constructor, device)
                steps, ep = play_episode(
                    env, policy, opp, main_pid, opp_type,
                    gamma=task['gamma'],
                    shaping_anneal=task['shaping_anneal'],
                    holding_reward_rate=task['holding_reward_rate'],
                    max_t=task['max_t'],
                )
                obs_l.extend(steps['obs'])
                act_l.extend(steps['actions'])
                lp_l.extend(float(x) for x in steps['log_probs'])
                rew_l.extend(steps['rewards'])
                opp_l.extend(steps['opp_onehots'])
                priv_l.extend(steps['privileged'])
                ends.append(len(rew_l))
                t_env += ep.pop('t_env')
                t_model += ep.pop('t_model_play')
                episode_dicts.append(ep)

            payload = {
                'boards': np.stack([o['board'] for o in obs_l]),
                'globals': np.stack([o['global'] for o in obs_l]),
                'masks': np.stack([o['valid_action_mask'] for o in obs_l]),
                'actions': np.array(act_l, dtype=np.int64),
                'log_probs': np.array(lp_l, dtype=np.float32),
                'rewards': np.array(rew_l, dtype=np.float32),
                'opp_onehots': np.stack(opp_l),
                'privileged': np.stack(priv_l),
                'episode_ends': ends,
                'episode_dicts': episode_dicts,
                't_env': t_env,
                't_model_play': t_model,
            }
            result_q.put((worker_id, 'OK', payload))
        except Exception:
            import traceback
            result_q.put((worker_id, 'ERROR', traceback.format_exc()))


class ParallelRolloutCollector:
    """Owns the persistent worker pool; broadcasts weights and gathers transitions per batch."""

    def __init__(self, n_workers, *, policy_hidden_dim, pool_max_size, seed_base):
        self._ctx = mp.get_context('spawn')
        self._n = n_workers
        self._task_qs = [self._ctx.Queue() for _ in range(n_workers)]
        self._result_q = self._ctx.Queue()
        self._procs = []
        cfg = {
            'policy_hidden_dim': policy_hidden_dim,
            'pool_max_size': pool_max_size,
            'seed_base': seed_base,
        }
        for wid in range(n_workers):
            p = self._ctx.Process(
                target=_worker_loop,
                args=(wid, self._task_qs[wid], self._result_q, cfg),
                daemon=True,
            )
            p.start()
            self._procs.append(p)
        self._seen_snapshots = 0
        logger.info(f'ParallelRolloutCollector: spawned {n_workers} rollout workers')

    def collect(self, policy, pool, n_episodes, *, gamma, shaping_anneal,
                holding_reward_rate, max_t):
        """Broadcast the current policy + pool delta, run n_episodes across workers.

        Returns the per-worker payloads in worker-id order (deterministic concat order).
        """
        policy_sd = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        new_snaps, cnt = pool.new_snapshots_since(self._seen_snapshots)
        self._seen_snapshots = cnt
        weights = pool.weights

        base, rem = divmod(n_episodes, self._n)
        counts = [base + (1 if i < rem else 0) for i in range(self._n)]

        for wid in range(self._n):
            self._task_qs[wid].put({
                'policy_sd': policy_sd,
                'new_snapshots': new_snaps,
                'weights': weights,
                'n_episodes': counts[wid],
                'gamma': gamma,
                'shaping_anneal': shaping_anneal,
                'holding_reward_rate': holding_reward_rate,
                'max_t': max_t,
            })

        results = {}
        for _ in range(self._n):
            wid, status, payload = self._result_q.get()
            if status == 'ERROR':
                self.shutdown()
                raise RuntimeError(f'rollout worker {wid} failed:\n{payload}')
            results[wid] = payload
        return [results[w] for w in range(self._n)]

    def shutdown(self):
        for q in self._task_qs:
            try:
                q.put(None)
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self._procs = []
