"""Parallel rollout collection across persistent worker processes.

See docs/parallel_rollouts.md. Workers are CPU-only and run ONLY the policy (act) + the
opponent move + env.step — the critic never leaves the main process (its values are computed
in one batched GPU pass after collection). Each batch the main process broadcasts the small
policy state_dict plus, incrementally, any new pool snapshot; workers pull episodes from a
shared counter (dynamic load balancing — P11a) and return pre-stacked numpy transitions which
the main process concatenates into the RolloutBuffer.

The API is split into submit() / gather() so the caller can overlap the next batch's collection
with the current batch's GPU update (P11b).

Kept import-light at module top so `spawn` children re-import it cheaply; torch/env/policy are
imported inside the worker function.
"""

import logging
import multiprocessing as mp
import time

logger = logging.getLogger('warchest')


def _worker_loop(worker_id, task_q, result_q, counter, cfg):
    """Persistent worker: load weights, claim episodes from the shared counter, return numpy."""
    import os
    import sys
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
    policy_arch = cfg['policy_arch']

    def policy_constructor():
        return Policy(device=device, hidden_dim=hidden_dim, arch=policy_arch)

    env = WarChestEnv(save_game_history=False, debug_mode=False)
    policy = policy_constructor()
    policy.eval()
    pool = OpponentPool(max_size=cfg['pool_max_size'], snapshot_every=10 ** 9,
                        p_random=1.0, p_greedy=0.0, p_pool=0.0, p_lookahead_critic=0.0,
                        lookahead_critic_time_budget=cfg['lookahead_critic_time_budget'],
                        lookahead_critic_device='cpu', p_puct=0.0, p_puct_live=0.0,
                        puct_time_budget=cfg.get('puct_time_budget', 0.1), puct_device='cpu',
                        puct_max_simulations=cfg.get('puct_max_simulations'),
                        puct_live_time_budget=cfg.get('puct_live_time_budget', 1.0),
                        puct_live_max_simulations=cfg.get('puct_live_max_simulations', 100),
                        puct_blind=cfg.get('puct_blind', True),
                        puct_forced_playouts_k=cfg.get('puct_forced_playouts_k', 2.0),
                        p_random_eval=0.0,
                        # Per-worker θ stream (docs/IDEAS.md B1). Without the offset every
                        # worker would replay the *same* sequence of sampled playstyles, so
                        # a batch would cover a handful of θ instead of one per episode.
                        random_eval_seed=seed,
                        random_eval_reply_branching=cfg.get('random_eval_reply_branching', 2))

    def claim_episode():
        # Atomic decrement of the shared remaining-episode counter (dynamic load balancing:
        # fast workers naturally take more episodes, so the batch waits on total work / N
        # rather than on the slowest fixed share).
        with counter.get_lock():
            if counter.value <= 0:
                return False
            counter.value -= 1
            return True

    while True:
        task = task_q.get()
        if task is None:
            break
        try:
            t_start = time.perf_counter()
            policy.load_state_dict(task['policy_sd'])
            for sd in task['new_snapshots']:
                pool.append_snapshot(sd)
            pool.set_weights(**task['weights'])

            obs_l, act_l, lp_l, rew_l, opp_l, priv_l = [], [], [], [], [], []
            oid_l = []
            ends, episode_dicts = [], []
            collect_dense = cfg.get('collect_dense', False)
            aux_parts = []  # per-episode dicts of dense aux samples (collect_dense only)
            t_env = t_model = 0.0

            while claim_episode():
                main_pid = int(np.random.choice([1, 2]))
                opp, opp_type = pool.sample(policy_constructor, device)
                steps, ep = play_episode(
                    env, policy, opp, main_pid, opp_type,
                    gamma=task['gamma'],
                    shaping_anneal=task['shaping_anneal'],
                    base_shaping_anneal=task['base_shaping_anneal'],
                    holding_reward_rate=task['holding_reward_rate'],
                    max_t=task['max_t'],
                    collect_dense=collect_dense,
                )
                obs_l.extend(steps['obs'])
                act_l.extend(steps['actions'])
                lp_l.extend(float(x) for x in steps['log_probs'])
                rew_l.extend(steps['rewards'])
                opp_l.extend(steps['opp_onehots'])
                oid_l.extend(steps['opp_ids'])
                priv_l.extend(steps['privileged'])
                ends.append(len(rew_l))
                if collect_dense and 'aux_targets' in steps:
                    aux_parts.append({
                        'boards': steps['aux_boards'], 'globals': steps['aux_globals'],
                        'opp_onehots': steps['aux_opp_onehots'],
                        'privileged': steps['aux_privileged'], 'targets': steps['aux_targets'],
                    })
                t_env += ep.pop('t_env')
                t_model += ep.pop('t_model_play')
                episode_dicts.append(ep)

            # A worker may claim zero episodes (all taken by faster peers) — return an empty,
            # concat-safe payload the buffer ingest skips.
            if obs_l:
                payload = {
                    'boards': np.stack([o['board'] for o in obs_l]),
                    'globals': np.stack([o['global'] for o in obs_l]),
                    'masks': np.stack([o['valid_action_mask'] for o in obs_l]),
                    'actions': np.array(act_l, dtype=np.int64),
                    'log_probs': np.array(lp_l, dtype=np.float32),
                    'rewards': np.array(rew_l, dtype=np.float32),
                    'opp_onehots': np.stack(opp_l),
                    'opp_ids': np.array(oid_l, dtype=np.int64),
                    'privileged': np.stack(priv_l),
                    'episode_ends': ends,
                    'episode_dicts': episode_dicts,
                }
            else:
                payload = {'episode_ends': [], 'episode_dicts': []}
            # Concatenate this worker's dense aux samples into one block for the buffer
            # (None when the flag is off or no opponent nodes were seen — ingest skips it).
            if aux_parts:
                payload['aux'] = {
                    'boards': np.concatenate([p['boards'] for p in aux_parts]),
                    'globals': np.concatenate([p['globals'] for p in aux_parts]),
                    'opp_onehots': np.concatenate([p['opp_onehots'] for p in aux_parts]),
                    'privileged': np.concatenate([p['privileged'] for p in aux_parts]),
                    'targets': np.concatenate([p['targets'] for p in aux_parts]),
                }
            payload['t_env'] = t_env
            payload['t_model_play'] = t_model
            payload['t_worker_wall'] = time.perf_counter() - t_start
            result_q.put((worker_id, 'OK', payload))
        except Exception:
            import traceback
            result_q.put((worker_id, 'ERROR', traceback.format_exc()))


class ParallelRolloutCollector:
    """Owns the persistent worker pool; broadcasts weights and gathers transitions per batch.

    submit()/gather() are separate so the caller can overlap the next batch's collection with
    the current batch's GPU update (P11b). collect() = submit()+gather() for the non-overlap
    path. Episodes are handed out via a shared atomic counter (P11a dynamic balancing), so
    which worker plays how many episodes varies with timing — runs are NOT bit-reproducible
    even for a fixed seed (documented tradeoff).
    """

    def __init__(self, n_workers, *, policy_hidden_dim, policy_arch, pool_max_size, seed_base,
                 lookahead_critic_time_budget=0.1, puct_time_budget=0.1, collect_dense=False,
                 random_eval_reply_branching=2, puct_max_simulations=None,
                 puct_live_time_budget=1.0, puct_live_max_simulations=100,
                 puct_blind=True, puct_forced_playouts_k=2.0):
        self._ctx = mp.get_context('spawn')
        self._n = n_workers
        self._task_qs = [self._ctx.Queue() for _ in range(n_workers)]
        self._result_q = self._ctx.Queue()
        self._counter = self._ctx.Value('i', 0)  # remaining episodes this batch
        self._procs = []
        cfg = {
            'policy_hidden_dim': policy_hidden_dim,
            'policy_arch': policy_arch,
            'pool_max_size': pool_max_size,
            'seed_base': seed_base,
            'lookahead_critic_time_budget': lookahead_critic_time_budget,
            'puct_time_budget': puct_time_budget,
            'puct_max_simulations': puct_max_simulations,
            'puct_live_time_budget': puct_live_time_budget,
            'puct_live_max_simulations': puct_live_max_simulations,
            'puct_blind': puct_blind,
            'puct_forced_playouts_k': puct_forced_playouts_k,
            'collect_dense': collect_dense,
            'random_eval_reply_branching': random_eval_reply_branching,
        }
        for wid in range(n_workers):
            p = self._ctx.Process(
                target=_worker_loop,
                args=(wid, self._task_qs[wid], self._result_q, self._counter, cfg),
                daemon=True,
            )
            p.start()
            self._procs.append(p)
        self._seen_snapshots = 0
        logger.info(f'ParallelRolloutCollector: spawned {n_workers} rollout workers')

    def submit(self, policy, pool, n_episodes, *, gamma, shaping_anneal,
               base_shaping_anneal, holding_reward_rate, max_t):
        """Broadcast the policy + pool delta and hand out n_episodes for the workers to claim."""
        policy_sd = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        new_snaps, cnt = pool.new_snapshots_since(self._seen_snapshots)
        self._seen_snapshots = cnt
        weights = pool.weights

        with self._counter.get_lock():
            self._counter.value = n_episodes

        task = {
            'policy_sd': policy_sd,
            'new_snapshots': new_snaps,
            'weights': weights,
            'gamma': gamma,
            'shaping_anneal': shaping_anneal,
            'base_shaping_anneal': base_shaping_anneal,
            'holding_reward_rate': holding_reward_rate,
            'max_t': max_t,
        }
        for wid in range(self._n):
            self._task_qs[wid].put(task)

    def gather(self):
        """Block for all worker payloads; return (chunks, timing).

        timing keys (seconds): rollout (critical-path worker wall = max over workers), env and
        model_play (aggregate across workers), ipc (gather wall not explained by the slowest
        worker ≈ result serialization + transfer + dispatch).
        """
        # Time the main thread is BLOCKED in the get-loop (not since submit): in the overlap
        # path the rollout ran during the previous update, so this measures only the
        # unhidden tail + transfer, not the whole intervening update.
        t0 = time.perf_counter()
        results = {}
        for _ in range(self._n):
            wid, status, payload = self._result_q.get()
            if status == 'ERROR':
                self.shutdown()
                raise RuntimeError(f'rollout worker {wid} failed:\n{payload}')
            results[wid] = payload
        gather_wall = time.perf_counter() - t0

        chunks = [results[w] for w in range(self._n) if results[w]['episode_ends']]
        worker_walls = [results[w]['t_worker_wall'] for w in range(self._n)]
        max_worker_wall = max(worker_walls) if worker_walls else 0.0
        timing = {
            'rollout': max_worker_wall,
            'env': sum(results[w]['t_env'] for w in range(self._n)),
            'model_play': sum(results[w]['t_model_play'] for w in range(self._n)),
            'ipc': max(0.0, gather_wall - max_worker_wall),
        }
        return chunks, timing

    def collect(self, policy, pool, n_episodes, **kw):
        self.submit(policy, pool, n_episodes, **kw)
        return self.gather()

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
