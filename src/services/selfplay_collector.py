"""Parallel self-play collection for expert iteration, across persistent worker processes.

Mirrors `rollout_collector.py`'s conventions almost exactly (persistent `spawn`-context
workers, `torch.set_num_threads(1)`, a shared atomic counter for dynamic load balancing,
heavy imports kept inside the worker function so a spawned child doesn't pay for them at
module-import time) — the difference is what a worker actually does: instead of one PPO
policy playing episodes against a sampled opponent, each worker builds its own `PuctBot`
from the (policy_path, critic_path, value_mode, search-knob) config in the task and plays
whole self-play games (`expert_iteration.play_selfplay_game`), returning a stacked
`SelfPlayDataset` plus per-game stats for logging.

Like `ParallelRolloutCollector`, the worker pool is spawned once and reused across many
`submit()`/`gather()` rounds — `app/expert_iteration.py`'s `loop` command builds one
collector before its round loop and reuses it every round, rebuilding each worker's
`PuctBot` only when the checkpoint/config actually changed (a new round's freshly
distilled nets), not on every task.

`gather()` shows a live `rich` progress bar (mirrors `gauntlet_parallel.py`'s use of
`rich.progress.Progress`): each worker increments a shared "games completed" counter
after every game it finishes (not just once at the very end), so the bar advances
continuously — self-play games are ~seconds each and a round is many minutes, the
one thing a user watching the console actually wants to see move.
"""
import logging
import multiprocessing as mp
import time
from queue import Empty

from rich.progress import Progress

logger = logging.getLogger('warchest')

# Fields identifying "does this worker need to rebuild its PuctBot" — everything that
# affects the bot's behavior, not `n_games`/`temperature`/`temp_moves`/`max_turns` (pure
# per-round play parameters that don't require a new bot instance).
_BOT_KEYS = ('policy_path', 'critic_path', 'value_mode', 'c_puct', 'max_branching',
            'time_budget', 'dirichlet_alpha', 'dirichlet_frac', 'see_opponent_hand',
            'n_determinizations', 'forced_playouts_k')


def _worker_loop(worker_id, task_q, result_q, counter, completed, seed_base):
    """Persistent worker: (re)build a PuctBot on demand, claim games from the shared
    counter, play them, and return a stacked SelfPlayDataset + per-game stats.
    """
    import os
    import sys
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if root not in sys.path:
        sys.path.insert(0, root)

    import numpy as np
    import torch
    torch.set_num_threads(1)  # CRITICAL: else N workers × intra-op threads oversubscribe cores

    from src.services.environment.warchest_env import WarChestEnv
    from src.services.bots.puct_bot import PuctBot
    from src.services.expert_iteration import SelfPlayDataset, play_selfplay_game

    np.random.seed(seed_base + worker_id)
    torch.manual_seed(seed_base + worker_id)

    cached_key = None
    bot = None
    env = None

    def claim_game():
        # Atomic decrement of the shared remaining-games counter (dynamic load
        # balancing — mirrors rollout_collector.py's claim_episode): fast workers
        # naturally take more games, so a round waits on total work / N rather than
        # on the slowest worker's fixed share.
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
            key = tuple(task[k] for k in _BOT_KEYS)
            if key != cached_key:
                bot = PuctBot(
                    policy_path=task['policy_path'], critic_path=task['critic_path'],
                    value_mode=task['value_mode'], c_puct=task['c_puct'],
                    max_branching=task['max_branching'], time_budget=task['time_budget'],
                    dirichlet_alpha=task['dirichlet_alpha'], dirichlet_frac=task['dirichlet_frac'],
                    see_opponent_hand=task['see_opponent_hand'],
                    n_determinizations=task['n_determinizations'],
                    forced_playouts_k=task['forced_playouts_k'],
                    device='cpu', stats_log_every=0,
                )
                # PolicyCriticBot.__init__ already resolved the obs encoder for this
                # policy checkpoint — reuse it rather than reloading the checkpoint
                # again just to read obs_version.
                env = WarChestEnv(save_game_history=False, obs_encoder=bot._policy_encoder)
                cached_key = key

            dataset = SelfPlayDataset()
            game_stats = []
            while claim_game():
                stats = play_selfplay_game(
                    bot, env, dataset, temperature=task['temperature'],
                    temp_moves=task['temp_moves'], max_turns=task['max_turns'],
                    apprentice_frac=task['apprentice_frac'],
                )
                game_stats.append(stats)
                with completed.get_lock():
                    completed.value += 1

            n_samples = len(dataset)
            payload = {
                'n_samples': n_samples, 'game_stats': game_stats,
                't_worker_wall': time.perf_counter() - t_start,
            }
            if n_samples > 0:
                payload['dataset'] = dataset.stack()
            result_q.put((worker_id, 'OK', payload))
        except Exception:
            import traceback
            result_q.put((worker_id, 'ERROR', traceback.format_exc()))


class ParallelSelfPlayCollector:
    """Owns the persistent self-play worker pool; broadcasts a round's config and
    gathers the combined dataset. `submit()`/`gather()` are separate (mirroring
    `ParallelRolloutCollector`) so a caller could overlap rounds if it ever needed to,
    though `app/expert_iteration.py` currently just calls `collect()`.
    """

    def __init__(self, n_workers, seed_base=0):
        self._ctx = mp.get_context('spawn')
        self._n = n_workers
        self._task_qs = [self._ctx.Queue() for _ in range(n_workers)]
        self._result_q = self._ctx.Queue()
        self._counter = self._ctx.Value('i', 0)  # remaining games this round (claim)
        self._completed = self._ctx.Value('i', 0)  # games finished this round (progress bar)
        self._n_games = 0
        self._desc = 'self-play'
        self._procs = []
        for wid in range(n_workers):
            p = self._ctx.Process(
                target=_worker_loop,
                args=(wid, self._task_qs[wid], self._result_q, self._counter, self._completed, seed_base),
                daemon=True,
            )
            p.start()
            self._procs.append(p)
        logger.info('ParallelSelfPlayCollector: spawned %d self-play workers', n_workers)

    def submit(self, *, policy_path, critic_path, value_mode, n_games, c_puct, max_branching,
               time_budget, dirichlet_alpha, dirichlet_frac, temperature, temp_moves, max_turns,
               see_opponent_hand=True, n_determinizations=1, forced_playouts_k=0.0,
               apprentice_frac=0.0, desc='self-play'):
        with self._counter.get_lock():
            self._counter.value = n_games
        with self._completed.get_lock():
            self._completed.value = 0
        self._n_games = n_games
        self._desc = desc
        task = {
            'policy_path': policy_path, 'critic_path': critic_path, 'value_mode': value_mode,
            'c_puct': c_puct, 'max_branching': max_branching, 'time_budget': time_budget,
            'dirichlet_alpha': dirichlet_alpha, 'dirichlet_frac': dirichlet_frac,
            'see_opponent_hand': see_opponent_hand,
            'n_determinizations': n_determinizations,
            'forced_playouts_k': forced_playouts_k,
            'temperature': temperature, 'temp_moves': temp_moves, 'max_turns': max_turns,
            'apprentice_frac': apprentice_frac,
        }
        for wid in range(self._n):
            self._task_qs[wid].put(task)

    def gather(self):
        """Block for all worker payloads (showing a live per-game progress bar
        meanwhile); return `(dataset, game_stats, timing)`.

        `timing['rollout']` is the critical-path worker wall (max over workers);
        `timing['ipc']` is gather wall not explained by the slowest worker
        (result serialization + transfer + dispatch) — same convention as
        `ParallelRolloutCollector.gather`.
        """
        from .expert_iteration import SelfPlayDataset  # local: keep module-top import-light
        # (see module docstring — a spawned child re-imports this file to find
        # _worker_loop, so a module-top import here would pull torch/env/policy into
        # every worker before it even receives a task).

        t0 = time.perf_counter()
        results = {}
        # Poll for final per-worker payloads with a short timeout instead of a plain
        # blocking get(), so the wait can be interleaved with reading the shared
        # `_completed` counter to advance the progress bar in real time — workers bump
        # it after every game, well before their final (all-games-done) payload lands.
        with Progress() as progress:
            task_id = progress.add_task(self._desc, total=self._n_games)
            while len(results) < self._n:
                try:
                    wid, status, payload = self._result_q.get(timeout=0.2)
                except Empty:
                    pass
                else:
                    if status == 'ERROR':
                        self.shutdown()
                        raise RuntimeError(f'self-play worker {wid} failed:\n{payload}')
                    results[wid] = payload
                progress.update(task_id, completed=self._completed.value)
            progress.update(task_id, completed=self._n_games)
        gather_wall = time.perf_counter() - t0

        parts = [results[w]['dataset'] for w in range(self._n) if results[w]['n_samples'] > 0]
        game_stats = [gs for w in range(self._n) for gs in results[w]['game_stats']]
        worker_walls = [results[w]['t_worker_wall'] for w in range(self._n)]
        max_worker_wall = max(worker_walls) if worker_walls else 0.0
        if not parts:
            raise RuntimeError('self-play round produced zero samples across all workers '
                               '(n_games=0? every game ended with no recordable move?)')
        dataset = SelfPlayDataset.concat(parts)
        timing = {'rollout': max_worker_wall, 'ipc': max(0.0, gather_wall - max_worker_wall)}
        return dataset, game_stats, timing

    def collect(self, **kw):
        self.submit(**kw)
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
