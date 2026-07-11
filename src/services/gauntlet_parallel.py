"""Parallel round-robin gauntlet across persistent worker processes.

Mirrors `rollout_collector.py`'s conventions (persistent `spawn`-context workers,
`torch.set_num_threads(1)`, never pickling live `nn.Module`s / bots across the
process boundary) but is simpler: the agent field is static for the whole run —
no per-batch broadcast — so a single shared task queue pre-loaded with the full,
deterministic task list gives dynamic load balancing for free (fast workers
naturally drain more of the queue; a per-worker-queue+counter scheme would force
either static partitioning — bad, since a LookaheadCriticBot game is much slower
than a greedy/random one — or reimplementing the counter dance for no benefit).

Every agent is rebuilt once per worker from a small picklable spec
(`gauntlet.build_agent`), not received as a live object: `LookaheadBot`/
`LookaheadCriticBot` instances are unconditionally unpicklable (both monkeypatch
`_sim_env._draw_one` with a bound method whose `__name__` doesn't match the
attribute it's stored under, which breaks pickle's bound-method reduction), so
specs are used uniformly for every agent kind rather than special-casing the
kinds that happen to be picklable.
"""

import logging
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

from .gauntlet import build_task_list, record_result, _finalize_report

logger = logging.getLogger('warchest')

# Slower agent kinds first (lower rank = dispatched earlier): a lookahead/critic game can run
# many times longer than a greedy/random one, so starting them first avoids the common
# straggler pattern where all the fast games finish early and every worker but one goes idle
# waiting on a handful of slow games that were left until last.
_KIND_RANK = {'lookahead_critic': 0, 'lookahead': 1}


def _prioritize(tasks, agent_specs):
    """Stable-sort `tasks` so games touching a slow agent kind are dispatched first.

    Does not change which seed goes to which (i, j) pairing (already fixed by
    `build_task_list`) — only the order workers pull tasks off the queue, which is
    otherwise unconstrained (see module docstring on dynamic load balancing).
    """
    def rank(task):
        i, j, _seed, _p1_is_i = task
        return min(_KIND_RANK.get(agent_specs[i]['kind'], 2),
                   _KIND_RANK.get(agent_specs[j]['kind'], 2))

    return sorted(tasks, key=rank)


def _worker_loop(worker_id, task_q, result_q, agent_specs):
    """Persistent worker: build the whole agent field once, then play games until told to stop."""
    import os
    import sys
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if root not in sys.path:
        sys.path.insert(0, root)

    # A spawned process (`mp.get_context('spawn')`, module docstring) is a fresh
    # interpreter — it does not inherit gauntlet.py main()'s logging.basicConfig,
    # so every actual game (and every bot's act()-time logging, e.g.
    # LookaheadCriticBot's per-move/aggregate search stats) would otherwise log to
    # nothing. `worker_id` (not just the OS pid) tags each line so interleaved
    # output from multiple workers stays attributable.
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s [%(levelname)s] [worker {worker_id}] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    import torch
    torch.set_num_threads(1)  # CRITICAL: else N workers x intra-op threads oversubscribe cores
    # (doubly so for LookaheadCriticBot: wall-clock-budgeted *and* torch-driven per search node)

    from src.services.gauntlet import build_agent, play_game

    device = torch.device('cpu')
    try:
        agents = [build_agent(spec, device=device) for spec in agent_specs]
    except Exception:
        import traceback
        result_q.put((worker_id, 'ERROR', traceback.format_exc()))
        return

    while True:
        task = task_q.get()
        if task is None:
            break
        i, j, game_seed, p1_is_i = task
        try:
            if p1_is_i:
                res = play_game(agents[i], agents[j], seed=game_seed)
            else:
                res = play_game(agents[j], agents[i], seed=game_seed)
            result_q.put((worker_id, 'OK', (i, j, p1_is_i, res)))
        except Exception:
            import traceback
            result_q.put((worker_id, 'ERROR', traceback.format_exc()))


def round_robin_parallel(agent_specs, names, *, k_games=20, seed=0, n_workers):
    """Parallel equivalent of `gauntlet.round_robin`, given agent specs (not live agents).

    `names` must be the display names in the same order `agent_specs` implies
    (i.e. what `[build_agent(s, device=...).name for s in agent_specs]` would
    produce) — passed in rather than recomputed so the caller isn't forced to
    build a throwaway agent field in the main process just to read `.name`.

    Returns the same dict shape as `round_robin`. With the same `seed`/`k_games`
    and a field of budget-free agents (no wall-clock search), the result matrix
    is bit-identical to `round_robin`'s regardless of `n_workers`, since the task
    list (and its seed assignment) is built once, up front, independent of
    dispatch order.
    """
    n = len(agent_specs)
    tasks = _prioritize(build_task_list(n, k_games=k_games, seed=seed), agent_specs)

    ctx = mp.get_context('spawn')
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    for t in tasks:
        task_q.put(t)
    for _ in range(n_workers):
        task_q.put(None)  # one sentinel per worker so every worker exits cleanly

    procs = [
        ctx.Process(target=_worker_loop, args=(wid, task_q, result_q, agent_specs), daemon=True)
        for wid in range(n_workers)
    ]
    for p in procs:
        p.start()

    wins = np.zeros((n, n), dtype=np.float64)
    games = np.zeros((n, n), dtype=np.float64)

    def _shutdown():
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    try:
        with tqdm(total=len(tasks), desc='gauntlet', unit='game') as bar:
            for _ in range(len(tasks)):
                worker_id, status, payload = result_q.get()
                if status == 'ERROR':
                    _shutdown()
                    raise RuntimeError(f'gauntlet worker {worker_id} failed:\n{payload}')
                i, j, p1_is_i, res = payload
                record_result(wins, games, i, j, p1_is_i, res)
                bar.update(1)
    except BaseException:
        _shutdown()
        raise
    _shutdown()

    return _finalize_report(names, wins, games)
