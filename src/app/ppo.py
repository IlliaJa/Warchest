import glob
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

# Match every other entry point under src/app (and the invocation README/CLAUDE.md document)
# so `python src/app/ppo.py` works without PYTHONPATH being set by hand.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.services.policy.policy import Policy, Critic
from src.services.policy.checkpoint import (
    save_policy_checkpoint, save_critic_checkpoint, load_policy_checkpoint,
    CRITIC_ARCHS, CURRENT_CRITIC_ARCH, POLICY_ARCHS, CURRENT_ARCH,
)
from src.services.environment.obs_encoders import get_encoder
from src.services.environment.warchest_env import WarChestEnv
from src.services.environment.rollout_core import (
    play_episode, SHAPING_C, C_MAT, OPP_TYPE_IDX, OPP_GROUP_IDX,
)
from src.services.opponent_pool import OpponentPool
from src.services.rollout_collector import ParallelRolloutCollector
from src.utils.rollout_buffer import RolloutBuffer
from src.services.bots import GreedyBot, RandomBot
from src.utils.elo import EloTracker

# SHAPING_C / C_MAT / OPP_TYPE_IDX are defined in rollout_core (they belong to the per-step
# reward that lives there) and re-imported above; kept referenceable from this module.
# The holding reward and the material PBRS term are linearly annealed from
# SHAPING_ANNEAL_INIT down to SHAPING_ANNEAL_FINAL over the first
# SHAPING_ANNEAL_HALF_FRAC of the run, then held at the floor. This keeps the dense
# guidance early (weak critic, high entropy) and hands the final policy back toward
# the true terminal objective — the over-shaping antidote (see docs/decision.md,
# 2026-07-03). Base-diff PBRS (SHAPING_C) is intentionally left constant.
SHAPING_ANNEAL_INIT = 1.0
SHAPING_ANNEAL_FINAL = 0.1
SHAPING_ANNEAL_HALF_FRAC = 0.5
# Reverse of OPP_GROUP_IDX, so the per-opponent advantage offsets log by name.
OPP_GROUP_NAME = {v: k for k, v in OPP_GROUP_IDX.items()}
# Where saved policies live; the eval reference opponent is picked from here.
POLICY_CKPT_GLOB = 'data/warchest_ppo_*.pth'
use_wandb = False

logger = logging.getLogger('warchest')


def latest_policy_checkpoint(pattern=POLICY_CKPT_GLOB):
    """Newest saved policy checkpoint, or None if none exist.

    Newest by modification time rather than by filename: the run-final checkpoints
    happen to sort by their timestamp suffix, but intermediate checkpoints saved
    mid-run need not, and this is chosen once at startup where a wrong pick would
    silently mislabel the whole run's eval baseline.
    """
    paths = glob.glob(pattern)
    if not paths:
        return None
    return max(paths, key=lambda p: (os.path.getmtime(p), p))


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

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std


class ReferenceOpponent:
    """A frozen policy checkpoint played as a fixed eval yardstick.

    Speaks the `act(obs) -> (action, log_prob, entropy)` bot protocol `_eval_episode`
    expects, so it drops in beside `GreedyBot`/`RandomBot` with no special-casing.

    `encoder` is set only when the checkpoint was trained under an *older* obs version
    than the env is emitting: the checkpoint's input layers are sized to its own
    encoder, so the env's observation is not a valid input for it and has to be
    re-derived (which is exactly what the gauntlet's `PolicyAgent` does). When the
    versions match this stays None and the env's already-computed obs is used as-is —
    re-encoding it would be pure waste in a loop that runs `eval_episodes` games.
    """

    def __init__(self, policy, *, path, obs_version, encoder=None, env=None):
        self.policy = policy
        self.path = path
        self.obs_version = obs_version
        self.name = os.path.splitext(os.path.basename(path))[0]
        self._encoder = encoder
        self._env = env

    def act(self, obs):
        if self._encoder is not None:
            obs = self._encoder.encode(self._env)
        return self.policy.act(obs)


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

    KL_TARGET = 0.03  # skip a minibatch whose approx-KL exceeds this (see _update_actor).
    # Raised from 0.015 now that offending minibatches are skipped individually rather
    # than aborting the whole update: at 0.015, once epoch 0 moves the policy ~0.015,
    # every later-epoch minibatch trips the gate and gets skipped, so almost no extra
    # data is used. 0.03 lets epochs 1-3 actually contribute while the PPO clip still
    # bounds each step.

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
        # Separate bonus on the verb-marginal entropy, annealed to a non-zero floor so
        # low-cardinality verbs stay in the repertoire (docs/IDEAS.md #R8). 0.0 => disabled.
        self._verb_entropy_coeff_init = hp.get('verb_entropy_coeff', 0.0)
        self._verb_entropy_coeff_final = hp.get('verb_entropy_coeff_final', self._verb_entropy_coeff_init)
        self._verb_entropy_coeff = self._verb_entropy_coeff_init
        # learning rates are linearly decayed init -> init*lr_final_frac (0 => decay to 0)
        self._lr_actor_init = hp['lr_actor']
        self._lr_critic_init = hp['lr_critic']
        self._lr_final_frac = hp.get('lr_final_frac', 0.0)
        self._holding_reward_rate = hp['holding_reward_rate']
        # anneal multiplier applied to holding + material shaping; set per batch.
        self._shaping_anneal = SHAPING_ANNEAL_INIT
        self._minibatch_size = hp['minibatch_size']
        # Parallel rollout collection (docs/parallel_rollouts.md). n_workers<=1 => in-process
        # path. Capped at collect_episodes so every worker gets >=1 episode.
        self._n_workers = min(hp.get('n_workers', 1), hp['collect_episodes'])
        # Overlap batch N+1's (CPU worker) collection with batch N's GPU update. Hides the
        # rollout wall behind the update at the cost of 1-step-stale behavior weights + a
        # second in-flight buffer in RAM. Only meaningful when n_workers > 1.
        self._overlap = hp.get('overlap_collection', False)
        self._rollout_seed = hp.get('rollout_seed', 0)
        self._policy_hidden_dim = hp['hidden_dim']
        self._policy_arch = hp.get('policy_arch', CURRENT_ARCH)
        self._pool_max_size = hp.get('pool_max_size', 20)
        self._collector = None
        self._print_every = hp['print_every']
        self._eval_every = hp.get('eval_every', 10)
        self._eval_episodes = hp.get('eval_episodes', 20)
        self._wr_finetune_threshold = hp['wr_greedy_finetune_threshold']
        self._opp_weights_initial = {
            'p_random': hp.get('p_random_initial', 0.0),
            'p_greedy': hp['p_greedy_initial'],
            'p_pool': hp['p_pool_initial'],
            'p_lookahead_critic': hp['p_lookahead_critic_initial'],
            'p_puct': hp.get('p_puct_initial', 0.0),
            'p_random_eval': hp.get('p_random_eval_initial', 0.0),
            'p_policy_theta': hp.get('p_policy_theta_initial', 0.0),
        }
        self._opp_weights_finetune = {
            'p_random': hp['p_random_finetune'],
            'p_greedy': hp['p_greedy_finetune'],
            'p_pool': hp['p_pool_finetune'],
            'p_lookahead_critic': hp['p_lookahead_critic_finetune'],
            'p_puct': hp.get('p_puct_finetune', 0.0),
            'p_random_eval': hp.get('p_random_eval_finetune', 0.0),
            'p_policy_theta': hp.get('p_policy_theta_finetune', 0.0),
        }
        # Advantage normalisation (docs/next_iteration.md §5 row 6). 'per_opponent' centres
        # advantages inside each opponent group, which is what lets `critic_v3` drop the
        # opponent one-hot without putting the opponent-identity offset back into the policy
        # gradient — the two are a matched pair. 'global' reproduces every pre-2026-08-09 run.
        self._adv_norm = hp.get('adv_norm', 'per_opponent')
        if not critic.uses_opp_onehot and self._adv_norm != 'per_opponent':
            logger.warning(
                f'critic arch {critic.arch} has no opponent one-hot but adv_norm='
                f'{self._adv_norm!r}: nothing removes the per-opponent advantage offset, so '
                f'actions taken against weak opponents get a systematically positive '
                f'advantage. Use --adv-norm per_opponent unless this is a deliberate ablation.'
            )
        self._lookahead_critic_time_budget = hp['lookahead_critic_time_budget']
        self._puct_time_budget = hp.get('puct_time_budget', 0.1)
        self._random_eval_reply_branching = hp.get('random_eval_reply_branching', 2)
        # Dense critic targets (docs/IDEAS.md #12): also regress the critic on opponent-
        # decision nodes via an auxiliary MC-return loss, on top of the unchanged main GAE
        # targets. Off by default — an opt-in experiment; the aux loss is scaled by
        # `aux_critic_coeff` so it stays a secondary supervision signal.
        self._dense_critic = hp.get('dense_critic_targets', False)
        self._aux_critic_coeff = hp.get('aux_critic_coeff', 0.5)

        # Board-only auxiliary value loss (docs/next_iteration.md §2 step 2, §3.4).
        # `critic_v2` only. The main head reaches its target through 299 non-spatial inputs
        # and draws 76% of its sensitivity from globals against 14% from the board, so the
        # board pathway gets almost no gradient and its ReLU trunk falls into an absorbing
        # state it can never leave. This term regresses the SAME return from the pooled
        # board block alone, so it cannot be satisfied from globals and the trunk has to
        # carry signal. Kept small — it is a pressure source, not a second objective.
        self._aux_board_coeff = hp.get('aux_board_coeff', 0.1)
        # How often to log per-conv trunk health. This bug ran undetected for a whole
        # generation of checkpoints and voided every search measurement taken with them.
        self._trunk_health_every = hp.get('trunk_health_every', 10)

        # Shaped-return dump (docs/next_iteration.md §5 row 2a). Off unless a directory is
        # given. Collects the dataset the critic-target A/B needs and nothing else touches.
        self._dump_returns_dir = hp.get('dump_returns_dir')
        self._dump_every_samples = hp.get('dump_returns_every_samples', 16000)
        self._dump_parts = []
        self._dump_n = 0
        self._dump_shard = 0

        # training-lifetime state (persists across batches).
        # Snapshotting rarely (vs every batch) makes the fixed-size pool span a wide skill
        # range — old/weak to recent/strong — instead of 20 near-identical recent copies.
        # The current policy then beats the weak snapshots (positive advantage) and ties the
        # strong ones, so self-play games carry a real learning signal instead of ~0-advantage
        # mirror matches. Pool spans roughly max_size * snapshot_every batches.
        self._pool = OpponentPool(
            max_size=hp.get('pool_max_size', 20),
            snapshot_every=hp.get('pool_snapshot_every', 15),
            p_random=hp.get('p_random_initial', 0.0),
            p_greedy=hp['p_greedy_initial'],
            p_pool=hp['p_pool_initial'],
            p_lookahead_critic=hp['p_lookahead_critic_initial'],
            lookahead_critic_time_budget=hp['lookahead_critic_time_budget'],
            p_puct=hp.get('p_puct_initial', 0.0),
            puct_time_budget=hp.get('puct_time_budget', 0.1),
            p_random_eval=hp.get('p_random_eval_initial', 0.0),
            random_eval_seed=hp.get('rollout_seed', 0),
            random_eval_reply_branching=hp.get('random_eval_reply_branching', 2),
            p_policy_theta=hp.get('p_policy_theta_initial', 0.0),
        )
        self._buffer = RolloutBuffer()
        self._greedy_bot = GreedyBot()
        self._elo = EloTracker()
        self._score_deque = deque(maxlen=self._print_every * self._collect_episodes)
        self._wr_vs_pool = deque(maxlen=100)
        self._wr_vs_greedy = deque(maxlen=100)
        self._wr_vs_lookahead_critic = deque(maxlen=100)
        self._wr_vs_puct = deque(maxlen=100)
        self._wr_vs_random_eval = deque(maxlen=100)
        self._wr_vs_policy_theta = deque(maxlen=100)

        # pre-computed once; actor-side params are needed for separate gradient clipping
        self._actor_side_params = list(self._policy.parameters())

        self._ret_normalizer = ReturnNormalizer()

        # Frozen previous-generation policy the eval phase plays against, so a run answers
        # "is this policy better than the last one I saved?" directly instead of only
        # "how does it do against greedy/random?" — which both saturate long before the
        # policy stops improving. Loaded ONCE here, before the first batch, from a path
        # resolved once at startup (see `latest_policy_checkpoint`): re-globbing at every
        # eval call would start picking up checkpoints THIS run wrote, making the baseline
        # a moving target that reports ~50 % forever no matter how much the policy gains.
        self._eval_reference = None
        self._init_reference_opponent(hp.get('reference_policy_path'))

        # batch-temporary; written by _collect_batch, read by _log_batch
        self._batch_eps: list = []
        self._batch_start: float = 0.0
        self._t_env: float = 0.0
        self._t_model_play: float = 0.0
        # per-batch timing (seconds) surfaced in the timing log line
        self._t_rollout: float = 0.0
        self._t_rollout_env: float = 0.0
        self._t_rollout_model: float = 0.0
        self._t_ipc: float = 0.0
        self._t_value_pass: float = 0.0
        self._t_actor_grad: float = 0.0
        self._t_critic_grad: float = 0.0
        self._t_eval: float = 0.0

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
        self._verb_entropy_coeff = (
            self._verb_entropy_coeff_init
            + frac * (self._verb_entropy_coeff_final - self._verb_entropy_coeff_init)
        )
        lr_scale = 1.0 - frac * (1.0 - self._lr_final_frac)
        for group in self._actor_optimizer.param_groups:
            group['lr'] = self._lr_actor_init * lr_scale
        for group in self._critic_optimizer.param_groups:
            group['lr'] = self._lr_critic_init * lr_scale

        # Holding + material shaping anneal: 1.0 -> SHAPING_ANNEAL_FINAL over the first
        # SHAPING_ANNEAL_HALF_FRAC of the run, then held at the floor. Used at COLLECTION
        # time; kept pure (see _compute_shaping_anneal) so the overlap path can compute the
        # next batch's value without disturbing this batch's LR/entropy.
        self._shaping_anneal = self._compute_shaping_anneal(batch_num)

    def _compute_shaping_anneal(self, batch_num: int) -> float:
        half = max(self._n_batches * SHAPING_ANNEAL_HALF_FRAC, 1.0)
        anneal_frac = min((batch_num - 1) / half, 1.0)
        return SHAPING_ANNEAL_INIT + anneal_frac * (SHAPING_ANNEAL_FINAL - SHAPING_ANNEAL_INIT)

    def train(self):
        overlap = self._n_workers > 1 and self._overlap
        if overlap:
            logger.warning(
                'overlap_collection=True: batch N+1 is collected with pre-update (1-step-stale) '
                'weights while the GPU updates batch N. Hides rollout wall behind the update, '
                'but adds off-policy staleness (interacts with KL-skip) and a second in-flight '
                'buffer in RAM. A/B learning quality, not just speed; disable if RAM-bound.'
            )
        try:
            if overlap:
                # Prime the pipeline: batch 1's rollout is launched before the loop so it is
                # ready to gather on the first iteration.
                self._submit_parallel(1)
            for batch_num in range(1, self._n_batches + 1):
                self._update_schedules(batch_num)  # LR/entropy for THIS batch's update
                self._batch_start = time.time()

                # --- collection: get this batch's transitions into the buffer ---
                if overlap:
                    # Rollout was submitted in the previous iteration and ran concurrently
                    # with the previous update; just gather it.
                    roll_timing = self._gather_and_ingest()
                elif self._n_workers > 1:
                    roll_timing = self._collect_parallel_barrier(batch_num)
                else:
                    roll_timing = self._collect_serial()

                # --- values + GAE (main process / GPU) ---
                tv = time.perf_counter()
                self._compute_values_batched()
                self._buffer.compute_gae(self._gamma, self._lam, self._device,
                                         adv_norm=self._adv_norm)
                self._ret_normalizer.update(self._buffer.returns)
                self._maybe_dump_returns(batch_num)
                self._t_value_pass = time.perf_counter() - tv

                # --- overlap: launch next batch's rollout NOW (pre-update weights) so the
                #     CPU workers run it while the GPU does this batch's update below ---
                if overlap and batch_num < self._n_batches:
                    self._submit_parallel(batch_num + 1)

                # --- update (GPU); sets self._t_actor_grad / self._t_critic_grad ---
                update_stats = self._run_ppo_update(batch_num)

                self._pool.maybe_snapshot(self._policy)
                te = time.perf_counter()
                self._maybe_eval(batch_num)
                self._t_eval = time.perf_counter() - te
                self._store_roll_timing(roll_timing)
                self._log_batch(batch_num, update_stats)
        finally:
            if self._collector is not None:
                self._collector.shutdown()

    # ------------------------------------------------------------------
    # Episode collection
    # ------------------------------------------------------------------

    # Cap on states per critic forward chunk; keeps the batched value pass off the
    # OOM edge for very long batches while staying large enough to amortise launch cost.
    _VALUE_CHUNK = 4096

    def _compute_values_batched(self):
        """Fill the buffer's per-step V(s) in a single batched pass on self._device.

        The critic is unused during rollout collection; instead every stored state is
        valued here in a few large batched forwards (GPU) rather than thousands of
        batch=1 forwards during the episode loop. Critic weights are unchanged since
        collection began (updates only happen in _run_ppo_update), so these values are
        identical to the old per-step ones up to negligible batch-vs-single numerics.
        """
        self._critic.eval()
        raw_values = []
        with torch.no_grad():
            for chunk in self._buffer.value_input_chunks(self._device, self._VALUE_CHUNK):
                val_norm = self._critic.value_batch(chunk)
                val_raw = self._ret_normalizer.denormalize(val_norm)
                raw_values.append(val_raw.detach().cpu())
        self._critic.train()
        self._buffer.set_values(torch.cat(raw_values).numpy() if raw_values else [])

    def _lazy_init_collector(self):
        if self._collector is None:
            self._collector = ParallelRolloutCollector(
                self._n_workers,
                policy_hidden_dim=self._policy_hidden_dim,
                policy_arch=self._policy_arch,
                pool_max_size=self._pool_max_size,
                seed_base=self._rollout_seed,
                lookahead_critic_time_budget=self._lookahead_critic_time_budget,
                puct_time_budget=self._puct_time_budget,
                collect_dense=self._dense_critic,
                random_eval_reply_branching=self._random_eval_reply_branching,
            )

    def _submit_parallel(self, batch_num: int):
        """Broadcast weights + hand out episodes for `batch_num` to the worker pool (async)."""
        self._lazy_init_collector()
        self._collector.submit(
            self._policy, self._pool, self._collect_episodes,
            gamma=self._gamma,
            shaping_anneal=self._compute_shaping_anneal(batch_num),
            holding_reward_rate=self._holding_reward_rate,
            max_t=self._max_t,
        )

    def _gather_and_ingest(self) -> dict:
        """Wait for the submitted batch, ingest worker transitions, return rollout timing."""
        chunks, timing = self._collector.gather()
        self._batch_eps = []
        self._buffer.ingest_chunks(chunks)
        for c in chunks:
            self._batch_eps.extend(c['episode_dicts'])
        return timing

    def _collect_parallel_barrier(self, batch_num: int) -> dict:
        """Non-overlap parallel path: submit then immediately gather (main blocks)."""
        self._submit_parallel(batch_num)
        return self._gather_and_ingest()

    def _collect_serial(self) -> dict:
        """In-process collection. Policy goes to CPU (dodges batch=1 CUDA overhead), back after."""
        self._policy.to('cpu')
        self._batch_eps = []
        self._t_env = 0.0
        self._t_model_play = 0.0
        self._policy.train()
        self._critic.train()
        t0 = time.perf_counter()
        self._buffer.clear()
        for _ in range(self._collect_episodes):
            main_pid = np.random.choice([1, 2])
            opp, opp_type = self._pool.sample(self._policy_constructor, self._policy.device)
            ep = self._collect_episode(opp, main_pid, opp_type)
            self._batch_eps.append(ep)
        self._buffer.stack()
        wall = time.perf_counter() - t0
        self._policy.to(self._device)
        return {'rollout': wall, 'env': self._t_env, 'model_play': self._t_model_play, 'ipc': 0.0}

    def _store_roll_timing(self, t: dict):
        self._t_rollout = t['rollout']
        self._t_rollout_env = t['env']
        self._t_rollout_model = t['model_play']
        self._t_ipc = t['ipc']

    def _collect_episode(self, opp, main_pid, opp_type) -> dict:
        """Run one episode via the shared rollout core, then append its steps to the buffer.

        The episode logic itself lives in rollout_core.play_episode (shared with the future
        parallel workers). Here we only feed the returned transitions into the buffer and
        fold the per-episode timing into the batch accumulators.
        """
        steps, ep = play_episode(
            self._env, self._policy, opp, main_pid, opp_type,
            gamma=self._gamma,
            shaping_anneal=self._shaping_anneal,
            holding_reward_rate=self._holding_reward_rate,
            max_t=self._max_t,
            collect_dense=self._dense_critic,
        )
        for obs, action, log_prob, reward, opp_onehot, privileged, opp_id in zip(
            steps['obs'], steps['actions'], steps['log_probs'],
            steps['rewards'], steps['opp_onehots'], steps['privileged'], steps['opp_ids'],
        ):
            self._buffer.add_step(obs, action, log_prob, reward, opp_onehot, privileged, opp_id)
        if self._dense_critic and 'aux_targets' in steps:
            self._buffer.add_aux_steps(
                steps['aux_boards'], steps['aux_globals'], steps['aux_opp_onehots'],
                steps['aux_privileged'], steps['aux_targets'],
            )
        self._buffer.end_episode()

        self._t_env += ep.pop('t_env')
        self._t_model_play += ep.pop('t_model_play')
        return ep

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _run_ppo_update(self, batch_num: int) -> dict:
        """Run actor and critic updates independently over the current buffer (timed separately)."""
        t0 = time.perf_counter()
        actor_stats = self._update_actor(batch_num)
        self._t_actor_grad = time.perf_counter() - t0
        t0 = time.perf_counter()
        critic_stats = self._update_critic(batch_num)
        self._t_critic_grad = time.perf_counter() - t0
        return {**actor_stats, **critic_stats}

    def _update_actor(self, batch_num: int) -> dict:
        kl_accum = 0.0
        actor_accum = 0.0
        entropy_accum = 0.0
        verb_entropy_accum = 0.0
        clip_frac_accum = 0.0
        last_actor_grad = 0.0
        n_actor_updates = 0
        n_actor_skipped = 0

        for epoch in range(self._ppo_epochs):
            for batch in self._buffer.iter_minibatches(self._minibatch_size, self._device):
                lp_new, ent, verb_ent = self._policy.evaluate_actions_batch(batch)
                lp_old = batch['log_probs_old']
                ratio = (lp_new - lp_old).exp()
                approx_kl = ((ratio - 1) - (lp_new - lp_old)).detach().mean().item()
                if approx_kl > self.KL_TARGET:
                    # Skip only this over-shooting minibatch, don't abort the whole
                    # update. Aborting on the first offender wasted most collected data
                    # (n_actor fell to ~4-7 updates/batch); this keeps applying the
                    # in-target minibatches across all epochs. PPO clipping still bounds
                    # each applied step, so per-minibatch KL is a safe gate.
                    n_actor_skipped += 1
                    continue

                adv = batch['advantages']
                clipped_ratio = ratio.clamp(1 - self._ppo_eps, 1 + self._ppo_eps)
                actor_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()
                loss = (
                    actor_loss
                    - self._entropy_coeff * ent.mean()
                    - self._verb_entropy_coeff * verb_ent.mean()
                )

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
                verb_entropy_accum += verb_ent.detach().mean().item()
                clip_frac_accum += ((ratio - 1.0).abs() > self._ppo_eps).float().mean().item()
                n_actor_updates += 1

        if n_actor_skipped:
            logger.debug(
                f'batch={batch_num} actor applied={n_actor_updates} '
                f'skipped_over_kl={n_actor_skipped}'
            )
        denom = max(n_actor_updates, 1)
        return {
            'avg_kl': kl_accum / denom,
            'avg_actor': actor_accum / denom,
            'avg_entropy': entropy_accum / denom,
            'avg_verb_entropy': verb_entropy_accum / denom,
            'avg_clip_frac': clip_frac_accum / denom,
            'last_actor_grad': last_actor_grad,
            'n_actor_updates': n_actor_updates,
            'n_actor_skipped': n_actor_skipped,
        }

    def _maybe_dump_returns(self, batch_num: int) -> None:
        """Append this batch's (critic input -> shaped GAE return) pairs to a shard on disk.

        Enabled by `dump_returns_dir`. Exists because the shaped return — the target that
        makes a critic able to rank siblings (docs/next_iteration.md §3.3b) — was computed
        every batch and thrown away, so the A/B against ExIt's outcome `z` had no dataset to
        run on. Shards are written with the ExIt key names (`z` holds the shaped return), so
        `eval_board_value.py fit --data '<dir>/round*.npz'` consumes them unchanged.
        """
        if not self._dump_returns_dir:
            return
        arrays = self._buffer.critic_target_arrays()
        if arrays is None:
            return
        self._dump_parts.append(arrays)
        self._dump_n += len(arrays['z'])
        if self._dump_n < self._dump_every_samples and batch_num < self._n_batches:
            return
        os.makedirs(self._dump_returns_dir, exist_ok=True)
        merged = {k: np.concatenate([p[k] for p in self._dump_parts])
                  for k in self._dump_parts[0]}
        # `round{n}` matches the ExIt shard naming that `fit`'s round-wise held-out split
        # keys on, so consecutive states from one batch cannot leak across the split.
        path = os.path.join(self._dump_returns_dir, f'round{self._dump_shard}.npz')
        np.savez_compressed(path, **merged)
        logger.info(f'dumped {len(merged["z"])} (state -> shaped return) samples to {path}')
        self._dump_parts.clear()
        self._dump_n = 0
        self._dump_shard += 1

    def _update_critic(self, batch_num: int) -> dict:
        critic_accum = 0.0
        critic_mae_accum = 0.0
        critic_mean_accum = 0.0
        critic_std_accum = 0.0
        last_critic_grad = 0.0
        n_critic_updates = 0
        board_aux_accum = 0.0
        n_board_aux = 0
        trunk_alive = None
        trunk_out_std = None
        health = None
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

                # Board-only pressure on the trunk (see _aux_board_coeff). No value clip:
                # this head has no v_old to clip against, and it is deliberately weak.
                if self._critic.board_only_head is not None and self._aux_board_coeff > 0:
                    bval_n = self._critic.board_only_value(batch['board'])
                    board_aux = 0.5 * ((bval_n - ret_n) ** 2).mean()
                    board_aux_accum += board_aux.item()
                    n_board_aux += 1
                    critic_loss = critic_loss + self._aux_board_coeff * board_aux

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

        # Auxiliary dense-target regression (dense_critic_targets only): plain MSE against
        # the opponent-node MC targets (no PPO value clip — these states have no v_old), run
        # for the same epoch count and scaled by aux_critic_coeff so it stays secondary. A
        # no-op when the flag is off (iter_aux_minibatches yields nothing).
        aux_accum = 0.0
        n_aux_updates = 0
        if self._dense_critic:
            for _ in range(self._ppo_epochs):
                for abatch in self._buffer.iter_aux_minibatches(self._minibatch_size, self._device):
                    tgt_n = self._ret_normalizer.normalize(abatch['targets'])
                    aval_n = self._critic.value_batch(abatch)
                    aux_loss = self._aux_critic_coeff * 0.5 * ((aval_n - tgt_n) ** 2).mean()

                    self._critic_optimizer.zero_grad(set_to_none=True)
                    aux_loss.backward()
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
                        logger.error(f'batch={batch_num} aux-critic NaN gradient, skipping step')
                    aux_accum += aux_loss.item()
                    n_aux_updates += 1

        # --- trunk health guard (docs/next_iteration.md §3.4) ---------------------- #
        # A conv block reading exactly 0.0 means its ReLU output is identically zero for
        # every state: the trunk is dead, `_split_pool` feeds the head hard zeros, and the
        # critic is BLIND TO THE BOARD. It cannot recover — the ReLU gradient is exactly 0
        # from then on. Logged so a run that produces a board-blind critic is visible while
        # it is still running, instead of being discovered a generation of checkpoints later.
        if self._trunk_health_every and batch_num % self._trunk_health_every == 0:
            for probe in self._buffer.iter_minibatches(self._minibatch_size, self._device):
                health = self._critic.trunk_health(probe['board'])
                break
            if health is not None:
                trunk_alive = health['alive']
                trunk_out_std = health['out_std']
                # Two distinct failures, see Critic.trunk_health: v1 dies into the ReLU
                # absorbing state (alive -> 0); either arch can collapse to a constant
                # output (out_std -> 0), which the alive fraction alone does not catch.
                if min(trunk_alive) == 0.0 or trunk_out_std < 1e-6:
                    logger.error(
                        f'batch={batch_num} BOARD-BLIND CRITIC: per-conv alive '
                        f'{["%.4f" % a for a in trunk_alive]} pooled-output std '
                        f'{trunk_out_std:.3g}. The value head is receiving no board '
                        f'information, so it cannot rank two positions and every '
                        f'measurement taken with this checkpoint is void — see '
                        f'docs/next_iteration.md §3.4.'
                    )

        denom = max(n_critic_updates, 1)
        return {
            'avg_critic': critic_accum / denom,
            'avg_critic_mae': critic_mae_accum / denom,
            'avg_critic_mean': critic_mean_accum / denom,
            'avg_critic_std': critic_std_accum / denom,
            'avg_critic_board_aux': board_aux_accum / max(n_board_aux, 1),
            **({f'critic_trunk_alive_conv{i + 1}': a for i, a in enumerate(trunk_alive)}
               if trunk_alive else {}),
            **({'critic_trunk_out_std': trunk_out_std} if trunk_out_std is not None else {}),
            'last_critic_grad': last_critic_grad,
            'n_critic_updates': n_critic_updates,
            'avg_aux_critic': aux_accum / max(n_aux_updates, 1),
            'n_aux_samples': self._buffer.n_aux(),
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _init_reference_opponent(self, path):
        """Load the frozen checkpoint the eval phase plays against.

        `path` is resolved by the caller (`latest_policy_checkpoint()` unless overridden)
        and passing None disables the reference match entirely. A checkpoint that cannot
        be rebuilt with the current code — an old arch, incompatible dims — downgrades to
        a warning rather than killing the run: the reference is a measurement, and losing
        it is not a reason to lose the training.
        """
        if path is None:
            logger.info(
                '[eval] no reference policy: eval runs against greedy/random only. '
                f'(nothing matched {POLICY_CKPT_GLOB}, or it was disabled explicitly)'
            )
            return
        try:
            meta = load_policy_checkpoint(path, map_location=self._device)
            encoder = get_encoder(meta['obs_version'])
            ref = Policy(device=self._device, hidden_dim=meta['hidden_dim'],
                         obs_encoder=encoder, arch=meta['arch']).to(self._device)
            ref.load_state_dict(meta['state_dict'])
        except Exception as e:
            logger.warning(
                f'[eval] could not load reference policy {path!r}: {e}. Eval runs against '
                f'greedy/random only; pass --reference-policy to point at a loadable one.'
            )
            return
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        env_version = self._env._obs_encoder.version
        self._eval_reference = ReferenceOpponent(
            ref,
            path=path,
            obs_version=meta['obs_version'],
            encoder=encoder if meta['obs_version'] != env_version else None,
            env=self._env,
        )
        logger.info(
            f'[eval] reference opponent = {path} '
            f'(obs v{meta["obs_version"]}, hidden_dim={meta["hidden_dim"]})'
            + ('' if meta['obs_version'] == env_version else
               f' — re-encoding the env at v{meta["obs_version"]} for its turns, the env '
               f'emits v{env_version}')
        )

    def _maybe_eval(self, batch_num: int):
        if batch_num % self._eval_every != 0:
            return

        self._policy.eval()
        self._critic.eval()
        greedy_wins = 0
        random_eval_wins = 0
        ref_wins = 0
        ref_losses = 0
        ref_draws = 0

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

            if self._eval_reference is not None:
                # Deliberately NOT fed into self._elo: the reference plays nobody but the
                # current policy, so the pair would float freely and drag elo_policy off
                # the greedy/random anchor it is comparable across runs by. The score
                # below is the whole signal, and it needs no rating system.
                outcome = self._eval_episode(self._eval_reference, main_pid)
                if outcome == 'win':
                    ref_wins += 1
                elif outcome == 'lose':
                    ref_losses += 1
                else:
                    ref_draws += 1

        self._policy.train()
        self._critic.train()

        elo_pol = self._elo.rating('policy')
        elo_grdy = self._elo.rating('greedy')
        elo_rnd = self._elo.rating('random')
        wr_random_eval = random_eval_wins / self._eval_episodes
        wr_greedy_eval = greedy_wins / self._eval_episodes

        if wr_greedy_eval >= self._wr_finetune_threshold:
            self._pool.set_weights(**self._opp_weights_finetune)
        else:
            self._pool.set_weights(**self._opp_weights_initial)

        # Score, not win rate, is the "is this policy better than the saved one" number:
        # truncations count for neither side and are common in a near-mirror match, so a
        # bare win rate sags as games get longer even when nothing regressed. 0.5 is
        # parity with the reference; above it the current policy is ahead.
        ref_stats = {}
        if self._eval_reference is not None:
            n = self._eval_episodes
            ref_stats = {
                'wr_vs_reference_eval': ref_wins / n,
                'score_vs_reference_eval': (ref_wins + 0.5 * ref_draws) / n,
                'draw_rate_vs_reference_eval': ref_draws / n,
            }

        logger.info(
            f'[eval] batch={batch_num} '
            f'wr_greedy={wr_greedy_eval:.3f} '
            f'wr_random={wr_random_eval:.3f} '
            f'elo_policy={elo_pol:.0f} elo_greedy={elo_grdy:.0f}'
            + (f' | vs {self._eval_reference.name}: '
               f'score={ref_stats["score_vs_reference_eval"]:.3f} '
               f'({ref_wins}W/{ref_losses}L/{ref_draws}D)'
               if self._eval_reference is not None else '')
        )
        if use_wandb:
            wandb.log({
                'elo_policy': elo_pol,
                'wr_vs_greedy_eval': wr_greedy_eval,
                'wr_vs_random_eval': wr_random_eval,
                **ref_stats,
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
            elif ep['opp_type'] == 'lookahead_critic':
                self._wr_vs_lookahead_critic.append(int(ep['outcome'] == 'win'))
            elif ep['opp_type'] == 'puct':
                self._wr_vs_puct.append(int(ep['outcome'] == 'win'))
            elif ep['opp_type'] == 'policy_theta':
                # One number over the whole verified family — θ is redrawn per episode, so
                # this is the win rate against a *distribution* of strong opponents.
                self._wr_vs_policy_theta.append(int(ep['outcome'] == 'win'))
            elif ep['opp_type'] == 'random_eval':
                # One number over the whole θ family, not per playstyle: each episode draws
                # its own θ, so this is the win rate against a *distribution* of opponents.
                # A drop here is coverage arriving, not a regression.
                self._wr_vs_random_eval.append(int(ep['outcome'] == 'win'))

        wr_pool = float(np.mean(self._wr_vs_pool)) if self._wr_vs_pool else 0.0
        wr_greedy = float(np.mean(self._wr_vs_greedy)) if self._wr_vs_greedy else 0.0
        wr_lookahead = (float(np.mean(self._wr_vs_lookahead_critic))
                        if self._wr_vs_lookahead_critic else 0.0)
        wr_puct = float(np.mean(self._wr_vs_puct)) if self._wr_vs_puct else 0.0
        wr_random_eval = (float(np.mean(self._wr_vs_random_eval))
                          if self._wr_vs_random_eval else 0.0)
        wr_policy_theta = (float(np.mean(self._wr_vs_policy_theta))
                           if self._wr_vs_policy_theta else 0.0)

        s = update_stats
        avg_turns = float(np.mean([ep['turns'] for ep in self._batch_eps]))
        total_invalid = sum(ep['invalid_count'] for ep in self._batch_eps)
        n_eps = len(self._batch_eps)
        # Bolster usage (docs/IDEAS.md #R8 — measured as near-zero pre-Material-PBRS).
        # Tracked per batch so a training run shows whether/when the rate moves.
        bolster_per_ep = sum(ep['bolster_count'] for ep in self._batch_eps) / n_eps
        bolster_fully_available_per_ep = (
            sum(ep['bolster_fully_available_count'] for ep in self._batch_eps) / n_eps)

        # per-episode mean of each score component (sums to score)
        r_attack = float(np.mean([ep['r_attack'] for ep in self._batch_eps]))
        r_shaping = float(np.mean([ep['r_shaping'] for ep in self._batch_eps]))
        r_holding = float(np.mean([ep['r_holding'] for ep in self._batch_eps]))
        r_material = float(np.mean([ep['r_material'] for ep in self._batch_eps]))
        r_terminal = float(np.mean([ep['r_terminal'] for ep in self._batch_eps]))
        r_other = float(np.mean([ep['r_other'] for ep in self._batch_eps]))
        r_tempo = float(np.mean([ep['r_tempo'] for ep in self._batch_eps]))
        # entropy ceiling: mean over all main-player decisions of log(n_legal).
        # ent_frac = entropy / max_entropy makes "how decisive" readable without mental math.
        tot_dec = sum(ep['n_decisions'] for ep in self._batch_eps)
        max_entropy = (
            sum(ep['sum_log_nlegal'] for ep in self._batch_eps) / tot_dec if tot_dec else 0.0
        )
        entropy_frac = s['avg_entropy'] / max_entropy if max_entropy > 0 else 0.0

        logger.info(
            f'batch={batch_num} n_actor={s["n_actor_updates"]} n_critic={s["n_critic_updates"]} '
            f'adv mean={self._buffer.raw_adv_mean:.4f} std={self._buffer.raw_adv_std:.4f} '
            f'ret mean={self._buffer.raw_ret_mean:.4f} std={self._buffer.raw_ret_std:.4f} '
            f'critic mean={s["avg_critic_mean"]:.4f} std={s["avg_critic_std"]:.4f} '
            f'clip_frac={s["avg_clip_frac"]:.3f} critic_mae={s["avg_critic_mae"]:.4f}'
            # Trunk health on the console, not only in W&B: a board-blind critic must be
            # visible in a plain log tail, which is where this failure was eventually found.
            + (''.join(f' alive{i + 1}={s[k]:.3f}'
                       for i, k in enumerate(sorted(k for k in s if k.startswith('critic_trunk_alive'))))
               )
            + (f' out_std={s["critic_trunk_out_std"]:.4f}'
               if 'critic_trunk_out_std' in s else '')
            + (f' board_aux={s["avg_critic_board_aux"]:.4f}'
               if s.get('avg_critic_board_aux') else '')
        )
        # rollout = worker critical-path wall (max over workers in parallel; the serial wall
        # otherwise). env/model_play are aggregate CPU-seconds across workers in parallel (so
        # they can exceed `rollout`). In overlap mode `rollout` ran concurrently with the
        # previous update, so total < rollout + gradients — that gap is the overlap win.
        total_wall = time.time() - self._batch_start
        overlap_note = ' (overlapped)' if (self._n_workers > 1 and self._overlap) else ''
        agg = f' aggregate/{self._n_workers}w' if self._n_workers > 1 else ''
        logger.info(
            f'batch={batch_num} timing: '
            f'rollout={self._t_rollout:.2f}s{overlap_note} '
            f'(env={self._t_rollout_env:.2f}s model_play={self._t_rollout_model:.2f}s{agg}) | '
            f'value_pass={self._t_value_pass:.2f}s '
            f'actor_gradient={self._t_actor_grad:.2f}s critic_gradient={self._t_critic_grad:.2f}s '
            f'eval={self._t_eval:.2f}s | '
            f'IPC={self._t_ipc:.2f}s | total={total_wall:.2f}s'
        )
        logger.info(
            f'batch={batch_num}/{self._n_batches} '
            f'score={np.mean(self._score_deque):.2f} '
            f'wr_pool={wr_pool:.3f} wr_greedy={wr_greedy:.3f} wr_lookahead={wr_lookahead:.3f} '
            f'actor={s["avg_actor"]:.3e} critic={s["avg_critic"]:.4f} '
            + (f'aux_critic={s["avg_aux_critic"]:.4f} n_aux={s["n_aux_samples"]} '
               if self._dense_critic else '')
            + f'kl={s["avg_kl"]:.4f} ent={s["avg_entropy"]:.3f} '
            f'ent_max={max_entropy:.3f} ent_frac={entropy_frac:.2f} '
            f'verb_ent={s["avg_verb_entropy"]:.3f} '
            f'ent_c={self._entropy_coeff:.4f} verb_ent_c={self._verb_entropy_coeff:.4f} '
            f'lr={self._actor_optimizer.param_groups[0]["lr"]:.2e} '
            f'grad_a={s["last_actor_grad"]:.3f} grad_c={s["last_critic_grad"]:.3f} '
            f'pool={len(self._pool)} turns={avg_turns:.0f} invalid={total_invalid} '
            f't={time.time() - self._batch_start:.2f}s'
        )
        # Per-opponent advantage offsets (docs/next_iteration.md §5 row 6). `adv_spread` is
        # the quantity that justifies the whole change: it is how much of the raw advantage
        # was pure opponent identity rather than action quality. It should be clearly
        # non-zero in the initial phase (random vs greedy vs pool are very different
        # opponents) and shrink in finetune as the pool narrows. A spread that stays large
        # while `adv_norm='global'` means that bias is going straight into the policy
        # gradient.
        offsets = self._buffer.adv_group_offsets
        spread = (max(offsets.values()) - min(offsets.values())) if offsets else 0.0
        # As a fraction of the raw advantage std, which is the readable form: "the opponent
        # gap was 0.4 sigma of the advantage signal". Raw units alone say nothing without
        # knowing the batch's scale, and that scale drifts over a run.
        spread_frac = spread / max(self._buffer.raw_adv_std, 1e-8)
        if offsets:
            logger.info(
                f'batch={batch_num} adv_norm={self._adv_norm} adv_spread={spread:.4f} '
                f'({spread_frac:.2f} of raw adv std) '
                + ' '.join(f'{OPP_GROUP_NAME.get(g, g)}={m:+.4f}'
                           for g, m in sorted(offsets.items()))
            )
        logger.info(
            f'batch={batch_num} score_parts (per-ep mean): '
            f'attack={r_attack:.3f} shaping={r_shaping:.3f} holding={r_holding:.3f} '
            f'material={r_material:.3f} terminal={r_terminal:.3f} other={r_other:.3f} '
            f'tempo={r_tempo:.3f} anneal={self._shaping_anneal:.3f}'
        )
        logger.info(
            f'batch={batch_num} bolster_per_ep={bolster_per_ep:.3f} '
            f'bolster_fully_available_per_ep={bolster_fully_available_per_ep:.3f}'
        )

        if use_wandb:
            wandb.log({
                'score_main': float(np.mean(self._score_deque)),
                'wr_vs_pool_train': wr_pool,
                'wr_vs_greedy_train': wr_greedy,
                'wr_vs_lookahead_critic_train': wr_lookahead,
                'wr_vs_puct_train': wr_puct,
                'wr_vs_random_eval_train': wr_random_eval,
                'wr_vs_policy_theta_train': wr_policy_theta,
                'actor_loss': s['avg_actor'],
                'critic_loss': s['avg_critic'],
                'approx_kl': s['avg_kl'],
                'entropy': s['avg_entropy'],
                'verb_entropy': s['avg_verb_entropy'],
                'verb_entropy_coeff': self._verb_entropy_coeff,
                'grad_norm_actor': s['last_actor_grad'],
                'grad_norm_critic': s['last_critic_grad'],
                'clip_frac': s['avg_clip_frac'],
                'critic_mae': s['avg_critic_mae'],
                # Board-trunk health per conv, and the board-only aux loss that keeps it
                # alive (docs/next_iteration.md §3.4). A conv at 0.0 means a board-blind
                # critic — watch these, not just the loss curves.
                **{k: v for k, v in s.items() if k.startswith('critic_trunk_alive')},
                'critic_board_aux': s['avg_critic_board_aux'],
                'advantage_std': self._buffer.raw_adv_std,
                # How much of the raw advantage was opponent identity rather than action
                # quality (docs/next_iteration.md §5 row 6). 0.0 under adv_norm='global'.
                # `_frac` is the same number in units of the raw advantage std — the one to
                # plot, since the raw scale drifts as the reward shaping anneals.
                'adv_group_spread': spread,
                'adv_group_spread_frac': spread_frac,
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
                'score_tempo': r_tempo,
                'shaping_anneal': self._shaping_anneal,
                'bolster_per_ep': bolster_per_ep,
                'bolster_fully_available_per_ep': bolster_fully_available_per_ep,
                **({'aux_critic_loss': s['avg_aux_critic'], 'n_aux_samples': s['n_aux_samples']}
                   if self._dense_critic else {}),
            })


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train the Warchest agent with PPO.')
    parser.add_argument(
        '--dense-critic-targets', action='store_true',
        help='Also regress the critic on opponent-decision nodes via an auxiliary MC-return '
             'loss (docs/IDEAS.md #12). Off by default; opt-in experiment. Save both this '
             "run's and a baseline run's critic to compare (see the gauntlet measurement plan).")
    parser.add_argument(
        '--aux-critic-coeff', type=float, default=0.5,
        help='Weight of the dense auxiliary critic loss (only used with --dense-critic-targets).')
    parser.add_argument(
        '--critic-arch', choices=CRITIC_ARCHS, default=CURRENT_CRITIC_ARCH,
        help="Critic architecture. 'critic_v5' (default) is critic_v4 plus the shared "
             'unit-type embedding on the unit planes and the per-type global vectors '
             '(docs/IDEAS.md A1); it deliberately has no FiLM, which would leak the globals '
             'into board_only_head and void the critic_v2 fix. critic_v4 is critic_v3 with '
             'the flank-average '
             'readout replaced by a gather at the 10 fixed base cells plus masked mean+max '
             'over own/opponent unit cells and the whole board (docs/IDEAS.md A2), aimed at '
             'the sibling-pair ties in docs/next_iteration.md §3.4. critic_v3 is the same net '
             'with the flank-average pool; it and critic_v2 also drop/keep the opponent '
             'one-hot (§5 row 6 — that offset now comes out of the advantage instead, see '
             '--adv-norm). The un-normalised critic_v1 trunk provably dies (§3.4). Pass an '
             'older arch only to reproduce a baseline.')
    parser.add_argument(
        '--policy-arch', choices=POLICY_ARCHS, default=CURRENT_ARCH,
        help="Policy architecture. 'policy_factored_v2' (default) contracts the unit-stack "
             'planes and the per-type global vectors against a shared unit-type table whose '
             'first 10 columns are frozen roster.py attributes (docs/IDEAS.md A1), and feeds '
             'the globals to the trunk as FiLM (gamma, beta) after each conv block instead of '
             'broadcasting them across all 49 cells into the spatial head (A3). '
             'policy_factored_v1 is the pre-A1/A3 net; pass it to reproduce a baseline.')
    parser.add_argument(
        '--aux-board-coeff', type=float, default=0.1,
        help='Weight of the board-only auxiliary value loss (critic_v2 and later). This is what '
             'gives the board trunk gradient pressure the main head does not supply; 0 '
             'disables it and lets the trunk drift toward the v1 failure mode.')
    parser.add_argument(
        '--lam', type=float, default=0.90,
        help="GAE lambda. Lowered 0.97 -> 0.90 (docs/IDEAS.md L2) now that the critic's trunk "
             'is alive: V(s_t+1) enters the advantage at gamma*(1-lam), so at 0.97 only ~3 %% '
             'of the discriminative signal was the critic and repairing it could not show up. '
             '0.90 is 3.3x that weight, with an effective horizon of ~9 main-actor decisions '
             'against a ~42-decision episode. Pass --lam 0.97 for the pre-2026-08-09 baseline '
             'arm. NOTE: lam also changes the critic\'s regression target, so critic_mae is '
             'NOT comparable across lam arms.')
    parser.add_argument(
        '--adv-norm', choices=('per_opponent', 'global'), default='per_opponent',
        help="How advantages are normalised. 'per_opponent' (default) subtracts each "
             'opponent group\'s own mean before applying one shared std, removing the '
             'per-opponent offset a state-only critic cannot predict (win rates are '
             '1.000/0.825/0.525 vs random/greedy/self). Pair it with --critic-arch '
             "critic_v3/critic_v4. 'global' is the historical single mean/std — use it as the A/B "
             'baseline, or with critic_v1/critic_v2, which carry the one-hot instead.')
    parser.add_argument(
        '--p-random-eval-finetune', type=float, default=0.0,
        help='Share of finetune-phase episodes played against the B1 randomised-coefficient '
             'family (docs/IDEAS.md B1): a SimGreedyBot whose 8 leaf-evaluator coefficients '
             'are redrawn every episode, giving a continuum of policy-independent playstyles '
             'for ~18 ms/move. Off by default. The finetune schedule is otherwise ~100 %% '
             'policy-derived (p_random = p_greedy = 0), which is the self-play collapse '
             'docs/independent_opponents.md diagnoses; this is the cheap lever against it. '
             'The share is taken out of p_pool, not added on top.')
    parser.add_argument(
        '--reference-policy', default=None,
        help='Policy checkpoint the eval phase plays the current policy against, so a run '
             'reports whether it beat the last saved generation rather than only how it '
             f'does against greedy/random. Default: the newest {POLICY_CKPT_GLOB} by mtime. '
             'Resolved ONCE at startup — checkpoints this run saves later can never become '
             'their own baseline. Pass --no-reference-eval to skip the match.')
    parser.add_argument(
        '--no-reference-eval', action='store_true',
        help='Skip the frozen-checkpoint eval opponent (saves eval_episodes games per eval).')
    parser.add_argument(
        '--p-policy-theta-finetune', type=float, default=0.0,
        help='Share of finetune-phase episodes played against the verified PolicyThetaBot '
             'family (docs/bots.md): six θ, each measured to beat lookahead_critic '
             '(0.53-0.78) at ~1/20th of its per-move cost, redrawn per episode. Off by '
             'default. Taken out of p_pool, which is the point — the finetune schedule is '
             'otherwise ~100 %% policy-derived. Needs a policy checkpoint on disk.')
    parser.add_argument(
        '--dump-returns-dir', default=None,
        help='If set, dump (critic input -> shaped GAE return) shards here as round*.npz, '
             "with the shaped return under the key 'z'. Feeds the critic-target A/B: "
             "`eval_board_value.py fit --data '<dir>/round*.npz'` reads them unchanged "
             '(docs/next_iteration.md §5 row 2a).')
    cli_args = parser.parse_args()

    use_wandb = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    run_id = time.strftime('%Y%m%d-%H%M%S')
    setup_run_logger(run_id)
    if device.type == 'cuda':
        logger.info(f'run_id={run_id} device=cuda ({torch.cuda.get_device_name(0)})')
    else:
        logger.info(f'run_id={run_id} device=cpu')

    environment = WarChestEnv(save_game_history=False, debug_mode=False)

    # Sized against the measured main-actor turn count, not the max_rounds worst case —
    # see WarChestEnv.default_holding_reward_rate and docs/IDEAS.md L8.
    holding_reward_rate = environment.default_holding_reward_rate()

    # Pin the eval baseline before a single batch runs, and log it. Deciding this here
    # (rather than inside the eval loop) is what makes the number mean "better than the
    # generation I started from" once intermediate checkpoints are being written to
    # data/ during the run — otherwise the run would progressively re-baseline onto its
    # own recent output and the score would sit at ~0.5 by construction.
    reference_policy_path = (
        None if cli_args.no_reference_eval
        else (cli_args.reference_policy or latest_policy_checkpoint())
    )

    hp = {
        'n_batches': 1500,
        'collect_episodes': 64,
        'max_t': 1000,
        'gamma': 0.99,
        # GAE-lambda raised 0.95 -> 0.97 (docs/IDEAS.md #R7): propagates the terminal
        # win/loss signal further back with less bias, densifying credit for
        # delayed-payoff actions (bolster -> durable stack -> later trade/base) without
        # touching the reward. Reward-neutral, so A/B against 0.95 in isolation.
        'lam': cli_args.lam,
        'ppo_epochs': 4,
        'ppo_eps': 0.2,
        'entropy_coeff': 0.025,
        'entropy_coeff_final': 0.003,  # linearly annealed from entropy_coeff over the run
        # Dedicated entropy bonus on the top-level verb marginal P(verb) (docs/IDEAS.md #R8).
        # The flat-joint entropy above is dominated by the many spatial actions, so it
        # barely constrains the 11-way verb head and lets rare verbs (BOLSTER, TACTIC)
        # collapse out of the repertoire in the first ~80 batches (see the bolster_per_ep
        # 10.2 -> 0.3 crash in ppo_20260713-144024). Unlike the flat coeff this anneals to a
        # meaningful floor, not near-zero, so the rare verbs stay sampled long enough for a
        # stack-punishing opponent / material PBRS to reinforce them. Set to 0.0 to disable.
        'verb_entropy_coeff': 0.02,
        'verb_entropy_coeff_final': 0.01,
        'holding_reward_rate': holding_reward_rate,
        'minibatch_size': 64,
        # Parallel rollout collection (docs/parallel_rollouts.md). 1 = in-process; 6 leaves
        # cores for the GPU update + IPC + OS on this 12-core box. Capped at collect_episodes.
        'n_workers': 6,
        # Overlap next-batch collection with the GPU update (docs/parallel_rollouts.md P11b).
        # Hides rollout wall behind the update; adds 1-step off-policy staleness + a second
        # in-flight buffer in RAM. A/B learning quality, and disable if RAM-bound.
        'overlap_collection': True,
        'rollout_seed': 0,
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'lr_final_frac': 0.1,  # LR decays linearly to lr_*_init * this. Was 0.0 (decay to
        # zero): the last ~25% of training then did no learning and the elo plateau in
        # those batches was purely the vanishing LR. 0.1 keeps a small floor so late
        # self-play refinement still moves the weights.
        # IDEAS.md #R5: widen the policy the same way the critic was widened, one clean
        # step (64 -> 128) so a policy-capacity gain stays attributable — no added conv
        # depth (the 3-layer trunk's radius-3 receptive field is deliberate, policy.py),
        # no jump straight to 192. The policy/critic board encoders are independent during
        # PPO rollout, so this does not touch the critic. Arch change: start a fresh run
        # (old OBS_VERSION=9 pool snapshots + PuctBot prior checkpoint are incompatible),
        # and A/B against the 64-wide baseline per the standing rule (IDEAS.md header).
        'hidden_dim': 128,
        # Step 5 (docs/rewards_improvements.md): strengthen the *densifier*. The critic
        # is what turns the terminal reward into a per-step signal, so widen it alone
        # (policy left at hidden_dim) to keep the capacity A/B attributable. Safe because
        # the critic's board encoder is independent of the policy's during PPO rollout.
        # Raised 128 -> 192: critic_mae held at ~0.5x the return std (explains only ~half
        # the value variance), the classic underfit signature — width is the first lever.
        'critic_hidden_dim': 192,
        'print_every': 10,
        # opponent sampling weights — initial phase (random opponent included).
        # lookahead_critic takes a fixed 15% slice; the other three keep their prior
        # relative balance (×0.85). It is a search-based, critic-guided opponent
        # (eval-scoped by design) so it runs at a small per-move budget in training —
        # see lookahead_critic_time_budget below and docs/bots.md.
        'p_greedy_initial': 0.4,
        'p_pool_initial': 0.30,
        'p_lookahead_critic_initial': 0.30,
        # opponent sampling weights — fine-tune phase (random removed from training).
        # Greedy is a small fixed anchor; the rest is self-play against the wide-skill
        # pool, with lookahead_critic holding the same 15% slice.
        'p_random_finetune': 0.00,
        'p_greedy_finetune': 0.00,
        'p_pool_finetune': 0.5,
        'p_lookahead_critic_finetune': 0.3,
        # per-move search budget for the lookahead_critic training opponent. Small on
        # purpose: at its own 0.5s eval default a 15%-sampled search opponent would
        # dominate rollout wall-clock. 0.1s keeps it a distinct, tougher opponent
        # without wrecking throughput.
        'lookahead_critic_time_budget': 0.1,
        # PuctBot (full PUCT/MCTS) training opponent — OFF by default. Unlike
        # lookahead_critic it also needs a *policy* checkpoint (data/warchest_ppo_*.pth)
        # for its priors, which it loads (frozen) from disk on first sample; a fresh
        # run with no checkpoint yet must therefore keep this at 0. Give it a slice
        # (e.g. 0.15, trimming the others) to train against it, and note each of its
        # moves runs a real 0.1s search — heavier than lookahead_critic per move.
        'p_puct_initial': 0.00,
        'p_puct_finetune': 0.2,
        'puct_time_budget': 0.1,
        # RandomEvalBot — the B1 randomised-coefficient family (docs/IDEAS.md B1), OFF by
        # default pending the coverage measurement it is supposed to buy. Unlike every
        # other entry here it is not one opponent: θ is redrawn per episode, so the slice
        # is a distribution over policy-independent playstyles (recruit-economy, bolster
        # brawler, turtle, base racer, ...) that self-play provably cannot generate. It
        # needs no checkpoint and costs SimGreedyBot time (~18 ms/move at
        # reply_branching=2), an order of magnitude under lookahead_critic's — so a slice
        # here is cheap in a way the other search opponents are not. Give it a share of
        # the *finetune* phase (where p_random = p_greedy = 0 and the schedule is ~100 %
        # policy-derived — the collapse docs/independent_opponents.md diagnoses) rather
        # than the initial one, which already has `random` for coverage.
        # PolicyThetaBot — the strong, fast branch of the B1 family (docs/bots.md).
        # Every member beats `lookahead_critic` (0.53-0.78, verified on a disjoint seed
        # block) at ~4.5 ms/move against its ~99, so unlike `p_lookahead_critic` a slice
        # here is nearly free. Needs a `data/warchest_ppo_*.pth` checkpoint for its
        # candidate prior, so a fresh run with none on disk must leave this at 0.
        # OFF by default: the strength and variety are measured, the *training* benefit
        # is not — that needs an A/B, which is the one thing this work has not run.
        'p_policy_theta_initial': 0.00,
        'p_policy_theta_finetune': cli_args.p_policy_theta_finetune,
        'p_random_eval_initial': 0.00,
        'p_random_eval_finetune': cli_args.p_random_eval_finetune,
        # 2nd-ply reply cap for that bot; the base SimGreedyBot uses 8. Only the opponent's
        # reply is capped, never the bot's own action set, so no verb is pruned from its
        # choice — see SimGreedyBot.reply_branching.
        'random_eval_reply_branching': 2,
        # win-rate vs greedy that triggers the phase switch
        'wr_greedy_finetune_threshold': 0.75,
        # self-play pool cadence: snapshot rarely so the max_size-slot pool spans a wide
        # skill range (~pool_max_size * pool_snapshot_every batches) rather than near-copies.
        'pool_max_size': 20,
        'pool_snapshot_every': 15,
        # Dense critic targets (opt-in via --dense-critic-targets; docs/IDEAS.md #12). Adds
        # an auxiliary MC-return regression on opponent-decision nodes, leaving the policy
        # path and main critic targets unchanged.
        'dense_critic_targets': cli_args.dense_critic_targets,
        'aux_critic_coeff': cli_args.aux_critic_coeff,
        # Critic trunk fix (docs/next_iteration.md §2 step 2): GroupNorm + a board-only
        # auxiliary head, so the trunk cannot enter the ReLU absorbing state and has a
        # gradient source the main head does not give it.
        'critic_arch': cli_args.critic_arch,
        # Policy trunk/input rework (docs/IDEAS.md A1 + A3): shared unit-type embedding
        # in place of one-hot type indices, and FiLM conditioning of the trunk on the
        # globals in place of the 245 broadcast constant planes.
        'policy_arch': cli_args.policy_arch,
        'adv_norm': cli_args.adv_norm,
        'aux_board_coeff': cli_args.aux_board_coeff,
        'trunk_health_every': 10,
        'dump_returns_dir': cli_args.dump_returns_dir,
        # Frozen checkpoint the eval phase plays against (None => that match is skipped).
        # Already resolved above, on purpose — see the comment there.
        'reference_policy_path': reference_policy_path,
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
                # Which checkpoint `score_vs_reference_eval` is scored against — the metric
                # is meaningless when comparing two runs that used different baselines.
                'reference_policy': hp['reference_policy_path'],
            }
        )
        logger.info(f'wandb_run={run.url}')

    def policy_constructor():
        return Policy(device=device, hidden_dim=hp['hidden_dim'], arch=hp['policy_arch'])

    warchest_policy = policy_constructor().to(device)
    warchest_critic = Critic(device=device, hidden_dim=hp['critic_hidden_dim'],
                             arch=hp['critic_arch']).to(device)
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
                critic_filename = f'warchest_critic_{timestamp}.pth'
                os.makedirs('data', exist_ok=True)
                save_policy_checkpoint(
                    warchest_policy, f'data/{filename}',
                    obs_version=environment._obs_encoder.version,
                    hidden_dim=hp['hidden_dim'],
                    arch=hp['policy_arch'],
                )
                save_critic_checkpoint(
                    warchest_critic, f'data/{critic_filename}',
                    obs_version=environment._obs_encoder.version,
                    hidden_dim=hp['critic_hidden_dim'],
                    arch=hp['critic_arch'],
                    return_mean=trainer._ret_normalizer.mean,
                    return_std=trainer._ret_normalizer.std,
                )
                logger.info(f'Model saved to data/{filename}')
                logger.info(f'Critic saved to data/{critic_filename}')
