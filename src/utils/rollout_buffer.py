import numpy as np
import torch


class RolloutBuffer:
    """Collects main-actor transitions across multiple episodes for one PPO update.

    Episode boundaries are tracked so GAE is computed correctly within each episode.
    Observations are stored as copied dicts to prevent aliasing with the live env state.
    obs['board'] is the pre-encoded [8,7,7] float32 array produced by generate_observation.
    """

    def __init__(self):
        self._obs = []
        self._actions = []
        self._log_probs_old = []
        self._rewards = []
        self._values = []          # stored as Python floats (not tensors)
        self._opp_onehots = []
        self._privileged = []
        self._episode_ends = []
        self.advantages = None
        self.returns = None
        # Pre-stacked arrays (populated by compute_gae, consumed by iter_minibatches).
        self._boards = None
        self._globals = None
        self._masks = None
        self._opp_onehots_arr = None
        self._privileged_arr = None
        self._actions_arr = None
        self._lp_old = None
        self._vals_old = None
        # Auxiliary dense-critic-target stream (opponent-decision nodes; see
        # rollout_core.play_episode collect_dense). Value-only: no action/log_prob/advantage,
        # trained by a separate MC-return regression in the critic update. Empty unless the
        # dense-targets flag is on. `_aux_parts` accumulates per-episode dicts (serial path);
        # stack()/ingest_chunks() concatenate them into the `_aux_*_arr` arrays.
        self._aux_parts = []
        self._aux_boards_arr = None
        self._aux_globals_arr = None
        self._aux_opp_arr = None
        self._aux_priv_arr = None
        self._aux_targets_arr = None

    def add_step(self, obs, action, log_prob, reward, opp_onehot, privileged):
        self._obs.append(obs)
        self._actions.append(action)
        self._log_probs_old.append(log_prob.detach().cpu())
        self._rewards.append(float(reward))
        self._opp_onehots.append(opp_onehot)
        self._privileged.append(privileged)

    def add_aux_steps(self, boards, globals_, opp_onehots, privileged, targets):
        """Append one episode's dense auxiliary samples (serial path). Arrays are the
        `steps['aux_*']` numpy blocks produced by play_episode(collect_dense=True)."""
        self._aux_parts.append({
            'boards': boards, 'globals': globals_, 'opp_onehots': opp_onehots,
            'privileged': privileged, 'targets': targets,
        })

    def _stack_aux(self, parts):
        """Concatenate accumulated aux parts into the `_aux_*_arr` arrays (or leave None)."""
        parts = [p for p in parts if p is not None and len(p['targets'])]
        if not parts:
            return
        self._aux_boards_arr = np.concatenate([p['boards'] for p in parts])
        self._aux_globals_arr = np.concatenate([p['globals'] for p in parts])
        self._aux_opp_arr = np.concatenate([p['opp_onehots'] for p in parts])
        self._aux_priv_arr = np.concatenate([p['privileged'] for p in parts])
        self._aux_targets_arr = np.concatenate([p['targets'] for p in parts]).astype(np.float32)

    def stack(self):
        """Stack per-step lists into contiguous arrays for batched forward passes.

        Called after collection, before value computation and GAE. Splitting this out
        of compute_gae lets the critic run one batched pass over _boards/_globals/etc.
        (see value_input_chunks) to fill self._values, instead of a per-step forward
        during rollout.
        """
        self._boards = np.stack([o['board'] for o in self._obs])
        self._globals = np.stack([o['global'] for o in self._obs])
        self._masks = np.stack([o['valid_action_mask'] for o in self._obs])
        self._opp_onehots_arr = np.stack(self._opp_onehots)
        self._privileged_arr = np.stack(self._privileged)
        self._actions_arr = np.array(self._actions, dtype=np.int64)
        self._lp_old = torch.stack(self._log_probs_old)    # cached once; sliced per minibatch
        self._stack_aux(self._aux_parts)

    def value_input_chunks(self, device, chunk_size):
        """Yield critic-input batch dicts over all stored states, in fixed order.

        Consumed once by the trainer to compute V(s) in a single batched pass. Requires
        stack() (single-process) or ingest_chunks() (parallel) to have populated the arrays.
        Order matches the stored transitions, so the values line up with rewards for GAE.
        """
        n = len(self._boards)
        for start in range(0, n, chunk_size):
            sl = slice(start, start + chunk_size)
            yield {
                'board':      torch.from_numpy(self._boards[sl]).to(device),
                'global':     torch.from_numpy(self._globals[sl]).to(device),
                'opp_onehot': torch.from_numpy(self._opp_onehots_arr[sl]).to(device),
                'privileged': torch.from_numpy(self._privileged_arr[sl]).to(device),
            }

    def set_values(self, values):
        """Store per-step V(s) (raw return scale) computed by the batched critic pass."""
        self._values = [float(v) for v in values]
        self._vals_old = torch.tensor(self._values, dtype=torch.float32)

    def ingest_chunks(self, chunks):
        """Populate the buffer from pre-stacked per-worker arrays (parallel rollout path).

        Replaces the add_step + stack() route: parallel workers already return numpy arrays,
        so we concatenate across workers here and shift each worker's episode_ends into the
        combined index space. After this, the buffer is in the same state stack() would leave
        it in (arrays + _rewards + _episode_ends), ready for value computation and GAE.

        Each chunk is a dict with keys: boards, globals, masks, actions, log_probs, rewards,
        opp_onehots, privileged, episode_ends. Chunks must be passed in a deterministic order
        (e.g. sorted by worker id) so runs are reproducible for a fixed seed.
        """
        self.clear()
        self._boards = np.concatenate([c['boards'] for c in chunks])
        self._globals = np.concatenate([c['globals'] for c in chunks])
        self._masks = np.concatenate([c['masks'] for c in chunks])
        self._opp_onehots_arr = np.concatenate([c['opp_onehots'] for c in chunks])
        self._privileged_arr = np.concatenate([c['privileged'] for c in chunks])
        self._actions_arr = np.concatenate([c['actions'] for c in chunks]).astype(np.int64)
        self._lp_old = torch.from_numpy(
            np.concatenate([c['log_probs'] for c in chunks]).astype(np.float32)
        )
        self._rewards = list(np.concatenate([c['rewards'] for c in chunks]).astype(np.float32))
        offset = 0
        ends = []
        for c in chunks:
            ends.extend(int(e + offset) for e in c['episode_ends'])
            offset += len(c['rewards'])
        self._episode_ends = ends
        # Dense aux stream: workers that collected any emit an 'aux' block; concatenate them.
        self._stack_aux([c.get('aux') for c in chunks])

    def end_episode(self):
        self._episode_ends.append(len(self._rewards))

    def append_terminal_reward(self, reward):
        if self._rewards:
            self._rewards[-1] += reward

    def compute_gae(self, gamma, lam, device):
        adv_chunks = []
        ret_chunks = []
        ep_start = 0

        for ep_end in self._episode_ends:
            rewards = self._rewards[ep_start:ep_end]
            values = self._values[ep_start:ep_end]   # already floats

            n = len(rewards)
            ep_adv = np.empty(n, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(n)):
                next_val = values[t + 1] if t + 1 < n else 0.0
                delta = rewards[t] + gamma * next_val - values[t]
                gae = delta + gamma * lam * gae
                ep_adv[t] = gae                          # O(1) — no list insert shift

            ep_ret = ep_adv + np.array(values, dtype=np.float32)
            adv_chunks.append(ep_adv)
            ret_chunks.append(ep_ret)
            ep_start = ep_end

        adv_t = torch.from_numpy(np.concatenate(adv_chunks))
        ret_t = torch.from_numpy(np.concatenate(ret_chunks))

        self.raw_adv_mean = adv_t.mean().item()
        self.raw_adv_std = adv_t.std().item()
        self.raw_ret_mean = ret_t.mean().item()
        self.raw_ret_std = ret_t.std().item()

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        self.advantages = adv_t.to(device)
        self.returns = ret_t.to(device)

    def iter_minibatches(self, batch_size, device):
        """Yield mini-batch dicts in random order.

        Uses pre-stacked arrays from compute_gae for O(1) per-minibatch numpy fancy-index
        instead of per-minibatch np.stack over list-indexed dicts.
        """
        N = len(self._boards)
        perm = np.random.permutation(N)

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            yield {
                'board':          torch.from_numpy(self._boards[idx]).to(device),
                'global':         torch.from_numpy(self._globals[idx]).to(device),
                'mask':           torch.from_numpy(self._masks[idx]).to(device),
                'opp_onehot':     torch.from_numpy(self._opp_onehots_arr[idx]).to(device),
                'privileged':     torch.from_numpy(self._privileged_arr[idx]).to(device),
                'actions':        torch.from_numpy(self._actions_arr[idx]).to(device),
                'log_probs_old':  self._lp_old[idx].to(device),
                'values_old':     self._vals_old[idx].to(device),
                'advantages':     self.advantages[idx],
                'returns':        self.returns[idx],
            }

    def n_aux(self):
        """Number of stored auxiliary (dense) samples."""
        return 0 if self._aux_targets_arr is None else len(self._aux_targets_arr)

    def iter_aux_minibatches(self, batch_size, device):
        """Yield critic-input minibatches for the dense auxiliary value regression.

        Keys match Critic.value_batch (board/global/opp_onehot/privileged) plus `targets`
        (raw-return-scale MC targets). No advantages/log-probs — these states are value-only.
        Yields nothing when the dense stream is empty (flag off), so the caller's loop is a
        no-op in that case.
        """
        if self._aux_targets_arr is None:
            return
        n = len(self._aux_targets_arr)
        perm = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            yield {
                'board':      torch.from_numpy(self._aux_boards_arr[idx]).to(device),
                'global':     torch.from_numpy(self._aux_globals_arr[idx]).to(device),
                'opp_onehot': torch.from_numpy(self._aux_opp_arr[idx]).to(device),
                'privileged': torch.from_numpy(self._aux_priv_arr[idx]).to(device),
                'targets':    torch.from_numpy(self._aux_targets_arr[idx]).to(device),
            }

    def iterate(self):
        """Yield (obs, action, log_prob_old, advantage, return_) in random order."""
        lp_old = self._lp_old if self._lp_old is not None else torch.stack(self._log_probs_old)
        for i in np.random.permutation(len(self._obs)):
            yield (
                self._obs[i],
                self._actions[i],
                lp_old[i],
                self.advantages[i],
                self.returns[i],
            )

    def clear(self):
        self._obs.clear()
        self._actions.clear()
        self._log_probs_old.clear()
        self._rewards.clear()
        self._values.clear()
        self._opp_onehots.clear()
        self._privileged.clear()
        self._episode_ends.clear()
        self.advantages = None
        self.returns = None
        self._boards = None
        self._globals = None
        self._masks = None
        self._opp_onehots_arr = None
        self._privileged_arr = None
        self._actions_arr = None
        self._lp_old = None
        self._vals_old = None
        self._aux_parts = []
        self._aux_boards_arr = None
        self._aux_globals_arr = None
        self._aux_opp_arr = None
        self._aux_priv_arr = None
        self._aux_targets_arr = None

    def __len__(self):
        # _boards is authoritative once populated (parallel path has no _obs dicts).
        if self._boards is not None:
            return len(self._boards)
        return len(self._obs)
