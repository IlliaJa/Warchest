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

    def add_step(self, obs, action, log_prob, reward, value, opp_onehot, privileged):
        self._obs.append(obs)
        self._actions.append(action)
        self._log_probs_old.append(log_prob.detach().cpu())
        self._rewards.append(float(reward))
        self._values.append(value.detach().cpu().item())   # float, not tensor
        self._opp_onehots.append(opp_onehot)
        self._privileged.append(privileged)

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

        # Stack all buffer arrays once; iter_minibatches fancy-indexes into these.
        self._boards = np.stack([o['board'] for o in self._obs])
        self._globals = np.stack([o['global'] for o in self._obs])
        self._masks = np.stack([o['valid_action_mask'] for o in self._obs])
        self._opp_onehots_arr = np.stack(self._opp_onehots)
        self._privileged_arr = np.stack(self._privileged)
        self._actions_arr = np.array(self._actions, dtype=np.int64)
        self._lp_old = torch.stack(self._log_probs_old)    # cached once; sliced per minibatch
        self._vals_old = torch.tensor(self._values, dtype=torch.float32)

    def iter_minibatches(self, batch_size, device):
        """Yield mini-batch dicts in random order.

        Uses pre-stacked arrays from compute_gae for O(1) per-minibatch numpy fancy-index
        instead of per-minibatch np.stack over list-indexed dicts.
        """
        N = len(self._obs)
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

    def __len__(self):
        return len(self._obs)
