import numpy as np
import torch


class RolloutBuffer:
    """Collects main-actor transitions across multiple episodes for one PPO update.

    Episode boundaries are tracked so GAE is computed correctly within each episode.
    Observations are stored as copied dicts to prevent aliasing with the live env state.
    """

    def __init__(self):
        self._obs = []
        self._actions = []
        self._log_probs_old = []
        self._rewards = []
        self._values = []
        self._opp_onehots = []
        self._episode_ends = []
        self.advantages = None
        self.returns = None

    def add_step(self, obs, action, log_prob, reward, value, opp_onehot):
        self._obs.append(obs)
        self._actions.append(action)
        self._log_probs_old.append(log_prob.detach().cpu())
        self._rewards.append(float(reward))
        self._values.append(value.detach().cpu())
        self._opp_onehots.append(opp_onehot)

    def end_episode(self):
        """Mark the end of the current episode for GAE boundary tracking."""
        self._episode_ends.append(len(self._rewards))

    def append_terminal_reward(self, reward):
        """Add a terminal reward to the last recorded step (opponent win or truncation)."""
        if self._rewards:
            self._rewards[-1] += reward

    def compute_gae(self, gamma, lam, device):
        adv_list = []
        ret_list = []
        ep_start = 0

        for ep_end in self._episode_ends:
            rewards = self._rewards[ep_start:ep_end]
            values = [v.item() for v in self._values[ep_start:ep_end]]

            gae = 0.0
            ep_adv = []
            for t in reversed(range(len(rewards))):
                next_val = values[t + 1] if t + 1 < len(values) else 0.0
                delta = rewards[t] + gamma * next_val - values[t]
                gae = delta + gamma * lam * gae
                ep_adv.insert(0, gae)

            ep_ret = [a + v for a, v in zip(ep_adv, values)]
            adv_list.extend(ep_adv)
            ret_list.extend(ep_ret)
            ep_start = ep_end

        adv_t = torch.tensor(adv_list, dtype=torch.float32)
        ret_t = torch.tensor(ret_list, dtype=torch.float32)

        self.raw_adv_mean = adv_t.mean().item()
        self.raw_adv_std = adv_t.std().item()
        self.raw_ret_mean = ret_t.mean().item()
        self.raw_ret_std = ret_t.std().item()

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        # returns stay in original reward scale — critic must predict V in the same scale
        # as the rewards; normalizing here would create a scale mismatch in GAE's delta

        self.advantages = adv_t.to(device)
        self.returns = ret_t.to(device)

    def get_batch(self, device):
        """Return a randomly shuffled batch dict of all buffer steps as tensors.

        Numpy arrays that still require policy-side encoding (board, exploration_map,
        active_player) are returned as numpy; everything else lands on device as a tensor.
        Call once per PPO epoch so each epoch sees a different shuffle.
        """
        N = len(self._obs)
        perm = np.random.permutation(N)
        obs = self._obs  # local alias to avoid repeated attribute lookup

        boards = np.stack([obs[i]['board'] for i in perm])
        exploration_maps = np.stack([obs[i]['exploration_map'] for i in perm])
        active_players = np.array([obs[i]['active_player'] for i in perm])

        lp_old = torch.stack(self._log_probs_old)
        return {
            'boards': boards,
            'exploration_maps': exploration_maps,
            'active_players': active_players,
            'global': torch.tensor(
                np.stack([obs[i]['global'] for i in perm]), dtype=torch.float32
            ).to(device),
            'units': torch.tensor(
                np.stack([obs[i]['units'] for i in perm]), dtype=torch.float32
            ).to(device),
            'mask': torch.tensor(
                np.stack([obs[i]['valid_action_mask'].astype(bool) for i in perm]),
                dtype=torch.bool,
            ).to(device),
            'opp_onehot': torch.tensor(
                np.stack([self._opp_onehots[i] for i in perm]), dtype=torch.float32
            ).to(device),
            'actions': torch.tensor(
                [self._actions[i] for i in perm], dtype=torch.long
            ).to(device),
            'log_probs_old': lp_old[perm].to(device),
            'advantages': self.advantages[perm],
            'returns': self.returns[perm],
        }

    def iter_minibatches(self, batch_size, device):
        """Yield mini-batch dicts of size batch_size in random order.

        Advantages and returns are pre-normalized buffer-wide in compute_gae.
        Boards/exploration_maps/active_players are returned as numpy for
        policy-side encoding in the training loop.
        """
        N = len(self._obs)
        perm = np.random.permutation(N)
        obs = self._obs
        lp_old = torch.stack(self._log_probs_old)
        vals_old = torch.stack(self._values)

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            yield {
                'boards': np.stack([obs[i]['board'] for i in idx]),
                'exploration_maps': np.stack([obs[i]['exploration_map'] for i in idx]),
                'active_players': np.array([obs[i]['active_player'] for i in idx]),
                'global': torch.tensor(
                    np.stack([obs[i]['global'] for i in idx]), dtype=torch.float32
                ).to(device),
                'units': torch.tensor(
                    np.stack([obs[i]['units'] for i in idx]), dtype=torch.float32
                ).to(device),
                'mask': torch.tensor(
                    np.stack([obs[i]['valid_action_mask'].astype(bool) for i in idx]),
                    dtype=torch.bool,
                ).to(device),
                'opp_onehot': torch.tensor(
                    np.stack([self._opp_onehots[i] for i in idx]), dtype=torch.float32
                ).to(device),
                'actions': torch.tensor(
                    [self._actions[i] for i in idx], dtype=torch.long
                ).to(device),
                'log_probs_old': lp_old[idx].to(device),
                'values_old': vals_old[idx].to(device),
                'advantages': self.advantages[idx],
                'returns': self.returns[idx],
            }

    def iterate(self):
        """Yield (obs, action, log_prob_old, advantage, return_) in random order."""
        lp_old = torch.stack(self._log_probs_old)
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
        self._episode_ends.clear()
        self.advantages = None
        self.returns = None

    def __len__(self):
        return len(self._obs)
