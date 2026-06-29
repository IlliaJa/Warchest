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
        self._values = []
        self._opp_onehots = []
        self._privileged = []
        self._episode_ends = []
        self.advantages = None
        self.returns = None

    def add_step(self, obs, action, log_prob, reward, value, opp_onehot, privileged):
        self._obs.append(obs)
        self._actions.append(action)
        self._log_probs_old.append(log_prob.detach().cpu())
        self._rewards.append(float(reward))
        self._values.append(value.detach().cpu())
        self._opp_onehots.append(opp_onehot)
        self._privileged.append(privileged)

    def end_episode(self):
        self._episode_ends.append(len(self._rewards))

    def append_terminal_reward(self, reward):
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

        self.advantages = adv_t.to(device)
        self.returns = ret_t.to(device)

    def iter_minibatches(self, batch_size, device):
        """Yield mini-batch dicts in random order.

        obs['board'] is already a [8,7,7] float32 array; it is stacked directly
        into the batch tensor without any further encoding.
        """
        N = len(self._obs)
        perm = np.random.permutation(N)
        obs = self._obs
        lp_old = torch.stack(self._log_probs_old)
        vals_old = torch.stack(self._values)

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            yield {
                'board': torch.from_numpy(
                    np.stack([obs[i]['board'] for i in idx])
                ).to(device),
                'global': torch.from_numpy(
                    np.stack([obs[i]['global'] for i in idx])
                ).to(device),
                'mask': torch.from_numpy(
                    np.stack([obs[i]['valid_action_mask'] for i in idx])
                ).to(device),
                'opp_onehot': torch.from_numpy(
                    np.stack([self._opp_onehots[i] for i in idx])
                ).to(device),
                'privileged': torch.from_numpy(
                    np.stack([self._privileged[i] for i in idx])
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
        self._privileged.clear()
        self._episode_ends.clear()
        self.advantages = None
        self.returns = None

    def __len__(self):
        return len(self._obs)
