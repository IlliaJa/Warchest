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
        self._episode_ends = []
        self.advantages = None
        self.returns = None

    def add_step(self, obs, action, log_prob, reward, value):
        self._obs.append(obs)
        self._actions.append(action)
        self._log_probs_old.append(log_prob.detach().cpu())
        self._rewards.append(float(reward))
        self._values.append(value.detach().cpu())

    def end_episode(self):
        """Mark the end of the current episode for GAE boundary tracking."""
        self._episode_ends.append(len(self._rewards))

    def append_terminal_reward(self, reward):
        """Add a terminal reward to the last recorded step (opponent win or truncation)."""
        if self._rewards:
            self._rewards[-1] += reward

    def compute_gae(self, gamma, lam, returns_rms, advantages_rms, device):
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

        advantages_rms.update(adv_t.numpy())
        adv_t = advantages_rms.normalize(adv_t)
        returns_rms.update(ret_t.numpy())
        ret_t = returns_rms.normalize(ret_t)

        self.advantages = adv_t.to(device)
        self.returns = ret_t.to(device)

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
        self.__init__()

    def __len__(self):
        return len(self._obs)
