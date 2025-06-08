import torch
import matplotlib.pyplot as plt
from torch.distributions import Categorical

n_samples = 1000
n_actions = 10

entropies_uniform = []
entropies_biased = []
entropies_random = []

for _ in range(n_samples):
    # Uniform-like: small noise around equal logits
    logits_uniform = torch.ones(n_actions) + 0.01 * torch.randn(n_actions)
    dist_uniform = Categorical(logits=logits_uniform)
    entropies_uniform.append(dist_uniform.entropy().item())

    # Biased: one logit high, others low
    logits_biased = torch.randn(n_actions) * 0.1
    logits_biased[0] += 3.0  # strong preference for action 0
    dist_biased = Categorical(logits=logits_biased)
    entropies_biased.append(dist_biased.entropy().item())

    # Random: general case
    logits_random = torch.randn(n_actions)
    dist_random = Categorical(logits=logits_random)
    entropies_random.append(dist_random.entropy().item())

plt.hist(entropies_uniform, bins=50, alpha=0.6, label='Uniform-like')
plt.hist(entropies_biased, bins=50, alpha=0.6, label='Biased')
plt.hist(entropies_random, bins=50, alpha=0.6, label='Random')
plt.xlabel('Entropy')
plt.ylabel('Frequency')
plt.title('Entropy Distributions for Different Categorical Policies')
plt.legend()
plt.grid(True)
plt.show()
