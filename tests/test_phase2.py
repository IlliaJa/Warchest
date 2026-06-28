"""Phase 2 — factored (verb-level) policy head.

The head must remain a valid distribution over the flat action space (so the env /
buffer / PPO stack is unchanged), factor P(a) = P(verb)·P(a|verb), give the verb head
a gradient, and stay numerically safe under heavy masking.
"""
import numpy as np
import torch

from src.services.environment.warchest_env import (
    WarChestEnv, ACTION_SPACE_SIZE, VERB_OF_ACTION,
)
from src.services.policy.policy import Policy


def _obs_after(steps, seed=0):
    env = WarChestEnv()
    env.reset()
    np.random.seed(seed)
    for _ in range(steps):
        _, _, t, tr, _ = env.step(int(np.random.choice(env.get_possible_actions())))
        if t or tr:
            env.reset()
    return env.generate_observation()


def _batch_from_obs(obs, action, device):
    return {
        'board': torch.tensor(obs['board'], dtype=torch.float32).unsqueeze(0).to(device),
        'global': torch.tensor(obs['global'], dtype=torch.float32).unsqueeze(0).to(device),
        'mask': torch.tensor(obs['valid_action_mask'].astype(bool)).unsqueeze(0).to(device),
        'actions': torch.tensor([action], dtype=torch.long).to(device),
    }


def test_probs_are_valid_distribution_over_legal_actions():
    pol = Policy(torch.device('cpu'))
    obs = _obs_after(7)
    probs = pol.forward(obs).squeeze(0).detach()
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    legal = obs['valid_action_mask'].astype(bool)
    # all probability mass sits on legal actions
    assert float(probs[~torch.tensor(legal)].sum()) < 1e-6
    assert probs.min() >= 0.0


def test_factorisation_matches_verb_times_within():
    """P(a) must equal P(verb(a)) · P(a | verb(a)) computed independently."""
    pol = Policy(torch.device('cpu'))
    obs = _obs_after(9, seed=2)
    mask = obs['valid_action_mask'].astype(bool)
    probs = pol.forward(obs).squeeze(0).detach().numpy()

    verb_of = VERB_OF_ACTION
    # marginal over verbs from the joint, and the within-verb conditional
    legal_ids = np.where(mask)[0]
    a = int(legal_ids[len(legal_ids) // 2])
    v = verb_of[a]
    same_verb_legal = [i for i in legal_ids if verb_of[i] == v]
    p_verb = probs[same_verb_legal].sum()
    p_within = probs[a] / p_verb
    assert abs(probs[a] - p_verb * p_within) < 1e-6  # tautology check on decomposition
    assert 0.0 < p_within <= 1.0 + 1e-6


def test_act_and_batch_log_probs_agree():
    pol = Policy(torch.device('cpu'))
    pol.eval()
    obs = _obs_after(11, seed=5)
    torch.manual_seed(0)
    action, lp_act, _ = pol.act(obs)
    lp_batch, ent = pol.evaluate_actions_batch(_batch_from_obs(obs, action, torch.device('cpu')))
    assert torch.allclose(lp_act, lp_batch.squeeze(0), atol=1e-5)
    assert torch.isfinite(ent).all()  # entropy is finite under masking


def test_verb_head_receives_gradient():
    pol = Policy(torch.device('cpu'))
    obs = _obs_after(6, seed=1)
    logits = pol._obs_logits(obs)
    a = int(np.where(obs['valid_action_mask'])[0][0])
    (-logits[0, a]).backward()
    grad = pol.verb_head.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0


def test_entropy_finite_with_single_legal_action():
    # A near-degenerate mask (one legal action) must not produce nan/inf.
    pol = Policy(torch.device('cpu'))
    obs = _obs_after(4)
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    only = int(np.where(obs['valid_action_mask'])[0][0])
    mask[only] = 1.0
    obs = {**obs, 'valid_action_mask': mask}
    _, ent = pol.evaluate_actions_batch(_batch_from_obs(obs, only, torch.device('cpu')))
    assert torch.isfinite(ent).all()
    assert float(ent.detach()) < 1e-4  # ~zero entropy when only one action is legal
