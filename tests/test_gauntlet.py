"""Round-robin gauntlet: rating math + end-to-end agent play (docs/next_steps.md Step 1)."""
import numpy as np
import torch

from src.services.gauntlet import (
    play_game, round_robin, greedy_agent, random_agent,
    PolicyAgent, _bradley_terry_elo, _intransitive_fraction,
)
from src.services.policy.policy import Policy
from src.services.environment.obs_encoders import latest_encoder


# --------------------------------------------------------------------------- #
# Rating math (pure, deterministic)
# --------------------------------------------------------------------------- #
def test_bradley_terry_orders_by_dominance():
    # 3 agents, strict pecking order A>B>C, everyone plays everyone 10 games.
    wins = np.array([
        [0, 9, 10],
        [1, 0, 8],
        [0, 2, 0],
    ], dtype=float)
    elo = _bradley_terry_elo(wins)
    assert elo[0] > elo[1] > elo[2]
    assert abs(np.mean(elo) - 1000.0) < 1e-6  # anchored to 1000


def test_intransitive_fraction_detects_cycle():
    # Perfect rock-paper-scissors: A>B>C>A → the single triple is a cycle.
    wr = np.array([
        [np.nan, 1.0, 0.0],
        [0.0, np.nan, 1.0],
        [1.0, 0.0, np.nan],
    ])
    assert _intransitive_fraction(wr) == 1.0


def test_intransitive_fraction_zero_when_transitive():
    wr = np.array([
        [np.nan, 1.0, 1.0],
        [0.0, np.nan, 1.0],
        [0.0, 0.0, np.nan],
    ])
    assert _intransitive_fraction(wr) == 0.0


# --------------------------------------------------------------------------- #
# End-to-end play
# --------------------------------------------------------------------------- #
def test_play_game_returns_valid_outcome():
    res = play_game(random_agent(), greedy_agent(), seed=0)
    assert res in (0, 1, 2)


def test_policy_agent_plays_a_full_game():
    # Random-init policy vs random bot — just exercise encode -> act -> remap -> step.
    enc = latest_encoder()
    policy = Policy(device=torch.device('cpu'), hidden_dim=32, obs_encoder=enc)
    policy.eval()
    agent = PolicyAgent('policy', policy, enc)
    res = play_game(agent, random_agent(), seed=1)
    assert res in (0, 1, 2)


def test_round_robin_matrix_and_greedy_beats_random():
    agents = [random_agent('random'), greedy_agent('greedy')]
    out = round_robin(agents, k_games=6, seed=0)
    assert out['wins'].shape == (2, 2)
    assert out['games'].shape == (2, 2)
    # Balanced colors: each pair plays exactly k_games total.
    assert out['games'][0, 1] == 6 and out['games'][1, 0] == 6
    # GreedyBot dominates RandomBot, so it should rate strictly higher.
    assert out['ratings']['greedy'] > out['ratings']['random']
    assert 0.0 <= out['intransitive_fraction'] <= 1.0
