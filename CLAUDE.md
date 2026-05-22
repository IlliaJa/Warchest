# Warchest

Turn-based hex-grid strategy game with a reinforcement learning agent (REINFORCE + GAE, actor-critic).

## Documentation

- [Architecture overview](docs/architecture.md) — component map, data flow, design decisions
- [Game mechanics](docs/game_mechanics.md) — board, actions, rewards, win conditions
- [Policy network](docs/policy_network.md) — CNN + MLP architecture, encoding, hyperparameters
- [Training guide](docs/training.md) — algorithm, hyperparameters, W&B metrics, cloud training
- [Environment API](docs/environment_api.md) — Gymnasium interface, observation/action spaces, Board API

## Quick orientation

```
environment/    game engine (Gymnasium env, board, units, renderer)
policy.py       actor-critic neural network
reinforce.py    training entry point
test.py         evaluation + interactive replay
demo.py / main.py  minimal smoke tests
Dockerfile      cloud training container
launch-agent.yaml  W&B Agents queue config
```

## Running the project

```bash
# Train
python reinforce.py

# Evaluate against random opponents + replay
python test.py

# Quick random-action smoke test
python demo.py
```

## Stack

Python 3.11 · PyTorch · Gymnasium 1.1 · NumPy · Matplotlib · Weights & Biases
