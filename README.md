# Warchest

A two-player turn-based strategy game on a hexagonal grid, with an AI agent trained via reinforcement learning (REINFORCE + Generalized Advantage Estimation).

Two players manoeuvre units across a hex board, capturing bases. The first to control 6 bases wins. The RL agent learns entirely through self-play and matches against random opponents — no hand-crafted heuristics.

## Setup

```bash
pip install -r requirements.txt
```

Weights & Biases is used for experiment tracking. Log in once before training:

```bash
wandb login
```

## Train a model

```bash
python reinforce.py
```

Training runs for 3 000 episodes and logs metrics to your W&B project (`warchest`). The trained model is saved automatically:

```
data/warchest_policy_YYYYMMDD-HH:MM.pth
```

## Replay a trained model

```bash
python test.py
```

This loads the latest checkpoint, evaluates it against 10 random opponents, prints win/draw/loss counts, then opens an interactive game replay window.

Use the **Previous / Next** buttons or keyboard shortcuts `←` / `→` (or `A` / `D`) to step through the game turn by turn.

## Further reading

- [Architecture](docs/architecture.md)
- [Game mechanics](docs/game_mechanics.md)
- [Policy network](docs/policy_network.md)
- [Training details](docs/training.md)
- [Environment API](docs/environment_api.md)
