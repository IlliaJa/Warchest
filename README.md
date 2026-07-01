# Warchest

A two-player turn-based strategy game on a hexagonal grid, with an AI agent trained via reinforcement learning (PPO actor-critic with Generalized Advantage Estimation).

Two players manoeuvre units across a hex board, capturing bases. The first to control 6 bases wins. The RL agent learns through self-play against an opponent pool (random, greedy, and frozen policy snapshots).

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Create the virtual environment and install everything from `pyproject.toml`:

```bash
uv sync
```

This creates a `.venv/` and installs all dependencies (including a CUDA build of PyTorch). Prefix any command with `uv run` to execute it inside the environment.

Weights & Biases is used for experiment tracking. Log in once before training:

```bash
uv run wandb login
```

To run training without W&B networking (no login, no cloud sync), set `WANDB_MODE=offline`.

> **Note:** all entry points must be run as **modules** (`-m`) from the project root, e.g. `uv run python -m src.app.ppo`. Running a script by file path (`uv run python src/app/ppo.py`) fails with `ModuleNotFoundError: No module named 'src'`.

## Train a model (PPO — recommended)

```bash
uv run python -m src.app.ppo
```

The PPO actor-critic trainer (`PPOTrainer`). Runs for `n_batches` (default 600; ~100s/batch on a laptop GPU) and logs metrics to your W&B project (`warchest`). Stop early with **Ctrl+C** — you'll be prompted to save the model. The trained model is saved automatically to:

```
data/warchest_ppo_YYYYMMDD-HH:MM.pth
```

Tune `n_batches` and other hyperparameters in the `hp` dict near the bottom of `src/app/ppo.py`.

## Train a model (REINFORCE — deprecated)

```bash
uv run python -m src.app.reinforce
```

The legacy REINFORCE + GAE trainer, kept for reference. PPO is the primary training path; use this only to compare against the older algorithm.

## Replay a trained model

```bash
uv run python -m src.app.demo
```

Loads the latest `data/warchest_ppo_*.pth` checkpoint, evaluates it against 10 random opponents, prints win/draw/loss counts, then opens an interactive game replay window.

Options:

```bash
uv run python -m src.app.demo --model-path data/warchest_ppo_20260628-1010.pth --opponent greedy
```

- `--model-path` — checkpoint to load (defaults to the latest `data/warchest_ppo_*.pth`)
- `--opponent` — `random` (default) or `greedy` for the rendered game
- `--hidden-dim` — network width, must match the trained model (default 64)

Use the **Previous / Next** buttons or keyboard shortcuts `←` / `→` (or `A` / `D`) to step through the game turn by turn.

## Further reading

- [Architecture](docs/architecture.md)
- [Game mechanics](docs/game_mechanics.md)
- [Policy network](docs/policy_network.md)
- [Training details](docs/training.md)
- [Environment API](docs/environment_api.md)
- [Reward design](docs/rewards.md)
- [RL algorithms](docs/rl_algorithms.md)
- [Metrics reference](docs/METRICS.md)
- [Ideas / open issues](docs/IDEAS.md)
- [Training history](docs/history.md)
