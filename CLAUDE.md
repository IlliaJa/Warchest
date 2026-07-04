# Warchest

Turn-based hex-grid strategy game with a reinforcement learning agent (PPO + GAE, actor-critic).

## Documentation

- [Architecture overview](docs/architecture.md) — component map, data flow, design decisions
- [Game mechanics](docs/game_mechanics.md) — board, actions, win conditions
- [Reward design](docs/rewards.md) — current reward table, sparsity analysis, improvement ideas
- [Policy network](docs/policy_network.md) — CNN + MLP architecture, encoding, hyperparameters
- [Training guide](docs/training.md) — algorithm, hyperparameters, W&B metrics, cloud training
- [Environment API](docs/environment_api.md) — Gymnasium interface, observation/action spaces, Board API
- [Ideas](docs/IDEAS.md) — numbered open items + REINFORCE-era archive (bottom)
- [Next steps](docs/next_steps.md) — live strategic plan: measurement-first roadmap (round-robin, exploitability/Nash, online play)
- [Training history](docs/history.md) — implemented fixes and their observed effects
- [RL algorithms](docs/rl_algorithms.md) — GAE, PPO, DQN, and alternatives with Warchest-specific trade-offs
- [Metrics reference](docs/METRICS.md) — W&B metrics explained: ideal ranges, trends, warning signs
- [Web agent](docs/web_agent.md) — design for driving warchestonline.com with a trained checkpoint via Playwright (not yet implemented; `config/web_agent.sample.toml` is the sketch)

## Quick orientation

```
src/
  services/
    environment/    game engine (Gymnasium env, board, units, renderer)
    policy/         actor-critic neural network (Policy + Critic)
    bots/           Bot ABC, RandomBot, GreedyBot
    opponent_pool.py  weighted opponent sampler (random / greedy / pool snapshots)
  app/
    ppo.py          PPO training entry point (PPOTrainer class)
    reinforce.py    legacy REINFORCE+GAE trainer (retained for reference, not the primary path)
    demo.py         evaluation vs random + interactive replay
    main.py         minimal random-action smoke test
    eval_bucketed.py  per-composition eval bucketing (see docs/IDEAS.md #1)
    policy_viz.py   export policy graph to TensorBoard
    test.py         entropy distribution visualiser
  utils/
    elo.py          Elo rating tracker
    rollout_buffer.py  GAE rollout buffer for PPO
config/              web_agent.sample.toml — sample config for the (not yet implemented) web agent, docs/web_agent.md
Dockerfile          cloud training container
launch-agent.yaml   W&B Agents queue config
```

## Running the project

Scripts add the project root to `sys.path` automatically, so run them from
the project root with either of these forms:

```bash
# Train with PPO (recommended)
python src/app/ppo.py

# Legacy REINFORCE trainer
python src/app/reinforce.py

# Evaluate a saved model + interactive replay
python src/app/demo.py

# Quick random-action smoke test
python src/app/main.py
```

## Stack

Python 3.11 · PyTorch · Gymnasium 1.1 · NumPy · Matplotlib · Weights & Biases
