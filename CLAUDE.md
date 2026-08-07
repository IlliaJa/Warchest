# Warchest

Turn-based hex-grid strategy game with a reinforcement learning agent (PPO + GAE, actor-critic).

## Documentation

- [Architecture overview](docs/architecture.md) — component map, data flow, design decisions
- [Game mechanics](docs/game_mechanics.md) — board, actions, win conditions
- [Reward design](docs/rewards.md) — current reward table, sparsity analysis, improvement ideas
- [Policy network](docs/policy_network.md) — CNN + MLP architecture, encoding, hyperparameters
- [Training guide](docs/training.md) — algorithm, hyperparameters, W&B metrics, cloud training
- [Environment API](docs/environment_api.md) — Gymnasium interface, observation/action spaces, Board API
- [Next iteration](docs/next_iteration.md) — **the live plan.** Thesis (§1): *Warchest value lives one to two moves ahead, and nothing in the system computes one to two moves ahead*. **Heavily amended 2026-08-07 — read the header box before citing anything.** Four retractions, all traceable to the measuring instrument rather than the game: (a) **§3.1 is RETRACTED** — the within-state metric pooled two disjoint sub-problems (pairs only globals can rank, pairs only the board can rank) and reported their average, which is what made "every method fails equally" look like a fact about the target; bucketed, `board_solo` (board only, no globals) *out-predicts every non-board feature*, R² 0.1846 vs 0.1633, and on same-verb pairs board-blind evaluators sit at or **below** chance while board-reading ones are above (§3.1a). (b) **§2 step 1's two consolations are RETRACTED** — at 16 playouts instead of 4, `HeuristicEvaluator`'s Spearman falls 0.246 → **0.149** and the "signal grows under lookahead" claim inverts; the hand-written leaf is the *worst* evaluator on clean labels. (c) The binding constraint was **label precision**, not the playout bot — never measure within-state below `--playouts 16`. (d) **New top lever, §3.3b:** a board-blind critic trained on shaped GAE returns outranks a board-reading net trained on ExIt's `z` by ~2×, and **ExIt trains on `z`**. Consequences: the **critic's training, not the search, is the best-supported target**; quiescence slipped from first to fourth; do **not** delete the critic's board trunk (dead, it ties 89–93% of positional pairs by arithmetic — §3.4 — and it is the critic behind the gauntlet, ExIt, `PuctBot`, `LookaheadCriticBot`); shared policy/critic encoder is demoted to a parameter-count optimisation. **§5 rows 2a and 3 are implemented (2026-08-07)** — `critic_v2` is now the default critic arch (GroupNorm + a board-only auxiliary head, `--aux-board-coeff`), a per-batch trunk-health guard logs `alive1..3`/`out_std`/`board_aux` and alarms on collapse, and `--dump-returns-dir` writes shaped-return targets that `eval_board_value.py fit --data` reads unchanged. `--critic-arch critic_v1` reproduces the old dying trunk for baselines; all pre-2026-08-07 checkpoints are v1 and still load, keyed on their recorded `arch`. Note a subtlety pinned by `tests/test_critic_arch.py`: GroupNorm makes the alive fraction alone a **useless** guard for v2 (a collapsed v2 trunk reads 1.0 alive while carrying no information), which is why `out_std` exists and is the condition to watch. What remains is to *run* it — one `python src/app/ppo.py --dump-returns-dir data/ppo_returns` verifies the trunk and fills the target-A/B dataset at once. All intervals in §3.1a resample **states**, not pairs. §3.1b lists what was verified in the tool plus its blind spots (tactic verb entirely, attacks 0.7% of the sample). §5 is the sequencing table. Supersedes the sequencing in `IDEAS.md` and `independent_opponents.md`
- [Ideas](docs/IDEAS.md) — the open-item list: measurement-first guiding principle, numbered items (#10–#22: exploitability/Nash, puzzle suite, curriculum, online play), the observation→probe→lever method, retired-item map (#1–#9). Plus **§ New directions (2026-08-07)** — a block written from outside the within-state investigation, grouped **B** (fast opponent pool) / **A** (architecture) / **L** (learning process), with its own order in §N.4 and three grounding measurements in §N.0. Two of those are reproducible via `src/app/probe_costs.py` and are new: **Table A**, per-decision cost (`GreedyBot` 0.83 ms — as expensive as a policy forward, because it BFSes per move × per target; `SimGreedyBot` 18 ms; `LookaheadBot`@0.1 **104 ms**; policy 0.86 ms; env step 0.35 ms), and **Table B**, per-decision observation sparsity (**62 % of board planes and 82 % of global dims are exactly zero**, 24 planes structurally so — only 4 of 16 unit types are drafted per side and *which* 24 changes every game). **Table C** is arithmetic on `logs/ppo_20260726-203902.log`: model inference is ~89 % of rollout core-time and roughly **two thirds of all rollout compute goes to the 25 % `lookahead_critic` slice**. Headline proposals: distil one small net that is simultaneously the fast pool opponent, the PPO warm start and an *independent* PUCT prior (B4); a `RaceBot` built on the claim-and-park structure the rules impose (`is_valid_claim` + move-blocking make a parked unit an absolute lock; 0 of 53 observed steals came from an empty base) (B3); a base-cell gather readout replacing `_split_pool` (A2); unit-type embeddings for the 63 %-zero input (A1); recording both sides of self-play for a free 2× (L1); and dropping λ *together with* the shipped critic fix, without which the fix cannot show up in the gauntlet (L2)
- [Independent opponents](docs/independent_opponents.md) — why self-play ExIt collapses (teacher ≡ student), and a plan for policy-independent exploiter bots to fix the bolster/tactic blind spots
- [Training history](docs/history.md) — implemented fixes and their observed effects
- [RL algorithms](docs/rl_algorithms.md) — GAE, PPO, DQN, and alternatives with Warchest-specific trade-offs
- [Metrics reference](docs/METRICS.md) — W&B metrics explained: ideal ranges, trends, warning signs
- [Web agent](docs/web_agent.md) — design for driving warchestonline.com with a trained checkpoint via Playwright (not yet implemented; `config/web_agent.sample.toml` is the sketch)
- [Bots](docs/bots.md) — non-learned/search-based bots overview; `LookaheadCriticBot` bugs found, fixes, and experiment log
- [Search under uncertainty](docs/search_under_uncertainty.md) — what is actually hidden (3-way coin partition), why single-determinization search is flawed (strategy fusion / non-locality), and the belief/IS-MCTS/CFR fix list. §8 holds the 2026-08-02 measurement that **closed** the belief track (seeing the opponent's hand is worth ~0) and, in passing, found the **v11 critics' spatial trunk is dead**

## Quick orientation

```
src/
  services/
    environment/    game engine (Gymnasium env, board, units, renderer)
      obs_encoders/   versioned observation encoders (v10.py + registry); env delegates encoding
    policy/         actor-critic neural network (Policy + Critic)
      checkpoint.py   checkpoint (de)serialization with obs-version + arch metadata
    bots/           Bot ABC, RandomBot, GreedyBot
    opponent_pool.py  weighted opponent sampler (random / greedy / pool snapshots)
    gauntlet.py     round-robin agents + Bradley-Terry/Elo ratings + transitivity
  app/
    ppo.py          PPO training entry point (PPOTrainer class)
    reinforce.py    legacy REINFORCE+GAE trainer (retained for reference, not the primary path)
    demo.py         evaluation vs random + interactive replay
    main.py         minimal random-action smoke test
    gauntlet.py     round-robin gauntlet CLI (WR matrix, Elo/BT ranking, transitivity)
    eval_bucketed.py  per-composition eval bucketing (see docs/IDEAS.md #R1)
    eval_info_value.py  measures what the opponent's hidden hand is worth to a search
                        bot (cheat vs blind, paired arms) — docs/search_under_uncertainty.md §8
    eval_privileged_ablation.py  is the Critic actually reading its privileged features,
                        and is its spatial trunk alive at all — §8.2
    eval_move_agreement.py  do the cheating and blind PuctBot teachers pick different
                        moves (the thing ExIt actually distills) — §8.3
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
