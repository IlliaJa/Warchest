# Decisions

Architectural decisions for the Warchest project. Each section captures one
decision, when it was made, why, and what it supersedes or unblocks.

---

## 2026-05-28 — Hex-correct convolution for the board encoder

**Decision.** Replace the 2D 3×3 convolutions in `Policy.board_encoder` and
`Critic.board_encoder` with a custom `HexConv2d` module (defined in
`src/services/policy/policy.py`) that gathers only the 6 hex neighbours plus
center (7 cells) and projects them with a 1×1 conv. The two anti-diagonal
positions — `(-1, +1)` and `(+1, -1)` in `(Δr, Δq)` — are excluded because
they are not hex-adjacent under the axial coordinate convention defined by
`Board.offsets` (`src/services/environment/board.py:8-15`).

**Why.** The board is a hexagon stored in a 7×7 square array. A standard 3×3
grid conv covers all 6 real hex neighbours but also includes 2 "phantom"
diagonals that are not adjacent under the hex topology. The network can
learn to drown those weights out, but it spends capacity on a symmetry
mismatch that is structurally avoidable. Hex-correct conv removes the bias
and aligns the encoder's receptive field with the game's adjacency rules.

This is `C13` from `docs/improvement_ideas.md`.

**Alternative considered.** Flat MLP over the full `6·7·7 = 294` inputs.
Simpler to implement but loses translation invariance — the same neighbour
pattern at different positions would become two unrelated features. Rejected
in favour of keeping the spatial structure.

**Implementation.** `HexConv2d` uses `F.unfold(kernel_size=3, padding=1)` to
extract the 3×3 patch per spatial location, then `index_select` over the 7
hex-valid indices `(0, 1, 3, 4, 5, 7, 8)`, then a `Conv2d(in*7, out, 1)`
projection. The hex-index tensor is a non-persistent buffer so it moves with
`.to(device)` but does not enter `state_dict`. Both `Policy.board_encoder`
and `Critic.board_encoder` use it; the rest of the network is unchanged.

**Compatibility note.** Parameter names changed (`board_encoder.0.weight` →
`board_encoder.0.proj.weight`), so any pre-existing `.pth` checkpoints will
not load into the new network. Within a single training run this is not an
issue — the `OpponentPool` only contains snapshots from the current run.

**Expected effect.** Modest — call it 5–10% faster convergence at best. The
larger payoff is unlocked when unit positions move into spatial channels
(see *Future work: spatial unit channels* below), at which point hex
topology governs unit-adjacency reasoning directly.

---

## 2026-05-29 — Focus shift: expand rule set toward original Warchest

**Decision.** With `wr_vs_greedy_eval` peaking at **80%** on a ~650-batch run after the C13 hex-conv change (see `docs/experiments.md` § 2026-05-29), the current prototype rule set is sufficiently solved. The next phase shifts effort away from RL tuning on the current target and toward expanding the game's rule set to bring it closer to the original tabletop Warchest.

**Why.** The 60% WR vs greedy goal is comfortably met. Further encoder / hyperparameter tuning on a 2-unit, move-only game is optimising against a saturated target — the marginal returns on RL changes shrink, and any architectural lesson learned here may not transfer to the full game. Conversely, every rule added (attacks, multiple unit types, coin mechanics, recruit / bolster / deploy) re-introduces hard learning problems and exercises the RL stack against a moving target that is far from solved.

**Scope of the next phase.** Likely additions, in rough order of priority:

- **Attack / strike actions.** First-class adjacency-to-enemy as a valid action target. This is the trigger condition described in *Future work: spatial unit channels* below — when this lands, unit positions should move into spatial channels so `HexConv2d` can do unit-adjacency reasoning natively.
- **Multiple unit types.** Swordsman is already in the code; add archer / cavalry / etc. with distinct stats and movement rules. Per-unit-type spatial channels become useful here.
- **Coin / bag mechanics.** Recruit, bolster, deploy. Introduces hidden information and a draw-based action economy — this is also when the algorithm choice may need to be revisited (see `docs/rl_algorithms.md` — DQN was ruled out partly *because* the full game has a factored action space and hidden state; that argument now starts to bite).
- **Win conditions beyond base count.** Original Warchest has additional victory paths (e.g. eliminating opponent units).

**What this means for open improvement ideas.** Items in `docs/improvement_ideas.md` that target the current prototype's RL dynamics (C7 entropy schedule, C8 LR decay, C11 hidden_dim, C12 batch size) drop in priority — they are tuning a solved problem. Items that expand the architecture (C14 DQN, C15 MCTS) also drop until the rule set is more mature, because the right algorithm choice depends on what the full action space and information structure actually look like. Foundational fixes already applied (C1–C6, C10's resolution, C13) stay applied.

**How this interacts with the hex-conv decision above.** `HexConv2d` is the substrate for unit-adjacency reasoning. Attack actions are the first feature that actually needs that substrate. The compounding payoff predicted in the previous decision starts being collected as soon as attacks land alongside *Future work: spatial unit channels*.

---

## 2026-05-30 — Attack + deploy actions and spatial conv action head

**Decision.** Added `attack` (instant-kill adjacent enemy) and `deploy` (place a new unit on a controlled empty base, lifetime cap `MAX_DEPLOYS=4`) as new action types, and simultaneously migrated the policy to a **spatial, cell-keyed convolutional action head** — the architecture flagged in *Future work: spatial unit channels* below. Unit positions now occupy two dedicated board planes (channels 6–7 for own/opp units); the flat actor MLP and separate `unit_encoder` are replaced by a `Conv2d → [14, 7, 7]` logit map (`action_dim = 686`), with global features broadcast as extra planes. P2 rotation reduces to a single `rot90` + direction-channel flip with no per-action-type tables. See `docs/attack_deploy_plan.md` for the full design rationale and decision log, and `docs/experiments.md` § *2026-05-30* for results.

**Why now.** Attack is the first action whose validity depends on a unit being adjacent to another unit — exactly the trigger stated in *Future work* below. Pulling in the spatial head at the same time avoids a double migration later, and the chess comparison (AlphaZero: 4672-dim spatial head, Leela: 1858) confirmed that head size is not the concern; translation-equivariant location logic is. The result validated the timing: `wr_vs_greedy_train` reached **~90%** by step ~270, surpassing the 80% peak of the move-only game.

**What it supersedes.** The *Future work: spatial unit channels* section below is now implemented. The legacy `unit_encoder` MLP, flat action table, and separate slot-keyed unit observation array are all removed. `reinforce.py` and `policy_viz.py` are deprecated (incompatible with the new interfaces).

---

## 2026-07-03 — Why a reward 100+ turns away is learnable: the critic densifies, not the reward

**Decision (settled understanding, recorded to avoid re-litigating).** The
sparsity of Warchest's terminal reward (a win pays out ~150 turns after the
moves that earned it) is **not** the fundamental obstacle it appears to be, and
the answer is not "add more reward." The mechanism that turns one distant
reward into a usable per-move learning signal is the **value function (critic) +
temporal-difference bootstrapping (GAE)** — the same machinery Warchest already
runs. This entry captures the reasoning so it doesn't get re-researched; the
prompt was "how does DeepNash learn from a reward that only fires 100+ moves
later?"

**The core mechanism.** The agent does not wait for the win to assign credit.
The critic `V(s)` learns to estimate "win probability from here." On each move,
the *change* in `V` (the TD error / advantage) is a **dense, immediate**
signal: a move that raises estimated win-probability gets positive credit now,
not 150 turns later. The terminal reward only has to correct `V` at the very
end of a game; that correction then propagates backward across many self-play
games — the endgame's value becomes accurate first, then the midgame, then the
opening. Analogy: you don't replay a chess game to mate to know a queen blunder
was bad — you've *learned* "losing the queen ≈ losing," and `V` encodes exactly
that shortcut. **Game length stops mattering** because credit is handed out
per-step through `V`, not smeared equally over all 100 moves (the latter is
Monte-Carlo return, which has ruinous variance precisely on long games).

**Corollary — the critic is the real "densifier," so critic quality is a first-
class lever.** If `V` is under-capacity or poorly bootstrapped, the terminal
reward propagates badly and one is *tempted* to paper over it with hand-crafted
shaping. Strengthening the critic (`hidden_dim`, smoother/lower-variance value
targets, GAE-λ tuning) is the principled alternative to adding reward terms, and
it's why the critic-strengthening step (implemented 2026-07-03 as a critic-only
widening, `critic_hidden_dim=128` — see `docs/rewards.md`) is not cosmetic. This is the same reason
TD-Gammon (self-play, no search, TD(λ)) learned expert backgammon from a purely
terminal reward: eligibility traces + a value net did the densifying.

**How DeepNash (Stratego — imperfect info, long games) does it, concretely.**
- **A value head does the propagation.** DeepNash uses five U-Net networks — a
  shared board embedding plus heads for value, deployment, piece-selection, and
  movement. The value head estimates expected future reward; this is the
  bootstrapping engine above.
- **It also used a *small* material shaping reward**, not a purely terminal one:
  per secondary write-ups (deeplearning.ai) it received a small reward for
  capturing an opponent piece and a larger reward for winning. This *supports*
  Warchest's material-PBRS direction (`docs/rewards.md` § *Material potential-based
  shaping*): even the RL flagship propped the rare terminal signal with a small material term.
- **No search at play time.** Unlike AlphaZero, DeepNash uses no MCTS at all —
  expert play is compiled entirely into the network weights during training, so
  every bit of credit assignment happens at train time via value + policy
  gradient. (Feasibility came partly from enormous self-play volume: scale is a
  direct substitute for reward density.)
- **R-NaD solves a *different* problem — do not conflate it with sparsity.**
  Regularized Nash Dynamics is about **convergence/unexploitability**, not the
  delayed reward. Naive self-play in two-player zero-sum imperfect-information
  games *cycles* (rock-paper-scissors dynamics) and never settles. R-NaD adds a
  regularizer that pulls the current policy toward a recent snapshot of itself
  (a KL-style penalty — "predict the same action-advantages as your previous
  version"), which damps the cycling and converges to a Nash equilibrium; the
  snapshot is then updated and the process repeats. This is orthogonal to how
  the terminal reward is propagated.

**What this means for Warchest, and what it forecloses.**
- The delayed-win reward is **already** densified by the actor-critic + GAE
  stack. So the plateau is not "reward too sparse in principle."
- Therefore the reward-side work is *targeted* (fix the non-PBRS `holding_reward`
  distortion; add material PBRS on the dark coin/economy axis — see
  `docs/rewards.md`), **not** "add generic dense reward," and the
  highest-leverage non-reward lever is **critic capacity/quality**.
- Warchest is compute-limited relative to DeepNash, which is exactly why modest
  explicit shaping is more justified here than the AlphaZero/DeepNash "terminal-
  first" precedent alone would suggest — but it stays *targeted and policy-
  invariant (PBRS)*, and annealed toward the true objective.

**Cross-project sources.** DeepNash: [arXiv 2206.15378](https://arxiv.org/abs/2206.15378),
[DeepMind blog](https://deepmind.google/blog/mastering-stratego-the-classic-game-of-imperfect-information/),
[deeplearning.ai summary](https://www.deeplearning.ai/the-batch/deepnash-the-rl-system-that-plays-stratego-like-a-master/).
TD-Gammon: [Tesauro 1995](https://www.csd.uwo.ca/~xling/cs346a/extra/tdgammon.pdf).
Full comparative analysis: `docs/rewards.md` § *What is discussed*.

---

## 2026-08-16 — `IDEAS.md` A6 (two value heads) is DEFERRED; ship two Platt floats instead

**Decision.** Do **not** build A6's second value head. Instead save a 2-parameter
Platt fit (`a`, `b` such that `sigmoid(a*v + b)` is a win probability) alongside the
existing `return_mean`/`return_std` in `save_critic_checkpoint`, let the search bots
read that, and delete `LookaheadCriticBot._calibrate_value_scale`. A6 stays on the
list but behind a much higher bar than it had, for the reasons below.

**What A6 claimed.** Two findings pointed opposite ways: §3.3b (a shaped-return
target ranks siblings ~2× better than `z`) and §3.5 (the critic's raw output is a
z-score of a shaped return and is therefore "meaningless to every search bot", which
is why the moment-matching calibration hack exists). A6 proposed resolving this with
two heads on one trunk — `V_shaped` for GAE, `V_win` for search — plus a
categorical/HL-Gauss loss.

**Why it is deferred — the measurement.** Two gates were built and run.

*Gate 1, row 2b — the premise (`src/app/eval_critic_target_ab.py`, new).* §3.3b's ~2×
was measured at `hidden_dim` 192 vs 96 on different data budgets, and the document
said so. At matched arch, matched `hidden_dim=96`, matched 120 k samples and matched
seed, scored on `data/la16_labels.pt`:

| arm | target | corr | spearman | same-verb acc |
|---|---|---|---|---|
| globals (control) | `z` | 0.101 | 0.140 | 46.8 % |
| globals (control) | shaped return | 0.206 | 0.181 | 48.0 % |
| board | `z` | 0.142 | 0.157 | 52.8 % |
| board | shaped return | 0.220 | 0.210 | 56.1 % |

The gap survives in **direction** but not in **size**: the spearman ratio is ~1.3×,
not ~2×. And the board-blind `globals` control improves almost as much (1.29×) as
`board` (1.33×), so most of the pooled effect is "shaped returns are a less noisy
target for everyone", not "the board ranks better under them". The board-specific
part is real but modest — on the isolating same-verb bucket `board` gains +3.3 pp
against the control's +1.2 pp. Note also that this confirms a decision already in
production rather than opening a new one: `ppo.py::_update_critic` already regresses
on `batch['returns']`, not on `z`. What it actually re-scopes is the claim about
**ExIt**, which does train on `z`.

*Gate 2 — the A6 decision itself (`src/app/eval_value_calibration.py`, new).* The
naive form of this test is rigged: a shaped return scored against `z ∈ {0,1}` has a
terrible Brier score by construction, whatever it knows, so "a z-head beats the raw
critic" proves nothing. The decomposition that does decide it is **AUC, which no
monotone rescaling can change**. Three arms are monotone maps of the *same* critic
scalar and therefore share one AUC by construction — `as_is`, `platt`, and
`isotonic` (PAV, the ceiling on *every possible* post-hoc recalibration). Only
`zhead` — A6-lite, an MLP on the frozen critic trunk + globals + privileged, trained
on `z` — reads the trunk, so it is the only arm whose AUC can move. Held-out 40 k
samples, split by round, critic `warchest_critic_20260810-0802.pth` (`critic_v4`):

| arm | brier | logloss | ECE | AUC | what it is |
|---|---|---|---|---|---|
| `as_is` | 0.2041 | 0.6102 | 0.1179 | 0.7861 | shipped scale, read as a probability |
| `platt` | 0.1897 | 0.5588 | 0.0313 | 0.7861 | 2 floats in the checkpoint |
| `isotonic` | 0.1883 | 0.5540 | **0.0180** | 0.7859 | best possible rescaling of that scalar |
| `zhead` | 0.1865 | 0.5489 | 0.0364 | **0.7940** | A6-lite: new head on the frozen trunk |

Three readings, and the third is the one that settles it:

- **Calibration was genuinely broken, and is cheaply fixable.** ECE 0.1179 → 0.0313
  from two floats. Of the total achievable Brier gain (`as_is` → `zhead`, −0.0184),
  Platt captures 78 % and isotonic 86 %.
- **Discrimination barely moves.** `zhead` buys +0.0081 AUC (+3.5 se — detectable
  only because n = 40 k). The figure is a real ceiling and not undertraining: three
  configs were run (lr 3e-4/4 epochs, lr 1e-4/8 epochs at hidden 128 and 256) and
  AUC saturates at 0.7940 in the latter two.
- **A6 loses at its own stated goal.** Its purpose was "search gets a *calibrated*
  win probability". The `zhead` is **worse calibrated** than simply rescaling the
  existing scalar — ECE 0.0364 (0.0518 at hidden 128) against isotonic's 0.0180. It
  wins only on ranking, by a hair. So the two-float fix achieves A6's objective
  *better* than A6's own mechanism does.

*The consequence check (`eval_value_calibration.py puct`).* A bad probability only
matters if it changes a move. `PuctBot._select` sums `sign*Q` with
`c_puct*P*sqrt(ΣN)/(1+N)`, and `c_puct` is tuned on AlphaZero's assumption that Q is
a win probability in [−1, 1]. Measured: Q's p5..p95 span is 1.74 and Q-spread /
typical-U = **1.27**. Q and exploration are commensurate — the shaped-return scale
landed near [−1, 1] by accident. So the "the hack distorts search" half of A6's
argument is not supported either.

**Where AlphaZero actually sits in this.** Worth recording because it is easy to cite
backwards: vanilla AlphaZero does not solve this problem, it *never has it*. With no
proxy reward, `z` is simultaneously the GAE-free training target, the MCTS backup
value and the PUCT input — one object, one scale. A6-shaped tension is a consequence
of Warchest's own choice to use shaped rewards, and the closer precedents are MuZero
(value target is an n-step bootstrapped return, i.e. the same move §3.3b describes,
made once intermediate rewards exist) and **KataGo**, which is the real A6: shared
trunk, separate `winrate` / `score margin` / `ownership` heads, kept separate
precisely because each target is best at a different job. If A6 is ever revived it
should be built KataGo-style **together with A7** (auxiliary heads) as one
"add structure to the value trunk" change, not as a standalone arch.

**The bounded blind spot, stated because it is the one thing that could overturn
this.** Gate 2's `zhead` reads a **frozen** trunk. It therefore measures "what is
still recoverable from the existing representation", not "what a jointly-trained
two-head critic could reach", where the trunk could learn features serving both
targets. That is a real limitation. But the burden of proof has moved: A6 now has to
argue the joint training buys something, and that costs a full PPO run rather than
40 seconds.

**Also fixed along the way.** `eval_board_value.load_exit_dataset` concatenated every
training shard *before* subsampling to `max_samples`. On `data/exit` (30 shards) that
was ~5 GB and went unnoticed; on `data/ppo_returns` (252 shards, 4.34 M samples) it
tries to materialise 45+ GB and gets OOM-killed. It now counts each shard cheaply
(reading only the small `z` array), picks the global sample indices first, and loads
only the selected rows. Same subsample for a given seed; peak RSS on the default
`--max-samples 120000` fell to 3.9 GB. `eval_board_value.py fit` inherits the fix.

**Status of the two gates.** Both are reproducible:

```bash
python src/app/eval_critic_target_ab.py \
    --shaped-data 'data/ppo_returns/round*.npz' --labels data/la16_labels.pt
python src/app/eval_value_calibration.py calib \
    --critic data/warchest_critic_20260810-0802.pth
python src/app/eval_value_calibration.py puct \
    --critic data/warchest_critic_20260810-0802.pth
```

---

