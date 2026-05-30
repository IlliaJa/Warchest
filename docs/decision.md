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

