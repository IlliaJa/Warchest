import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from ..environment.warchest_env import (
    N_VERBS, BOARD_DIM, ACTION_SPACE_SIZE, SPATIAL_SIZE, FACEDOWN_SIZE,
    N_FACTORED_VERBS, VERB_OF_ACTION,
)
from ..environment.board import Board
from ..environment.obs_encoders import latest_encoder


class HexConv2d(nn.Module):
    """Topology-correct convolution for hex grids stored in a 2D axial array.

    3×3 Conv2d where the two non-hex-adjacent kernel positions ((-1,+1) and
    (+1,-1) in (Δr,Δq)) are kept permanently zero. Zeros are enforced at init;
    a backward hook zeroes their gradient so the optimizer never moves them.
    Functionally identical to the prior unfold+index_select approach but uses
    the optimised BLAS/MKLDNN Conv2d path (~2× faster on CPU at batch size 1).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        with torch.no_grad():
            self.conv.weight[:, :, 0, 2] = 0  # (-1, +1): not hex-adjacent
            self.conv.weight[:, :, 2, 0] = 0  # (+1, -1): not hex-adjacent
        self.conv.weight.register_hook(self._zero_non_hex_grad)

    @staticmethod
    def _zero_non_hex_grad(grad: torch.Tensor) -> torch.Tensor:
        grad[:, :, 0, 2] = 0
        grad[:, :, 2, 0] = 0
        return grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _split_pool(feat: torch.Tensor) -> torch.Tensor:
    """[B,C,7,7] -> [B,2C] location-preserving pool, split along the flank (q) axis.

    A single global mean pool is location-blind — it can tell a threat exists
    somewhere but not which flank (see docs/IDEAS.md "the agent can't see the
    board as one position"). Column 3 (the board's true center) is included in
    both halves so the contested middle stays visible to each side.
    """
    left = feat[..., 0:4].mean(dim=(-2, -1))
    right = feat[..., 3:7].mean(dim=(-2, -1))
    return torch.cat([left, right], dim=-1)


# The 10 fixed base-cell (row, col) indices win/loss is actually decided on (docs/IDEAS.md
# A2). `Board.default_bases` is the single source of truth; the layout never moves and is
# symmetric under the P2 ego-rotation (each cell's 180°-rotated counterpart is also a base
# cell), so one constant index set is correct for both players' encoded boards.
_BASE_CELLS = tuple(sorted({cell for cells in Board.default_bases.values() for cell in cells}))


def _masked_mean_max(feat, mask):
    """feat: [B,C,7,7], mask: [B,7,7] bool -> (mean [B,C], max [B,C]) over True cells.

    A sample with no True cells (e.g. before any unit is deployed) falls back to zero for
    both rather than propagating -inf/NaN.
    """
    m = mask.unsqueeze(1).to(feat.dtype)  # [B,1,7,7]
    count = m.sum(dim=(-2, -1)).clamp_min(1.0)  # [B,1]
    mean = (feat * m).sum(dim=(-2, -1)) / count  # [B,C]
    masked = feat.masked_fill(~mask.unsqueeze(1), torch.finfo(feat.dtype).min)
    max_, _ = masked.flatten(2).max(dim=-1)  # [B,C]
    empty = mask.flatten(1).sum(dim=-1) == 0  # [B]
    if empty.any():
        mean = mean.masked_fill(empty.unsqueeze(-1), 0.0)
        max_ = max_.masked_fill(empty.unsqueeze(-1), 0.0)
    return mean, max_


def _global_mean_max(feat):
    """feat: [B,C,7,7] -> (mean [B,C], max [B,C]) over the whole board."""
    mean = feat.mean(dim=(-2, -1))
    max_, _ = feat.flatten(2).max(dim=-1)
    return mean, max_


class Policy(nn.Module):
    """Actor with a factored (verb-level) action head — Phase 2.

    The action space is grouped into top-level verbs (move / attack / control /
    deploy / bolster / claim_initiative / pass / recruit). The policy factors:

        P(a) = P(verb = v(a) | s) · P(a | v(a), s)

    where P(verb) comes from a dedicated verb head, and P(a | verb) is a softmax
    over that verb's legal flat actions, using the existing spatial-conv +
    face-down logits. Both masks are conditional (verbs with no legal action are
    masked out; within-verb is masked to the verb's legal actions). The result is
    still a single distribution over ACTION_SPACE_SIZE flat ids, so sampling,
    log-prob, entropy and the whole PPO/buffer/env stack are unchanged — only the
    way the per-action log-probs are computed differs.

    The verb head therefore receives a gradient on every step (it shapes P(verb)
    for all actions), which is the structure that pays off as unit types grow.
    All sub-field agency (direction, deploy-type, pay-coin, take-type) is retained
    inside the within-verb softmax, so there is no loss of expressiveness vs the
    flat head.

    Board trunk: HexConv2d stack on BOARD_CHANNELS input planes → [B, Cf, 7, 7],
    3 layers deep (receptive-field radius 3 — exactly covers the Lancer's
    distance-3 charge; see docs/IDEAS.md "the agent can't see the board as one
    position"). Input planes (BOARD_CHANNELS, ego-centric):
        0: invalid  1: empty  2: uncontrolled base
        3: own base  4: opp base  5: exploration
        6-21: own unit-type stack planes (index = unit id - 1)
        22-37: opponent unit-type stack planes (index = unit id - 1)
        38-40: own threat (melee, ranged, charge) — graded hit-count this side
               could land on each cell this turn
        41-43: enemy threat (melee, ranged, charge)
        44: row_coord  45: col_coord — static ego-centric position planes
    """

    def __init__(self, device, hidden_dim=64, *, obs_encoder=None):
        super().__init__()

        # Observation dims come from the (versioned) encoder this net is paired
        # with, not from a hardcoded env constant — so a policy built for one obs
        # version sizes its input layers to that version.
        enc = obs_encoder or latest_encoder()
        self.board_channels = enc.board_channels
        self.global_dim = enc.global_dim

        self.board_encoder = nn.Sequential(
            HexConv2d(in_channels=self.board_channels, out_channels=32),
            nn.ReLU(),
            HexConv2d(in_channels=32, out_channels=hidden_dim),
            nn.ReLU(),
            HexConv2d(in_channels=hidden_dim, out_channels=hidden_dim),
            nn.ReLU(),
        )

        # Within-verb logits: 1×1 conv → N_VERBS spatial planes (flattened) plus a
        # linear head for the face-down actions. The spatial head sees the full
        # per-cell feature map directly, so it was never location-blind; only the
        # pooled path (facedown_head/verb_head) needed the split pool below.
        self.policy_head = nn.Conv2d(hidden_dim + self.global_dim, N_VERBS, kernel_size=1)
        self.facedown_head = nn.Linear(2 * hidden_dim + self.global_dim, FACEDOWN_SIZE)
        # Top-level verb head.
        self.verb_head = nn.Linear(2 * hidden_dim + self.global_dim, N_FACTORED_VERBS)

        # Static flat-id -> verb-index map, and a per-verb membership matrix.
        verb_index = torch.tensor(VERB_OF_ACTION, dtype=torch.long)
        group_mat = torch.zeros(N_FACTORED_VERBS, ACTION_SPACE_SIZE, dtype=torch.bool)
        group_mat[verb_index, torch.arange(ACTION_SPACE_SIZE)] = True
        self.register_buffer('_verb_index', verb_index, persistent=False)
        self.register_buffer('_group_mat', group_mat, persistent=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def _encode_board(self, board: torch.Tensor) -> torch.Tensor:
        """board: [B,C,7,7] → feat: [B, hidden_dim, 7, 7]"""
        return self.board_encoder(board)

    def _logits_from_feat(self, feat: torch.Tensor, global_feats: torch.Tensor):
        """feat: [B,hidden_dim,7,7], global_feats: [B,G] → (flat_logits [B,A], verb_logits [B,V])"""
        B = feat.shape[0]
        g = global_feats.view(B, self.global_dim, 1, 1).expand(B, self.global_dim, BOARD_DIM, BOARD_DIM)
        spatial = self.policy_head(torch.cat([feat, g], dim=1)).flatten(1)  # [B, SPATIAL_SIZE]
        pooled = _split_pool(feat)  # [B, 2*hidden_dim]
        pg = torch.cat([pooled, global_feats], dim=-1)
        facedown = self.facedown_head(pg)  # [B, FACEDOWN_SIZE]
        flat_logits = torch.cat([spatial, facedown], dim=1)  # [B, ACTION_SPACE_SIZE]
        verb_logits = self.verb_head(pg)  # [B, N_FACTORED_VERBS]
        return flat_logits, verb_logits

    def _features(self, board: torch.Tensor, global_feats: torch.Tensor):
        """board: [B,C,7,7], global_feats: [B,G] → (flat_logits [B,A], verb_logits [B,V])"""
        return self._logits_from_feat(self._encode_board(board), global_feats)

    def _joint_log_probs(self, flat_logits, verb_logits, mask):
        """Factored joint log-probs over the flat action space. [B, ACTION_SPACE_SIZE].

        log P(a) = log P(verb=v(a)) + log P(a | verb=v(a)); illegal ids -> NEG.
        """
        NEG = -1e9
        B = flat_logits.shape[0]
        ml = flat_logits.masked_fill(~mask, NEG)  # [B, A]
        g = self._verb_index.unsqueeze(0).expand(B, -1)  # [B, A]

        # Per-verb max via scatter reduce (replaces Python loop over N_FACTORED_VERBS).
        gmax = torch.full((B, N_FACTORED_VERBS), NEG, dtype=ml.dtype, device=ml.device)
        gmax.scatter_reduce_(1, g, ml, reduce='amax', include_self=True)

        shifted = ml - gmax.gather(1, g)  # [B, A]; ≤0 for legal members
        exp_shifted = shifted.exp() * mask.float()  # zero out illegal ids

        # Per-verb exp sum via scatter add (replaces second Python loop).
        gsum = torch.zeros(B, N_FACTORED_VERBS, dtype=ml.dtype, device=ml.device)
        gsum.scatter_add_(1, g, exp_shifted)

        within_logp = (shifted - gsum.clamp_min(1e-12).log().gather(1, g)).masked_fill(~mask, NEG)

        verb_mask = gsum > 0  # verbs with at least one legal action
        verb_logp = F.log_softmax(verb_logits.masked_fill(~verb_mask, NEG), dim=1)  # [B, V]

        joint = verb_logp.gather(1, g) + within_logp  # [B, A]
        return joint.masked_fill(~mask, NEG)

    def _obs_logits(self, obs):
        board = torch.from_numpy(obs['board']).unsqueeze(0).to(self.device)
        global_feats = torch.from_numpy(obs['global']).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(obs['valid_action_mask']).bool().unsqueeze(0).to(self.device)
        flat_logits, verb_logits = self._features(board, global_feats)
        return self._joint_log_probs(flat_logits, verb_logits, mask)  # [1, A]

    def forward(self, obs):
        """Return action probabilities for a single observation dict."""
        return F.softmax(self._obs_logits(obs), dim=-1)

    def act(self, obs):
        with torch.inference_mode():
            dist = Categorical(logits=self._obs_logits(obs))
            action = dist.sample()
            return action.item(), dist.log_prob(action).squeeze(0), dist.entropy().squeeze(0)

    def act_with_encoded(self, obs):
        """Like act(), but returns encoded board features [1,H,7,7] for critic reuse."""
        with torch.inference_mode():
            board = torch.from_numpy(obs['board']).unsqueeze(0)
            global_feats = torch.from_numpy(obs['global']).unsqueeze(0)
            mask = torch.from_numpy(obs['valid_action_mask']).bool().unsqueeze(0)
            feat = self._encode_board(board)  # [1, hidden_dim, 7, 7]
            flat_logits, verb_logits = self._logits_from_feat(feat, global_feats)
            joint = self._joint_log_probs(flat_logits, verb_logits, mask)
            dist = Categorical(logits=joint)
            action = dist.sample()
            return action.item(), dist.log_prob(action).squeeze(0), dist.entropy().squeeze(0), feat, global_feats

    def evaluate_actions(self, obs, action):
        dist = Categorical(logits=self._obs_logits(obs))
        return dist.log_prob(action), dist.entropy()

    def _verb_marginal_entropy(self, verb_logits, mask):
        """Entropy of the top-level verb marginal P(verb), over legal verbs only. [B].

        Distinct from the flat-joint entropy: the joint distribution is dominated by
        the many spatial (cell, direction) actions, so a bonus on its entropy barely
        constrains the 11-way verb marginal and lets low-cardinality verbs (BOLSTER,
        TACTIC) collapse out of the repertoire before reward can reinforce them. A
        dedicated bonus on this quantity keeps rare verbs alive (docs/IDEAS.md #R8).
        """
        NEG = -1e9
        B = mask.shape[0]
        g = self._verb_index.unsqueeze(0).expand(B, -1)  # [B, A]
        legal_per_verb = torch.zeros(B, N_FACTORED_VERBS, dtype=verb_logits.dtype, device=mask.device)
        legal_per_verb.scatter_add_(1, g, mask.to(verb_logits.dtype))
        verb_mask = legal_per_verb > 0  # verbs with at least one legal action
        verb_logp = F.log_softmax(verb_logits.masked_fill(~verb_mask, NEG), dim=1)  # [B, V]
        return Categorical(logits=verb_logp).entropy()

    def evaluate_actions_batch(self, batch):
        """Single forward pass over a full buffer batch.

        batch keys (all on device): board (N,C,7,7), global (N,GLOBAL_DIM),
        mask (N,A) bool, actions (N,) long. Returns log_probs (N,), flat-joint
        entropies (N,), verb-marginal entropies (N,).
        """
        flat_logits, verb_logits = self._features(batch['board'], batch['global'])
        joint = self._joint_log_probs(flat_logits, verb_logits, batch['mask'])
        dist = Categorical(logits=joint)
        verb_ent = self._verb_marginal_entropy(verb_logits, batch['mask'])
        return dist.log_prob(batch['actions']), dist.entropy(), verb_ent

    def joint_log_probs_batch(self, batch):
        """Full [N, A] masked joint log-prob matrix for a batch (board, global, mask).

        Same forward as `evaluate_actions_batch` but returns the whole per-action
        distribution instead of gathering one action's log-prob — for a distillation
        cross-entropy against a target distribution over the action space:
        `loss = -(target * joint).sum(dim=1).mean()`. Illegal ids sit at NEG (-1e9)
        in `joint`, but are annihilated by `target[a] = 0` (a visit-count target only
        puts mass on legal moves), so they never contribute to the loss or its grad.
        Kept differentiable (no inference_mode) for training use.
        """
        flat_logits, verb_logits = self._features(batch['board'], batch['global'])
        return self._joint_log_probs(flat_logits, verb_logits, batch['mask'])  # [N, A]


CRITIC_ARCH_V1 = 'critic_v1'
CRITIC_ARCH_V2 = 'critic_v2'
CRITIC_ARCH_V3 = 'critic_v3'
CRITIC_ARCH_V4 = 'critic_v4'
CRITIC_GROUPS = 8  # GroupNorm groups; must divide both 32 and hidden_dim
_KNOWN_CRITIC_ARCHS = (CRITIC_ARCH_V1, CRITIC_ARCH_V2, CRITIC_ARCH_V3, CRITIC_ARCH_V4)


class Critic(nn.Module):
    """Separate value network with its own spatial encoder.

    Privileged at training time, in two ways the policy never sees:
      - a 3-d opponent one-hot (random / greedy / pool);
      - a PRIV_DIM vector of the opponent's *true* hidden coin split
        (opp hand / bag / face-down discard per coin). Discarded at inference.

    Board trunk: HexConv2d stack → [B, Cf, 7, 7]. Split flank pool (see
    `_split_pool`) → [B, 2*Cf]. Concatenate with global features, opp_onehot and
    the privileged vector → scalar value.

    Four architectures, selected by `arch`, because checkpoints of every generation
    exist on disk and the gauntlet has to keep reconstructing them (docs/history.md):

    `critic_v1` — the original `HexConv2d → ReLU` ×3 with no normalisation. **This
    trunk dies.** Measured on the shipped v11 checkpoint: every pre-activation of
    the final ReLU is <= 0 for all 1085 probe states and all 192 channels (max
    -0.003), so the trunk outputs exactly zero, `_split_pool` feeds `head[0]` a
    block of hard zeros, and the critic is blind to the board. Once there, the ReLU
    gradient is exactly 0 and Adam's moments stay 0 — it never recovers. Cost:
    89-93 % of sibling pairs that differ only in position get identical values and
    cannot be ranked at all (docs/next_iteration.md §3.4).

    `critic_v2` — the fix, two parts:
      * **GroupNorm before every ReLU.** Re-centres each sample's channels, so a
        whole channel cannot sit permanently below zero. Measured on 8 fresh seeds:
        removes the absorbing state and raises |out|max 0.083 -> 5.03. NOT
        BatchNorm — `ppo.py` toggles `critic.eval()/.train()` around the rollout, so
        batch statistics would desynchronise the values the critic is fit to from
        the ones used in the rollout.
      * **A board-only auxiliary value head** (`board_only_head`). GroupNorm removes
        the trap but gives the trunk no *reason* to learn: the main head draws 76 %
        of its sensitivity from globals and 14 % from the board, so the board
        pathway sees almost no gradient. This head reads the pooled board block
        alone, so its loss cannot be satisfied from globals, and the trunk must
        carry signal. `ppo.py` adds it at `aux_board_coeff`. The evidence it is the
        right mechanism: the `board_solo` probe — same stack, same data, but with no
        globals to fall back on — trains to a 43.2 % alive trunk and pooled
        R^2 0.1846, above a globals-only control's 0.1633.

    `critic_v3` — v2's trunk and auxiliary head, **minus the opponent one-hot**
    (docs/next_iteration.md §5 row 6). The 3-wide `opp_onehot` block told the critic
    which opponent family it was facing, and it moved the output more than the
    position did: `V(start)` spanned 0.747 across the three slots against a 0.44 std
    of `V` across *positions* (§3.5). That was not a bug in the critic — win rates
    really are 1.000 / 0.825 / 0.525 vs random / greedy / self — but it has three
    costs. It is dead weight during finetune (`p_random = p_greedy = 0`, and the
    search bots are mapped onto the `pool` slot anyway), it makes the raw output
    meaningless to any consumer outside training (every search bot has to pick a slot
    arbitrarily, on top of return normalisation), and it lets the head satisfy a large
    part of the loss without reading the state at all. The opponent-identity offset it
    was carrying is removed from the *advantage* instead, where it actually mattered:
    `RolloutBuffer.compute_gae(adv_norm='per_opponent')` centres advantages within
    each opponent group, so the per-opponent mean cancels out of the policy gradient
    without the critic needing to know who it is playing. The two changes are a
    matched pair — dropping the input without the grouped centring would put the
    opponent bias straight back into the advantage.

    Consequence for consumers: `opp_onehot` is **ignored** by v3 (and may be `None`),
    so every call site keeps its signature and v1/v2 checkpoints keep working.

    `critic_v4` — v3's trunk and auxiliary head, with `_split_pool`'s flank-average
    readout replaced by a task-relevant gather (docs/IDEAS.md A2). Motivation: against
    a win condition that is a function of 10 fixed base cells with at most a handful of
    units per side, a two-number-per-channel average is close to throwing the board
    away, and it is exactly the readout §3.4 measured tying 89-93 % of sibling pairs
    that differ only in position. The new readout concatenates the trunk features at
    the 10 static base-cell indices (`Board.default_bases` — free, a constant index
    set, the layout never moves), masked mean+max over own- and opponent-occupied unit
    cells (mean alone would erase "my Berserker is hanging on the far flank"; occupancy
    comes from the *input* board's unit-stack planes via `ObsEncoder.own/opp_unit_
    channels`, since the trunk output no longer carries a clean per-cell occupancy
    signal), and a whole-board mean+max — `[B, 16*hidden_dim]` in place of
    `[B, 2*hidden_dim]`. Trunk and auxiliary head are otherwise identical to v3
    (GroupNorm, no opponent one-hot). Needs the raw board tensor, not just trunk
    features, so `value_from_features` (which only ever receives pre-encoded features)
    does not support it.
    """

    OPP_DIM = 3

    def __init__(self, device, hidden_dim=64, *, obs_encoder=None, arch=CRITIC_ARCH_V4):
        super().__init__()
        if arch not in _KNOWN_CRITIC_ARCHS:
            raise ValueError(f'unknown critic arch {arch!r}')
        # Obs dims (incl. the privileged vector) come from the paired encoder.
        enc = obs_encoder or latest_encoder()
        self.board_channels = enc.board_channels
        self.global_dim = enc.global_dim
        self.priv_dim = enc.priv_dim
        self.arch = arch

        if arch == CRITIC_ARCH_V1:
            # Module layout preserved EXACTLY so existing state_dicts load unchanged.
            self.board_encoder = nn.Sequential(
                HexConv2d(self.board_channels, 32),
                nn.ReLU(),
                HexConv2d(32, hidden_dim),
                nn.ReLU(),
                HexConv2d(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            if 32 % CRITIC_GROUPS or hidden_dim % CRITIC_GROUPS:
                raise ValueError(
                    f'{arch} needs hidden_dim divisible by {CRITIC_GROUPS}, got {hidden_dim}')
            self.board_encoder = nn.Sequential(
                self._conv_block(self.board_channels, 32),
                self._conv_block(32, hidden_dim),
                self._conv_block(hidden_dim, hidden_dim),
            )

        # v3 and v4 drop the opponent one-hot from the head input (see the class docstring).
        self.uses_opp_onehot = arch not in (CRITIC_ARCH_V3, CRITIC_ARCH_V4)
        # v4's gather readout is wider than the flank-average pool it replaces: 10
        # concatenated base cells + own/opp unit mean+max + whole-board mean+max = 16
        # channel-widths, vs. 2 for `_split_pool` (see `_gathered_pool`).
        self.pool_width = 16 * hidden_dim if arch == CRITIC_ARCH_V4 else 2 * hidden_dim
        head_in = self.pool_width + self.global_dim + self.priv_dim
        if self.uses_opp_onehot:
            head_in += self.OPP_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # v1 checkpoints have no such parameter, so it only exists on v2 and later.
        self.board_only_head = (nn.Linear(self.pool_width, 1)
                                if arch != CRITIC_ARCH_V1 else None)

        if arch == CRITIC_ARCH_V4:
            rows = torch.tensor([r for r, q in _BASE_CELLS], dtype=torch.long)
            cols = torch.tensor([q for r, q in _BASE_CELLS], dtype=torch.long)
            self.register_buffer('_base_rows', rows, persistent=False)
            self.register_buffer('_base_cols', cols, persistent=False)
            self._own_unit_channels = enc.own_unit_channels
            self._opp_unit_channels = enc.opp_unit_channels

    @staticmethod
    def _conv_block(cin, cout):
        return nn.Sequential(HexConv2d(cin, cout), nn.GroupNorm(CRITIC_GROUPS, cout), nn.ReLU())

    @property
    def device(self):
        return next(self.parameters()).device

    def trunk_health(self, board):
        """Is the board trunk carrying information? -> {'alive': [f1, f2, f3], 'out_std': float}

        `alive[i]` is the fraction of POSITIVE pre-activations at conv block `i`; `out_std` is
        the standard deviation of the pooled trunk output across the batch. **Both are needed,
        because the two architectures fail differently:**

        * `critic_v1` dies by the ReLU **absorbing state** — every pre-activation goes <= 0, so
          `alive` hits exactly 0.0 and the output is exactly zero. Once there the ReLU gradient
          is 0 and Adam's moments stay 0, so it never recovers.
        * `critic_v2` structurally *cannot* reach that state — GroupNorm re-centres each
          sample, so a whole channel cannot sit below zero. Verified: force the last conv to a
          constant -50 and v1 reports `alive` 0.0 with output exactly 0, while v2 reports
          `alive` **1.0**. That is the fix working, but it means the alive fraction alone is a
          useless guard for v2: an all-positive **constant** output carries exactly as little
          information as an all-zero one. `out_std` is what catches it.

        So: `min(alive) == 0` diagnoses the v1 failure, `out_std ~ 0` diagnoses either. A
        board-blind critic voids every measurement taken with it (docs/next_iteration.md §3.4),
        which is why `ppo.py` logs both every run. Healthy `alive` is roughly 20-50 %.
        """
        alive, x = [], board
        with torch.no_grad():
            for block in self.board_encoder:
                if isinstance(block, nn.Sequential):     # v2: [conv, norm, relu]
                    for layer in block[:-1]:
                        x = layer(x)
                    alive.append(float((x > 0).float().mean()))
                    x = block[-1](x)
                elif isinstance(block, nn.ReLU):         # v1: flat [conv, relu, ...]
                    alive.append(float((x > 0).float().mean()))
                    x = block(x)
                else:
                    x = block(x)
            # Variation of what the value head actually receives, across the batch. This is
            # the quantity that matters: the head sees `pool(trunk(board))` (`_split_pool`
            # for v1-v3, `_gathered_pool` for v4), and if that does not vary with the board
            # the critic cannot rank two positions no matter how many pre-activations are
            # positive.
            pooled = self._gathered_pool(board, x) if self.arch == CRITIC_ARCH_V4 else _split_pool(x)
            out_std = float(pooled.std()) if pooled.numel() > 1 else 0.0
        return {'alive': alive, 'out_std': out_std}

    def _gathered_pool(self, board_enc, feat):
        """Task-relevant readout (docs/IDEAS.md A2). -> [B, 16*hidden_dim]  (critic_v4 only.)

        Concatenates the trunk features at the 10 fixed base-cell indices, masked
        mean+max over own- and opponent-occupied unit cells, and a whole-board mean+max.
        Occupancy comes from `board_enc` (the input planes), not `feat` (the trunk
        output no longer carries a clean per-cell occupancy signal after convolution).
        """
        base = feat[:, :, self._base_rows, self._base_cols]  # [B, C, 10]
        base = base.reshape(base.shape[0], -1)  # [B, 10*C]
        own_occ = board_enc[:, self._own_unit_channels, :, :].sum(dim=1) > 0  # [B,7,7]
        opp_occ = board_enc[:, self._opp_unit_channels, :, :].sum(dim=1) > 0
        own_mean, own_max = _masked_mean_max(feat, own_occ)
        opp_mean, opp_max = _masked_mean_max(feat, opp_occ)
        glob_mean, glob_max = _global_mean_max(feat)
        return torch.cat(
            [base, own_mean, own_max, opp_mean, opp_max, glob_mean, glob_max], dim=-1)

    def _pooled(self, board_enc):
        """Board trunk + readout. -> [B, pool_width]"""
        feat = self.board_encoder(board_enc)
        if self.arch == CRITIC_ARCH_V4:
            return self._gathered_pool(board_enc, feat)
        return _split_pool(feat)

    def _head_input(self, pooled, global_feats, opp_onehot, privileged):
        """Assemble the value head's input, with or without the opponent one-hot.

        `critic_v3` drops the one-hot (class docstring), so it accepts `None` there and
        ignores anything passed — which is what lets every existing call site keep its
        signature while v1/v2 checkpoints still load and still receive their block.
        """
        if not self.uses_opp_onehot:
            return torch.cat([pooled, global_feats, privileged], dim=-1)
        if opp_onehot is None:
            raise ValueError(
                f'{self.arch} was trained with the opponent one-hot and cannot be evaluated '
                f'without one; pass a (B, {self.OPP_DIM}) tensor, or use a critic_v3 '
                f'checkpoint if you want an opponent-independent value.'
            )
        return torch.cat([pooled, global_feats, opp_onehot, privileged], dim=-1)

    def _forward(self, board_enc, global_feats, opp_onehot, privileged):
        pooled = self._pooled(board_enc)  # [B, 2*hidden_dim]
        return self.head(self._head_input(pooled, global_feats, opp_onehot, privileged)).squeeze(-1)

    def board_only_value(self, board_enc):
        """Value predicted from the BOARD ALONE. -> [B]  (critic_v2 and later.)

        The auxiliary target that keeps the trunk alive: because this head sees no
        globals, opp_onehot or privileged features, its loss is unsatisfiable without a
        board representation that carries signal. See the class docstring.
        """
        if self.board_only_head is None:
            raise RuntimeError(
                f'board_only_value requires {CRITIC_ARCH_V2} or later, this is {self.arch}')
        return self.board_only_head(self._pooled(board_enc)).squeeze(-1)

    def value_single(self, obs, opp_onehot, privileged):
        """V(s) for one raw observation dict.

        opp_onehot: (1, OPP_DIM) tensor; privileged: (1, PRIV_DIM) tensor — both on device.
        """
        board_t = torch.from_numpy(obs['board']).unsqueeze(0).to(self.device)
        global_t = torch.from_numpy(obs['global']).unsqueeze(0).to(self.device)
        return self._forward(board_t, global_t, opp_onehot, privileged).squeeze(0)

    def value_from_tensors(self, board_t, global_t, opp_onehot, privileged):
        """V(s) from raw board tensor — runs board_encoder internally."""
        return self._forward(board_t, global_t, opp_onehot, privileged).squeeze(0)

    def value_from_features(self, feat, global_t, opp_onehot, privileged):
        """V(s) reusing pre-encoded board features from act_with_encoded, skipping board_encoder.

        feat must come from a Policy with the same hidden_dim as this Critic.
        Used during rollout collection to avoid running the board encoder twice per step.
        The critic's own board_encoder is still used (and trained) via value_batch.

        Not supported on `critic_v4`: its readout needs the raw board tensor for unit
        occupancy (see `_gathered_pool`), which this fast path never receives.
        """
        if self.arch == CRITIC_ARCH_V4:
            raise NotImplementedError(
                f'{self.arch} readout needs the raw board tensor for unit occupancy; '
                'value_from_features only has pre-encoded features. Use value_batch or '
                'value_from_tensors instead (docs/IDEAS.md A2).'
            )
        pooled = _split_pool(feat)  # [B, 2*hidden_dim]
        combined = self._head_input(pooled, global_t, opp_onehot, privileged)
        return self.head(combined).squeeze(-1).squeeze(0)

    def value_batch(self, batch):
        """V(s) for a pre-encoded batch.

        Expects batch keys: board, global, privileged (N,PRIV_DIM), plus opp_onehot
        (N,3) — required by `critic_v1`/`critic_v2`, ignored (and optional) on
        `critic_v3`/`critic_v4`.
        """
        return self._forward(batch['board'], batch['global'], batch.get('opp_onehot'),
                             batch['privileged'])
