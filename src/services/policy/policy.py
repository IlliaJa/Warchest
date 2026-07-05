import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from ..environment.warchest_env import (
    N_VERBS, BOARD_DIM, ACTION_SPACE_SIZE, SPATIAL_SIZE, FACEDOWN_SIZE,
    N_FACTORED_VERBS, VERB_OF_ACTION,
)
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

    def evaluate_actions_batch(self, batch):
        """Single forward pass over a full buffer batch.

        batch keys (all on device): board (N,C,7,7), global (N,GLOBAL_DIM),
        mask (N,A) bool, actions (N,) long. Returns log_probs (N,), entropies (N,).
        """
        flat_logits, verb_logits = self._features(batch['board'], batch['global'])
        joint = self._joint_log_probs(flat_logits, verb_logits, batch['mask'])
        dist = Categorical(logits=joint)
        return dist.log_prob(batch['actions']), dist.entropy()


class Critic(nn.Module):
    """Separate value network with its own spatial encoder.

    Privileged at training time, in two ways the policy never sees:
      - a 3-d opponent one-hot (random / greedy / pool);
      - a PRIV_DIM vector of the opponent's *true* hidden coin split
        (opp hand / bag / face-down discard per coin). Discarded at inference.

    Board trunk: same HexConv2d stack as Policy → [B, Cf, 7, 7].
    Split flank pool (see `_split_pool`) → [B, 2*Cf]. Concatenate with global
    features, opp_onehot and the privileged vector → scalar value.
    """

    OPP_DIM = 3

    def __init__(self, device, hidden_dim=64, *, obs_encoder=None):
        super().__init__()
        # Obs dims (incl. the privileged vector) come from the paired encoder.
        enc = obs_encoder or latest_encoder()
        self.board_channels = enc.board_channels
        self.global_dim = enc.global_dim
        self.priv_dim = enc.priv_dim
        self.board_encoder = nn.Sequential(
            HexConv2d(self.board_channels, 32),
            nn.ReLU(),
            HexConv2d(32, hidden_dim),
            nn.ReLU(),
            HexConv2d(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        head_in = 2 * hidden_dim + self.global_dim + self.OPP_DIM + self.priv_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def _forward(self, board_enc, global_feats, opp_onehot, privileged):
        feat = self.board_encoder(board_enc)  # [B, hidden_dim, 7, 7]
        pooled = _split_pool(feat)  # [B, 2*hidden_dim]
        combined = torch.cat([pooled, global_feats, opp_onehot, privileged], dim=-1)
        return self.head(combined).squeeze(-1)

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
        """
        pooled = _split_pool(feat)  # [B, 2*hidden_dim]
        combined = torch.cat([pooled, global_t, opp_onehot, privileged], dim=-1)
        return self.head(combined).squeeze(-1).squeeze(0)

    def value_batch(self, batch):
        """V(s) for a pre-encoded batch.

        Expects batch keys: board, global, opp_onehot (N,3), privileged (N,PRIV_DIM).
        """
        return self._forward(batch['board'], batch['global'], batch['opp_onehot'], batch['privileged'])
