import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from ..environment.warchest_env import (
    N_VERBS, BOARD_DIM, ACTION_SPACE_SIZE, GLOBAL_DIM,
    BOARD_CHANNELS, SPATIAL_SIZE, FACEDOWN_SIZE, PRIV_DIM,
    N_FACTORED_VERBS, VERB_OF_ACTION,
)


class HexConv2d(nn.Module):
    """Topology-correct convolution for hex grids stored in a 2D axial array.

    Equivalent to a 3×3 conv with padding=1, except the receptive field is the
    6 hex neighbours plus center (7 cells) instead of all 9 grid cells. Two
    anti-diagonal positions — (-1, +1) and (+1, -1) in (Δr, Δq) — are not
    hex-adjacent under the axial convention in Board.offsets, so they are
    excluded from the kernel.

    Window index map for the 3×3 patch returned by F.unfold (row-major):
        0: (-1,-1)  hex       3: ( 0,-1)  hex       6: (+1,-1)  excluded
        1: (-1, 0)  hex       4: ( 0, 0)  center    7: (+1, 0)  hex
        2: (-1,+1)  excluded  5: ( 0,+1)  hex       8: (+1,+1)  hex
    """

    HEX_INDICES = (0, 1, 3, 4, 5, 7, 8)

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = nn.Conv2d(in_channels * 7, out_channels, kernel_size=1)
        self.register_buffer(
            '_hex_idx',
            torch.tensor(self.HEX_INDICES, dtype=torch.long),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        unfolded = F.unfold(x, kernel_size=3, padding=1).view(n, c, 9, h, w)
        hex_window = unfolded.index_select(dim=2, index=self._hex_idx)
        hex_window = hex_window.reshape(n, c * 7, h, w)
        return self.proj(hex_window)


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

    Board trunk: HexConv2d stack on BOARD_CHANNELS input planes → [B, Cf, 7, 7].
    Input planes (BOARD_CHANNELS, ego-centric):
        0: invalid  1: empty  2: uncontrolled base
        3: own base  4: opp base  5: exploration
        6: own sword  7: own knight  8: opp sword  9: opp knight
    """

    def __init__(self, device, hidden_dim=64):
        super().__init__()

        self.board_encoder = nn.Sequential(
            HexConv2d(in_channels=BOARD_CHANNELS, out_channels=32),
            nn.ReLU(),
            HexConv2d(in_channels=32, out_channels=hidden_dim),
            nn.ReLU(),
        )

        # Within-verb logits: 1×1 conv → N_VERBS spatial planes (flattened) plus a
        # linear head for the face-down actions.
        self.policy_head = nn.Conv2d(hidden_dim + GLOBAL_DIM, N_VERBS, kernel_size=1)
        self.facedown_head = nn.Linear(hidden_dim + GLOBAL_DIM, FACEDOWN_SIZE)
        # Top-level verb head.
        self.verb_head = nn.Linear(hidden_dim + GLOBAL_DIM, N_FACTORED_VERBS)

        # Static flat-id -> verb-index map, and a per-verb membership matrix.
        verb_index = torch.tensor(VERB_OF_ACTION, dtype=torch.long)
        group_mat = torch.zeros(N_FACTORED_VERBS, ACTION_SPACE_SIZE, dtype=torch.bool)
        group_mat[verb_index, torch.arange(ACTION_SPACE_SIZE)] = True
        self.register_buffer('_verb_index', verb_index, persistent=False)
        self.register_buffer('_group_mat', group_mat, persistent=False)

        self.device = device

    def _features(self, board: torch.Tensor, global_feats: torch.Tensor):
        """board: [B,C,7,7], global_feats: [B,G] → (flat_logits [B,A], verb_logits [B,V])"""
        B = board.shape[0]
        feat = self.board_encoder(board)  # [B, hidden_dim, 7, 7]
        g = global_feats.view(B, GLOBAL_DIM, 1, 1).expand(B, GLOBAL_DIM, BOARD_DIM, BOARD_DIM)
        spatial = self.policy_head(torch.cat([feat, g], dim=1)).flatten(1)  # [B, SPATIAL_SIZE]
        pooled = feat.mean(dim=(-2, -1))  # [B, hidden_dim]
        pg = torch.cat([pooled, global_feats], dim=-1)
        facedown = self.facedown_head(pg)  # [B, FACEDOWN_SIZE]
        flat_logits = torch.cat([spatial, facedown], dim=1)  # [B, ACTION_SPACE_SIZE]
        verb_logits = self.verb_head(pg)  # [B, N_FACTORED_VERBS]
        return flat_logits, verb_logits

    def _joint_log_probs(self, flat_logits, verb_logits, mask):
        """Factored joint log-probs over the flat action space. [B, ACTION_SPACE_SIZE].

        log P(a) = log P(verb=v(a)) + log P(a | verb=v(a)); illegal ids -> NEG.
        """
        NEG = -1e9
        B = flat_logits.shape[0]
        ml = flat_logits.masked_fill(~mask, NEG)  # [B, A]
        g = self._verb_index.unsqueeze(0).expand(B, -1)  # [B, A]

        # Per-verb max for a numerically stable within-group softmax.
        gmax = torch.stack(
            [ml[:, self._group_mat[v]].max(dim=1).values for v in range(N_FACTORED_VERBS)],
            dim=1,
        )  # [B, V]
        shifted = ml - gmax.gather(1, g)  # [B, A]; ≤0 for legal members
        exp_shifted = shifted.exp() * mask.float()  # zero out illegal ids
        gsum = torch.stack(
            [exp_shifted[:, self._group_mat[v]].sum(dim=1) for v in range(N_FACTORED_VERBS)],
            dim=1,
        )  # [B, V]
        within_logp = (shifted - gsum.clamp_min(1e-12).log().gather(1, g)).masked_fill(~mask, NEG)

        verb_mask = gsum > 0  # verbs with at least one legal action
        verb_logp = F.log_softmax(verb_logits.masked_fill(~verb_mask, NEG), dim=1)  # [B, V]

        joint = verb_logp.gather(1, g) + within_logp  # [B, A]
        return joint.masked_fill(~mask, NEG)

    def _obs_logits(self, obs):
        board = torch.tensor(obs['board'], dtype=torch.float32).unsqueeze(0).to(self.device)
        global_feats = torch.tensor(obs['global'], dtype=torch.float32).unsqueeze(0).to(self.device)
        mask = torch.tensor(obs['valid_action_mask'].astype(bool)).unsqueeze(0).to(self.device)
        flat_logits, verb_logits = self._features(board, global_feats)
        return self._joint_log_probs(flat_logits, verb_logits, mask)  # [1, A]

    def forward(self, obs):
        """Return action probabilities for a single observation dict."""
        return F.softmax(self._obs_logits(obs), dim=-1)

    def act(self, obs):
        dist = Categorical(logits=self._obs_logits(obs))
        action = dist.sample()
        return action.item(), dist.log_prob(action).squeeze(0), dist.entropy().squeeze(0)

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
    Global average pool → [B, Cf]. Concatenate with global features, opp_onehot and
    the privileged vector → scalar value.
    """

    OPP_DIM = 3

    def __init__(self, device, hidden_dim=64):
        super().__init__()
        self.board_encoder = nn.Sequential(
            HexConv2d(BOARD_CHANNELS, 32),
            nn.ReLU(),
            HexConv2d(32, hidden_dim),
            nn.ReLU(),
        )
        head_in = hidden_dim + GLOBAL_DIM + self.OPP_DIM + PRIV_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.device = device

    def _forward(self, board_enc, global_feats, opp_onehot, privileged):
        feat = self.board_encoder(board_enc)  # [B, hidden_dim, 7, 7]
        pooled = feat.mean(dim=(-2, -1))  # global avg pool → [B, hidden_dim]
        combined = torch.cat([pooled, global_feats, opp_onehot, privileged], dim=-1)
        return self.head(combined).squeeze(-1)

    def value_single(self, obs, opp_onehot, privileged):
        """V(s) for one raw observation dict.

        opp_onehot: (1, OPP_DIM) tensor; privileged: (1, PRIV_DIM) tensor — both on device.
        """
        board_t = torch.tensor(obs['board'], dtype=torch.float32).unsqueeze(0).to(self.device)
        global_t = torch.tensor(obs['global'], dtype=torch.float32).unsqueeze(0).to(self.device)
        return self._forward(board_t, global_t, opp_onehot, privileged).squeeze(0)

    def value_batch(self, batch):
        """V(s) for a pre-encoded batch.

        Expects batch keys: board, global, opp_onehot (N,3), privileged (N,PRIV_DIM).
        """
        return self._forward(batch['board'], batch['global'], batch['opp_onehot'], batch['privileged'])
