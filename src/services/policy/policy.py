import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from ..environment.warchest_env import N_VERBS, BOARD_DIM, ACTION_SPACE_SIZE, GLOBAL_DIM


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
    """Actor network with a spatial cell-keyed convolutional head.

    Board trunk: HexConv2d stack on 8 input planes → [B, Cf, 7, 7] feature map.
    Global features are broadcast as extra planes (tiled over the grid) and
    concatenated with the trunk output before the policy head.
    Policy head: 1×1 Conv2d → [B, N_VERBS, 7, 7] logit map, flattened to
    [B, 686], masked, softmax.

    Action encoding: id = verb * 49 + r * 7 + q.
    Input planes (8 channels, ego-centric):
        0: invalid  1: empty  2: uncontrolled base
        3: own base  4: opp base  5: exploration
        6: own units  7: opp units
    """

    def __init__(self, device, hidden_dim=64):
        super().__init__()

        self.board_encoder = nn.Sequential(
            HexConv2d(in_channels=8, out_channels=32),
            nn.ReLU(),
            HexConv2d(in_channels=32, out_channels=hidden_dim),
            nn.ReLU(),
        )

        # 1×1 conv maps (hidden_dim + GLOBAL_DIM) planes → N_VERBS logit planes
        self.policy_head = nn.Conv2d(hidden_dim + GLOBAL_DIM, N_VERBS, kernel_size=1)

        self.device = device

    def _logits(self, board: torch.Tensor, global_feats: torch.Tensor) -> torch.Tensor:
        """board: [B,8,7,7], global_feats: [B,G] → logits [B, N_VERBS*49]"""
        B = board.shape[0]
        feat = self.board_encoder(board)  # [B, hidden_dim, 7, 7]
        g = global_feats.view(B, GLOBAL_DIM, 1, 1).expand(B, GLOBAL_DIM, BOARD_DIM, BOARD_DIM)
        combined = torch.cat([feat, g], dim=1)  # [B, hidden_dim+G, 7, 7]
        return self.policy_head(combined).flatten(1)  # [B, 686]

    def forward(self, obs):
        board = torch.tensor(obs['board'], dtype=torch.float32).unsqueeze(0).to(self.device)
        global_feats = torch.tensor(obs['global'], dtype=torch.float32).unsqueeze(0).to(self.device)

        logits = self._logits(board, global_feats)  # [1, 686]

        mask = torch.tensor(obs['valid_action_mask'].astype(bool)).unsqueeze(0).to(self.device)
        return F.softmax(logits.masked_fill(~mask, -1e9), dim=-1)

    def act(self, obs):
        probs = self.forward(obs)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).squeeze(0), dist.entropy().squeeze(0)

    def evaluate_actions(self, obs, action):
        probs = self.forward(obs)
        dist = Categorical(probs)
        return dist.log_prob(action), dist.entropy()

    def evaluate_actions_batch(self, batch):
        """Single forward pass over a full buffer batch.

        batch keys (all on device):
            board   (N, 8, 7, 7) float
            global  (N, GLOBAL_DIM) float
            mask    (N, 686) bool
            actions (N,) long
        Returns log_probs (N,), entropies (N,).
        """
        logits = self._logits(batch['board'], batch['global'])  # [N, 686]
        masked = logits.masked_fill(~batch['mask'], -1e9)
        dist = Categorical(F.softmax(masked, dim=-1))
        return dist.log_prob(batch['actions']), dist.entropy()


class Critic(nn.Module):
    """Separate value network with its own spatial encoder.

    Privileged at training time: receives a 3-d opponent one-hot (random/greedy/pool)
    that the policy never sees.

    Board trunk: same HexConv2d stack as Policy → [B, Cf, 7, 7].
    Global average pool → [B, Cf]. Concatenate with global features and opp_onehot
    → scalar value.
    """

    OPP_DIM = 3

    def __init__(self, device, hidden_dim=64):
        super().__init__()
        self.board_encoder = nn.Sequential(
            HexConv2d(8, 32),
            nn.ReLU(),
            HexConv2d(32, hidden_dim),
            nn.ReLU(),
        )
        head_in = hidden_dim + GLOBAL_DIM + self.OPP_DIM
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

    def _forward(self, board_enc, global_feats, opp_onehot):
        feat = self.board_encoder(board_enc)  # [B, hidden_dim, 7, 7]
        pooled = feat.mean(dim=(-2, -1))  # global avg pool → [B, hidden_dim]
        return self.head(torch.cat([pooled, global_feats, opp_onehot], dim=-1)).squeeze(-1)

    def value_single(self, obs, opp_onehot):
        """V(s) for one raw observation dict. opp_onehot: (1, OPP_DIM) tensor on device."""
        board_t = torch.tensor(obs['board'], dtype=torch.float32).unsqueeze(0).to(self.device)
        global_t = torch.tensor(obs['global'], dtype=torch.float32).unsqueeze(0).to(self.device)
        return self._forward(board_t, global_t, opp_onehot).squeeze(0)

    def value_batch(self, batch):
        """V(s) for a pre-encoded batch.

        Expects batch keys: board (N,8,7,7), global (N,GLOBAL_DIM), opp_onehot (N,3).
        """
        return self._forward(batch['board'], batch['global'], batch['opp_onehot'])
