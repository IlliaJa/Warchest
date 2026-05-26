import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from environment.cell_ids import *

class Policy(nn.Module):
    def __init__(self, action_dim, device, hidden_dim=128):
        super(Policy, self).__init__()

        # Board encoder (CNN for spatial data)
        self.board_encoder = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim)
        )

        # Global features encoder (fully connected layer)
        # self.global_encoder = nn.Linear(4, hidden_dim)

        # Unit encoder (flattened units data)
        self.unit_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim + 3 + 64, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.device = device

    def forward(self, obs):
        board = Policy.encode_board(obs['board'], obs['exploration_map'], obs['active_player']).astype(float)
        board = torch.tensor(board, dtype=torch.float32).unsqueeze(0).to(self.device)
        board_features = self.board_encoder(board)

        global_feats = torch.tensor(obs['global'].astype(float), dtype=torch.float32).unsqueeze(0).to(self.device)

        unit_coords = torch.tensor(obs['units'], dtype=torch.float32).to(self.device)
        unit_coords = unit_coords.view(2, 2, 2)
        encoded_units = self.unit_encoder(unit_coords.view(-1, 2)).view(2, 2, -1)
        my_unit_features = encoded_units[0].mean(dim=0, keepdim=True)
        opp_unit_features = encoded_units[1].mean(dim=0, keepdim=True)
        unit_features = torch.cat([my_unit_features, opp_unit_features], dim=-1)

        combined = torch.cat([board_features, global_feats, unit_features], dim=-1)

        logits = self.actor_head(combined)

        action_mask = np.expand_dims(obs['valid_action_mask'].astype(bool), 0)
        masked_logits = logits.clone()
        masked_logits[~action_mask] = -1e9

        probs = F.softmax(masked_logits, dim=-1)
        return probs

    def act(self, obs):
        probs = self.forward(obs)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def evaluate_actions(self, obs, action):
        probs = self.forward(obs)
        dist = Categorical(probs)
        return dist.log_prob(action), dist.entropy()

    def evaluate_actions_batch(self, batch):
        """Single forward pass over a full buffer batch.

        batch keys (all on device):
            board   (N, 6, 7, 7) float — already encoded by encode_board_batch
            global  (N, 3)       float
            units   (N, 2, 2, 2) float
            mask    (N, 14)      bool
            actions (N,)         long
        Returns log_probs (N,), entropies (N,).
        """
        N = batch['board'].shape[0]

        board_features = self.board_encoder(batch['board'])

        encoded = self.unit_encoder(batch['units'].view(N * 4, 2)).view(N, 2, 2, -1)
        unit_features = torch.cat([
            encoded[:, 0].mean(dim=1),
            encoded[:, 1].mean(dim=1),
        ], dim=-1)

        combined = torch.cat([board_features, batch['global'], unit_features], dim=-1)

        masked_logits = self.actor_head(combined).masked_fill(~batch['mask'], -1e9)
        dist = Categorical(F.softmax(masked_logits, dim=-1))

        return dist.log_prob(batch['actions']), dist.entropy()

    @staticmethod
    def encode_board_batch(boards, exploration_maps, active_players):
        """Vectorized board encoding for a batch of observations.

        boards          (N, 7, 7) int
        exploration_maps (N, 7, 7) float
        active_players  (N,) int  — 1 or 2
        Returns (N, 6, 7, 7) float32.
        """
        N = len(boards)
        enc = np.zeros((N, 6, 7, 7), dtype=np.float32)
        enc[:, 0] = (boards == INVALID_CELL_ID)
        enc[:, 1] = (boards == EMPTY_CELL_ID)
        enc[:, 2] = (boards == UNCONTROLLED_BASE_CELL_ID)
        p1 = (active_players == 1)[:, None, None]
        enc[:, 3] = np.where(p1, boards == CONTROLLED_BASE_PLAYER_1_CELL_ID, boards == CONTROLLED_BASE_PLAYER_2_CELL_ID)
        enc[:, 4] = np.where(p1, boards == CONTROLLED_BASE_PLAYER_2_CELL_ID, boards == CONTROLLED_BASE_PLAYER_1_CELL_ID)
        visits = np.clip(exploration_maps, 0, None).astype(np.float32)
        enc[:, 5] = visits / (visits.max(axis=(1, 2), keepdims=True) + 1e-5)
        return enc

    @staticmethod
    def encode_board(board, exploration_map, active_player):
        one_hot = np.zeros((6, 7, 7), dtype=np.float32)

        one_hot[0] = (board == INVALID_CELL_ID)
        one_hot[1] = (board == EMPTY_CELL_ID)
        one_hot[2] = (board == UNCONTROLLED_BASE_CELL_ID)
        if active_player == 1:
            one_hot[3] = (board == CONTROLLED_BASE_PLAYER_1_CELL_ID)
            one_hot[4] = (board == CONTROLLED_BASE_PLAYER_2_CELL_ID)
        else:
            one_hot[3] = (board == CONTROLLED_BASE_PLAYER_2_CELL_ID)
            one_hot[4] = (board == CONTROLLED_BASE_PLAYER_1_CELL_ID)

        visits = exploration_map.astype(np.float32)
        visits[visits < 0] = 0
        visits = visits / (visits.max() + 1e-5)
        one_hot[5] = visits
        return one_hot


class Critic(nn.Module):
    """Separate value network with its own spatial and unit encoders.

    Independent encoders let the critic develop value-optimized representations
    without conflicting with the actor's policy gradient.
    """

    def __init__(self, device, hidden_dim=128):
        super().__init__()
        self.board_encoder = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim),
        )
        self.unit_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 3 + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.device = device

    def _forward(self, board_enc, global_feats, units):
        N = units.shape[0]
        board_features = self.board_encoder(board_enc)
        encoded = self.unit_encoder(units.view(N * 4, 2)).view(N, 2, 2, -1)
        unit_features = torch.cat([encoded[:, 0].mean(1), encoded[:, 1].mean(1)], dim=-1)
        return self.head(
            torch.cat([board_features, global_feats, unit_features], dim=-1)
        ).squeeze(-1)

    def value_single(self, obs):
        """V(s) for one raw observation dict (used during rollout collection)."""
        board = Policy.encode_board(obs['board'], obs['exploration_map'], obs['active_player'])
        board_t = torch.tensor(board, dtype=torch.float32).unsqueeze(0).to(self.device)
        global_t = torch.tensor(obs['global'].astype(float), dtype=torch.float32).unsqueeze(0).to(self.device)
        units_t = torch.tensor(obs['units'], dtype=torch.float32).unsqueeze(0).to(self.device)
        return self._forward(board_t, global_t, units_t).squeeze(0)

    def value_batch(self, batch):
        """V(s) for a pre-encoded batch (used during PPO update).

        Expects batch['board'] (N,6,7,7), batch['global'] (N,3), batch['units'] (N,2,2,2).
        """
        return self._forward(batch['board'], batch['global'], batch['units'])