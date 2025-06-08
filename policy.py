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
            nn.Linear(hidden_dim + 4 + 32, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic_head = nn.Linear(hidden_dim + 4 + 32, 1)

        self.device = device

    def forward(self, obs):
        # Extract inputs from observation dict # (1, 5, 7, 7)
        board = self.encode_board(obs['board'], exploration_map=obs['exploration_map']).astype(float)
        board = torch.tensor(board, dtype=torch.float32).unsqueeze(0).to(self.device)
        board_features = self.board_encoder(board)

        global_feats = torch.tensor(obs['global'].astype(float), dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 2)

        unit_coords = torch.tensor(obs['units'], dtype=torch.float32).to(self.device)
        unit_coords = unit_coords.view(2, 2, 2)  # [player, unit, coord]
        encoded_units = self.unit_encoder(unit_coords.view(-1, 2))
        unit_features = encoded_units.mean(dim=0, keepdim=True)

        combined = torch.cat([board_features, global_feats, unit_features], dim=-1)

        # Policy head (action logits)
        logits = self.actor_head(combined)
        value = self.critic_head(combined)

        action_mask = np.expand_dims(obs['valid_action_mask'].astype(bool), 0)
        masked_logits = logits.clone()
        masked_logits[~action_mask] = -1e9

        probs = F.softmax(masked_logits, dim=-1)
        return probs, value

    def act(self, obs):
        probs, value = self.forward(obs)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value, dist.entropy()

    def evaluate_actions(self, obs, action):
        probs, value = self.forward(obs)
        dist = Categorical(probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value

    def encode_board(self, board, exploration_map):
        one_hot = np.zeros((6, 7, 7), dtype=np.float32)

        one_hot[0] = (board == INVALID_CELL_ID)
        one_hot[1] = (board == EMPTY_CELL_ID)
        one_hot[2] = (board == UNCONTROLLED_BASE_CELL_ID)
        one_hot[3] = (board == CONTROLLED_BASE_PLAYER_1_CELL_ID)
        one_hot[4] = (board == CONTROLLED_BASE_PLAYER_2_CELL_ID)

        visits = exploration_map.astype(np.float32)
        visits[visits < 0] = 0
        visits = visits / (visits.max() + 1e-5)
        one_hot[5] = visits
        return one_hot