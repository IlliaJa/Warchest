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
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim)
        )

        # Global features encoder (fully connected layer)
        self.global_encoder = nn.Linear(4, hidden_dim)

        # Unit encoder (flattened units data)
        self.unit_encoder = nn.Linear(2 * 2 * 2, hidden_dim)

        # Final policy head
        self.fc = nn.Linear(hidden_dim * 3, action_dim)

        self.device = device

    def forward(self, obs):
        # Extract inputs from observation dict # (1, 5, 7, 7)
        board = self.encode_board(obs['board']).astype(float)
        board = torch.tensor(board, dtype=torch.float32).unsqueeze(0).to(self.device)
        global_feats = torch.tensor(obs['global'].astype(float), dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 2)
        units = torch.tensor(obs['units'].astype(float).flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 8)

        # Forward through encoders
        board_features = self.board_encoder(board)
        global_features = self.global_encoder(global_feats)
        unit_features = self.unit_encoder(units)

        # Concatenate all features
        combined = torch.cat([board_features, global_features, unit_features], dim=-1)

        # Policy head (action logits)
        logits = self.fc(combined)

        action_mask = np.expand_dims(obs['valid_action_mask'].astype(bool), 0)
        masked_logits = logits.clone()
        masked_logits[~action_mask] = -1e9

        return F.softmax(masked_logits, dim=-1)

    def act(self, obs):
        probs = self.forward(obs)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def encode_board(self, board):
        one_hot = np.zeros((5, 7, 7), dtype=np.float32)

        one_hot[0] = (board == INVALID_CELL_ID)
        one_hot[1] = (board == EMPTY_CELL_ID)
        one_hot[2] = (board == UNCONTROLLED_BASE_CELL_ID)
        one_hot[3] = (board == CONTROLLED_BASE_PLAYER_1_CELL_ID)
        one_hot[4] = (board == CONTROLLED_BASE_PLAYER_2_CELL_ID)

        return one_hot