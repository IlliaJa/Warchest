import torch
from policy import Policy
from environment.warchest_env import WarChestEnv
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

def trace_model(model, obs):
    # This function will wrap the original forward pass to fit TensorBoard
    class WrappedModel(nn.Module):
        def __init__(self, model, obs):
            super().__init__()
            self.model = model
            self.obs = obs

        def forward(self, dummy_input):
            return self.model(self.obs)

    return WrappedModel(model, obs)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    environment = WarChestEnv(save_game_history=False, debug_mode=False)
    obs, _ = environment.reset()

    training_hyperparameters = {
        'device': device,
        "n_training_episodes": 1500,
        "n_evaluation_episodes": 10,
        "max_t": 1000,
        "gamma": 0.95,
        "lr": 1e-3,
        "action_space": environment.action_space.n,
        'hidden_dim': 256,
    }

    warchest_policy = Policy(
        action_dim=training_hyperparameters["action_space"],
        device=training_hyperparameters["device"],
        hidden_dim=training_hyperparameters["hidden_dim"]).to(device)
    observation, _ = environment.reset()

    wrapped = trace_model(warchest_policy, observation)

    dummy_input = torch.zeros(1)  # Not used, just required by TensorBoard API
    writer = SummaryWriter()
    writer.add_graph(wrapped, dummy_input)
    writer.close()

    # RIUN tensorboard --logdir=runs