import torch
import torch.nn as nn
import torch.nn.functional as f


class BasicNet(nn.Module):
    def __init__(self, obs_dim, output_dim, hidden_dim, **kwargs):
        super(BasicNet, self).__init__()

        self.e = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, **kwargs):
        return self.e(x), None
