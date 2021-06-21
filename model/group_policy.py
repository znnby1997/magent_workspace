import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupPolicy(nn.Module):
    def __init__(self, input_shape, n_groups, hidden_dim):
        super().__init__()

        self.tokens_fc = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_groups)
        )

    def forward(self, x, h=None):
        """
            x: [batch, num, tokens],
            h: [batch, hidden_dim]
        """
        if h is not None:
            x = torch.cat([x, h], dim=2) # shape: nums, batch, 2*(input_shape+h_dim)
        return f.softmax(self.tokens_fc(x))

        


