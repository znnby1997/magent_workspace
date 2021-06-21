import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class AlwAttNet(nn.Module):
    def __init__(self, obs_dim, output_dim, hidden_dim, agent_num, nonlin='softmax', **kwargs):
        super(AlwAttNet, self).__init__()
        self.agent_num = agent_num
        self.att_weight = None
        self.nonlin = 'softmax'

        # agent level attention
        # self info: 36bits  opp info: 28bits  partner info: 28bits noisy info: 28bits
        self.al_att_layer = nn.Linear(obs_dim, agent_num * 3 - 1)
        self.output = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, **kwargs):
        if self.nonlin == 'softmax':
            self.att_weight = f.softmax(self.al_att_layer(x), dim=1)
        elif self.nonlin == 'sigmoid':
            self.att_weight = torch.sigmoid(self.al_att_layer(x)) # size: [batch, other agent num]

        self_info = x[:, 0:36] # size: [batch, 36]
        other_info = x[:, 36:] # size: [batch, other num * 28]
        agents_info = torch.stack(other_info.split(28, dim=1)).permute(1, 0, 2) # size: [batch, agent num, 28]

        batch = self_info.shape[0]

        other_we = (self.att_weight.unsqueeze(2) * agents_info).reshape(batch, -1) # size: [batch, (other agent num) * 28]
        obs_embedding = torch.cat([other_we, self_info], dim=1) # size: [batch, obs_dim]
        return self.output(obs_embedding), self.att_weight # size: [batch, n_actions]
    
