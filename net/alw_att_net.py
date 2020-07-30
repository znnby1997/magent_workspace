import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import sys
sys.path.append('..')
from net.basic_net import BasicNet

class AlwAttNet(BasicNet):
    def __init__(self, obs_dim, n_actions, hidden_dim, em_dim=32, nonlin='softmax', **kwargs):
        super().__init__(obs_dim, n_actions, hidden_dim, em_dim)
        self.mlp = nn.Linear(obs_dim, hidden_dim)
        self.nonlin = nonlin

        # agent level attention
        # self info: 37bits  opp info: 28bits  partner info: 28bits
        self.al_att_layer = nn.Linear(obs_dim, agent_num * 2)
        self.att_output_encoder = nn.Linear(em_dim, hidden_dim)

    def att_layer(self, x):
        if self.nonlin == 'softmax':
            self.att_weight = f.softmax(self.al_att_layer(x), dim=1) # size: [batch, agent_num]
        elif self.nonlin == 'sigmoid':
            self.att_weight = torch.sigmoid(self.al_att_layer(x))
        elif self.nonlin == 'tanh':
            self.att_weight = torch.tanh(self.al_att_layer(x))

        self_info = x[:, 0:37]
        other_info = x[:, 37:]
        agents_info = other_info.split(28, dim=1)
        encodings = []
        encodings.append(self.self_encoder(self_info))
        for agent_info in agents_info:
            encodings.append(f.relu(self.other_encoder(agent_info)))
        encodings = torch.stack(encodings).permute(1, 0, 2) # size: [batch, agent_num, embedding_dim]
        att_output = torch.bmm(self.att_weight.unsqueeze(1), encodings).squeeze(1) # size: [batch, em_dim]
        return f.relu(self.att_output_encoder(att_output))