import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import sys
sys.path.append('..')
from net.basic_net import BasicNet

class AlwAttNet(BasicNet):
    def __init__(self, obs_dim, n_actions, hidden_dim, agent_num, em_dim=32, nonlin='softmax', **kwargs):
        super().__init__(obs_dim, n_actions, hidden_dim, em_dim)
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
        agents_info = torch.stack(other_info.split(28, dim=1)) # size: [other agents num, batch, 28]
        other_embedding = self.other_encoder(agents_info) # size: [other agents num, batch, em_dim]
        self_embedding = self.self_encoder(self_info) # size: [batch, em_dim]
        encodings = torch.cat([self_embedding.unsqueeze(0), other_embedding], dim=0) # size: [agents num, batch, em_dim]
        att_output = torch.bmm(self.att_weight.unsqueeze(1), encodings.permute(1, 0, 2)).squeeze(1) # size: [batch, em_dim]
        return self.att_output_encoder(att_output)