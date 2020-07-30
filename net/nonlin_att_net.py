import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import sys
sys.path.append('..')
from net.basic_net import BasicNet

class NonlinAttNet(BasicNet):
    def __init__(self, obs_dim, n_actions, hidden_dim, nonlin='softmax', **kwargs):
        super().__init__(obs_dim, n_actions, hidden_dim)
        self.mlp = nn.Linear(obs_dim, hidden_dim)
        self.nonlin = nonlin

        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)

    def att_layer(self, x):
        att_output = None
        if self.nonlin == 'softmax':
            self.att_weight = f.softmax(self.att_layer(x), dim=1)
            att_output = self.att_weight * x
        elif self.nonlin == 'sigmoid':
            self.att_weight = torch.sigmoid(self.att_layer(x))
            att_output = self.att_weight * x
        elif self.nonlin == 'tanh':
            self.att_weight = torch.tanh(self.att_layer(x))
            att_output = self.att_weight * x
        return f.relu(self.mlp2(att_output))
