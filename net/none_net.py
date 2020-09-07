import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import sys
sys.path.append('..')
from net.basic_net import BasicNet

class NoneNet(BasicNet):
    def __init__(self, obs_dim, n_actions, hidden_dim, **kwargs):
        super().__init__(obs_dim, n_actions, hidden_dim)
        self.mlp1 = nn.Linear(obs_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)

    def att_layer(self, x, greedy_group=False):
        e1 = f.relu(self.mlp1(x))
        e2 = f.relu(self.mlp2(e1))
        return e2


if __name__ == '__main__':
    n = NoneNet(3, 4, 10)
    a = torch.randn(1, 3)
    print(n.att_layer(a))