import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class QnetM(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim, net, agent_num, nonlin):
        super(QnetM, self).__init__()
        
        self.e = net(obs_dim, n_actions, hidden_dim, agent_num=agent_num, nonlin=nonlin)

        self.n_actions = n_actions

    def forward(self, x):
        att_weight = None
        em, att_weight = self.e(x)
        return em, att_weight
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs.reshape(1, -1))

        coin = random.random()
        if coin < epsilon:
            return random.randint(0, self.n_actions - 1), out[1]
        else: 
            return out[0].argmax().item(), out[1]