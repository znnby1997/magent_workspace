import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import sys
sys.path.append('..')
from net.basic_net import BasicNet

class GruDSANet(BasicNet):
    def __init__(self, obs_dim, n_actions, hidden_dim, agent_num, em_dim=32, **kwargs):
        super().__init__(obs_dim, n_actions, hidden_dim, em_dim)
        # gru attention
        self.gru_layer = nn.GRU(input_size=em_dim, hidden_size=hidden_dim // 2, bidirectional=True)
        self.w = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.fix_query = nn.Parameter(torch.Tensor(hidden_dim, 1))
    
    def att_layer(self, x):
        # gru attention
        self_info = x[:, 0:37]
        other_info = x[:, 37:]
        agents_info = other_info.split(28, dim=1)
        encodings = []
        encodings.append(self.self_encoder(self_info))
        for agent_info in agents_info:
            encodings.append(f.relu(self.other_encoder(agent_info)))
        gru_input = torch.stack(encodings) # size:[agent_num, batch, em_dim]
        gru_out, _ = self.gru_layer(gru_input)
        x = gru_out.permute(1, 0, 2) # size: [batch, agent_num, hidden]
        u = torch.tanh(torch.matmul(x, self.w)) # size: [batch, agent_num, hidden]
        att = torch.matmul(u, self.fix_query) # size: [batch, agent_num, 1]
        self.att_weight = f.softmax(att, dim=1) # size: [batch, agent_num, 1]
        scored_x = x * self.att_weight # size: [batch, agent_num, hidden]
        att_output = torch.sum(scored_x, dim=1) # size: [batch, hidden]
        return att_output