import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import sys
sys.path.append('..')
from net.basic_net import BasicNet

class DotScaleAttNet(BasicNet):
    def __init__(self, obs_dim, n_actions, hidden_dim, em_dim=32, **kwargs):
        super().__init__(obs_dim, n_actions, hidden_dim, em_dim)
        
        self.query_w = nn.Linear(em_dim, hidden_dim)
        self.key_w = nn.Linear(em_dim, hidden_dim)
        self.value_w = nn.Linear(em_dim, hidden_dim)

        self.output_q = nn.Linear(hidden_dim, n_actions)

    def att_layer(self, x):
        # scale dot attention
        self_info = x[:, 0:37]
        other_info = x[:, 37:]
        agents_info = other_info.split(28, dim=1)
        encodings = []
        encodings.append(self.self_encoder(self_info))
        for agent_info in agents_info:
            encodings.append(f.relu(self.other_encoder(agent_info)))
        query = self.query_w(encodings[0]).unsqueeze(1) # size: [batch, 1, hidden]
        keys, values = [], []
        for encoding in encodings:
            keys.append(self.key_w(encoding))
            values.append(self.value_w(encoding))
        
        keys_matrix = torch.stack(keys).permute(1, 2, 0) # size: [batch, hidden, agent_num]
        values_matrix = torch.stack(values).permute(1, 0, 2) # size: [batch, agent_num, hidden]
        self.att_weight = f.softmax((torch.bmm(query, keys_matrix) / np.sqrt(self.hidden_dim)), dim=2) # size: [batch, 1, agent_num]
        att_output = torch.bmm(self.att_weight, values_matrix).squeeze(1) # size: [batch, hidden]
        return att_output