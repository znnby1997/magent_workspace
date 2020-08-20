import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import sys
sys.path.append('..')
from net.basic_net import BasicNet

class DotScaleAttNet(BasicNet):
    def __init__(self, obs_dim, n_actions, hidden_dim, heads_num=4, em_dim=32, **kwargs):
        super().__init__(obs_dim, n_actions, hidden_dim, em_dim)
        
        self.query_w = nn.Linear(em_dim, em_dim)
        self.key_w = nn.Linear(em_dim, em_dim)
        self.value_w = nn.Linear(em_dim, em_dim)

        self.align_layer = nn.Linear(em_dim, hidden_dim)

    def att_layer(self, x):
        # scale dot attention
        self_info = x[:, 0:37]
        other_info = x[:, 37:]
        agents_info = torch.stack(other_info.split(28, dim=1)) # size: [other agents num, batch, 28]
        other_embedding = f.relu(self.other_encoder(agents_info)) # size: [other agents num, batch, em_dim]
        self_embedding = f.relu(self.self_encoder(self_info)) # size: [batch, em_dim]
        encodings = torch.cat([self_embedding.unsqueeze(0), other_embedding], dim=0) # size: [agent num, batch, em_dim]
        
        query = self.query_w(self_embedding).unsqueeze(1) # size: [batch, 1, em_dim]
        keys = self.key_w(encodings) # size: [agents num, batch, em_dim]
        values = self.value_w(encodings) # size: [agents num, batch, em_dim]

        self.att_weight = f.softmax((torch.bmm(query, keys.permute(1, 2, 0)) / np.sqrt(self.em_dim)), dim=2) # size: [batch, 1, agents_num]
        att_output = torch.bmm(self.att_weight, values.permute(1, 0, 2)).squeeze(1) # size: [batch, em_dim]
        return self.align_layer(att_output)