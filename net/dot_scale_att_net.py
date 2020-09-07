import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import sys
sys.path.append('..')
from net.basic_net import BasicNet

class DotScaleAttNet(BasicNet):
    def __init__(self, obs_dim, n_actions, hidden_dim, heads_num=4, em_dim=32, concatenation=False, **kwargs):
        super().__init__(obs_dim, n_actions, hidden_dim, em_dim)
        
        self.query_w = nn.Linear(37, em_dim)
        self.key_w = nn.Linear(28, em_dim)
        self.value_w = nn.Linear(28, em_dim)

        self.concatenation = concatenation

        self.align_layer = nn.Linear(2 * em_dim, hidden_dim)

    def att_layer(self, x, greedy_group=False):
        # scale dot attention
        self_info = x[:, 0:37]
        other_info = x[:, 37:]
        agents_info = torch.stack(other_info.split(28, dim=1)) # size: [other agents num, batch, 28]
        # other_embedding = f.relu(self.other_encoder(agents_info)) # size: [other agents num, batch, em_dim]
        # self_embedding = f.relu(self.self_encoder(self_info)) # size: [batch, em_dim]
        # encodings = torch.cat([self_embedding.unsqueeze(0), other_embedding], dim=0) # size: [agent num, batch, em_dim]
        query = self.query_w(self_info) # size: [batch, em_dim]
        keys = self.key_w(agents_info) # size: [other agents num, batch, em_dim]
        values = self.value_w(agents_info) # size: [other agents num, batch, em_dim]

        self.att_weight = f.softmax((torch.bmm(query.unsqueeze(1), keys.permute(1, 2, 0)) / np.sqrt(self.em_dim)), dim=2) # size: [batch, 1, other agents_num]
        
        batch = self_info.shape[0]
        if self.concatenation:
            self.att_weight = self.att_weight.squeeze(1) # size: [batch, agent_num]
            weights = self.att_weight.split(1, dim=1) # 每个的size: [batch, 1]

            other_w = torch.stack(weights) # size: [other agents num, batch, 1]
            other_we = (agents_info * other_w).permute(1, 0, 2).reshape(batch, -1) # size: [batch, other agent num * 28]
            obs_embedding = torch.cat([other_we, self_info], dim=1) # size: [batch, obs_dim]
            return f.relu(self.basic_mlp1(obs_embedding)) # size: [batch, hidden]
        else:
            att_output = torch.bmm(self.att_weight, values.permute(1, 0, 2)).squeeze(1) # size: [batch, em_dim]
            obs_embedding = torch.cat([att_output, query], dim=1) # size: [batch, 2 * em_dim]
            return f.relu(self.align_layer(obs_embedding))

    def get_mask(self):
        return self.att_weight