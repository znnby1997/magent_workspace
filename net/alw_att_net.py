import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import sys
sys.path.append('..')
from net.basic_net import BasicNet

class AlwAttNet(BasicNet):
    def __init__(self, obs_dim, n_actions, hidden_dim, agent_num, em_dim=32, nonlin='softmax', concatenation=False, **kwargs):
        super().__init__(obs_dim, n_actions, hidden_dim, em_dim)
        self.nonlin = nonlin
        self.concatenation = concatenation  # 直接将得到的weight与观测对应元素相乘，而非聚合
        self.agent_num = agent_num

        # agent level attention
        # self info: 37bits  opp info: 28bits  partner info: 28bits
        self.al_att_layer = nn.Linear(obs_dim, agent_num * 2 - 1)
        self.att_output_encoder = nn.Linear(37 + 28, hidden_dim)


    def att_layer(self, x, greedy_group=False):
        if self.nonlin == 'softmax':
            self.att_weight = f.softmax(self.al_att_layer(x), dim=1) # size: [batch, other agent num]
        elif self.nonlin == 'sigmoid':
            self.att_weight = torch.sigmoid(self.al_att_layer(x))
        elif self.nonlin == 'tanh':
            self.att_weight = torch.tanh(self.al_att_layer(x))

        self_info = x[:, 0:37] # size: [batch, 37]
        other_info = x[:, 37:] # size: [batch, 39 * 28]
        agents_info = torch.stack(other_info.split(28, dim=1)) # size: [other agents num, batch, 28]
        # agents_num = agents_info.shape[0] # 这里可能agents_num != self.agent，因为环境中可能加入了额外的agent信息作为噪声
        batch = self_info.shape[0]

        if self.concatenation:
            weights = self.att_weight.split(1, dim=1) # 每个的size: [batch, 1]

            other_w = torch.stack(weights) # size: [other agents num, batch, 1]
            other_we = (agents_info * other_w).permute(1, 0, 2).reshape(batch, -1) # size: [batch, other agent num * 28]
            obs_embedding = torch.cat([other_we, self_info], dim=1) # size: [batch, obs_dim]
            return f.relu(self.basic_mlp1(obs_embedding)) # size: [batch, hidden]
        else:
            # 感觉有点问题，因为weight是根据原始观测生成的，因此weight聚合也应该直接聚合在原始观测上
            att_output = torch.bmm(self.att_weight.unsqueeze(1), agents_info.permute(1, 0, 2)).squeeze(1) # size: [batch, 28]
            obs_embedding = torch.cat([att_output, self_info], dim=1) # size: [batch, 37 + 28]
            return f.relu(self.att_output_encoder(obs_embedding))

    def get_mask(self):
        return self.att_weight