import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import sys
sys.path.append('..')
from net.basic_net import BasicNet

class GruGenAttNet(BasicNet):
    def __init__(self, obs_dim, n_actions, hidden_dim, em_dim=32, **kwargs):
        super().__init__(obs_dim, n_actions, hidden_dim, em_dim)
        # 新的基于gru的attention生成方式
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, bidirectional=True)
        self.mlp2 = nn.Linear(hidden_dim * 2, 1) # 直接输出一个对应的weight

    def att_layer(self, x):
        # 尝试自己想到的attention方式
        # 1.每个info向量进行编码
        self_info = x[:, 0:37]
        other_info = x[:, 37:]
        agents_info = other_info.split(28, dim=1)
        infos_encoder = []
        infos_encoder.append(self.self_encoder(self_info))
        for agent_info in agents_info:
            infos_encoder.append(f.relu(self.other_encoder(agent_info)))
        infos_encoder = torch.stack(infos_encoder) # size: [agent_num, batch, hidden]

        # 2.将所有的info通过双向的gru
        gru_output, _ = self.gru(infos_encoder) # gru output size: [agent_num, batch, hidden * 2]
        weights = []
        for output in gru_out:
            weights.append(torch.tanh(self.mlp2(output))) # weight size: [batch, 1]
        weights = torch.stack(weights).permute(1, 0, 2) # size: [batch, agent_num, 1]
        self.att_weight = f.softmax(weights, dim=1) # size: [batch, agent_num, 1]
        att_output = torch.bmm(self.att_weight.permute(0, 2, 1), infos_encoder.permute(1, 0, 2))
        return att_output.squeeze() # size : [batch, hidden]