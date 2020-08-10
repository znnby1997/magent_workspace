import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import sys
sys.path.append('..')
from net.basic_net import BasicNet

class GruGenAttNet(BasicNet):
    def __init__(self, obs_dim, n_actions, hidden_dim, heads_num=4, em_dim=32, **kwargs):
        super().__init__(obs_dim, n_actions, hidden_dim, em_dim)
        # 新的基于gru的attention生成方式
        self.gru = nn.GRU(input_size=em_dim, hidden_size=em_dim, bidirectional=True)

        # 转换参数,训练获得
        self.trans_params = nn.Linear(2 * em_dim, em_dim)
        # self.trans_params = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 2))
        # 多头参数,训练多头attention
        self.multiheads_params = nn.Linear(em_dim, heads_num)
        # self.multiheads_params = nn.Parameter(torch.Tensor(heads_num, hidden_dim))

        self.align_layer = nn.Linear(heads_num * em_dim * 2, hidden_dim)

    def att_layer(self, x):
        # 1.每个info向量进行编码
        self_info = x[:, 0:37]
        other_info = x[:, 37:]
        agents_info = other_info.split(28, dim=1)
        infos_encoder = []
        infos_encoder.append(self.self_encoder(self_info))
        for agent_info in agents_info:
            infos_encoder.append(f.relu(self.other_encoder(agent_info)))
        infos_encoder = torch.stack(infos_encoder) # size: [agent_num, batch, em_dim]

        # 2.将所有的info通过双向的gru
        gru_output, _ = self.gru(infos_encoder) # gru output size: [agent_num, batch, em_dim * 2]
        h = gru_output.permute(1, 0, 2) # size: [batch, agent_num, em_dim*2]
        self.att_weight = f.softmax(
            self.multiheads_params(
                torch.tanh(self.trans_params(h))), dim=2).permute(0, 2, 1)  # size: [batch, head_num, agent_num]
        multi_output = torch.bmm(self.att_weight, h) # size: [batch, head_num, em_dim*2]
        # 多头拼接
        att_output = multi_output.view(multi_output.shape[0], -1) # size: [batch, head_num * em_dim * 2]
        return  self.align_layer(att_output) # size : [batch, hidden]