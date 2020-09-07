import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import sys
sys.path.append('..')
from net.basic_net import BasicNet

class GruGenAttNet(BasicNet):
    def __init__(self, obs_dim, n_actions, hidden_dim, heads_num=4, em_dim=32, concatenation=False, **kwargs):
        super().__init__(obs_dim, n_actions, hidden_dim, em_dim)
        self.concatenation = concatenation
        # 新的基于gru的attention生成方式
        self.gru = nn.GRU(input_size=em_dim, hidden_size=em_dim, bidirectional=True)

        # 转换参数,训练获得
        self.trans_params = nn.Linear(2 * em_dim, hidden_dim)
        # self.trans_params = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 2))
        # 多头参数,训练多头attention
        self.multiheads_params = nn.Linear(hidden_dim, heads_num)
        # self.multiheads_params = nn.Parameter(torch.Tensor(heads_num, hidden_dim))

        self.align_layer = nn.Linear(heads_num * em_dim * 2, hidden_dim)

    def att_layer(self, x, greedy_group=False):
        # 1.每个info向量进行编码
        self_info = x[:, 0:37]
        other_info = x[:, 37:]
        agents_info = torch.stack(other_info.split(28, dim=1)) # size: [other agents num, batch, 28]
        other_embedding = f.relu(self.other_encoder(agents_info)) # size: [other agents num, batch, em_dim]
        self_embedding = f.relu(self.self_encoder(self_info)) # size: [batch, em_dim]
        encodings = torch.cat([self_embedding.unsqueeze(0), other_embedding], dim=0) # size: [agents num, batch, em_dim]
        # 2.将所有的info通过双向的gru
        gru_output, _ = self.gru(encodings) # gru output size: [agent_num, batch, em_dim * 2]
        h = gru_output.permute(1, 0, 2) # size: [batch, agent_num, em_dim*2]
        self.att_weight = f.softmax(
            self.multiheads_params(
                torch.tanh(self.trans_params(h))), dim=2).permute(0, 2, 1)  # size: [batch, head_num, agent_num]

        batch = self_info.shape[0]
        if self.concatenation:
            # 把多头attention weight取平均值看看
            self.att_weight = torch.mean(self.att_weight, dim=1) # size: [batch, agent_num]
            weights = self.att_weight.split(1, dim=1) # 每个的size: [batch, 1]
            self_we = self_info * weights[0] # size: [batch, 37]

            other_w = torch.stack(weights[1:]) # size: [other agents num, batch, 1]
            other_we = (agents_info * other_w).permute(1, 0, 2).reshape(batch, -1) # size: [batch, other agent num * 28]
            obs_embedding = torch.cat([self_we, other_we], dim=1) # size: [batch, obs_dim]
            return f.relu(self.basic_mlp1(obs_embedding)) # size: [batch, hidden]
        else:
            multi_output = torch.bmm(self.att_weight, h) # size: [batch, head_num, em_dim*2]
            # 多头拼接
            att_output = multi_output.view(multi_output.shape[0], -1) # size: [batch, head_num * em_dim * 2]
            return  self.align_layer(att_output) # size : [batch, hidden]

    def get_mask(self):
        return self.att_weight.numpy()


class GruGenAttNetNew(BasicNet):
    """这个类用于解决之前提到的GRU attention weight上自身信息和其他信息
        信息量不同的从而双向GRU存在不合理因素的问题(目前思考的解决方式是
        其他信息进入双向GRU，自身信息作为一个上下文来用，相当于query)
    """
    def __init__(self, obs_dim, n_actions, hidden_dim, heads_num=4, em_dim=32, concatenation=False, **kwargs):
        super().__init__(obs_dim, n_actions, hidden_dim, em_dim)
        self.concatenation = concatenation
        # gru中只输入其他信息
        self.gru = nn.GRU(input_size=em_dim, hidden_size=em_dim, bidirectional=True)

        # other info 's trans
        self.trans_params = nn.Linear(2 * em_dim, em_dim)

        self.align_layer = nn.Linear(2 * em_dim, hidden_dim)

    def att_layer(self, x, greedy_group=False):
        # 编码过程不变
        self_info = x[:, 0:37]
        other_info = x[:, 37:]
        agents_info = torch.stack(other_info.split(28, dim=1)) # size: [other agents num, batch, 28]
        other_embedding = f.relu(self.other_encoder(agents_info)) # size: [other agents num, batch, em_dim]
        self_embedding = f.relu(self.self_encoder(self_info)) # size: [batch, em_dim]

        # other embedding 送入GRU
        gru_output, _ = self.gru(other_embedding) # gru output size: [other agent num, batch, 2 * em_dim]
        other_u = torch.tanh(self.trans_params(gru_output)) # size: [other agent num, batch, em_dim]
        self.att_weight = f.softmax(torch.bmm(self_embedding.unsqueeze(1), other_u.permute(1, 2, 0)), dim=2) # size: [batch, 1, other num]
        batch = self_info.shape[0]
        # 拼接或者聚合
        if self.concatenation:
            weights = self.att_weight.squeeze(1).split(1, dim=1) # 每个的size: [batch, 1]
            other_w = torch.stack(weights) # size: [other agents num, batch, 1]
            other_we = (agents_info * other_w).permute(1, 0, 2).reshape(batch, -1) # size: [batch, other agent num * 28]
            obs_embedding = torch.cat([self_info, other_we], dim=1) # size: [batch, obs_dim]
            return f.relu(self.basic_mlp1(obs_embedding)) # size: [batch, hidden]
        else:
            other_e = torch.bmm(self.att_weight, other_embedding.permute(1, 0, 2)).squeeze(1) # size: [batch, em_dim]
            embedding = torch.cat([other_e, self_embedding], dim=1) # size: [batch, 2 * em_dim]
            return f.relu(self.align_layer(embedding)) # size: [batch, hidden]


    def get_mask(self):
        return self.att_weight