import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class BasicNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=32, em_dim=32, **kwargs):
        super(BasicNet, self).__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.em_dim = em_dim
        self.att_weight = None

        # 基本的网络中包含一层MLP，用于处理第一层的输出embedding
        self.basic_mlp2 = nn.Linear(self.hidden_dim, self.n_actions)

        # 编码观测信息向量的时候用
        # self info: 37bits  opp info: 28bits  partner info: 28bits
        self.self_encoder = nn.Linear(37, em_dim)
        self.other_encoder = nn.Linear(28, em_dim)

    def forward(self):
        raise NotImplementedError

    def att_layer(self, x):
        # 该部分每个网络继承basic net时需要重写
        return

    def q(self, obs):
        att_output = self.att_layer(obs)
        return self.basic_mlp2(att_output)

    def get_cur_weight(self):
        return self.att_weight.detach().numpy()

