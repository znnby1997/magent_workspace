import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import sys
sys.path.append('..')
from net.basic_net import BasicNet

class DyanGroup(BasicNet):
    def __init__(self, obs_dim, n_actions, hidden_dim, agent_num, em_dim=32, aggregate_form='mean', **kwargs):
        # super().__init__(obs_dim, n_actions, hidden_dim, em_dim)
        super(DyanGroup, self).__init__(obs_dim, n_actions, hidden_dim, em_dim)
        self.align_embedding = nn.Linear(4 * em_dim, hidden_dim)
        self.aggregate_form = aggregate_form
        self.agent_num = agent_num
        # self.att_weight = [0.9, 0.1, 0.05, 0.9, 0.1, 0.05]

    def get_group_mask(self, batch):
        group_mask = torch.zeros(4, self.agent_num * 2 - 1).cuda()
        cur_idx = 0
        for i in range(4):
            for j in range(10):
                if cur_idx < self.agent_num * 2 - 1:
                    group_mask[i][cur_idx] = 1.0 - (i + cur_idx) * 0.02438
                    cur_idx += 1
                else:
                    break
        return group_mask.repeat(batch, 1, 1)

    def att_layer(self, x, greedy_group=False):
        # 输入的观测已经按距离排好序了(由近及远)
        # agent观测信息,按距离由近及远分,一个组5个信息向量
        # 1.为每个info_vector编码
        self_info = x[:, 0:37]
        other_info = x[:, 37:]
        agents_info = torch.stack(other_info.split(28, dim=1)) # size: [other agents num, batch, 28]
        other_embedding = f.relu(self.other_encoder(agents_info)) # size: [other agents num, batch, em_dim]
        self_embedding = f.relu(self.self_encoder(self_info)) # size: [batch, em_dim]
        # 2.分组实际上就是定义一个mask矩阵，矩阵的size [batch, group_num, other agents_num]
        group_mask = self.get_group_mask(x.shape[0])
        # 3.分组聚合
        group_embeddings = torch.bmm(group_mask, other_embedding.permute(1, 0, 2)) # size: [batch, group_num, em_dim]
        # 4.组间拼接
        embeddings = group_embeddings.view(group_embeddings.shape[0], -1) # size: [batch, group_num * em_dim]
        att_output = f.relu(self.align_embedding(embeddings)) # size: [batch, hidden]
        return att_output
            

    def get_info_vectors(self, x):
        opp_index = 37 + 28 * self.agent_num
        opps_info = x[:, 37:opp_index].split(28, dim=1) # 20 agents
        partners_info = x[:, opp_index:].split(28, dim=1) # 19 agents
        
        groups = []
        groups.append(torch.stack(opps_info[:10])) # size: 10 agents * batch * 28
        groups.append(torch.stack(opps_info[10:])) # size: 10 agents * batch * 28
        
        groups.append(torch.stack(partners_info[:10])) # size: 10 agents * batch * 28
        groups.append(torch.stack(partners_info[10:])) # size: 9 agents * batch * 28

        return opp_groups, partner_groups

    def aggregate_vector(self, groups):
        # attention weights: 按距离分[0.9, 0.1, 0.05, 0.9, 0.1, 0.05]
        groups_embedding = []
        for g in groups:
            encoder = f.relu(self.feature_encoder(g)) # size: 5 * batch * hidden_dim
            aggregate_embedding = None
            if self.aggregate_form == 'mean':
                aggregate_embedding = torch.mean(encoder, dim=0)
            elif self.aggregate_form == 'sum':
                aggregate_embedding = torch.sum(encoder, dim=0)
            elif self.aggregate_form == 'max':
                aggregate_embedding = torch.max(encoder, dim=0)[0] # size: batch * hidden_dim
            
            groups_embedding.append(aggregate_embedding)
        return groups_embedding
            



        