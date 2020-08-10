import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import sys
sys.path.append('..')
from net.basic_net import BasicNet

class DyanGroup(BasicNet):
    def __init__(self, obs_dim, n_actions, hidden_dim, agent_num, em_dim=32, aggregate_form='mean', **kwargs):
        super().__init__(obs_dim, n_actions, hidden_dim, em_dim)
        self.feature_encoder = nn.Linear(28, hidden_dim)
        self.self_encoder = nn.Linear(37, hidden_dim)
        self.align_embedding = nn.Linear(2 * hidden_dim, hidden_dim)
        self.aggregate_form = aggregate_form
        self.agent_num = agent_num

    def att_layer(self, x):
        # 输入的观测已经按距离排好序了(由近及远)
        # agent观测信息,按距离由近及远分,一个组5个信息向量
        opp_groups, partner_groups = self.get_info_vectors(x)
        self_info = x[:, 0:37]
        opp_embeddings = self.aggregate_vector(opp_groups)
        partner_embeddings = self.aggregate_vector(partner_groups)
        self_embedding = f.relu(self.self_encoder(self_info)) # size: [batch, hidden]
        other_info = torch.stack(opp_embeddings + partner_embeddings) # size: [6 * batch * hidden]
        self.att_weight = torch.tensor([0.9, 0.1, 0.05, 0.9, 0.1, 0.05], dtype=torch.float).cuda().reshape(1, -1) # size: 1 * 6
        other_embedding = torch.matmul(self.att_weight, other_info.permute(1, 0, 2)).squeeze(1) # size: [batch, hidden]
        embedding = torch.cat([other_embedding, self_embedding], dim=1) # size: [batch, 2 * hidden]
        att_output = f.relu(self.align_embedding(embedding)) # size: [batch, hidden]
        return att_output
            

    def get_info_vectors(self, x):
        opp_index = 37 + 28 * self.agent_num
        opps_info = x[:, 37:opp_index].split(28, dim=1) # 20 agents
        partners_info = x[:, opp_index:].split(28, dim=1) # 19 agents
        
        opp_groups = []
        partner_groups = []
        opp_groups.append(torch.stack(opps_info[:5])) # size: 5 * batch * 28
        opp_groups.append(torch.stack(opps_info[5:10]))
        opp_groups.append(torch.stack(opps_info[10:]))
        
        partner_groups.append(torch.stack(partners_info[:5]))
        partner_groups.append(torch.stack(partners_info[5:10]))
        partner_groups.append(torch.stack(partners_info[10:]))
        return opp_groups, partner_groups

    def aggregate_vector(self, groups):
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
            



        