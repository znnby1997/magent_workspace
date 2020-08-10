import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import sys
sys.path.append('..')
from net.basic_net import BasicNet

class Dyan(BasicNet):
    def __init__(self, obs_dim, n_actions, hidden_dim, agent_num, em_dim=32, aggregate_form='mean', **kwargs):
        super().__init__(obs_dim, n_actions, hidden_dim, em_dim)
        self.feature_encoder = nn.Linear(28, hidden_dim)
        self.self_encoder = nn.Linear(37, hidden_dim)
        self.align_embedding = nn.Linear(3 * hidden_dim, hidden_dim)
        self.aggregate_form = aggregate_form
        self.agent_num = agent_num
        # self.output_q = nn.Linear(hidden_dim, n_actions)

    def att_layer(self, x):
        # DyAN聚合方式, 所有的对手观测聚合, 所有的队友观测聚合, 最后拼接三个信息
        self_info = x[:, 0:37]
        opp_index = 37 + 28 * self.agent_num
        opps_info = x[:, 37:opp_index].split(28, dim=1) # 20 agents
        partners_info = x[:, opp_index:].split(28, dim=1) # 19 agents
        opps_encoder = []
        partners_encoder = []
        for opp_info in opps_info:
            opps_encoder.append(f.relu(self.feature_encoder(opp_info)))
        for partner_info in partners_info:
            partners_encoder.append(f.relu(self.feature_encoder(partner_info)))
        opps_embedding = torch.stack(opps_encoder) # size: [agent_num, batch, hidden]
        partners_embedding = torch.stack(partners_encoder) # size: [agent_num, batch, hidden]
        aggregate_opp = None
        aggregate_partner = None
        if self.aggregate_form == 'mean':
            aggregate_opp = torch.mean(opps_embedding, dim=0)
            aggregate_partner = torch.mean(partners_embedding, dim=0)
        elif self.aggregate_form == 'sum':
            aggregate_opp = torch.sum(opps_embedding, dim=0)
            aggregate_partner = torch.sum(partners_embedding, dim=0)
        elif self.aggregate_form == 'max':
            aggregate_opp = torch.max(opps_embedding, dim=0)[0] # size: [batch, hidden] sum/mean/max
            aggregate_partner = torch.max(partners_embedding, dim=0)[0] # size: [batch, hidden]
        self_embedding = f.relu(self.self_encoder(self_info)) # size: [batch, hidden]
        embedding = torch.cat([aggregate_opp, aggregate_partner, self_embedding], dim=1) # size: [batch, 3 * hidden]
        att_output = f.relu(self.align_embedding(embedding)) # size: [batch, hidden]
        return att_output