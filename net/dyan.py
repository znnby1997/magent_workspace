import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class Dyan(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim, agent_num, aggregate_form='mean', **kwargs):
        super(Dyan, self).__init__()
        self.feature_encoder = nn.Linear(28, hidden_dim)
        self.self_encoder = nn.Linear(36, hidden_dim)
        self.align_embedding = nn.Linear(3 * hidden_dim, hidden_dim)
        self.aggregate_form = aggregate_form
        self.agent_num = agent_num
        # self.output_q = nn.Linear(hidden_dim, n_actions)
    
    def forward(self, x, **kwargs):
        self_info = x[:, 0:36]
        opp_index = 36 + 28 * self.agent_num
        opps_info = torch.stack(x[:, 36:opp_index].split(28, dim=1)) # 20 agents size: [20, batch, 28]
        partners_info = torch.stack(x[:, opp_index:].split(28, dim=1)) # 19 agents size: [19, batch, 28]
        self_embedding = f.relu(self.self_encoder(self_info)) # size: [batch, hidden_dim]
        opps_embedding = f.relu(self.feature_encoder(opps_info)) # size: [20, batch, hidden_dim]
        partners_embedding = f.relu(self.feature_encoder(partners_info)) # size: [19, batch, hidden_dim]
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
        
        embedding = torch.cat([aggregate_opp, aggregate_partner, self_embedding], dim=1) # size: [batch, 3 * hidden]
        return self.align_embedding(embedding)