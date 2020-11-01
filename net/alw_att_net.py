import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class AlwAttNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim, agent_num, concatenation=False, nonlin='softmax', **kwargs):
        super(AlwAttNet, self).__init__()
        self.concatenation = concatenation  # 直接将得到的weight与观测对应元素相乘，而非聚合
        self.agent_num = agent_num
        self.att_weight = None
        self.nonlin = nonlin

        # agent level attention
        # self info: 36bits  opp info: 28bits  partner info: 28bits noisy info: 28bits(3)
        self.al_att_layer = nn.Linear(obs_dim, agent_num * 2 - 1)
        if self.concatenation:
            self.output = nn.Linear(obs_dim, hidden_dim)
        else:
            self.output = nn.Linear(36 + 28, hidden_dim)

    def forward(self, x, **kwargs):
        if self.nonlin == 'softmax':
            self.att_weight = f.softmax(self.al_att_layer(x), dim=1)
        elif self.nonlin == 'sigmoid':
            self.att_weight = torch.sigmoid(self.al_att_layer(x)) # size: [batch, other agent num + noisy]

        self_info = x[:, 0:36] # size: [batch, 37]
        other_info = x[:, 36:] # size: [batch, 39 * 28]
        agents_info = torch.stack(other_info.split(28, dim=1)) # size: [other agents num + noisy, batch, 28]

        if self.concatenation:
            batch = self_info.shape[0]
            weights = self.att_weight.split(1, dim=1) # 每个的size: [batch, 1]

            other_w = torch.stack(weights) # size: [other agents num + noisy, batch, 1]
            other_we = (agents_info * other_w).permute(1, 0, 2).reshape(batch, -1) # size: [batch, (other agent num + noisy) * 28]
            obs_embedding = torch.cat([other_we, self_info], dim=1) # size: [batch, obs_dim]
            return self.output(obs_embedding), self.att_weight # size: [batch, hidden]
        else:
            att_output = torch.bmm(self.att_weight.unsqueeze(1), agents_info.permute(1, 0, 2)).squeeze(1) # size: [batch, 28]
            obs_embedding = torch.cat([att_output, self_info], dim=1) # size: [batch, 37 + 28]
            return self.output(obs_embedding), self.att_weight

    def get_weight(self):
        return self.att_weight


class AlwGAT(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim, agent_num, **kwargs):
        super(AlwGAT, self).__init__()
        self.other_att_trans = nn.Linear(36 + 28, 1)
        self.other_e_trans = nn.Linear(36 + 28, hidden_dim)
        self.self_e_trans = nn.Linear(36 + 36, hidden_dim)
        self.output = nn.Linear(2 * hidden_dim, hidden_dim)
        self.att_weight = None
        self.agent_num = agent_num
    
    def forward(self, x, **kwargs):
        self_info = x[:, 0:36] # size: [batch, 36]
        other_info = x[:, 36:] # size: [batch, 39 + 3 * 28]
        other_list = other_info.split(28, dim=1)
        other_num = len(other_list)
        self_info_ext = self_info.unsqueeze(1).repeat(1, other_num, 1) # size: [batch, other agents num + noisy, 36]
        agents_info = torch.stack(other_info.split(28, dim=1)).permute(1, 0, 2) # size: [batch, other agents num + noisy, 28]
        other_os = torch.cat([self_info_ext, agents_info], dim=2) # size: [batch, other agents num, 36 + 28]

        # attention weight
        self.att_weight = f.softmax(self.other_att_trans(other_os), dim=1) # size: [batch, other agents num, 1]
        # other info embedding
        other_es = self.other_e_trans(other_os) # size: [batch, other agents num, hidden_dim]
        other_embedding = torch.bmm(self.att_weight.permute(0, 2, 1), other_es).squeeze(1) # size: [batch, hidden_dim]
        self_embedding = self.self_e_trans(torch.cat([self_info, self_info], dim=1)) # size: [batch, hidden]
        obs_embedding = torch.cat([other_embedding, self_embedding], dim=1) # size: [batch, 2 * hidden]
        return self.output(obs_embedding)

    def get_weight(self):
        return self.get_weight.squeeze()
    
