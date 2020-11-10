import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class HandProcess(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim, agent_num, concatenation=True, **kwargs):
        super(HandProcess, self).__init__()
        self.concatenation = concatenation
        self.agent_num = agent_num
        if self.concatenation:
            self.layer = nn.Linear(obs_dim, hidden_dim)
        else:
            self.layer = nn.Linear(36 + 28, hidden_dim)

    
    def forward(self, x, **kwargs):
        self_info = x[:, 0:36] # size: [batch, 36]
        batch = self_info.shape[0]
        agents_index = 36 + 28 * (self.agent_num * 2 - 1)
        agents_info = torch.stack(x[:, 36:agents_index].split(28, dim=1)).permute(1, 0, 2) # size: [batch, number, 28]
        noisy_info = x[:, agents_index:]

        self.att_weight = self.hand_weight(agents_info) # size: [batch, number]

        if self.concatenation:
            agents_info = (agents_info * self.att_weight.unsqueeze(2)).reshape(batch, -1)
            noisy_info = noisy_info * 0.0
            obs = torch.cat([self_info, agents_info, noisy_info], dim=1)
            return self.layer(obs)
        else:
            # 噪声不需要加入了, 因为weight=0
            self.att_weight = self.att_weight.unsqueeze(1)
            other_info_es = torch.bmm(self.att_weight, agents_info).squeeze(1) # size: [batch, 28]
            obs = torch.cat([self_info, other_info_es], dim=1) # size: [batch, 36 + 28]
            return self.layer(obs)

    
    @staticmethod
    def hand_weight(agents_info, bench=1, tau=2):
        """obs提取距离, 输入size: [batch, agent number, 28]
           first bit: hp
           second bit: group id
           13 bits: x
           13 bits: y
        """
        loc_list = agents_info[:, :, 2:].split(13, dim=2) # per size: [batch, number, 13]
        x_set = loc_list[0].max(dim=2)[1] # size: [batch, number]
        y_set = loc_list[1].max(dim=2)[1] # size: [batch, number]
        xys = torch.stack([x_set, y_set]).permute(1, 2, 0).float() # size: [batch, number, 2]
        dis_set = torch.sum(torch.abs(6 - xys), dim=2) # size: [batch, number]
        att_weight = bench / (tau ** dis_set)
        return att_weight

    def get_weight(self):
        return self.att_weight.squeeze()


class HandProcessGroup(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim, agent_num, ig_num=5, **kwargs):
        super(HandProcessGroup, self).__init__()
        self.ig_num = ig_num

        self.unimportant_encoder = nn.Linear(28, hidden_dim)
        self.self_info_encoder = nn.Linear(36, hidden_dim)
        self.align_layer = nn.Linear(ig_num * 28 + 2 * hidden_dim, hidden_dim)
    
    def forward(self, x, **kwargs):
        self_info = f.relu(self.self_info_encoder(x[:, 0:36])) # size: [batch, hidden]
        batch = self_info.shape[0]
        agents_info = torch.stack(x[:, 36:].split(28, dim=1)).permute(1, 0, 2) # size: [batch, number, 28]

        i_group, u_group = self.hand_group(agents_info, self.ig_num) # per size: [batch, ig_num, 28]

        # important info concatenate
        i_group_e = i_group.reshape(batch, -1) # size: [batch, ig_num*28]
        # unimportant info aggregate
        u_group_e = f.relu(self.unimportant_encoder(u_group)).sum(dim=1) # size: [batch, hidden_dim]
        
        obs_e = torch.cat([i_group_e, self_info, u_group_e], dim=1) # size: [batch, ig_num * 28 + 2 * hidden_dim]
        return self.align_layer(obs_e), i_group

    @staticmethod
    def hand_group(agents_info, ig_num=3):
        """obs提取距离, 输入size: [batch, agent number, 28]
           first bit: hp
           second bit: group id
           13 bits: x
           13 bits: y
        """
        hidden_dim = agents_info.shape[2]
        loc_list = agents_info[:, :, 2:].split(13, dim=2) # per size: [batch, number, 13]
        x_set = loc_list[0].max(dim=2)[1] # size: [batch, number]
        y_set = loc_list[1].max(dim=2)[1] # size: [batch, number]
        xys = torch.stack([x_set, y_set]).permute(1, 2, 0).float() # size: [batch, number, 2]
        dis_set = torch.sum(torch.abs(6 - xys), dim=2) # size: [batch, number]

        important_degree, idx = torch.sort(dis_set, dim=1, descending=False)
        important_idx = idx[:, 0:ig_num].unsqueeze(2).repeat(1, 1, hidden_dim) # size: [batch, ig_num, 28]
        unimportant_idx = idx[:, ig_num:].unsqueeze(2).repeat(1, 1, hidden_dim) # size: [batch, agent_num - ig_num, 28]

        important_group = torch.gather(agents_info, 1, important_idx.long()) # size: [batch, ig_num, 28]
        unimportant_group = torch.gather(agents_info, 1, unimportant_idx.long()) # size: [batch, uig_num, 28]

        return important_group, unimportant_group



