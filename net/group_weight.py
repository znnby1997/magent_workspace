import torch
import torch.nn as nn
import torch.nn.functional as f

class GroupWeight(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim, agent_num, **kwargs):
        super(GroupWeight, self).__init__()
        self.agent_num = agent_num
        self.att_trans = nn.Linear(28 + 36, 1)
        self.e_trans = nn.Linear(28 + 36, hidden_dim)
        self.self_e_trans = nn.Linear(36 + 36, hidden_dim)
        self.group_att = nn.Linear(3 * hidden_dim, 3)
        self.output = nn.Linear(2 * hidden_dim, hidden_dim)
        self.att_weights = None

    def forward(self, x, **kwargs):
        # 提取信息，分成三组，一组是opp,一组是partner,一组是noisy
        self_info = x[:, 0:36] # size: [batch, 36]
        opp_index = 36 + 28 * self.agent_num
        opp_info_list = x[:, 36:opp_index].split(28, dim=1)
        par_index = opp_index + 28 * (self.agent_num - 1)
        par_info_list = x[:, opp_index:par_index].split(28, dim=1)
        noisy_info_list = x[:, par_index:].split(28, dim=1)
        opp_info_len = len(opp_info_list)
        par_info_len = len(par_info_list)
        noisy_info_len = len(noisy_info_list)

        self_info_exts = [
            self_info.unsqueeze(1).repeat(1, opp_info_len, 1), 
            self_info.unsqueeze(1).repeat(1, par_info_len, 1),
            self_info.unsqueeze(1).repeat(1, noisy_info_len, 1)]
        opps_info = torch.stack(opp_info_list).permute(1, 0, 2) # 20 agents size: [batch, 20, 28]
        partners_info = torch.stack(par_info_list).permute(1, 0, 2) # 19 agents size: [batch, 19, 28]
        noisy_info = torch.stack(noisy_info_list).permute(1, 0, 2) # e noisy: [batch, 3, 28]

        opp_os = torch.cat([self_info_exts[0], opps_info], dim=2) # size: [batch, 20, 28 + 36]
        par_os = torch.cat([self_info_exts[1], partners_info], dim=2) # size: [batch, 19, 28 + 36]
        noisy_os = torch.cat([self_info_exts[2], noisy_info], dim=2) # size: [batch, 3, 28 + 36]

        # attention weight
        opp_att_weights = f.softmax(self.att_trans(opp_os), dim=1) # size: [batch, 20, 1]
        par_att_weights = f.softmax(self.att_trans(par_os), dim=1) # size: [batch, 21, 1]
        noisy_att_weights = f.softmax(self.att_trans(noisy_os), dim=1) # size: [batch, 3, 1]
        # 组内聚合
        opp_es = self.e_trans(opp_os) # size: [batch, 20, hidden]
        par_es = self.e_trans(par_os) # size: [batch, 21, hidden]
        noisy_es = self.e_trans(noisy_os) # size: [batch, 3, hidden]
        opp_embedding = torch.bmm(opp_att_weights.permute(0, 2, 1), opp_es).squeeze(1) # size: [batch, hidden]
        par_embedding = torch.bmm(par_att_weights.permute(0, 2, 1), par_es).squeeze(1) # size: [batch, hidden]
        noisy_embedding = torch.bmm(noisy_att_weights.permute(0, 2, 1), noisy_es).squeeze(1) # size: [batch, hidden]

        # 组间直接通过一个网络生成weight，没有再与self拼接
        group_e = torch.stack([opp_embedding, par_embedding, noisy_embedding]) # size: [3, batch, hidden]
        group_att_weight = f.softmax(self.group_att(torch.cat([opp_embedding, par_embedding, noisy_embedding], dim=1)), dim=1) # size: [batch, 3]
        other_embedding = torch.bmm(group_att_weight.unsqueeze(1), group_e.permute(1, 0, 2)).squeeze(1) # size: [batch, hidden]

        self.att_weights = (opp_att_weights.squeeze(), par_att_weights.squeeze(), group_att_weight.squeeze())
        # 自身信息编码
        self_o = torch.cat([self_info, self_info], dim=1) # size: [batch, 36 + 36]
        self_e = self.self_e_trans(self_o) # size: [batch, hidden]
        
        obs_embedding = torch.cat([other_embedding, self_e], dim=1) # size: [batch, 2 * hidden]
        return self.output(obs_embedding)

    def get_weight(self):
        return self.att_weights



