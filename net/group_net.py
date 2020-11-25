import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class GroupNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim, group_num, **kwargs):
        super(GroupNet, self).__init__()

        self.ig_num = group_num
        self.hidden_dim = hidden_dim

        self.self_encoder = nn.Linear(36, hidden_dim)
        self.other_encoder = nn.Linear(28, hidden_dim)

        self.w = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.FloatTensor(hidden_dim))

        self.i_group_em = nn.Linear(group_num * 28, hidden_dim)
        # self.u_group_em = nn.Linear(28, hidden_dim)

        self.align_layer = nn.Linear(3 * hidden_dim, hidden_dim)

    def forward(self, x, **kwargs):
        self_info = x[:, 0:36]
        origional_other_info = x[:, 36:]
        batch = self_info.shape[0]

        other_info = torch.stack(origional_other_info.split(28, dim=1)).permute(1, 0, 2) # size: [batch, other_num, 28]
        other_num = other_info.shape[1]
        other_es = f.relu(self.other_encoder(other_info)) # size: [batch, other_num, hidden_dim]
        self_e = f.relu(self.self_encoder(self_info)) # size: [batch, hidden_dim]

        # group weight estimate for other info
        att = torch.tanh(self.w(torch.cat([other_es, self_e.unsqueeze(1).repeat(1, other_num, 1)], dim=2))) # size: [batch, other_num, hidden_dim]
        # self.v = self.v.unsqueeze(0).unsqueeze(2) # size: [1, hidden_dim, 1]
        weights = f.softmax(torch.matmul(att, self.v.unsqueeze(0).unsqueeze(2)), dim=1).squeeze(2) # size: [batch, other_num]

        # info process for important group and unimportant group
        i_g, i_w, u_g, u_w = self.group_mask_with_weight(weights, other_info, other_es, self.ig_num)
        # aggregate with weights in important group
        i_e = f.relu(self.i_group_em((i_g * i_w).reshape(batch, -1))) # size: [batch, hidden_dim]
        # concat with weights in unimportant group
        u_e = torch.bmm(u_w, u_g).squeeze(1) # size: [batch, hidden_dim]

        # info integrate
        obs_e = self.align_layer(torch.cat([self_e, i_e, u_e], dim=1))
        return obs_e, weights


    @staticmethod
    def group_mask_with_weight(weights, info, encoding_info, ig_num):
        """
            info is origional observation for other agents
            the shape of i_w: [batch, ig_num, 28] used to concat
            the shape of u_w: [batch, 1, uig_num] used to aggregate
        """
        

        ori_dim = info.shape[2]
        en_dim = encoding_info.shape[2]
        sort_weights, idx = weights.sort(dim=1, descending=True) # size: [batch, other_num]
        i_w, i_idx, u_w, u_idx = sort_weights[:, 0:ig_num].unsqueeze(2).repeat(1, 1, ori_dim), idx[:, 0:ig_num].unsqueeze(2).repeat(1, 1, ori_dim), \
                                    sort_weights[:, ig_num:].unsqueeze(1), idx[:, ig_num:].unsqueeze(2).repeat(1, 1, en_dim)
        
        i_group = torch.gather(info, 1, i_idx.long()) # size: [batch, ig_num, 28]
        u_group = torch.gather(encoding_info, 1, u_idx.long()) # size: [batch, uig_num, en_dim]

        return i_group, i_w, u_group, u_w

        
