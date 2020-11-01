import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class GAA(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim, agent_num, group_num, aggregate_form='mean', **kwargs):
        super(GAA, self).__init__()
        self.align_embedding = nn.Linear(2 * hidden_dim, hidden_dim)

        self.fc_group_layer = nn.Linear(28, group_num) # 直接在原始观测上进行分组

        self.obs_encoder = nn.Linear(28, hidden_dim)
        self.self_info_encoder = nn.Linear(36, hidden_dim)

        self.trans_w1 = nn.Linear(hidden_dim, hidden_dim)
        self.trans_w2 = nn.Linear(hidden_dim, 1)
        
        self.aggregate_form = aggregate_form
        self.agent_num = agent_num
        self.group_num = group_num
        
        self.group_mask = None

    def forward(self, x, detach=False, **kwargs):
        self_info = x[:, 0:36]
        self_e = f.relu(self.self_info_encoder(self_info))
        other_info = x[:, 36:]
        agents_info = torch.stack(other_info.split(28, dim=1)) # size: [other agents num, batch, 28]
        info_encoders = f.relu(self.obs_encoder(agents_info)) # size: [other agents num, batch, hidden_dim]
        
        group_output = self.fc_group_layer(agents_info) # size: [other agents num, batch, group_num]
        if detach:
            group_probs = f.softmax(group_output, dim=2)
            self.group_mask = self.get_greedy_group(group_probs)
        else:
            self.group_mask = f.gumbel_softmax(group_probs, 1.0, hard=True, dim=2) # size: [other agents num, batch, group num] one-hot

        # 组内聚合
        group_es = torch.bmm(self.group_mask.permute(1, 2, 0), info_encoders.permute(1, 0, 2)) # size:[batch, group, hidden_dim]
        group_att = f.softmax(self.trans_w2(torch.tanh(self.trans_w1(group_es))), dim=1) # size: [batch, group, 1]
        other_embedding = torch.bmm(group_att.permute(0, 2, 1), group_es).squeeze(1) # size: [batch, hidden_dim]
        obs_embedding = torch.cat([other_embedding, self_e], dim=1) # size: [batch, hidden_dim * 2]
        return self.align_embedding(obs_embedding), (self.group_mask, group_att)
    
    @staticmethod
    def get_greedy_group(group_probs):
        max_val, max_idx = torch.max(group_probs, dim=2)
        mask = torch.zeros(group_probs.shape, dtype=torch.float).cuda()
        return mask.scatter_(2, index=max_idx.unsqueeze(2).long(), value=1)

    def get_mask(self):
        return self.group_mask


class GroupNet(nn.Module):
    """
        大致过程：
            1.分组：输出分组矩阵(目前想到的分组算法，有监督的knn以及无监督的kmeans)
            2.learning的过程实际上是决定哪些group的信息不需要reduce(需要重点关注)，哪些信息需要reduce(知道个大概即可)
            3.最后应该是group拼接(包括自身信息)
    """
    def __init__(self, obs_dim, n_actions, hidden_dim, agent_num, group_num, aggregate_form='mean', **kwargs):
        super(GroupNet, self).__init__()

        self.i_group_num = group_num[0]
        self.o_group_num = group_num[1]
        self.group_num = self.i_group_num + self.o_group_num

        self.group_layer = nn.Sequential(
            nn.Linear(28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.group_num)
        )

        self.other_info_num = 2 * agent_num - 1 + 3
        self.align_layer = nn.Linear(36 + self.i_group_num * self.other_info_num * 28 + self.o_group_num * 28, hidden_dim)

    """
        B - batch_size, n1 - important group num, n2 - other group num, N - other agent num, H - hidden_dim
        other_agent_info: o_matrix(size: [B, N, H])
        实现过程:
            1.分组矩阵G，目前最简单的方式是直接将观测输入网络中输出一个group mask, 划分为igroup_mask(size: [B, n1, N]), ogroup_mask(size: [B, n2, N])
            2.聚合other groups内的观测信息: o_group_e = ogroup_mask * o_matrix (size: [B, n2, H]) (矩阵乘法)
            3.提取important groups内的观测信息: i_group_info = igroup_mask(size: [B, n1, N, 1]) * o_matrix(repeat, size: [B, n1, N, H]) (哈达玛积)(size: [B, n1, N, H])
            4.拼接所有group的features reshape(batch, -1)
    """
    def forward(self, x, detach=False, group_matrix=None, **kwargs):
        self_info = x[:, 0:36]
        other_info = x[:, 36:]
        batch = self_info.shape[0]
        o_matrix = torch.stack(other_info.split(28, dim=1)).permute(1, 0, 2) # size: [batch, other agents num, 28]

        # step 1
        split_policy = [self.i_group_num, self.o_group_num]
        group_masks = self.get_group(o_matrix, detach).split(split_policy, dim=1)
        igroup_mask = group_masks[0] # size: [batch, igroup num, other agents num]
        ogroup_mask = group_masks[1] # size: [batch, ogroup num, other agents num]

        # step 2
        ogroup_em = torch.bmm(ogroup_mask, o_matrix) # size:[batch, ogroup num, 28]

        # step 3
        igroup_em = igroup_mask.unsqueeze(3) * (o_matrix.unsqueeze(1).repeat(1, self.i_group_num, 1, 1)) # size: [batch, igroup num, other agents num, 28]

        # step 4
        em = torch.cat([self_info, igroup_em.reshape(batch, -1), ogroup_em.reshape(batch, -1)], dim=1) # size: [batch, 36 + igroup num * other agents num * 28 + ogroup num * 28]
        return self.align_layer(em), group_masks

    def get_group(self, o, detach):
        group_output = self.group_layer(o) # size: [batch, agent num, group num]
        if detach:
            # 采样
            group_probs = f.softmax(group_output, dim=2)
            group_mask = self.get_greedy_group(group_output).permute(0, 2, 1)
        else:
            # gumbel trick
            group_mask = f.gumbel_softmax(group_output, 1.0, hard=True, dim=2).permute(0, 2, 1)
        return group_mask # size: [batch, group num, agent_num] one-hot
    
    @staticmethod
    def get_greedy_group(group_probs):
        max_val, max_idx = torch.max(group_probs, dim=2)
        mask = torch.zeros(group_probs.shape, dtype=torch.float).cuda()
        return mask.scatter_(2, index=max_idx.unsqueeze(2).long(), value=1)

        




