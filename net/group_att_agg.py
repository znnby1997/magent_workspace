import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import sys
sys.path.append('..')
from net.basic_net import BasicNet
import utils.misc as misc

class GAA(BasicNet):
    def __init__(self, obs_dim, n_actions, hidden_dim, agent_num, group_num, group_greedy, em_dim=32, aggregate_form='mean', **kwargs):
        super().__init__(obs_dim, n_actions, hidden_dim, em_dim)
        self.align_embedding = nn.Linear(2 * em_dim, hidden_dim)

        self.gru = nn.GRU(input_size=em_dim, hidden_size=em_dim, bidirectional=True)
        self.group_layer = nn.Linear(2 * em_dim, group_num)

        self.trans_w1 = nn.Linear(em_dim, em_dim)
        self.trans_w2 = nn.Linear(em_dim, 1)
        
        self.aggregate_form = aggregate_form
        self.agent_num = agent_num
        self.group_num = group_num
        self.group_greedy = group_greedy
        self.tau = 0.0

    def att_layer(self, x):
        # 1.每个info向量进行编码
        self_info = x[:, 0:37]
        other_info = x[:, 37:]
        agents_info = torch.stack(other_info.split(28, dim=1)) # size: [other agents num, batch, 28]
        other_embedding = f.relu(self.other_encoder(agents_info)) # size: [other agents num, batch, em_dim]
        self_embedding = f.relu(self.self_encoder(self_info)) # size: [batch, em_dim]
        encodings = torch.cat([self_embedding.unsqueeze(0), other_embedding], dim=0) # size: [agents num, batch, em_dim]

        # 2.对每个other agent观测信息进行分组并依据attention进行聚合
        gru_output, _ = self.gru(encodings) # gru_output size: [agent_num, batch, em_dim*2]
        att_logit = None
        if not self.group_greedy:
            # 下面这两步是有问题的，因为两次的gumbel sample是不同的 ????????
            self.tau -= 0.
            group_mask = f.gumbel_softmax(self.group_layer(gru_output[1:]), 1.0, hard=True, dim=2) # size: [other agent_num, batch, group_num] dim=2 is one-hot
            # sample = f.gumbel_softmax(self.group_layer(gru_output[1:]), 1.0, hard=False, dim=2)
            # 这里使用分入一组的sample值作为权重logit
            # att_logit = sample * group_mask # size: [other agent_num, batch, group_num]
            att_logit = group_mask
        else:
            group_probs = f.softmax(self.group_layer(gru_output[1:]), dim=2) # size: [other agent_num, batch, group_num]
            max_val, max_idx = torch.max(group_probs, dim=2)
            mask = torch.zeros(group_probs.shape, dtype=torch.float).cuda()
            att_logit = torch.scatter_(2, index=max_idx.unsqueeze(2).long(), src=max_val.unsqueeze(2).float())
        # agent_level_att = f.softmax(att_logit, dim=0)
        # 组内聚合
        group_level_embeddings = torch.bmm(att_logit.permute(1, 2, 0), other_embedding.permute(1, 0, 2)) # size: [batch, group, em_dim]
        group_level_att = f.softmax(self.trans_w2(torch.tanh(self.trans_w1(group_level_embeddings))), dim=1) # size: [batch, group, 1]
        other_embeddings = torch.bmm(group_level_att.permute(0, 2, 1), group_level_embeddings).squeeze(1) # size: [batch, em_dim]

        # self.att_weight = (agent_level_att, group_level_att)

        # 3.将self info以及其他agent的观测聚合信息拼接
        embedding = torch.cat([self_embedding, other_embeddings], dim=1) # size: [batch, em_dim * 2]
        return self.align_embedding(embedding) # size: [batch, hidden]