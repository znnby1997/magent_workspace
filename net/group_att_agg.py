import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import sys
sys.path.append('..')
from net.basic_net import BasicNet
import utils.misc as misc

class GAA(BasicNet):
    def __init__(self, obs_dim, n_actions, hidden_dim, agent_num, group_num, em_dim=32, aggregate_form='mean', **kwargs):
        super().__init__(obs_dim, n_actions, hidden_dim, em_dim)
        self.align_embedding = nn.Linear(3 * em_dim, hidden_dim)

        self.gru = nn.GRU(input_size=em_dim, hidden_size=em_dim, bidirectional=True)
        self.group_layer = nn.Linear(2 * em_dim, group_num)

        self.fc_group_layer = nn.Linear(28, group_num) # 直接在原始观测上进行分组

        self.trans_w1 = nn.Linear(em_dim, em_dim)
        self.trans_w2 = nn.Linear(em_dim, 1)
        
        self.aggregate_form = aggregate_form
        self.agent_num = agent_num
        self.group_num = group_num
        self.tau = 0.0
        
        self.group_mask = None
    
    @staticmethod
    def get_greedy_group(group_probs):
        max_val, max_idx = torch.max(group_probs, dim=2)
        mask = torch.zeros(group_probs.shape, dtype=torch.float).cuda()
        return mask.scatter_(2, index=max_idx.unsqueeze(2).long(), value=1)

    def att_layer(self, x, greedy_group=False):
        # 1.每个info向量进行编码
        # self_info = x[:, 0:37]
        # other_info = x[:, 37:]
        # agents_info = torch.stack(other_info.split(28, dim=1)) # size: [other agents num, batch, 28]
        # other_embedding = f.relu(self.other_encoder(agents_info)) # size: [other agents num, batch, em_dim]
        # self_embedding = f.relu(self.self_encoder(self_info)) # size: [batch, em_dim]
        # encodings = torch.cat([self_embedding.unsqueeze(0), other_embedding], dim=0) # size: [agents num, batch, em_dim]
        self_info = x[:, 0:37]
        opp_index = 37 + 28 * self.agent_num
        opps_info = torch.stack(x[:, 37:opp_index].split(28, dim=1)) # 20 agents size: [20, batch, 28]
        partners_info = torch.stack(x[:, opp_index:].split(28, dim=1)) # 19 agents size: [19, batch, 28]
        self_embedding = f.relu(self.self_encoder(self_info)) # size: [batch, em_dim]
        opps_embedding = f.relu(self.other_encoder(opps_info)) # size: [20, batch, em_dim]
        partners_embedding = f.relu(self.other_encoder(partners_info)) # size: [19, batch, em_dim]

        # 生成分组分布并采样
        opp_group_mask = None
        par_group_mask = None
        if not greedy_group:
            opp_group_mask = f.gumbel_softmax(self.fc_group_layer(opps_info), 1.0, hard=True, dim=2) # size: [20, batch, group num] one-hot
            par_group_mask = f.gumbel_softmax(self.fc_group_layer(partners_info), 1.0, hard=True, dim=2)
        else:
            opp_group_probs = f.softmax(self.fc_group_layer(opps_info), dim=2)
            par_group_probs = f.softmax(self.fc_group_layer(partners_info), dim=2)
            opp_group_mask = self.get_greedy_group(opp_group_probs)
            par_group_mask = self.get_greedy_group(par_group_probs)

        self.group_mask = (opp_group_mask.squeeze(), par_group_mask.squeeze())
            
        # 组内聚合(这里分组聚合应该使用原始观测)
        opp_group_embeddings = torch.bmm(opp_group_mask.permute(1, 2, 0), opps_embedding.permute(1, 0, 2)) # size:[batch, group, em_dim]
        par_group_embeddings = torch.bmm(par_group_mask.permute(1, 2, 0), partners_embedding.permute(1, 0, 2))
        opp_group_att = f.softmax(self.trans_w2(torch.tanh(self.trans_w1(opp_group_embeddings))), dim=1) # size: [batch, group, 1]
        par_group_att = f.softmax(self.trans_w2(torch.tanh(self.trans_w1(par_group_embeddings))), dim=1) # size: [batch, group, 1]
        
        self.att_weight = (opp_group_att.squeeze(), par_group_att.squeeze())
        
        opp_embedding = torch.bmm(opp_group_att.permute(0, 2, 1), opp_group_embeddings).squeeze(1) # size: [batch, em_dim]
        par_embedding = torch.bmm(par_group_att.permute(0, 2, 1), par_group_embeddings).squeeze(1)
        embedding = torch.cat([opp_embedding, par_embedding, self_embedding], dim=1) # size: [batch, 3 * em_dim]
        # 2.对每个other agent观测信息进行分组并依据attention进行聚合
        # gru_output, _ = self.gru(encodings) # gru_output size: [agent_num, batch, em_dim*2]
        # att_logit = None
        # if not self.group_greedy:
        #     # 下面这两步是有问题的，因为两次的gumbel sample是不同的 ????????
        #     self.tau -= 0.
        #     group_mask = f.gumbel_softmax(self.group_layer(gru_output[1:]), 1.0, hard=True, dim=2) # size: [other agent_num, batch, group_num] dim=2 is one-hot
        #     # sample = f.gumbel_softmax(self.group_layer(gru_output[1:]), 1.0, hard=False, dim=2)
        #     # 这里使用分入一组的sample值作为权重logit
        #     # att_logit = sample * group_mask # size: [other agent_num, batch, group_num]
        #     att_logit = group_mask
        # else:
        #     group_probs = f.softmax(self.group_layer(gru_output[1:]), dim=2) # size: [other agent_num, batch, group_num]
        #     max_val, max_idx = torch.max(group_probs, dim=2)
        #     mask = torch.zeros(group_probs.shape, dtype=torch.float).cuda()
        #     att_logit = mask.scatter_(2, index=max_idx.unsqueeze(2).long(), src=max_val.unsqueeze(2).float())
        # agent_level_att = f.softmax(att_logit, dim=0)
        # 组内聚合
        # group_level_embeddings = torch.bmm(att_logit.permute(1, 2, 0), other_embedding.permute(1, 0, 2)) # size: [batch, group, em_dim]
        # group_level_att = f.softmax(self.trans_w2(torch.tanh(self.trans_w1(group_level_embeddings))), dim=1) # size: [batch, group, 1]
        # other_embeddings = torch.bmm(group_level_att.permute(0, 2, 1), group_level_embeddings).squeeze(1) # size: [batch, em_dim]

        # self.att_weight = (agent_level_att, group_level_att)

        # 3.将self info以及其他agent的观测聚合信息拼接
        # embedding = torch.cat([self_embedding, other_embeddings], dim=1) # size: [batch, em_dim * 2]
        return f.relu(self.align_embedding(embedding)) # size: [batch, hidden]

    def get_mask(self):
        return self.group_mask