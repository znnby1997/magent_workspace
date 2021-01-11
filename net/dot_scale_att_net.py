import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class DotScaleAttNet(nn.Module):
    def __init__(self, obs_dim, output_dim, hidden_dim, agent_num, **kwargs):
        super(DotScaleAttNet, self).__init__()

        self.query_w = nn.Linear(36, hidden_dim, bias=False)
        self.key_w = nn.Linear(28, hidden_dim, bias=False)
        self.value_w = nn.Linear(28, hidden_dim, bias=False)

        self.output = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x, **kwargs):
        self_info = x[:, 0:36]
        other_info = x[:, 36:]
        agents_info = torch.stack(other_info.split(28, dim=1)) # size: [other agents num, batch, 28]

        query = self.query_w(self_info) # size: [batch, hidden_dim]
        keys = self.key_w(agents_info) # size: [other agents num, batch, hidden_dim]
        values = self.value_w(agents_info) # size: [other agents num, batch, hidden_dim]

        att_weight = f.softmax((torch.bmm(query.unsqueeze(1), keys.permute(1, 2, 0)) / np.sqrt(self.hidden_dim)), dim=2) # size: [batch, 1, other agents_num]

        att_output = torch.bmm(self.att_weight, values.permute(1, 0, 2)).squeeze(1) # size: [batch, hidden_dim]
        obs_embedding = torch.cat([att_output, query], dim=1) # size: [batch, 2 * hidden_dim]
        return self.output(obs_embedding), att_weight
        


    