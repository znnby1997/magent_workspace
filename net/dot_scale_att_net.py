import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class DotScaleAttNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim, agent_num, concatenation=False, **kwargs):
        super(DotScaleAttNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.query_w = nn.Linear(36, hidden_dim, bias=False)
        self.key_w = nn.Linear(28, hidden_dim, bias=False)
        self.value_w = nn.Linear(28, hidden_dim, bias=False)

        self.concatenation = concatenation

        if self.concatenation:
            self.output = nn.Linear((agent_num * 2 + 3) * hidden_dim, hidden_dim)
        else:
            self.output = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, x, **kwargs):
        self_info = x[:, 0:36]
        other_info = x[:, 36:]
        agents_info = torch.stack(other_info.split(28, dim=1)) # size: [other agents num, batch, 28]

        query = self.query_w(self_info) # size: [batch, hidden_dim]
        keys = self.key_w(agents_info) # size: [other agents num, batch, hidden_dim]
        values = self.value_w(agents_info) # size: [other agents num, batch, hidden_dim]

        self.att_weight = f.softmax((torch.bmm(query.unsqueeze(1), keys.permute(1, 2, 0)) / np.sqrt(self.hidden_dim)), dim=2) # size: [batch, 1, other agents_num]

        if self.concatenation:
            batch = self_info.shape[0]
            self.att_weight = self.att_weight.squeeze(1) # size: [batch, agent_num]
            weights = self.att_weight.split(1, dim=1) # 每个的size: [batch, 1]

            other_w = torch.stack(weights) # size: [other agents num, batch, 1]
            other_we = (values * other_w).permute(1, 0, 2).reshape(batch, -1) # size: [batch, other agent num * hidden_dim]
            obs_embedding = torch.cat([other_we, query], dim=1) # size: [batch, agent num * hidden_dim]
            return self.output(obs_embedding), self.att_weight # size: [batch, hidden]
        else:
            att_output = torch.bmm(self.att_weight, values.permute(1, 0, 2)).squeeze(1) # size: [batch, hidden_dim]
            obs_embedding = torch.cat([att_output, query], dim=1) # size: [batch, 2 * hidden_dim]
            return self.output(obs_embedding), self.att_weight

    def get_weight(self):
        return self.att_weight

class ScaleDotAtt(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim, agent_num, **kwargs):
        super(ScaleDotAtt, self).__init__()
        self.hidden_dim = hidden_dim
        self.self_em_layer = nn.Linear(36, hidden_dim)
        self.other_em_layer = nn.Linear(28, hidden_dim)

        self.fc_mu = nn.Linear(obs_dim, hidden_dim)
        self.fc_var = nn.Linear(obs_dim, hidden_dim)

        self.query_w = nn.Linear(hidden_dim, hidden_dim)
        self.key_w = nn.Linear(hidden_dim, hidden_dim)
        self.value_w = nn.Linear(2 * hidden_dim, hidden_dim)

        self.align_layer = nn.Linear(2 * hidden_dim, hidden_dim)

    def reparameterize(self, mu, std):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def sample_z(self, mu, std):
        std = torch.exp(0.5 * std)
        return torch.normal(mean=mu, std=std)

    def forward(self, x, detach=False, **kwargs):
        # get a variable
        mu = self.fc_mu(x) # size: [batch, hidden]
        log_var = self.fc_var(x) # size: [batch, hidden]
        z = None

        if detach:
            # do not need gradient
            z = self.sample_z(mu, log_var)
        else:
            # need gradient
            # reparameterization trick
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = eps * std + mu # size: [batch, hidden]

        self_info = x[:, 0:36]
        other_info = x[:, 36:]
        agents_info = torch.stack(other_info.split(28, dim=1)) # size: [other num, batch, 28]

        self_em = f.relu(self.self_em_layer(self_info)) # size: [batch, hidden_dim]
        other_ems = f.relu(self.other_em_layer(agents_info)) # size: [other num, batch, hidden_dim]

        # scale dot product attention
        q = f.relu(self.query_w(z)) # size: [batch, hidden_dim]
        k = f.relu(self.key_w(other_ems)) # size: [other num, batch, hidden_dim]
        v = f.relu(self.value_w(torch.cat([other_ems, self_em.repeat(agents_info.shape[0], 1, 1)], dim=2))) # size: [other num, batch, hidden_dim]

        att_weight = f.softmax(torch.bmm(q.unsqueeze(1), k.permute(1, 2, 0)) / np.sqrt(self.hidden_dim), dim=2) # size: [batch, 1, other num]
        h = torch.bmm(att_weight, v.permute(1, 0, 2)) # size: [batch, 1, hidden_dim]
        output = torch.cat([h.squeeze(1), self_em], dim=1) # size: [batch, 2 * hidden_dim]
        return f.relu(self.align_layer(output)), att_weight
        


    