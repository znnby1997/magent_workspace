import torch
import torch.nn as nn
import torch.nn.functional as f

class IAN(nn.Module):
    """
        Attention-Based Information Process Network
        Attention Generation: 
            dot: score = query * keys
            general: score = query * W * keys
            concat: score = v * tanh(W * [query;keys])

        For every piece of information in other information, 
        take it and self information as input to generate a Q value or policy distributions.

        According to attention weights, aggregate all Q values or policy distributions and get the final Q or policy.
    """
    def __init__(self, obs_dim, output_dim, hidden_dim, agent_num, **kwargs):
        super(IAN, self).__init__()

        # generating Q or policy
        self.b_o2e = nn.Linear(obs_dim, hidden_dim)
        self.b_e2a = nn.Linear(hidden_dim, output_dim)
        self.o_o2e = nn.Linear(28, hidden_dim)
        self.o_e2a = nn.Linear(2 * hidden_dim, output_dim)

        # generating attention weight
        self.attn_layer = nn.Linear(obs_dim, agent_num * 3 - 1)

    def forward(self, x, **kwargs):
        self_info = x[:, 0:36]
        other_info = x[:, 36:]
        agents_info = torch.stack(other_info.split(28, dim=1)) # size: [other agents num, batch, 28]
        other_info_num = agents_info.shape[0]

        # obs to embeddings
        b_e = f.relu(self.b_o2e(x)) # size: [batch, hidden_dim]
        o_es = f.relu(self.o_o2e(agents_info)).permute(1, 0, 2) # size: [batch, other info num, hidden_dim]
        b_o_es = torch.cat([b_e.unsqueeze(1).repeat(1, other_info_num, 1), o_es], dim=2) # size: [batch, other info num, 2 * hidden_dim]

        attn = f.softmax(self.attn_layer(x), dim=1) # size: [batch, other info num]
        # print('attn output: ', attn)

        # generating attention weights
        b_v = self.b_e2a(b_e) # size: [batch, n_actions]
        o_vs = self.o_e2a(b_o_es) # size: [batch, other agents num, n_actions]

        # aggregate info-based q values or policies
        o_v = torch.bmm(attn.unsqueeze(1), o_vs).squeeze(1) # size: [batch, n_actions]
        value = (b_v + o_v) / 2.

        return value, attn

        
