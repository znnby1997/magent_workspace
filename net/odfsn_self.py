import torch
import torch.nn as nn
from net.base import BasicNet, LocalNet, Encoder, AttnAggV1, AttnAggV2

class ODFSNSelf(nn.Module):
    def __init__(self, obs_dim, output_dim, hidden_dim, agg, agent_num, **kwargs):
        super(ODFSNSelf, self).__init__()

        self.basic_net = BasicNet(obs_dim, output_dim, hidden_dim)
        self.encoder = Encoder(28, hidden_dim)
        self.local_net = LocalNet(hidden_dim, hidden_dim, output_dim, hidden_dim)
        self.agg = None
        self.token_num = 3 * agent_num - 1
        if agg == 'v1':
            self.agg = AttnAggV1(obs_dim, 3*agent_num-1, hidden_dim)
        elif agg == 'v2':
            self.agg = AttnAggV2(output_dim)

    def forward(self, x):
        self_info = x[:, 0:36]
        other_info = x[:, 36:]
        agents_info = torch.stack(other_info.split(28, dim=1)) # size: [other agents num, batch, 28]

        # basic q values
        basic_q, h = self.basic_net(x) # basic_q shape: batch, n_actions, h shape: batch, hidden
        local_es = self.encoder(agents_info) # nums, batch, hidden
        h = h.unsqueeze(0).repeat(self.token_num, 1, 1) # nums, batch, hidden
        local_qs = self.local_net(local_es, h).permute(1, 0, 2) # batch, nums, hidden

        assert self.agg, 'aggregate function cannot be null'

        return self.agg(basic_q, local_qs, x)



