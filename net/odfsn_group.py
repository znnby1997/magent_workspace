import torch
import torch.nn as nn
from net.base import BasicNet, LocalNet, Encoder, AttnAggV1, AttnAggV2, GroupNet

class ODFSNGroup(nn.Module):
    def __init__(self, obs_dim, output_dim, hidden_dim, agg, n_groups, **kwargs):
        super(ODFSNGroup, self).__init__()

        self.basic_net = BasicNet(obs_dim, output_dim, hidden_dim)
        self.local_net = LocalNet(hidden_dim, 0, output_dim, hidden_dim)
        self.agg = None
        if agg == 'v1':
            self.agg = AttnAggV1(obs_dim, n_groups, hidden_dim)
        elif agg == 'v2':
            self.agg = AttnAggV2(output_dim)

    def forward(self, x, gs):
        """
            x: obs
            gs: group_embeddings, batch, n_groups, embedding_dim
        """
        # basic q values
        basic_q, h = self.basic_net(x) # basic_q shape: batch, n_actions, h shape: batch, hidden
        local_qs = self.local_net(gs) # batch, nums, hidden

        assert self.agg, 'aggregate function cannot be null'

        return self.agg(basic_q, local_qs, x)



