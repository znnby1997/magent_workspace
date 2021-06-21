import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim, net, agent_num, agg, nonlin):
        super(ActorCritic, self).__init__()

        self.fc_pi = net(obs_dim, n_actions, hidden_dim, agg=agg, agent_num=agent_num,  nonlin=nonlin)
        self.fc_v = net(obs_dim, 1, hidden_dim, agg=agg, agent_num=agent_num, nonlin=nonlin)

    def forward(self):
        raise NotImplementedError

    def pi(self, x):
        em, att_weight = self.fc_pi(x)
        x = F.softmax(em, dim=1)
        return x, att_weight

    def v(self, x):
        return self.fc_v(x)
 


    

    