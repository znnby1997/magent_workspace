import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import sys
sys.path.append('..')

from net.alw_att_net import AlwAttNet, AlwGAT
from net.dot_scale_att_net import DotScaleAttNet, ScaleDotAtt
from net.dyan import Dyan
from net.gru_weight import GruGenAttNet, GruGenAttNetNew
from net.group_att_agg import GAA, GroupNet
from net.group_weight import GroupWeight
from net.hand_process import HandProcess, HandProcessGroup

# Hyperparameters
# learning_rate = 0.0005
gamma = 0.98
# lmbda = 0.95
# eps_clip = 0.1
# K_epoch = 3
# T_horizon = 20

net_dict = {
    'alw': AlwAttNet, 'alw_gat': AlwGAT, 'dot_scale': DotScaleAttNet, 'dyan': Dyan,
    'gruga': GruGenAttNet, 'none': None, 'ssd': ScaleDotAtt, 'gn': GroupNet,
    'gaa': GAA, 'gruga2': GruGenAttNetNew, 'gw': GroupWeight, 'hand_weight': HandProcess,
    'hand_group': HandProcessGroup
}

class PPO(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim, net_type, concatenation, agent_num, aggregate_form, group_num, nonlin):
        super(PPO, self).__init__()
        self.net_type = net_type
        self.obs_dim = obs_dim
        if net_dict[self.net_type]:
            self.e = net_dict[net_type](
                obs_dim, n_actions, hidden_dim, agent_num=agent_num, 
                concatenation=concatenation, aggregate_form=aggregate_form, group_num=group_num, nonlin=nonlin)
        else:
            self.e = nn.Linear(obs_dim, hidden_dim)

        self.fc_pi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

        self.fc_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self):
        raise NotImplementedError

    def pi(self, x, detach=False):
        att_weight = None
        if net_dict[self.net_type]:
            em, att_weight = self.e(x, detach=detach)
        else:
            em, att_weight = self.e(x), None

        x = F.relu(em)
        x = F.softmax(self.fc_pi(x), dim=1)
        return x, att_weight

    def v(self, x, detach=False):
        att_weight = None
        if net_dict[self.net_type]:
            em, att_weight = self.e(x, detach=detach)
        else:
            em, att_weight = self.e(x), None

        x = F.relu(em)
        x = self.fc_v(x)
        return x, att_weight


class DataSet():
    def __init__(self):
        self.data = []

    def put_data(self, transition):
        self.data.append(transition)
    
    def not_none(self):
        # print('data: ', self.data)
        return len(self.data) > 0 and len(self.data[0][3]) > 0

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).cuda(), torch.tensor(a_lst).cuda(), \
                                              torch.tensor(r_lst, dtype=torch.float).cuda(), torch.tensor(s_prime_lst, dtype=torch.float).cuda(), \
                                              torch.tensor(done_lst, dtype=torch.float).cuda(), torch.tensor(prob_a_lst, dtype=torch.float).cuda()

        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

def learn_ppo(model, data, optimizer, k_epoch, lmbda, eps_clip, entropy_w, print_log=False):
    s, a, r, s_prime, done_mask, prob_a = data.make_batch()

    for i in range(k_epoch):
        # print('s_prime shape: ', s_prime.shape)
        td_target = r + gamma * model.v(s_prime)[0] * done_mask
        delta = td_target - model.v(s)[0]
        delta = delta.detach().cpu().numpy()

        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = gamma * lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float).cuda()

        pi = model.pi(s)[0]
        pi_a = pi.gather(1, a)
        ratio = torch.exp(torch.log(pi_a + 1e-8) - torch.log(prob_a + 1e-8))

        entropy = Categorical(pi).entropy()

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage

        actor_loss = torch.min(surr1, surr2)
        critic_loss = F.smooth_l1_loss(model.v(s)[0], td_target.detach())
        loss = -actor_loss + critic_loss - entropy_w * entropy

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        if print_log:
            return loss.mean(), actor_loss.mean(), critic_loss.mean(), entropy.mean()
        else:
            return 0., 0., 0., 0.
