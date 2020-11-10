import collections
import random

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

#Hyperparameters
# learning_rate = 0.0005
gamma         = 0.98
# buffer_limit  = 50000
# batch_size    = 32
# beta = 0.1

net_dict = {
    'alw': AlwAttNet, 'alw_gat': AlwGAT, 'dot_scale': DotScaleAttNet, 'dyan': Dyan,
    'gruga': GruGenAttNet, 'none': None, 'ssd': ScaleDotAtt, 'gn': GroupNet,
    'gaa': GAA, 'gruga2': GruGenAttNetNew, 'gw': GroupWeight, 'hand_weight': HandProcess,
    'hand_group': HandProcessGroup
}

class ReplayBufferM():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float).cuda(), torch.tensor(a_lst).cuda(), \
               torch.tensor(r_lst, dtype=torch.float).cuda(), torch.tensor(s_prime_lst, dtype=torch.float).cuda(), \
               torch.tensor(done_mask_lst, dtype=torch.float).cuda()
    
    def size(self):
        return len(self.buffer)

class QnetM(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim, net_type, concatenation, agent_num, aggregate_form, group_num, nonlin):
        super(QnetM, self).__init__()
        self.net_type = net_type
        if net_dict[self.net_type]:
            self.e = net_dict[net_type](
                obs_dim, n_actions, hidden_dim, agent_num=agent_num, 
                concatenation=concatenation, aggregate_form=aggregate_form, group_num=group_num, nonlin=nonlin)
        else:
            self.e = nn.Linear(obs_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)
        self.n_actions = n_actions

    def forward(self, x, detach=False):
        att_weight = None
        if net_dict[self.net_type]:
            em, att_weight = self.e(x, detach=detach)
        else:
            em, att_weight = self.e(x), None

        x = F.relu(em)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, att_weight
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs.reshape(1, -1), detach=True)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,self.n_actions - 1), 0
        else: 
            return out[0].argmax().item(), out[1]
    
    def get_weight(self):
        if isinstance(self.e, nn.Module):
            return self.e.get_weight()
        else:
            return
            
def learn_m(q, q_target, memory, optimizer, batch_size, beta, need_diff=False):
    print('learning  ......')
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out, w_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime, detach=True)[0].max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * (1 - done_mask)
        
        loss = F.smooth_l1_loss(q_a, target)
        if need_diff:
            # weight's entropy, make sure some differences
            # w_entropy = Categorical(w_out).entropy()
            loss = loss + (beta * Categorical(w_out).entropy())
        
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()