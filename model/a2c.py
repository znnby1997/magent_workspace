import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import time
import numpy as np

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
# n_train_processes = 3
# learning_rate = 0.0002
# update_interval = 5
gamma = 0.98
# max_train_steps = 60000
# PRINT_INTERVAL = update_interval * 100

net_dict = {
    'alw': AlwAttNet, 'alw_gat': AlwGAT, 'dot_scale': DotScaleAttNet, 'dyan': Dyan,
    'gruga': GruGenAttNet, 'none': None, 'ssd': ScaleDotAtt, 'gn': GroupNet,
    'gaa': GAA, 'gruga2': GruGenAttNetNew, 'gw': GroupWeight, 'hand_weight': HandProcess,
    'hand_group': HandProcessGroup
}

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim, net_type, concatenation, agent_num, aggregate_form, group_num, nonlin):
        super(ActorCritic, self).__init__()
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

def worker(worker_id, master_end, worker_end, env):
    master_end.close()
    
    while True:
        cmd, data = worker_end.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            worker_end.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            worker_end.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            worker_end.send(ob)
        elif cmd == 'close':
            worker_end.close()
            break
        elif cmd == 'get_spaces':
            worker_end.send((env.observation_space, env.action_space))
        elif cmd == 'get_ids':
            alive_ids = env.get_group_agent_id(data)
            worker_end.send(alive_ids)
        else:
            raise NotImplementedError

class ParallelEnv:
    def __init__(self, n_train_processes, env):
        self.nenvs = n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = list()

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker, args=(worker_id, master_end, worker_end, env))
            p.daemon = True
            p.start()
            self.workers.append(p)

        for worker_end in worker_ends:
            worker_end.close()       

    def step_async(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action))
        self.waiting = True

    def step_wait(self):
        result = [master_end.recv() for master_end in self.master_ends]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack([master_end.recv() for master_end in self.master_ends])

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        return np.stack([master_end.recv() for master_end in self.master_ends])

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]
        for master_end in self.master_ends:
            master_end.send(('close', None))
        for worker in self.workers:
            worker.join()
            self.closed = True

def compute_target(v_final, r_lst, mask_lst):
    G = v_final.reshape(-1)
    td_target = list()

    for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
        G = r + gamma * G * mask
        td_target.append(G)

    return torch.tensor(td_target[::-1]).float()

def learn(model, s_prime, a_lst, s_lst, r_lst, mask_lst, optimizer, entropy_w):
    s_final = torch.from_numpy(s_prime).cuda().float()
    v_final = model.v(s_final)[0].detach().clone().cpu().numpy()
    td_target = compute_target(v_final, r_lst, mask_lst)

    td_target_vec = td_target.reshape(-1).cuda()
    s_vec = torch.tensor(s_lst).float().cuda().reshape(-1, model.obs_dim)
    a_vec = torch.tensor(a_lst).cuda().reshape(-1).unsqueeze(1)
    print('td_target_vec shape: ', td_target_vec.shape, ' model output shape: ', model.v(s_vec)[0].reshape(-1).shape)
    advantage  = td_target_vec - model.v(s_vec)[0].reshape(-1)

    pi, _ = model.pi(s_vec)
    pi_a = pi.gather(1, a_vec).reshape(-1)
    entropy = Categorical(pi).entropy()

    loss = -(torch.log(pi_a) * advantage.detach()) + F.smooth_l1_loss(model.v(s_vec)[0].reshape(-1), td_target_vec) - entropy_w * entropy

    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

    