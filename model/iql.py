import torch
import torch.optim as optim
import torch.nn.functional as f
from torch.distributions import Categorical

import sys
sys.path.append('..')


from utils.experience_memory import ExperienceMemory, Dynamics, PrioritisedBuffer
# from net.basic_net import QNet
from net.alw_att_net import AlwAttNet
from net.dot_scale_att_net import DotScaleAttNet
from net.dyan import Dyan
from net.gru_weight import GruGenAttNet, GruGenAttNetNew
from net.none_net import NoneNet
from net.nonlin_att_net import NonlinAttNet
from net.dyan_group import DyanGroup
from net.group_att_agg import GAA

import time
import random

net_dict = {
    'alw': AlwAttNet, 'dot_scale': DotScaleAttNet, 'dyan': Dyan,
    'gruga': GruGenAttNet, 'none': NoneNet, 'nonlinatt': NonlinAttNet, 'dyan_group': DyanGroup,
    'gaa': GAA, 'gruga2': GruGenAttNetNew
}

class IQL(object):
    def __init__(self, obs_dim, n_actions, net, agent_num, group_num, concatenation, gamma=0.98, batch_size=5000,
                capacity=100000, lr=1e-4, hidden_dim=32, em_dim=32, prioritised_replay=False, target_net=False,
                use_cuda=False, nonlin='softmax', aggregate_form='mean'):
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_net = target_net
        self.prioritised_replay = prioritised_replay
        self.use_cuda = use_cuda

        if self.prioritised_replay:
            print('Add prioritised_replay')
            self.pool = PrioritisedBuffer(capacity)
        else:
            self.pool = ExperienceMemory(capacity)

        self.n_actions = n_actions
        
        self.q_net = net_dict[net](obs_dim, n_actions, agent_num=agent_num, hidden_dim=hidden_dim,
                                    nonlin=nonlin, aggregate_form=aggregate_form, em_dim=em_dim, group_num=group_num, concatenation=concatenation)
        if self.target_net:
            print('Add double dqn')
            self.target_q_net = net_dict[net](obs_dim, n_actions, agent_num=agent_num, hidden_dim=hidden_dim,
                                                nonlin=nonlin, aggregate_form=aggregate_form, em_dim=em_dim, group_num=group_num, concatenation=concatenation)
            self.update_target()
        
        if self.use_cuda:
            self.q_net = self.q_net.cuda()
            self.target_q_net = self.target_q_net.cuda()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def update_target(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def infer_action(self, joint_obs, epsilon=1.0, greedy=False, print_mask=False):
        actions = []
        mask = []
        with torch.no_grad():
            joint_obs = torch.tensor(joint_obs, dtype=torch.float)
            if self.use_cuda:
                joint_obs = joint_obs.cuda()

            for obs in joint_obs:
                if random.uniform(0, 1) >= epsilon or greedy:
                    q = self.q_net.q(obs.reshape(1, -1), greedy_group=greedy)
                    actions.append(q.max(1)[1].item())
                    
                    if print_mask:
                        m = self.q_net.get_mask()
                        mask.append(m)
                else:
                    actions.append(random.randint(0, self.n_actions - 1))
        return actions, mask

    def push_data(self, joint_obs, joint_actions, rewards, next_joint_obs, dones):
        # 这里输入的joint_obs的shape[0] <= agent_num, 只保留活着的agent的数据
        for obs, action, reward, next_obs, done in zip(joint_obs, joint_actions, rewards, next_joint_obs, dones):
            self.pool.push(obs, action, reward, next_obs, done)

    def sample(self):
        if self.batch_size > len(self.pool):
            print('Data is not enough')
            return
        
        batch, indices, weights = self.pool.sample(self.batch_size)
        data = Dynamics(*zip(*batch))

        obs = torch.tensor(data.state, dtype=torch.float)
        action = torch.tensor(data.action, dtype=torch.long).reshape(-1, 1)
        reward = torch.tensor(data.reward, dtype=torch.float).reshape(-1, 1)
        next_obs = torch.tensor(data.next_state, dtype=torch.float)
        done = torch.tensor(data.is_end, dtype=torch.float).reshape(-1, 1)
        weights = torch.tensor(weights, dtype=torch.float).reshape(-1, 1)

        if self.use_cuda:
            obs = obs.cuda()
            action = action.cuda()
            reward = reward.cuda()
            next_obs = next_obs.cuda()
            done = done.cuda()
            weights = weights.cuda()

        return obs, action, reward, next_obs, done, indices, weights

    def learn(self):
        if not self.sample():
            return
        
        obs, action, reward, next_obs, done, indices, weights = self.sample()

        q_values = self.q_net.q(obs)
        # print('q values shape: ', q_values.shape)
        qa_values = q_values.gather(1, action)

        if self.target_net:
            q_next = self.q_net.q(next_obs) # size: [batch, n_actions]
            cur_max_as = q_next.max(1)[1].reshape(-1, 1) # size: [batch]
            next_q_values = self.target_q_net.q(next_obs) # size: [batch, n_actions]
            next_q_values = next_q_values.gather(1, cur_max_as)
            q_target = reward + self.gamma * next_q_values * (1 - done)
        else:
            q_next = self.q_net.q(next_obs)
            # print('q_next max shape: ', q_next.max(1)[0].shape)
            q_target = reward + self.gamma * q_next.max(1)[0].reshape(-1, 1) * (1 - done)
        
        if self.prioritised_replay:
            loss = ((qa_values - q_target.detach()).pow(2) * weights).squeeze()
            prios = loss + 1e-5
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.pool.update_priorities(indices, prios.data.cpu().numpy())
            self.optimizer.step()
        else:
            loss = f.smooth_l1_loss(input=qa_values, target=q_target.detach())
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def get_cur_weight(self, joint_obs):
        att_weight_list = []
        joint_obs = torch.tensor(joint_obs, dtype=torch.float16)
        if self.use_cuda:
            joint_obs = joint_obs.cuda()

        for obs in joint_obs:
            q = self.q_net.q(obs.reshape(1, -1))
            att_weight_list.append(self.q_net.get_cur_weight())
        return att_weight_list


    


        