import numpy as np

from tensorboardX import SummaryWriter
import time
import torch
import torch.optim as optim

import sys
sys.path.append('..')

from loss._simple import ac_one_td_loss
from model.ac import ActorCritic
from replay_buffer._on_policy_simple import OPSimpleReplayBuffer
import utils.data_process as dp
from interaction.exec import exec_for_coll, exec_

def train(model, buffer, optimizer, gamma, entropy_coef):
    traj = buffer.get()

    a_loss, c_loss, e_loss = ac_one_td_loss(model, traj, gamma, entropy_coef)

    loss = -a_loss + c_loss - e_loss
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

    buffer.clear()
    return loss.mean(), a_loss.mean(), c_loss.mean(), e_loss.mean()

def train_a2c(env, net, gamma=0.98, lr=1e-4, hidden_dim=32,
    agent_num=20, opp_policy=None, model_save_url='../../data/a2c/model/',
    epoch_num=1000, test_num=20, train_rate=20,  tensorboard_data='../../data/a2c/log/data_info_', data_buffer_limit=5000,
    save_data=True, csv_url='../../data/a2c/csv/', seed_flag=1, nonlin='softmax',
    entr_w=0.02, print_log=False, device=None):

    env_action_space = env.action_space.n
    env_obs_space = env.observation_space.shape[0]
    
    assert opp_policy != '', 'opp policy cannot be empty'
    opp = torch.load(opp_policy)
    
    agent = ActorCritic(obs_dim=env_obs_space, n_actions=env_action_space, hidden_dim=hidden_dim, net=net,
                agent_num=agent_num, nonlin=nonlin).to(device)

    data_buffer = OPSimpleReplayBuffer(data_buffer_limit, device)
    
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    total_reward_list = []
    ave_kill_num_list = []
    ave_survive_num_list = []

    writer = SummaryWriter(tensorboard_data)

    for epoch in range(epoch_num):
        print('Epoch %d -- trajectory data collection starts ...' % (epoch))
        for e_ in range(train_rate):
            traj_data = exec_for_coll(env, 'a2c', agent, 0.01, opp, device)
            data_buffer.push(traj_data)
            
            print('episode %d is over ...' % (e_))
        
        print('epoch %d ... learning ...' % (epoch))
        _, a_loss, c_loss, e_loss = train(agent, data_buffer, optimizer, gamma, entr_w)
        if print_log:
            writer.add_scalar('train_loss/actor_loss', a_loss, epoch)
            writer.add_scalar('train_loss/critic_loss', c_loss, epoch)
            writer.add_scalar('train_loss/entropy_loss', e_loss, epoch)

        print('epoch idx %d ... test stage starting ...' % epoch)
        total_kill_num = 0
        total_survive_num = 0
        total_reward = 0

        for test_episode in range(test_num):
            statistic_var = exec_(env, agent, 0.01, opp, device, False)
            total_reward += statistic_var[0]
            total_kill_num += statistic_var[1]
            total_survive_num += statistic_var[2]
            
        epoch_total_reward = total_reward / test_num
        epoch_total_kill_num = total_kill_num / test_num
        epoch_total_survive_num = total_survive_num / test_num
        print('epoch idx %d | total reward for group2: %0.2f | total kill num: %0.2f | total survive num: %0.2f' % (epoch, epoch_total_reward, epoch_total_kill_num, epoch_total_survive_num))
        writer.add_scalar('train/total_reward_for_group2', epoch_total_reward, epoch)
        writer.add_scalar('train/kill_num_for_group2', epoch_total_kill_num, epoch)
        writer.add_scalar('train/survive_num_for_group2', epoch_total_survive_num, epoch)

        total_reward_list.append(epoch_total_reward)
        ave_kill_num_list.append(epoch_total_kill_num)
        ave_survive_num_list.append(epoch_total_survive_num)

    torch.save(agent, model_save_url + 'model.th')
    print('model is saved.')
    writer.close()

    if save_data:
        data_dict = {}
        index = 'seed(' + str(seed_flag) + ')'
        data_dict[index + 'total_reward'] = total_reward_list
        data_dict[index + 'kill_num'] = ave_kill_num_list
        data_dict[index + 'survive_num'] = ave_survive_num_list
        dp.get_csv(csv_url + 'res.csv', data_dict)