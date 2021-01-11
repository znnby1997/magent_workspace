import numpy as np

from tensorboardX import SummaryWriter
import time
import torch
import torch.optim as optim

import sys
sys.path.append('..')

from loss._simple import q_one_td_loss
from model.dqn import QnetM
from replay_buffer._simple import SimpleReplayBuffer
import utils.data_process as dp
from interaction.exec import exec_for_coll, exec_

def train(q, q_target, buffer, optimizer, batch_size, gamma):
    last_loss = q_one_td_loss(q, q_target, buffer.get(batch_size), gamma)
    loss = last_loss
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()
    return loss.mean(), last_loss.mean()

def train_dqn(env: MagentEnv, net, gamma=0.98, batch_size=5000, capacity=100000, 
    lr=1e-4, hidden_dim=32, agent_num=20, opp_policy=None, model_save_url='',
    episodes_per_epoch=100, episodes_per_test=20, epoch_num=500, tensorboard_data='',
    save_data=True, csv_url='', seed_flag=1, nonlin='softmax',
    update_model_rate=100, print_log=True, device=None):
    
    env_action_space = env.action_space.n
    env_obs_space = env.observation_space.shape[0]

    assert opp_policy != '', 'opp policy cannot be empty'
    opp = torch.load(opp_policy)

    q = QnetM(env_obs_space, env_action_space, hidden_dim=hidden_dim, net=net,
        agent_num=agent_num, nonlin=nonlin).to(device)
    q_target = QnetM(env_obs_space, env_action_space, hidden_dim=hidden_dim, net=net,
        agent_num=agent_num, nonlin=nonlin).to(device)
    q_target.load_state_dict(q.state_dict())
    
    memory = SimpleReplayBuffer(capacity, device)
    optimizer = optim.Adam(q.parameters(), lr=lr)

    total_reward_list = []
    ave_kill_num_list = []
    ave_survive_num_list = []

    writer = SummaryWriter(tensorboard_data)

    for epoch in range(epoch_num):
        print('epoch %d training starts' % epoch)

        for episode in range(episodes_per_epoch):
            epsilon = max(0.01, 1.0 - 0.01 * episode)
            memory.push(exec_for_coll(env, q, epsilon, opp, device))

            if len(memory) > batch_size:
                _, last_loss = train(q, q_target, memory, optimizer, batch_size, gamma)
                if print_log:
                    writer.add_scalar('train/q_td_loss', last_loss, epoch)

            if episode % update_model_rate == 1:
                q_target.load_state_dict(q.state_dict())
            print('trainging ... epoch %d episode %d is over' % (epoch, episode))
        
        print('test stage for epoch %d' % epoch)
        total_kill_num = 0
        total_survive_num = 0
        total_reward = 0
 
        for test_episode in range(episodes_per_test):
            statistic_var = exec_(env, q_target, 0.01, opp, device, False)
            print('test ... epoch %d episode %d' % (epoch, test_episode))
            total_reward += statistic_var[0]
            total_kill_num += statistic_var[1]
            total_survive_num += statistic_var[2]
        
        epoch_total_reward = total_reward / episodes_per_test
        epoch_total_kill_num = total_kill_num / episodes_per_test
        epoch_total_survive_num = total_survive_num / episodes_per_test
        print('epoch %d | total reward for group2: %0.2f | total kill num: %0.2f | total survive num: %0.2f' % (epoch, epoch_total_reward, epoch_total_kill_num, epoch_total_survive_num))
        writer.add_scalar('train/total_reward_for_group2', epoch_total_reward, epoch)
        writer.add_scalar('train/kill_num_for_group2', epoch_total_kill_num, epoch)

        total_reward_list.append(epoch_total_reward)
        ave_kill_num_list.append(epoch_total_kill_num)
        ave_survive_num_list.append(epoch_total_survive_num)

    q_target.load_state_dict(q.state_dict())
    torch.save(q_target, model_save_url + 'model.th')
    print('model is saved.')
    writer.close()

    if save_data:
        print('saving data ....')
        data_dict = {}
        index = 'seed(' + str(seed_flag) + ')'
        data_dict[index + 'total_reward'] = total_reward_list
        data_dict[index + 'kill_num'] = ave_kill_num_list
        data_dict[index + 'survive_num'] = ave_survive_num_list
        dp.get_csv(csv_url + 'res.csv', data_dict)
        print('csv is saved.')