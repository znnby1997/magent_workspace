import pandas as pd
import numpy as np

from tensorboardX import SummaryWriter
import time
import os 
import torch
import random

from model.iql import IQL
from env_gym_wrap import MagentEnv
from magent.builtin.rule_model import RandomActor

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

agent_num = 20
map_size = 14
max_step = 200
net_structure_types = {
    'none': 0, 'softmax': 1, 'tanh': 2, 
    'sigmoid': 3, 'alw': 4, 'transformer': 5, 'fix_attention': 6, 'gru_attention': 7}
save_model_rate = 500

env = MagentEnv(agent_num=agent_num, map_size=map_size, max_step=max_step, opp_policy_random=True)

def train(env: MagentEnv, net_type, opp_policy=None, 
        episode_num=5000, epsilon=1.0, step_epsilon=0.01, final_epsilon=0.01, save_data=False, csv_url=None):
    env_action_space = env.action_space.n
    env_obs_space = env.observation_space.shape[0]
    n_group = 2
    group1 = IQL(env_obs_space, env_action_space, net_structure_types[net_type], agent_num)
    group2 = None
    if opp_policy:
        group2 = torch.load(opp_policy)
    else:
        group2 = RandomActor(env.env, env.handles[1])

    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    writer = SummaryWriter('log/data_info_' + net_type + '_' + timestamp)
    
    group1_win_num = 0
    group2_win_num = 0
    total_reward_list = []
    win_rate_list = []
    opp_win_rate_list = []
    for episode in range(episode_num):
        obs = env.reset(agent_num, use_random_init=False)
        done = False
        total_reward = 0
        alive_info = None

        while not done:
            group1_as = group1.infer_action(obs[0], epsilon)
            group2_as = group2.infer_action(obs[1])
            print('group1 actions: ', group1_as)
            next_obs, rewards, done, alive_info = env.step([group1_as, group2_as])
            # print(alive_info)
            print('rewards: ', rewards)
            
            alive_info = alive_info['agent_live']
            print('alive info: ', alive_info)
            alive_agent_ids = env.get_group_agent_id(0)
            d = []
            for alive_agent_id in alive_agent_ids:
                d.append(1 - alive_info[0][alive_agent_id])
            
            group1.push_data(obs[0], group1_as, rewards[0], next_obs[0], d)
            total_reward += sum(rewards[0])

            obs = next_obs
        
        if all(~alive_info[0]):
            group2_win_num += 1
        elif all(~alive_info[1]):
            group1_win_num += 1
        
        group1.learn()
        epsilon = max(epsilon - step_epsilon, final_epsilon)
        print('Episode %d | total reward for group1: %0.2f' % (episode, total_reward))
        writer.add_scalar('train/total_reward_per_episode_for_group1', total_reward, episode)
        writer.add_scalar('train/win_rate_for_group1', group1_win_num / (episode + 1), episode)
        writer.add_scalar('train/win_rate_for_group2', group2_win_num / (episode + 1), episode)

        total_reward_list.append(total_reward)
        win_rate_list.append(group1_win_num / (episode + 1))
        opp_win_rate_list.append(group2_win_num / (episode + 1))

    torch.save(group1, 'save_model/' + net_type + '_' + timestamp + '.th')
    print('model is saved.')
    writer.close()

    if save_data:
        data_dict = {}
        index = 'seed(' + str(seed_flag) + ')'
        data_dict[index + 'total_reward'] = total_reward_list
        data_dict[index + 'win_rate'] = win_rate_list
        data_dict[index + 'opp_win_rate'] = opp_win_rate_list
        dp.get_csv(csv_url, data_dict)

# seed = 1
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
train(env, 'none')


def test(env: MagentEnv, group1_url, opp_url=None, episode_num=10, epsilon=0.01):
    group1 = torch.load(group1_url)
    group2 = None
    
    if opp_url:
        group2 = torch.load(opp_url)
    else:
        group2 = RandomActor(env.env, env.handles[1])

    group1_win_num = 0
    group2_win_num = 0
    for episode in range(episode_num):
        obs = env.reset(agent_num)
        done = False
        while not done:
            group1_as = group1.infer_action(obs[0], episode)
            group2_as = group2.infer_action(obs[1])
            
            next_obs, rewards, done, alive_info = env.step([group1_as, group2_as], render=True)

            alive_info = alive_info['agent_live']
            obs = next_obs

        if all(~alive_info[0]):
            group2_win_num += 1
        elif all(~alive_info[1]):
            group1_win_num += 1
        
    print('group1 win num: ', group1_win_num)
    print('group2 win num: ', group2_win_num)

# test(env, 'save_model/none_20200709003409.th')