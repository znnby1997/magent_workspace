from tensorboardX import SummaryWriter
import time
import os 
import torch
import torch.optim as optim

import sys
sys.path.append('..')

from model.opp_model import Qnet, ReplayBuffer, learn

def train_opp_policy(env, gamma=0.98, batch_size=32, capacity=5000, 
        lr=1e-4, hidden_dim=32, model_save_url='', episode_num=5000, 
        tensorboard_data='', update_model_rate=20, print_info_rate=40, device=None):
    env_action_space = env.action_space.n
    env_obs_space = env.observation_space.shape[0]
    print('action space: ', env_action_space)
    print('obs space: ', env_obs_space)

    # 定义模型
    q = Qnet(env_obs_space, env_action_space, hidden_dim).to(device)
    q_target = Qnet(env_obs_space, env_action_space, hidden_dim).to(device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer(capacity, device)
    optimizer = optim.Adam(q.parameters(), lr=lr)

    group2 = RandomActor(env.env, env.handles[1])

    writer = SummaryWriter(tensorboard_data)
    
    group1_win_num = 0
    group2_win_num = 0

    for episode in range(episode_num):
        epsilon = max(0.01, 1.0 - 0.01 * episode)
        obs = env.reset()
        done = False
        total_reward = 0
        alive_info = env.get_live_agent()
        print('alive info: ', alive_info)

        while not done:
            # group 1决策
            group1_as = []
            for a_o_1 in obs[0]:
                group1_as.append(q.sample_action(torch.from_numpy(a_o_1).to(device).float(), epsilon))
            group2_as = group2.infer_action(obs[1])
            next_obs, rewards, done, alive_info = env.step([group1_as, group2_as])
            
            alive_info = alive_info['agent_live']
            alive_agent_ids = env.get_group_agent_id(0)

            cur_rewards = []
            for id, alive_agent_id in enumerate(alive_agent_ids):
                memory.put((obs[0][id], group1_as[id], rewards[0][alive_agent_id], next_obs[0][id], 1 - alive_info[0][alive_agent_id]))
                cur_rewards.append(rewards[0][alive_agent_id])
            total_reward += sum(cur_rewards)

            if env.step_num % print_info_rate == 0:
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup1 actions: ', group1_as)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup2 actions: ', group2_as)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\talive_info: ', alive_info)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\trewards: ', cur_rewards)

            obs = next_obs
        
        if not any(alive_info[0]):
            group2_win_num += 1
        elif not any(alive_info[1]):
            group1_win_num += 1
        
        if memory.size()>batch_size:
            learn(q, q_target, memory, optimizer, batch_size)

        if episode % update_model_rate == 1:
            q_target.load_state_dict(q.state_dict())
        print('Episode %d | total reward for group1: %0.2f | group1 win num: %d | group2 win num: %d' % (episode, total_reward, group1_win_num, group2_win_num))
        
        if episode % 500 == 499:
            torch.save(q_target, model_save_url + 'episode_' + str(episode) + '.th')

        writer.add_scalar('train/total_reward_per_episode_for_group1', total_reward, episode)
        writer.add_scalar('train/win_rate_for_group1', group1_win_num / (episode + 1), episode)
        writer.add_scalar('train/win_rate_for_group2', group2_win_num / (episode + 1), episode)

    q_target.load_state_dict(q.state_dict())
    torch.save(q_target, model_save_url + 'final_model.th')
    print('model is saved.')
    writer.close()

def test_opp(env, model=None, episode_num=20, render=True):
    print('test opponent policy')
    agent_1 = torch.load(model)
    group2 = RandomActor(env.env, env.handles[1])
    alive_info = env.get_live_agent()
    total_reward_1_list, total_reward_2_list = [], []
    kill_num_1_list, kill_num_2_list = [], []
    survive_num_1_list, survive_num_2_list = [], []

    for episode in range(episode_num):
        obs = env.reset()
        done = False
        total_reward_1, total_reward_2 = 0, 0
        kill_num_1, kill_num_2 = 0, 0
        survive_num_1, survive_num_2 = 0, 0

        while not done:
            # agent actions in group1
            group1_as = []
            for o in obs[0]:
                group1_as.append(agent_1.sample_action(torch.from_numpy(o).to(device).float(), 0.01))
            group2_as = group2.infer_action(obs[1])
            next_obs, rewards, done, alive_info = env.step([group1_as, group2_as], render=render)
            alive_info = alive_info['agent_live']
            total_reward_1 += sum(rewards[0])
            total_reward_2 += sum(rewards[1])
            obs = next_obs
        print('episode %d | group1 -- total reward %0.2f kill num %d survive num %d ' % (episode, total_reward_1, np.sum(alive_info[1] == 0), np.sum(alive_info[0] != 0)))
        print('episode %d | group2 -- total reward %0.2f kill num %d survive num %d ' % (episode, total_reward_2, np.sum(alive_info[0] == 0), np.sum(alive_info[1] != 0)))
        total_reward_1_list.append(total_reward_1)
        total_reward_2_list.append(total_reward_2)
        kill_num_1_list.append(np.sum(alive_info[1] == 0))
        kill_num_2_list.append(np.sum(alive_info[0] == 0))
        survive_num_1_list.append(np.sum(alive_info[0] != 0))
        survive_num_2_list.append(np.sum(alive_info[1] != 0))
    print('test is over !!!!!!')
    print('group1 -- ave total reward %0.2f kill num %0.2f survive num %0.2f' % (sum(total_reward_1_list) / episode_num, sum(kill_num_1_list) / episode_num, sum(survive_num_1_list) / episode_num))
    print('group2 -- ave total reward %0.2f kill num %0.2f survive num %0.2f' % (sum(total_reward_2_list) / episode_num, sum(kill_num_2_list) / episode_num, sum(survive_num_2_list) / episode_num))

