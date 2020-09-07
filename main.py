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
import utils.data_process as dp

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

"""
    net_dict = {
    'alw': AlwAttNet, 'dot_scale': DotScaleAttNet, 'dyan': Dyan,
    'gruga': GruGenAttNet, 'none', NoneNet, 'nonlinatt': NonlinAttNet, 'dyan_group': DyanGroup,
    'gaa': GAA
}
"""

env = MagentEnv(agent_num=20, map_size=15, max_step=200, opp_policy_random=True)

# 用于训练一个对手模型，自身对手为随机动作
def train_opp_policy(env: MagentEnv, net_type, gamma=0.98, batch_size=5000, capacity=100000, 
        lr=1e-4, hidden_dim=32, 
        agent_num=20, prioritised_replay=True, model_save_url='../../data/model/',
        episode_num=5000, epsilon=1.0, step_epsilon=0.01, tensorboard_data='../../data/log/data_info_',
        final_epsilon=0.01, save_data=True, csv_url='../../data/csv/', seed_flag=1, update_net=True,
        update_model_rate=100, print_info_rate=20, use_cuda=True):
    env_action_space = env.action_space.n
    env_obs_space = env.observation_space.shape[0]
    # group1作为对手，真正训练的是group2
    group1 = IQL(env_obs_space, env_action_space,
        net_type, agent_num, gamma, batch_size, capacity, lr, hidden_dim,
        use_cuda=use_cuda, prioritised_replay=prioritised_replay, target_net=update_net)

    group2 = RandomActor(env.env, env.handles[1])

    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    writer = SummaryWriter(tensorboard_data + 'opp_' + timestamp)
    
    group1_win_num = 0
    group2_win_num = 0
    total_reward_list = []
    win_rate_list = []
    opp_win_rate_list = []
    for episode in range(episode_num):
        obs = env.reset(use_random_init=False)
        done = False
        total_reward = 0
        alive_info = None

        while not done:
            # 由于我们已经训练好的对手习惯于从左上角开始攻击，因此将我们要训练的agent放到右下角
            # group1 采用环境中的id=1; group2 采用环境中的id=0
            group1_as = group1.infer_action(obs[0], epsilon)
            group2_as = group2.infer_action(obs[1])
            next_obs, rewards, done, alive_info = env.step([group1_as, group2_as])
            
            alive_info = alive_info['agent_live']
            alive_agent_ids = env.get_group_agent_id(0)
            d = []
            cur_rewards = []
            for alive_agent_id in alive_agent_ids:
                d.append(1 - alive_info[0][alive_agent_id])
                cur_rewards.append(rewards[0][alive_agent_id])
            
            group1.push_data(obs[0], group1_as, rewards[0], next_obs[0], d)
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
        
        group1.learn()

        if update_net and episode % update_model_rate == 1:
            group1.update_target()

        epsilon = max(epsilon - step_epsilon, final_epsilon)
        print('Episode %d | total reward for group1: %0.2f' % (episode, total_reward))
        writer.add_scalar('train/total_reward_per_episode_for_group1', total_reward, episode)
        writer.add_scalar('train/win_rate_for_group1', group1_win_num / (episode + 1), episode)
        writer.add_scalar('train/win_rate_for_group2', group2_win_num / (episode + 1), episode)

        total_reward_list.append(total_reward)
        win_rate_list.append(group1_win_num / (episode + 1))
        opp_win_rate_list.append(group2_win_num / (episode + 1))

    torch.save(group1, model_save_url + 'opp_' + timestamp + '.th')
    print('model is saved.')
    writer.close()

    if save_data:
        data_dict = {}
        index = 'seed(' + str(seed_flag) + ')'
        data_dict[index + 'total_reward'] = total_reward_list
        data_dict[index + 'win_rate'] = win_rate_list
        data_dict[index + 'opp_win_rate'] = opp_win_rate_list
        dp.get_csv(csv_url + 'opp_policy_' + timestamp + '.csv', data_dict)


# 对手为训练好的模型,而非随机动作
def train(env: MagentEnv, net_type, gamma=0.98, batch_size=5000, capacity=100000, group_num=4,
        lr=1e-4, hidden_dim=32, nonlin='softmax', aggregate_form='mean', 
        agent_num=20, opp_policy=None, prioritised_replay=True, model_save_url='../../data/model/',
        episode_num=1000, epsilon=1.0, step_epsilon=0.01, use_cuda=True, tensorboard_data='../../data/log/data_info_',
        final_epsilon=0.01, save_data=True, csv_url='../../data/csv/', seed_flag=1, update_net=True,
        update_model_rate=100, print_info_rate=20, em_dim=32, concatenation=False, print_info=True):
    env_action_space = env.action_space.n
    env_obs_space = env.observation_space.shape[0]
    # group1作为对手，真正训练的是group2
    group1 = torch.load(opp_policy)
    group2 = IQL(env_obs_space, env_action_space, net_type, agent_num, group_num, concatenation,
        gamma, batch_size, capacity, lr, hidden_dim, nonlin=nonlin, aggregate_form=aggregate_form,
        prioritised_replay=prioritised_replay, target_net=update_net, use_cuda=use_cuda, em_dim=em_dim)

    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    writer = SummaryWriter(tensorboard_data + net_type + '_' + timestamp)
    
    group1_win_num = 0
    group2_win_num = 0
    total_reward_list = []
    win_rate_list = []
    opp_win_rate_list = []
    for episode in range(episode_num):
        obs = env.reset(use_random_init=False)
        done = False
        total_reward = 0
        alive_info = env.get_live_agent()
        # print('episode ', str(episode), '\talive agent:', alive_info)

        while not done:
            # 由于我们已经训练好的对手习惯于从左上角开始攻击，因此将我们要训练的agent放到右下角
            # group1 采用环境中的id=1; group2 采用环境中的id=0
            group1_as, _ = group1.infer_action(obs[0], greedy=True)
            group2_as, _ = group2.infer_action(obs[1], epsilon=epsilon)
            next_obs, rewards, done, alive_info = env.step([group1_as, group2_as])
            
            alive_info = alive_info['agent_live']
            alive_agent_ids = env.get_group_agent_id(1)
            d = []
            cur_rewards = []
            for alive_agent_id in alive_agent_ids:
                d.append(1 - alive_info[1][alive_agent_id])
                cur_rewards.append(rewards[1][alive_agent_id])

            group2.push_data(obs[1], group2_as, cur_rewards, next_obs[1], d)

            total_reward += sum(cur_rewards)
            # print('step: ', env.step_num)

            if print_info and env.step_num % print_info_rate == 0:
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup1 actions: ', group1_as)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup2 actions: ', group2_as)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\talive_info: ', alive_info)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\trewards: ', rewards)

            obs = next_obs

        group2.learn()
        
        if not any(alive_info[0]):
            group2_win_num += 1
        elif not any(alive_info[1]):
            group1_win_num += 1

        if update_net and episode % update_model_rate == 1:
            group2.update_target()

        epsilon = max(epsilon - step_epsilon, final_epsilon)
        print('Episode %d | total reward for group2: %0.2f' % (episode, total_reward))
        writer.add_scalar('train/total_reward_per_episode_for_group2', total_reward, episode)
        writer.add_scalar('train/win_rate_for_group1', group1_win_num / (episode + 1), episode)
        writer.add_scalar('train/win_rate_for_group2', group2_win_num / (episode + 1), episode)

        total_reward_list.append(total_reward)
        win_rate_list.append(group2_win_num / (episode + 1))
        opp_win_rate_list.append(group1_win_num / (episode + 1))

    torch.save(group2, model_save_url + net_type + '_' + timestamp + '_' + str(seed_flag) + '.th')
    print('model is saved.')
    writer.close()

    if save_data:
        data_dict = {}
        index = 'seed(' + str(seed_flag) + ')'
        data_dict[index + 'total_reward'] = total_reward_list
        data_dict[index + 'win_rate'] = win_rate_list
        data_dict[index + 'opp_win_rate'] = opp_win_rate_list
        dp.get_csv(csv_url + net_type + '_' + timestamp + '.csv', data_dict)

def test_model(env: MagentEnv, model=None, episode_num=20, render=True, print_att_weight=False, 
                print_group_mask=False, csv_url='../../data/csv/', seed=0, save_data=True, print_info=True):
    print('test env')
    group1 = torch.load(model[0])  # 对手的模型
    group2 = torch.load(model[1])  # 测试模型
    alive_info = env.get_live_agent()
    win_num_1 = 0
    win_num_2 = 0
    test_time_flag = time.strftime('%Y%m%d%H%M%S', time.localtime())

    total_reward_1_list, total_reward_2_list = [], []
    kill_num_1_list, kill_num_2_list = [], []
    survive_num_1_list, survive_num_2_list = [], []

    for episode in range(episode_num):
        obs = env.reset(use_random_init=False)
        done = False
        total_reward_1 = 0
        total_reward_2 = 0
        csv_path = csv_url + test_time_flag + '_' + str(episode)
        os.mkdir(csv_path)

        step_flag = 1
        print('print group mask', print_group_mask)
        while not done:
            group1_as, _ = group1.infer_action(obs[0], greedy=True)
            group2_as, mask = group2.infer_action(obs[1], greedy=True, print_mask=print_group_mask)
            next_obs, rewards, done, alive_info = env.step([group1_as, group2_as], render=render)

            alive_info = alive_info['agent_live']
            total_reward_1 += sum(rewards[0])
            total_reward_2 += sum(rewards[1])
            if print_info:
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup1 actions: ', group1_as)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup2 actions: ', group2_as)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\trewards: ', rewards)
                print('step: ', str(step_flag), '\talive_info: ', alive_info)

            if print_group_mask and mask:
                opp_mask_list, par_mask_list = [], []
                for m in mask:
                    opp_mask_list.append(m[0])
                    par_mask_list.append(m[1])
                opp_mask = torch.stack(opp_mask_list).cpu().numpy()
                par_mask = torch.stack(par_mask_list).cpu().numpy()
                np.save(csv_path + '/step_' + str(step_flag) + 'opp_mask', opp_mask)
                np.save(csv_path + '/step_' + str(step_flag) + 'par_mask', par_mask)

            step_flag += 1
            obs = next_obs

        if not any(alive_info[0]):
            win_num_2 += 1
        elif not any(alive_info[1]):
            win_num_1 += 1
        print('Episode %d | total reward for group1: %0.2f   total reward for group2: %0.2f' % (episode, total_reward_1, total_reward_2))
        total_reward_1_list.append(total_reward_1)
        total_reward_2_list.append(total_reward_2)
        kill_num_1_list.append(np.sum(alive_info[1] == 0))
        kill_num_2_list.append(np.sum(alive_info[0] == 0))
        survive_num_1_list.append(np.sum(alive_info[0] != 0))
        survive_num_2_list.append(np.sum(alive_info[1] != 0))

    print('Test is over. group1 win num: %d   group2 win num: %d' % (win_num_1, win_num_2))
    if save_data:
        print('saving data ...')
        data_dict = {}
        index = 'seed(' + str(seed) + ')'
        data_dict[index + 'total_reward_1'] = total_reward_1_list
        data_dict[index + 'total_reward_2'] = total_reward_2_list
        data_dict[index + 'kill_num_1'] = kill_num_1_list
        data_dict[index + 'kill_num_2'] = kill_num_2_list 
        data_dict[index + 'survive_num_1'] = survive_num_1_list
        data_dict[index + 'survive_num_2'] = survive_num_2_list
        dp.get_csv(csv_url + net_type + '_' + timestamp + '_' + str(seed) + '.csv', data_dict)

# 统计的是total_reward以及每个epoch中测试时group2平均击杀数量
def epoch_train(env: MagentEnv, net_type, gamma=0.98, batch_size=5000, capacity=100000, 
    lr=1e-4, hidden_dim=32, nonlin='softmax', aggregate_form='mean', group_num=4,
    agent_num=20, opp_policy=None, prioritised_replay=True, model_save_url='../../data/model/',
    episodes_per_epoch=100, episodes_per_test=20, epoch_num=500, epsilon=1.0, step_epsilon=0.01, 
    use_cuda=True, tensorboard_data='../../data/log/data_info_',
    final_epsilon=0.01, save_data=True, csv_url='../../data/csv/', seed_flag=1, update_net=True,
    update_model_rate=100, print_info_rate=20, em_dim=32, print_info=True, concatenation=False):
    env_action_space = env.action_space.n
    env_obs_space = env.observation_space.shape[0]
    # group1作为对手，真正训练的是group2
    group1 = torch.load(opp_policy)
    group2 = IQL(env_obs_space, env_action_space, net_type, agent_num, group_num, concatenation,
                    gamma, batch_size, capacity, lr, hidden_dim, nonlin=nonlin, aggregate_form=aggregate_form,
                    prioritised_replay=prioritised_replay, target_net=update_net, use_cuda=use_cuda, em_dim=em_dim)

    # 测试时的平均值
    total_reward_list = []
    ave_kill_num_list = []
    ave_survive_num_list = []

    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    writer = SummaryWriter(tensorboard_data + net_type + '_' + timestamp)

    for epoch in range(epoch_num):
        print('epoch %d training starts' % epoch)

        for episode in range(episodes_per_epoch):
            obs = env.reset(use_random_init=False)
            done = False
            alive_info = env.get_live_agent()
            # print('episode ', str(episode), '\talive agent:', alive_info)

            while not done:
                # 由于我们已经训练好的对手习惯于从左上角开始攻击，因此将我们要训练的agent放到右下角
                # group1 采用环境中的id=1; group2 采用环境中的id=0
                group1_as, _ = group1.infer_action(obs[0], greedy=True)
                group2_as, _ = group2.infer_action(obs[1], epsilon=epsilon)
                next_obs, rewards, done, alive_info = env.step([group1_as, group2_as])
                
                alive_info = alive_info['agent_live']
                alive_agent_ids = env.get_group_agent_id(1)
                d = []
                cur_rewards = []
                for alive_agent_id in alive_agent_ids:
                    d.append(1 - alive_info[1][alive_agent_id])
                    cur_rewards.append(rewards[1][alive_agent_id])

                group2.push_data(obs[1], group2_as, cur_rewards, next_obs[1], d)

                # print('step: ', env.step_num)

                if print_info and env.step_num % print_info_rate == 0:
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup1 actions: ', group1_as)
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup2 actions: ', group2_as)
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\talive_info: ', alive_info)
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\trewards: ', rewards)

                obs = next_obs

            group2.learn()
            if update_net and episode % update_model_rate == 1:
                group2.update_target()
            epsilon = max(epsilon - step_epsilon, final_epsilon)
            print('trainging ... epoch %d episode %d is over' % (epoch, episode))
        
        print('test stage for epoch %d' % epoch)
        total_kill_num = 0
        total_survive_num = 0
        total_reward = 0
 
        for test_episode in range(episodes_per_test):
            obs = env.reset(use_random_init=False)
            done = False
            alive_info = None

            while not done:
                group1_as, _ = group1.infer_action(obs[0], greedy=True)
                group2_as, _ = group2.infer_action(obs[1], greedy=True)
                next_obs, rewards, done, alive_info = env.step([group1_as, group2_as])

                alive_info = alive_info['agent_live']
                # total_reward_1 += sum(rewards[0])
                total_reward += sum(rewards[1])

                if print_info and env.step_num % print_info_rate == 0:
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup1 actions: ', group1_as)
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup2 actions: ', group2_as)
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\talive_info: ', alive_info)
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\trewards: ', rewards)

                obs = next_obs

            total_kill_num += np.sum(alive_info[0] == 0)
            total_survive_num += np.sum(alive_info[1] != 0)
            print('test ... epoch %d episode %d' % (epoch, test_episode))
        
        epoch_total_reward = total_reward / episodes_per_test
        epoch_total_kill_num = total_kill_num / episodes_per_test
        epoch_total_survive_num = total_survive_num / episodes_per_test
        print('epoch %d | total reward for group2: %0.2f | total kill num: %0.2f | total survive num: %0.2f' % (epoch, epoch_total_reward, epoch_total_kill_num, epoch_total_survive_num))
        writer.add_scalar('train/total_reward_for_group2', epoch_total_reward, epoch)
        writer.add_scalar('train/kill_num_for_group2', epoch_total_kill_num, epoch)

        total_reward_list.append(epoch_total_reward)
        ave_kill_num_list.append(epoch_total_kill_num)
        ave_survive_num_list.append(epoch_total_survive_num)

    torch.save(group2, model_save_url + net_type + '_' + timestamp + '_' + str(seed_flag) + '.th')
    print('model is saved.')
    writer.close()

    if save_data:
        data_dict = {}
        index = 'seed(' + str(seed_flag) + ')'
        data_dict[index + 'total_reward'] = total_reward_list
        data_dict[index + 'kill_num'] = ave_kill_num_list
        data_dict[index + 'survive_num'] = ave_survive_num_list
        dp.get_csv(csv_url + net_type + '_' + timestamp + '.csv', data_dict)