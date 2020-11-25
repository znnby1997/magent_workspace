import pandas as pd
import numpy as np

from tensorboardX import SummaryWriter
import time
import os 
import torch
import torch.optim as optim
from torch.distributions import Categorical
import random

from model.opp_model import Qnet, ReplayBuffer, learn
from model.dqn import QnetM, ReplayBufferM, learn_m
from model.a2c import ActorCritic, ParallelEnv, learn_a2c
from model.ppo import PPO, learn_ppo, DataSet
from env_gym_wrap import MagentEnv
from magent.builtin.rule_model import RandomActor
import utils.data_process as dp

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

"""
    net_dict = {
    'alw': AlwAttNet, 'dot_scale': DotScaleAttNet, 'dyan': Dyan,
    'gruga': GruGenAttNet, 'none', NoneNet, 'nonlinatt': NonlinAttNet,
    'gaa': GAA
}
"""

# env = MagentEnv(agent_num=20, map_size=15, max_step=200, opp_policy_random=True)

# 用于训练一个对手模型，自身对手为随机动作
def train_opp_policy(env: MagentEnv, gamma=0.98, batch_size=32, capacity=5000, 
        lr=1e-4, hidden_dim=32, model_save_url='../../data/model/', episode_num=5000, 
        tensorboard_data='../../data/log/data_info_', update_model_rate=20, print_info_rate=40, random_init=False):
    env_action_space = env.action_space.n
    env_obs_space = env.observation_space.shape[0] - 28 * 5
    print('action space: ', env_action_space)
    print('obs space: ', env_obs_space)

    # 定义模型
    q = Qnet(env_obs_space, env_action_space, hidden_dim).cuda()
    q_target = Qnet(env_obs_space, env_action_space, hidden_dim).cuda()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer(capacity)
    optimizer = optim.Adam(q.parameters(), lr=lr)

    group2 = RandomActor(env.env, env.handles[1])

    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    writer = SummaryWriter(tensorboard_data + 'opp_' + timestamp)
    
    group1_win_num = 0
    group2_win_num = 0
    model_path = model_save_url + 'opp_' + timestamp
    os.mkdir(model_path)

    for episode in range(episode_num):
        epsilon = max(0.01, 1.0 - 0.01 * episode)
        obs = env.reset(use_random_init=random_init)
        done = False
        total_reward = 0
        alive_info = env.get_live_agent()
        print('alive info: ', alive_info)

        while not done:
            # group 1决策
            group1_as = []
            for a_o_1 in obs[0]:
                group1_as.append(q.sample_action(torch.from_numpy(a_o_1).cuda().float(), epsilon))
            group2_as = group2.infer_action(obs[1])
            next_obs, rewards, done, alive_info = env.step([group1_as, group2_as])
            
            alive_info = alive_info['agent_live']
            alive_agent_ids = env.get_group_agent_id(0)

            cur_rewards = []
            for id, alive_agent_id in enumerate(alive_agent_ids):
                memory.put((obs[0][id], group1_as[id], rewards[0][alive_agent_id], next_obs[0][id], 1 - alive_info[0][alive_agent_id]))
                cur_rewards.append(rewards[0][alive_agent_id])
            total_reward += sum(cur_rewards)

            # if env.step_num % print_info_rate == 0:
            #     print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup1 actions: ', group1_as)
            #     print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup2 actions: ', group2_as)
            #     print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\talive_info: ', alive_info)
            #     print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\trewards: ', cur_rewards)

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
            torch.save(q_target, model_path + '/episode_' + str(episode) + '.th')

        writer.add_scalar('train/total_reward_per_episode_for_group1', total_reward, episode)
        writer.add_scalar('train/win_rate_for_group1', group1_win_num / (episode + 1), episode)
        writer.add_scalar('train/win_rate_for_group2', group2_win_num / (episode + 1), episode)

    q_target.load_state_dict(q.state_dict())
    torch.save(q_target, model_path + '/final_model.th')
    print('model is saved.')
    writer.close()

def test_opp(env: MagentEnv, model=None, episode_num=20, render=True, random_init=False):
    print('test opponent policy')
    agent_1 = torch.load(model)
    group2 = RandomActor(env.env, env.handles[1])
    alive_info = env.get_live_agent()
    total_reward_1_list, total_reward_2_list = [], []
    kill_num_1_list, kill_num_2_list = [], []
    survive_num_1_list, survive_num_2_list = [], []

    for episode in range(episode_num):
        obs = env.reset(use_random_init=random_init)
        done = False
        total_reward_1, total_reward_2 = 0, 0
        kill_num_1, kill_num_2 = 0, 0
        survive_num_1, survive_num_2 = 0, 0

        while not done:
            # agent actions in group1
            group1_as = []
            for o in obs[0]:
                group1_as.append(agent_1.sample_action(torch.from_numpy(o).cuda().float(), 0.01))
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
    

def test_model(env: MagentEnv, model=None, basic_model='dqn', episode_num=20, render=True, print_att_weight=False, net_flag='none', random_init=False,
                print_group_mask=False, csv_url='../../data/csv/', seed=0, save_data=True, print_info=True):
    print('test env')
    agent_1 = torch.load(model[0])  # 对手的模型
    agent_2 = torch.load(model[1])  # 测试模型
    alive_info = env.get_live_agent()
    win_num_1 = 0
    win_num_2 = 0
    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    print('timestamp  ', timestamp)

    total_reward_1_list, total_reward_2_list = [], []
    kill_num_1_list, kill_num_2_list = [], []
    survive_num_1_list, survive_num_2_list = [], []

    for episode in range(episode_num):
        obs = env.reset(use_random_init=random_init)
        done = False
        total_reward_1 = 0
        total_reward_2 = 0
        # csv_path = csv_url + timestamp + '_' + str(episode)
        # os.mkdir(csv_path)

        step_flag = 0
        while not done:
            # agent actions in group1
            weights_list = []
            group1_as = []
            for o in obs[0]:
                group1_as.append(agent_1.sample_action(torch.from_numpy(o).cuda().float(), 0.01))
            # agent actions in group2
            group2_as = []
            mask_list = []
            if basic_model == 'dqn':
                for o in obs[1]:
                    out = agent_2.sample_action(torch.from_numpy(o).cuda().float(), 0.01)
                    mask_list.append(out[1])
                    group2_as.append(out[0])
            elif basic_model == 'a2c':
                for o in obs[1]:
                    out = agent_2.pi(torch.from_numpy(o).cuda().float().reshape(1, -1), detach=True)
                    a = Categorical(out[0]).sample().cpu().numpy()
                    group2_as.append(a)
                    mask_list.append(out[1])
            elif basic_model == 'ppo':
                for o in obs[1]:
                    out = agent_2.pi(torch.from_numpy(o).cuda().float().reshape(1, -1), detach=True)
                    m = Categorical(out[0]).sample().item()
                    group2_as.append(m)
                    mask_list.append(out[1])
                # group2_as.append(agent_2.sample_action(torch.from_numpy(o).cuda().float(), 0.01))
                # if print_att_weight:
                #     weights_list.append(agent_2.get_weight().cpu().numpy())
            # 保存输出的attention weight
            # weights_list = np.array(weights_list, dtype=float)
            # np.save(csv_path + '/episode_' + str(episode) + '_step_' + str(step_flag) + 'weights.npy', weights_list)
            next_obs, rewards, done, alive_info = env.step([group1_as, group2_as], render=render)

            alive_info = alive_info['agent_live']
            total_reward_1 += sum(rewards[0])
            total_reward_2 += sum(rewards[1])
            if print_info:
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup1 actions: ', group1_as)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup2 actions: ', group2_as)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\trewards: ', rewards)
                print('step: ', str(step_flag), '\talive_info: ', alive_info)
                print('mask log: ', mask_list)

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

    print('Test is over. group1 ave total reward: %0.2f   group2 ave total reward: %0.2f   group1 ave kill: %0.2f   group2 ave kill: %0.2f   group1 ave survive: %0.2f   group2 ave survive: %0.2f' % (
            sum(total_reward_1_list) / episode_num, sum(total_reward_2_list) / episode_num, sum(kill_num_1_list) / episode_num, sum(kill_num_2_list) / episode_num, sum(survive_num_1_list) / episode_num, sum(survive_num_2_list) / episode_num))
    # 最后一行统计平均每个episode的值
    total_reward_1_list.append(sum(total_reward_1_list) / episode_num)
    total_reward_2_list.append(sum(total_reward_2_list) / episode_num)
    kill_num_1_list.append(sum(kill_num_1_list) / episode_num)
    kill_num_2_list.append(sum(kill_num_2_list) / episode_num)
    survive_num_1_list.append(sum(survive_num_1_list) / episode_num)
    survive_num_2_list.append(sum(survive_num_2_list) / episode_num)
    
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
        dp.get_csv(csv_url + 'test_' + timestamp + net_flag + '_' + str(seed) + '.csv', data_dict)
        print('over !!')


# 统计的是total_reward以及每个epoch中测试时group2平均击杀数量
def epoch_train(env: MagentEnv, net_type, gamma=0.98, batch_size=5000, capacity=100000, 
    lr=1e-4, hidden_dim=32, aggregate_form='mean', random_init=False,
    agent_num=20, opp_policy=None, model_save_url='../../data/model/',
    episodes_per_epoch=100, episodes_per_test=20, epoch_num=500, tensorboard_data='../../data/log/data_info_',
    save_data=True, csv_url='../../data/csv/', seed_flag=1, update_net=True, nonlin='softmax',
    update_model_rate=100, print_info_rate=20, print_info=True, concatenation=False, beta=0.1, need_diff=False, ig_num=5):
    env_action_space = env.action_space.n
    env_obs_space = env.observation_space.shape[0]
    # group1作为对手，真正训练的是group2
    agent_1 = torch.load(opp_policy)

    # 定义模型
    q = QnetM(env_obs_space, env_action_space, hidden_dim=hidden_dim, net_type=net_type, concatenation=concatenation,
        agent_num=agent_num, aggregate_form=aggregate_form, group_num=ig_num, nonlin=nonlin).cuda()
    q_target = QnetM(env_obs_space, env_action_space, hidden_dim=hidden_dim, net_type=net_type, concatenation=concatenation,
        agent_num=agent_num, aggregate_form=aggregate_form, group_num=ig_num, nonlin=nonlin).cuda()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBufferM(capacity)
    optimizer = optim.Adam(q.parameters(), lr=lr)
    # 测试时的平均值
    total_reward_list = []
    ave_kill_num_list = []
    ave_survive_num_list = []

    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    writer = SummaryWriter(tensorboard_data + net_type + '_' + timestamp)

    for epoch in range(epoch_num):
        print('epoch %d training starts' % epoch)

        for episode in range(episodes_per_epoch):
            epsilon = max(0.01, 1.0 - 0.01 * episode)
            obs = env.reset(use_random_init=random_init)
            done = False
            alive_info = env.get_live_agent()

            while not done:
                group1_as = []
                for o in obs[0]:
                    group1_as.append(agent_1.sample_action(torch.from_numpy(o).cuda().float(), 0.01))

                group2_as = []
                for o in obs[1]:
                    out = q.sample_action(torch.from_numpy(o).cuda().float(), epsilon)
                    group2_as.append(out[0])

                next_obs, rewards, done, alive_info = env.step([group1_as, group2_as])
                
                alive_info = alive_info['agent_live']
                alive_agent_ids = env.get_group_agent_id(1)

                cur_rewards = []
                for id, alive_agent_id in enumerate(alive_agent_ids):
                    memory.put((obs[1][id], group2_as[id], rewards[1][alive_agent_id], next_obs[1][id], 1 - alive_info[1][alive_agent_id]))
                    cur_rewards.append(rewards[1][alive_agent_id])


                # if print_info and episode % print_info_rate == 0:
                #     print('epoch %d episode %d log -- ' % (epoch, episode))
                #     print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup1 actions: ', group1_as)
                #     print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup2 actions: ', group2_as)
                #     print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\talive_info: ', alive_info)
                #     print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\trewards: ', rewards)
                #     print('weight log: ', weight_list)

                obs = next_obs

            if memory.size()>batch_size:
                learn_m(q, q_target, memory, optimizer, batch_size, beta, need_diff)

            if update_net and episode % update_model_rate == 1:
                q_target.load_state_dict(q.state_dict())
            print('trainging ... epoch %d episode %d is over' % (epoch, episode))
        
        print('test stage for epoch %d' % epoch)
        total_kill_num = 0
        total_survive_num = 0
        total_reward = 0
 
        for test_episode in range(episodes_per_test):
            obs = env.reset(use_random_init=random_init)
            done = False
            alive_info = None

            while not done:
                group1_as = []
                for o in obs[0]:
                    group1_as.append(agent_1.sample_action(torch.from_numpy(o).cuda().float(), 0.01))

                group2_as = []
                mask_list = []
                for o in obs[1]:
                    out = q_target.sample_action(torch.from_numpy(o).cuda().float(), 0.01)
                    mask_list.append(out[1])
                    group2_as.append(out[0])

                next_obs, rewards, done, alive_info = env.step([group1_as, group2_as])

                alive_info = alive_info['agent_live']
                alive_agent_ids = env.get_group_agent_id(1)
                cur_rewards = []
                for alive_agent_id in alive_agent_ids:
                    cur_rewards.append(rewards[1][alive_agent_id])
                total_reward += sum(cur_rewards)

                if print_info and test_episode % print_info_rate == 0:
                    print('epoch %d episode %d log -- ' % (epoch, episode))
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup1 actions: ', group1_as)
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup2 actions: ', group2_as)
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\talive_info: ', alive_info)
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\trewards: ', rewards)
                    print('mask log: ', mask_list)

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

    q_target.load_state_dict(q.state_dict())
    torch.save(q_target, model_save_url + net_type + '_' + timestamp + '_' + str(seed_flag) + '.th')
    print('model is saved.')
    writer.close()

    if save_data:
        print('saving data ....')
        data_dict = {}
        index = 'seed(' + str(seed_flag) + ')'
        data_dict[index + 'total_reward'] = total_reward_list
        data_dict[index + 'kill_num'] = ave_kill_num_list
        data_dict[index + 'survive_num'] = ave_survive_num_list
        dp.get_csv(csv_url + net_type + '_' + timestamp + '_' + str(seed_flag) + '.csv', data_dict)
        print('csv is saved.')

def epoch_train_a2c(train_env: MagentEnv, test_env: MagentEnv, net_type, gamma=0.98, lr=1e-4, hidden_dim=32, aggregate_form='mean',
    agent_num=20, opp_policy=None, model_save_url='../../data/a2c/model/', update_interval = 5, group_num=3,
    max_train_steps=50000, test_num=20, test_rate=100,  tensorboard_data='../../data/a2c/log/data_info_',
    save_data=True, csv_url='../../data/a2c/csv/', seed_flag=1, nonlin='softmax', random_init=False,
    update_model_rate=100, print_info_rate=20, print_info=True, concatenation=False, entr_w=0.02, print_log=False, ig_num=5):
    env_action_space = train_env.action_space.n
    env_obs_space = train_env.observation_space.shape[0]
    print('env_obs_space: ', env_obs_space)
    # group1作为对手，真正训练的是group2
    agent_1 = torch.load(opp_policy)

    # envs = ParallelEnv(n_train_processes, env)
    
    agent_2 = ActorCritic(obs_dim=env_obs_space, n_actions=env_action_space, hidden_dim=hidden_dim, net_type=net_type, concatenation=concatenation,
                agent_num=agent_num, aggregate_form=aggregate_form, group_num=ig_num, nonlin=nonlin).cuda()
    
    optimizer = optim.Adam(agent_2.parameters(), lr=lr)

    total_reward_list = []
    ave_kill_num_list = []
    ave_survive_num_list = []

    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    writer = SummaryWriter(tensorboard_data + net_type + '_' + timestamp)

    step_idx = 0
    train_step = 0
    obs = train_env.reset(use_random_init=random_init)
    epoch = 0
    while step_idx < max_train_steps:
        s_lst, a_lst, r_lst, mask_lst = [[] for _ in range(agent_num)], [[] for _ in range(agent_num)], \
                                        [[] for _ in range(agent_num)], [[] for _ in range(agent_num)]
        print('step idx %d ... interval update ...' % step_idx)
        for _ in range(update_interval):
            group1_as = []
            for o in obs[0]:
                group1_as.append(agent_1.sample_action(torch.from_numpy(o).cuda().float(), 0.01))

            group2_as = []
            alive_agent_ids = train_env.get_group_agent_id(1)
            for o, id in zip(obs[1], alive_agent_ids):
                out = agent_2.pi(torch.from_numpy(o).cuda().float().reshape(1, -1), detach=True)
                a = Categorical(out[0]).sample().cpu().numpy()
                group2_as.append(a)

                s_lst[id].append(o)
                a_lst[id].append(a)

            # print('train_env done', train_env.done, ' test_env done', test_env.done)
            s_prime, r, done, info = train_env.step([group1_as, group2_as])

            alive_info = info['agent_live']
            # cur_alive_agent_ids = env.get_group_agent_id(1)

            for id, alive_agent_id in enumerate(alive_agent_ids):
                r_lst[alive_agent_id].append(r[1][alive_agent_id])
                mask_lst[alive_agent_id].append(alive_info[1][alive_agent_id])

            if done:
                s_prime = train_env.reset(use_random_init=random_init)
            
            obs = s_prime
            step_idx += 1

        # learning
        print('learning ...')
        total_loss, actor_loss, critic_loss, entropy = [], [], [], []
        for o, id in zip(s_prime[1], alive_agent_ids):
            # print('agent ', id, ' alist len', len(a_lst[id]), ' slist len', len(s_lst[id]), ' rlist len', len(r_lst[id]), 'mask lst len', len(mask_lst[id]))
            loss = learn_a2c(agent_2, o, a_lst[id], s_lst[id], r_lst[id], mask_lst[id], optimizer, entr_w, print_log)
            total_loss.append(loss[0])
            actor_loss.append(loss[1])
            critic_loss.append(loss[2])
            entropy.append(loss[3])
        
        if print_log:
            writer.add_scalar('train_loss/total_loss', torch.mean(torch.tensor(total_loss).cuda().float()).cpu().numpy(), train_step)
            writer.add_scalar('train_loss/actor_loss', torch.mean(torch.tensor(actor_loss).cuda().float()).cpu().numpy(), train_step)
            writer.add_scalar('train_loss/critic_loss', torch.mean(torch.tensor(critic_loss).cuda().float()).cpu().numpy(), train_step)
            writer.add_scalar('train_loss/entropy_loss', torch.mean(torch.tensor(entropy).cuda().float()).cpu().numpy(), train_step)
            train_step += 1

        if step_idx % test_rate == 0:
            total_reward, total_kill_num, total_survive_num = a2c_test(step_idx, test_env, test_num, agent_1, agent_2, print_info, print_info_rate, random_init)
            
            epoch_total_reward = total_reward / test_num
            epoch_total_kill_num = total_kill_num / test_num
            epoch_total_survive_num = total_survive_num / test_num
            print('step idx %d | total reward for group2: %0.2f | total kill num: %0.2f | total survive num: %0.2f' % (step_idx, epoch_total_reward, epoch_total_kill_num, epoch_total_survive_num))
            writer.add_scalar('train/total_reward_for_group2', epoch_total_reward, epoch)
            writer.add_scalar('train/kill_num_for_group2', epoch_total_kill_num, epoch)
            writer.add_scalar('train/survive_num_for_group2', epoch_total_survive_num, epoch)

            total_reward_list.append(epoch_total_reward)
            ave_kill_num_list.append(epoch_total_kill_num)
            ave_survive_num_list.append(epoch_total_survive_num)

            epoch += 1

    print('learning is over | epoch num ', epoch)
    torch.save(agent_2, model_save_url + net_type + '_' + timestamp + '_' + str(seed_flag) + '.th')
    print('model is saved.')
    writer.close()

    if save_data:
        data_dict = {}
        index = 'seed(' + str(seed_flag) + ')'
        data_dict[index + 'total_reward'] = total_reward_list
        data_dict[index + 'kill_num'] = ave_kill_num_list
        data_dict[index + 'survive_num'] = ave_survive_num_list
        dp.get_csv(csv_url + net_type + '_' + timestamp + '_' + str(seed_flag) + '.csv', data_dict)


def a2c_test(step_idx, test_env, test_num, agent_1, agent_2, print_info, print_info_rate, random_init):
    print('step idx %d ... test stage starting ...' % step_idx)
    total_kill_num = 0
    total_survive_num = 0
    total_reward = 0

    for test_episode in range(test_num):
        obs_t = test_env.reset(use_random_init=random_init)
        done = False
        alive_info = None

        while not done:
            group1_as = []
            for o in obs_t[0]:
                group1_as.append(agent_1.sample_action(torch.from_numpy(o).cuda().float(), 0.01))

            group2_as = []
            mask_list = []
            for o in obs_t[1]:
                out = agent_2.pi(torch.from_numpy(o).cuda().float().reshape(1, -1), detach=True)
                a = Categorical(out[0]).sample().cpu().numpy()
                group2_as.append(a)
                mask_list.append(out[1])

            next_obs_t, rewards, done, alive_info = test_env.step([group1_as, group2_as])

            alive_info = alive_info['agent_live']
            alive_agent_ids = test_env.get_group_agent_id(1)
            cur_rewards = []
            for alive_agent_id in alive_agent_ids:
                cur_rewards.append(rewards[1][alive_agent_id])
            total_reward += sum(cur_rewards)

            if print_info and test_episode % print_info_rate == 0:
                print('step idx %d episode %d log -- ' % (step_idx, test_episode))
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup1 actions: ', group1_as)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup2 actions: ', group2_as)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\talive_info: ', alive_info)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\trewards: ', rewards)
                print('mask log: ', mask_list)

            obs_t = next_obs_t

        total_kill_num += np.sum(alive_info[0] == 0)
        total_survive_num += np.sum(alive_info[1] != 0)
        print('test ... step idx %d episode %d' % (step_idx, test_episode))
        
    return total_reward, total_kill_num, total_survive_num


def epoch_train_ppo(env: MagentEnv, net_type, gamma=0.98, 
    lr=1e-4, hidden_dim=32, aggregate_form='mean', group_num=3,
    agent_num=20, opp_policy=None, model_save_url='../../data/model/', random_init=False,
    episodes_per_epoch=100, episodes_per_test=20, epoch_num=500, tensorboard_data='../../data/log/data_info_',
    save_data=True, csv_url='../../data/csv/', seed_flag=1, nonlin='softmax', 
    print_info_rate=20, print_info=True, concatenation=False, k_epoch=3, lmbda=0.95, eps_clip=0.1, t_horizon=20, beta=0.1, print_log=False, ig_num=5):
    env_action_space = env.action_space.n
    env_obs_space = env.observation_space.shape[0]
    print('env action space: ', env_action_space, ' env obs space: ', env_obs_space)
    # group1作为对手，真正训练的是group2
    agent_1 = torch.load(opp_policy)

    agent_2 = PPO(obs_dim=env_obs_space, n_actions=env_action_space, hidden_dim=hidden_dim, net_type=net_type, concatenation=concatenation,
                agent_num=agent_num, aggregate_form=aggregate_form, group_num=ig_num, nonlin=nonlin).cuda()
    
    # 这里为每个agent都定义一个Dataset，考虑到ppo on-policy
    data_sets = [DataSet() for _ in range(agent_num)]
    optimizer = optim.Adam(agent_2.parameters(), lr=lr)

    total_reward_list = []
    ave_kill_num_list = []
    ave_survive_num_list = []

    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    writer = SummaryWriter(tensorboard_data + net_type + '_' + timestamp)

    train_step = 0

    for epoch in range(epoch_num):
        print('epoch %d training starts' % epoch)

        for episode in range(episodes_per_epoch):
            obs = env.reset(use_random_init=random_init)
            done = False
            alive_info = env.get_live_agent()

            while not done:
                for t in range(t_horizon):
                    group1_as = []
                    for o in obs[0]:
                        group1_as.append(agent_1.sample_action(torch.from_numpy(o).cuda().float(), 0.01))

                    group2_as = []
                    group2_probs = []
                    for o in obs[1]:
                        out = agent_2.pi(torch.from_numpy(o).cuda().float().reshape(1, -1), detach=True)
                        m = Categorical(out[0]).sample().item()
                        group2_as.append(m)
                        group2_probs.append(out[0].squeeze())

                    next_obs, rewards, done, alive_info = env.step([group1_as, group2_as])
                
                    alive_info = alive_info['agent_live']
                    alive_agent_ids = env.get_group_agent_id(1)

                    # cur_rewards = []
                    # print('alive id: ', alive_agent_ids)
                    for id, alive_agent_id in enumerate(alive_agent_ids):
                        # print('group2 probs: ', len(group2_probs), '\t', group2_probs[0].shape)
                        data_sets[id].put_data((obs[1][id], group2_as[id], rewards[1][alive_agent_id], next_obs[1][id], group2_probs[id][group2_as[id]].item(), 1 - alive_info[1][alive_agent_id]))
                        # cur_rewards.append(rewards[1][alive_agent_id])

                    obs = next_obs
                    if done:
                        print('episode %d is over ...' % (episode))
                        break
                
                loss = None
                print('learning ...')
                for data in data_sets:
                    # print(data.data)
                    if data.not_none():
                        # s_prime有的才拿来训练
                        loss = learn_ppo(agent_2, data, optimizer, k_epoch, lmbda, eps_clip, beta, print_log)
        
                if print_log and loss != None:
                    writer.add_scalar('train_loss/total_loss', loss[0].item(), train_step)
                    writer.add_scalar('train_loss/actor_loss', loss[1].item(), train_step)
                    writer.add_scalar('train_loss/critic_loss', loss[2].item(), train_step)
                    writer.add_scalar('train_loss/entropy_loss', loss[3].item(), train_step)
                    train_step += 1

            print('trainging ... epoch %d episode %d is over' % (epoch, episode))
        
        print('test stage for epoch %d' % epoch)
        total_kill_num = 0
        total_survive_num = 0
        total_reward = 0
 
        for test_episode in range(episodes_per_test):
            obs = env.reset(use_random_init=random_init)
            done = False
            alive_info = None

            while not done:
                group1_as = []
                for o in obs[0]:
                    group1_as.append(agent_1.sample_action(torch.from_numpy(o).cuda().float(), 0.01))

                group2_as = []
                mask_list = []
                for o in obs[1]:
                    out = agent_2.pi(torch.from_numpy(o).cuda().float().reshape(1, -1), detach=True)
                    m = Categorical(out[0]).sample().item()
                    group2_as.append(m)
                    mask_list.append(out[1])
                
                next_obs, rewards, done, alive_info = env.step([group1_as, group2_as])

                alive_info = alive_info['agent_live']
                alive_agent_ids = env.get_group_agent_id(1)
                cur_rewards = []
                for alive_agent_id in alive_agent_ids:
                    cur_rewards.append(rewards[1][alive_agent_id])
                total_reward += sum(cur_rewards)

                if print_info and test_episode % print_info_rate == 0:
                    print('epoch %d episode %d log -- ' % (epoch, episode))
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup1 actions: ', group1_as)
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\tgroup2 actions: ', group2_as)
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\talive_info: ', alive_info)
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '\trewards: ', rewards)
                    print('mask log: ', mask_list)

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
        writer.add_scalar('train/survive_num_for_group2', epoch_total_survive_num, epoch)

        total_reward_list.append(epoch_total_reward)
        ave_kill_num_list.append(epoch_total_kill_num)
        ave_survive_num_list.append(epoch_total_survive_num)

    torch.save(agent_2, model_save_url + net_type + '_' + timestamp + '_' + str(seed_flag) + '.th')
    print('model is saved.')
    writer.close()

    if save_data:
        print('saving data ....')
        data_dict = {}
        index = 'seed(' + str(seed_flag) + ')'
        data_dict[index + 'total_reward'] = total_reward_list
        data_dict[index + 'kill_num'] = ave_kill_num_list
        data_dict[index + 'survive_num'] = ave_survive_num_list
        dp.get_csv(csv_url + net_type + '_' + timestamp + '_' + str(seed_flag) + '.csv', data_dict)
        print('csv is saved.')