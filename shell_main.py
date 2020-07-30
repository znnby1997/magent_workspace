import argparse
from env_gym_wrap import MagentEnv
import main
import os
import torch
import numpy as np
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
use_cuda = torch.cuda.is_available()
cuda_id = torch.cuda.current_device()
torch.set_num_threads(2)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--agent_num', type=int, default=20)
    parse.add_argument('--map_size', type=int, default=15)
    parse.add_argument('--max_step', type=int, default=200)
    parse.add_argument('--train_opp', type=int, default=0)
    
    parse.add_argument('--print_info_rate', type=int, default=20)

    parse.add_argument('--net_type', type=str, default='none')
    parse.add_argument('--prioritised_replay', type=int, default=1)
    parse.add_argument('--model_save_url', type=str, default='../../data/model/')
    parse.add_argument('--episode_num', type=int, default=5000)
    parse.add_argument('--epsilon', type=float, default=1.0)
    parse.add_argument('--epsilon_step', type=float, default=0.01)
    parse.add_argument('--tensorboard_data', type=str, default='../../data/log/data_info_')
    parse.add_argument('--final_epsilon', type=float, default=0.01)
    parse.add_argument('--save_data', type=int, default=1)
    parse.add_argument('--csv_url', type=str, default='../../data/csv/')
    parse.add_argument('--seed', type=int, default=0)
    parse.add_argument('--update_net', type=int, default=1)
    parse.add_argument('--update_model_rate', type=int, default=100)
    parse.add_argument('--opp_policy', type=str, default=None)

    parse.add_argument('--gamma', type=float, default=0.98)
    parse.add_argument('--batch_size', type=int, default=5000)
    parse.add_argument('--capacity', type=int, default=100000)
    parse.add_argument('--learning_rate', type=float, default=1e-4)
    parse.add_argument('--hidden_dim', type=int, default=32)

    args = parse.parse_args()
    agent_num = args.agent_num
    map_size = args.map_size
    max_step = args.max_step

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_opp = args.train_opp
    
    net_type = args.net_type
    prioritised_replay = args.prioritised_replay
    model_save_url = args.model_save_url
    episode_num = args.episode_num
    epsilon = args.epsilon
    epsilon_step = args.epsilon_step
    tensorboard_data = args.tensorboard_data
    final_epsilon = args.final_epsilon
    save_data = args.save_data
    csv_url = args.csv_url
    update_net = args.update_net
    update_model_rate = args.update_model_rate
    print_info_rate = args.print_info_rate
    opp_policy = args.opp_policy

    gamma = args.gamma
    batch_size = args.batch_size
    capacity = args.capacity
    learning_rate = args.learning_rate
    hidden_dim = args.hidden_dim

    print('current env info: ', 'GPU ', use_cuda, ' GPU id ', cuda_id)

    print('use parameters: {}'.format(args))

    if train_opp:
        env = MagentEnv(agent_num=agent_num, map_size=map_size, max_step=max_step, opp_policy_random=True)
        main.train_opp_policy(env, net_type, gamma, batch_size, capacity, learning_rate, hidden_dim, agent_num, 
            prioritised_replay, model_save_url, episode_num, epsilon, epsilon_step,
            tensorboard_data, final_epsilon, save_data, csv_url, seed, update_net, update_model_rate, 
            print_info_rate, use_cuda)
    else:
        env = MagentEnv(agent_num=agent_num, map_size=map_size, max_step=max_step, opp_policy_random=False)
        main.train(env, net_type, gamma, batch_size, capacity, learning_rate, hidden_dim, agent_num, opp_policy, 
            prioritised_replay, model_save_url, episode_num, epsilon,
            epsilon_step, use_cuda, tensorboard_data, final_epsilon, save_data, csv_url, seed, update_net,
            update_model_rate, print_info_rate)



    

