import argparse

def str_to_bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Must be true or false')

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default=3, type=int, help='configure which gpu to train')
parser.add_argument('--threads_num', default=1, type=int, help='configure threads num')
parser.add_argument('--seed', default=0, type=int, help='configure seed')
parser.add_argument('--root_url', default='./', type=str, help='in which data saved')

# environment configuration
parser.add_argument('--agent_num', default=5, type=int, help='num of agents in a group')
parser.add_argument('--map_size', default=10, type=int, help='width and height of the map')
parser.add_argument('--max_step', default=200, type=int, help='max step in an episode')
parser.add_argument('--noisy_num', default=5, type=int, help='noisy agents num')

# model configuration
parser.add_argument('--model_tag', default='ppo', type=str, help='basic rl model')
parser.add_argument('--net', default='none', type=str, help='net type')
parser.add_argument('--gamma', default=0.98, type=float, help='discount factor')
parser.add_argument('--batch_size', default=256, type=int, help='sample minibatch')
parser.add_argument('--capacity', default=5000, type=int, help='exp buffer limit')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--hidden_dim', default=32, type=int, help='hidden layer dim')
parser.add_argument('--episode_num', default=200, type=int, help='the number of episode for opponent training')
parser.add_argument('--update_rate', default=100, type=int, help='dqn target net update rate')
parser.add_argument('--print_rate', default=40, type=int, help='print log rate for opponent')
parser.add_argument('--opp_policy', default='', type=str, help='opp policy th')
parser.add_argument('--episodes_per_epoch', default=200, type=int, help='episodes per epoch')
parser.add_argument('--episodes_per_test', default=20, type=int, help='episodes per test')
parser.add_argument('--epoch_num', default=200, type=int, help='epoch num')
parser.add_argument('--save_data', default=True, type=str_to_bool, help='save data or not')
parser.add_argument('--print_log', default=True, type=str_to_bool, help='print log or not')
parser.add_argument('--e_coef', default=0.02, type=float, help='entropy coefficient')
parser.add_argument('--lmbda', default=0.95, type=float, help='gae lmbda')
parser.add_argument('--eps_clip', default=0.1, type=float, help='clip function coefficient')
parser.add_argument('--k_epoch', default=3, type=int, help='k times training for a sample data')
