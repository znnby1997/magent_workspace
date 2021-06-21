from env_gym_wrap import MagentEnv
import os
import torch
import numpy as np
import random
import argparse
from args import parser
from utils.settings import exp_name, set_seed, create_dir
from utils.net_config import net_cfg
from train.train_opp import train_opp_policy
from train.train_dqn import train_dqn
from train.train_a2c import train_a2c
from train.train_ppo import train_ppo

opt = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

if __name__ == "__main__":
    torch.set_num_threads(opt.threads_num)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dir_name = exp_name(
        opt, exclude_lst=[
            'gpu_id', 'threads_num', 'seed', 'root_url', 'max_step', 'episode_num', 'print_rate', 'opp_policy',
            'episodes_per_epoch', 'episodes_per_test', 'epoch_num', 'save_data', 'print_log', 'agent_num', 'map_size'
            'noisy_num', 'e_coef', 'lmbda', 'eps_clip', 'k_epoch'
        ], seed=opt.seed
    )

    # 设置相关数据保存路径
    root_url = opt.root_url
    cj_name = str(opt.agent_num) + 'v' + str(opt.agent_num) + 'v' + str(opt.noisy_num)
    render_url = os.path.join(root_url, dir_name, cj_name, 'render')
    model_url = os.path.join(root_url, dir_name, cj_name, 'model/')
    log_url = os.path.join(root_url, dir_name, cj_name, 'log/')
    csv_url = os.path.join(root_url, dir_name, cj_name, 'csv/')
    create_dir(render_url)
    create_dir(model_url)
    create_dir(log_url)
    create_dir(csv_url)

    # 保存参数设置
    hyper_file_name = log_url + 'h_para'
    with open(hyper_file_name, 'w') as f:
        f.write(str(opt))

    set_seed(opt.seed)

    model_tag = opt.model_tag

    if model_tag == 'opp':
        env = MagentEnv(
            agent_num=opt.agent_num, 
            map_size=opt.map_size, 
            max_step=opt.max_step, 
            opp_policy_random=True, 
            render_url=render_url, 
            noisy_agent_num=0
        )

        train_opp_policy(
            env=env,
            gamma=opt.gamma,
            batch_size=opt.batch_size,
            capacity=opt.capacity,
            lr=opt.lr,
            hidden_dim=opt.hidden_dim,
            model_save_url=model_url,
            episode_num=opt.episode_num,
            tensorboard_data=log_url,
            update_model_rate=opt.update_rate,
            print_info_rate=opt.print_rate,
            device=device
        )
    elif model_tag == 'dqn':
        env = MagentEnv(
            agent_num=opt.agent_num, 
            map_size=opt.map_size, 
            max_step=opt.max_step, 
            opp_policy_random=False, 
            render_url=render_url, 
            noisy_agent_num=opt.noisy_num
        )

        train_dqn(
            env=env,
            net=net_cfg[opt.net],
            gamma=opt.gamma, 
            batch_size=opt.batch_size, 
            capacity=opt.capacity, 
            lr=opt.lr, 
            hidden_dim=opt.hidden_dim,
            agent_num=opt.agent_num, 
            opp_policy=opt.opp_policy, 
            model_save_url=model_url,
            episodes_per_epoch=opt.episodes_per_epoch, 
            episodes_per_test=opt.episodes_per_test, 
            epoch_num=opt.epoch_num,  
            tensorboard_data=log_url, 
            save_data=opt.save_data, 
            csv_url=csv_url, 
            seed_flag=opt.seed, 
            update_model_rate=opt.update_rate,
            print_log=opt.print_log,
            device=device,
            agg=opt.agg_version
        )
    elif model_tag == 'a2c':
        env = MagentEnv(
            agent_num=opt.agent_num, 
            map_size=opt.map_size, 
            max_step=opt.max_step, 
            opp_policy_random=False, 
            render_url=render_url, 
            noisy_agent_num=opt.noisy_num
        )

        train_a2c(
            env=env, 
            net=net_cfg[opt.net], 
            gamma=opt.gamma, 
            lr=opt.lr, 
            hidden_dim=opt.hidden_dim,
            agent_num=opt.agent_num, 
            opp_policy=opt.opp_policy, 
            model_save_url=model_url, 
            epoch_num=opt.epoch_num,
            test_num=opt.episodes_per_test,
            train_rate=opt.episodes_per_epoch,
            tensorboard_data=log_url,
            data_buffer_limit=opt.capacity,
            save_data=opt.save_data,
            csv_url=csv_url,
            seed_flag=opt.seed,
            entr_w=opt.e_coef,
            print_log=opt.print_log,
            device=device,
            agg=opt.agg_version
        )
    elif model_tag == 'ppo':
        env = MagentEnv(
            agent_num=opt.agent_num, 
            map_size=opt.map_size, 
            max_step=opt.max_step, 
            opp_policy_random=False, 
            render_url=render_url, 
            noisy_agent_num=opt.noisy_num
        )

        train_ppo(
            env=env, 
            net=net_cfg[opt.net],
            gamma=opt.gamma, 
            lr=opt.lr, 
            hidden_dim=opt.hidden_dim, 
            agent_num=opt.agent_num,
            opp_policy=opt.opp_policy, 
            model_save_url=model_url, 
            episodes_per_epoch=opt.episodes_per_epoch, 
            episodes_per_test=opt.episodes_per_test, 
            epoch_num=opt.epoch_num,
            tensorboard_data=log_url, 
            data_buffer_limit=opt.capacity,
            save_data=opt.save_data, 
            csv_url=csv_url, 
            seed_flag=opt.seed, 
            lmbda=opt.lmbda, 
            eps_clip=opt.eps_clip, 
            entropy_coef=opt.e_coef,
            k_epoch=opt.k_epoch,
            print_log=opt.print_log,
            device=device,
            agg=opt.agg_version
        )
    else:
        raise argparse.ArgumentTypeError('argument model_tag is unknown')
