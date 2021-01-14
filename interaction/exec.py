import torch
import numpy as np

import sys
sys.path.append('..')

from utils.sample_action import sample_a_from_q, sample_a_from_pi
from model.ac import ActorCritic
from model.dqn import QnetM

def exec_for_coll(env, model_tag, model, epsilon, opp_policy, device):
    obs = env.reset()
    done = False
    alive_info = None
    data_buffer = [[] for _ in range(env.agent_num)] if model_tag == 'ppo' else []

    while not done:
        opp_as = []
        agent_as = []
        agent_probs = []

        for o in obs[0]:
            opp_as.append(opp_policy.sample_action(torch.from_numpy(o).to(device).float(), 0.01))
        
        if isinstance(model, ActorCritic):
            for o in obs[1]:
                action, a_prob, _ = sample_a_from_pi(model, torch.from_numpy(o).to(device).float())
                agent_as.append(action)
                agent_probs.append(a_prob)

        elif isinstance(model, QnetM):
            for o in obs[1]:
                action, _ = sample_a_from_q(model, torch.from_numpy(o).to(device).float(), epsilon)
                agent_as.append(action)

        next_obs, rewards, done, alive_info = env.step([opp_as, agent_as])

        alive_info = alive_info['agent_live']
        alive_agents_ids = env.get_group_agent_id(1)

        for id, alive_agent_id in enumerate(alive_agents_ids):
            if model_tag != 'ppo':
                data_buffer.append((obs[1][id], agent_as[id], rewards[1][alive_agent_id], next_obs[1][id], 1 - alive_info[1][alive_agent_id]))
            else:
                data_buffer[id].append((obs[1][id], agent_as[id], rewards[1][alive_agent_id], next_obs[1][id], agent_probs[id], 1 - alive_info[1][alive_agent_id]))

        obs = next_obs
    
    return data_buffer

def exec_(env, model, epsilon, opp_policy, device, render=False):
    obs = env.reset()
    done = False
    alive_info = None
    opp_total_reward, agent_total_reward = 0.0, 0.0
    
    while not done:
        opp_as = []
        agent_as = []
        attn_lst = []

        for o in obs[0]:
            opp_as.append(opp_policy.sample_action(torch.from_numpy(o).to(device).float(), 0.01))

        if isinstance(model, ActorCritic):
            for o in obs[1]:
                action, _, attn = sample_a_from_pi(model, torch.from_numpy(o).to(device).float())
                agent_as.append(action)
                attn_lst.append(attn)

        elif isinstance(model, QnetM):
            for o in obs[1]:
                action, attn = sample_a_from_q(model, torch.from_numpy(o).to(device).float(), 0.01)
                agent_as.append(action)
                attn_lst.append(attn)
        
        next_obs, rewards, done, alive_info = env.step([opp_as, agent_as], render)

        opp_total_reward += sum(rewards[0])
        agent_total_reward += sum(rewards[1])

        obs = next_obs

    alive_info = alive_info['agent_live']
    opp_kill_num = np.sum(alive_info[1] == 0)
    agent_kill_num = np.sum(alive_info[0] == 0)
    opp_survive_num = np.sum(alive_info[0] != 0)
    agent_survive_num = np.sum(alive_info[1] != 0)

    return agent_total_reward, agent_kill_num, agent_survive_num, opp_total_reward, opp_kill_num, opp_survive_num





        



        
