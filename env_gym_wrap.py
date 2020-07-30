import numpy as np
import magent
import itertools
from gym import spaces
import time

def generate_map(env, id_sets, pos_sets, handles):
    env.reset()

    left_pos = pos_sets[0]
    left_id = id_sets[0]
    env.add_agents(handles[left_id], method='custom', pos=left_pos)

    right_pos = pos_sets[1]
    right_id = id_sets[1]
    env.add_agents(handles[right_id], method='custom', pos=right_pos)

def obs_info_extract(views, features, group_id, max_agent, debug=False, have_env_info=False):
    # 这里的views和features表示了一个group的
    # group_id=0表示group1，1表示group2
    assert len(views.shape) == 4

    """提取后的观测向量：[agent自身信息，环境信息，观测到的各个对手信息，观测到的各个队友信息]
        agent自身信息：feature包含需要的agent自身信息(id embedding, last action, last reward, relative pos)
                    同时包含自身血量、自身所在group、自身minimap
        环境信息：观测到的wall的13 * 13维向量
        其他agent信息：血量, 所在group, 自身minimap，相对观测者的位置
    """
    self_infos, opp_infos, partner_infos = [], [], []
    env_info = views[0, :, :, 0].flatten() # 环境信息
    # 提取自身观测信息向量
    for view, feature in zip(views, features):
        self_view_vector = view[6][6] # 13*13观测视角中, 自身观测在中间位置
        self_view_info = np.stack([self_view_vector[2], group_id, self_view_vector[3]]) # 血量、group、minimap
        self_info = np.append(feature, self_view_info)
        self_infos.append(self_info)

    # 提取对手以及队友观测信息向量
    for view in views:
        opp_info = np.zeros((max_agent, 2 + view.shape[0] * 2))
        partner_info = np.zeros((max_agent - 1, 2 + view.shape[0] * 2))
        opp_index = 0
        partner_index = 0
        for row in range(view.shape[0]):
            for col in range(view.shape[1]):
                if row != 6 and col != 6 and view[row][col][1] == 1:
                    # 队友
                    partner_info[partner_index][0] = view[row][col][2] # 血量
                    partner_info[partner_index][1] = group_id
                    partner_info[partner_index][2 + row] = 1 # x
                    partner_info[partner_index][2 + view.shape[0] + col] = 1 # y
                    partner_index += 1
                elif row != 6 and col != 6 and view[row][col][4] == 1:
                    # 对手
                    opp_info[opp_index][0] = view[row][col][5]
                    opp_info[opp_index][1] = 1 - group_id
                    opp_info[opp_index][2 + row] = 1
                    opp_info[opp_index][2 + view.shape[0] + col] = 1
                    opp_index += 1
        opp_infos.append(opp_info.flatten())
        partner_infos.append(partner_info.flatten())
    
    # 整合
    self_infos = np.vstack(self_infos)
    opp_infos = np.vstack(opp_infos)
    partner_infos = np.vstack(partner_infos)
    env_infos = np.tile(env_info, (self_infos.shape[0], 1))
    # print('self infos ', self_infos.shape, ' opp infos ', opp_infos.shape, ' partner infos ', partner_infos.shape, ' env infos ', env_infos.shape)
    output = None
    if have_env_info:
        output = np.hstack((self_infos, env_infos, opp_infos, partner_infos))
    else:
        output = np.hstack((self_infos, opp_infos, partner_infos))
    return output


obs_input = obs_info_extract


class MagentEnv:
    def __init__(self, agent_num=20, map_size=15, id_set=[0, 1], shift_delta=0, max_step=100, have_env_info=False, opp_policy_random=True):
        self.agent_num = agent_num

        magent.utility.init_logger('battle')
        self.env = magent.GridWorld("battle", map_size=map_size)
        self.env.set_render_dir("../../data/render/render_" + time.strftime('%Y%m%d%H%M%S', time.localtime()))
        self.handles = self.env.get_handles()

        self.map_size = map_size
        self.agent_num = agent_num
        self.have_env_info = have_env_info

        self.id_set = id_set
        self.n_group = len(id_set)
        self.shift_delta = shift_delta
        self.opp_policy_random = opp_policy_random

        # i取值((map_size - 1)/2 - 5, (map_size - 1)/2)
        # (4-5, 4+5)
        # # 20vs20 15*15map
        if agent_num == 20 and map_size == 15:
            self.left_side_config = [[i - shift_delta, j - shift_delta] for j in range(1, 8) for i in range(1, 4)]
            self.right_side_config = [[i - shift_delta, j - shift_delta] for j in range(13, 6, -1) for i in range(13, 10, -1)]
        elif agent_num == 10 and map_size == 10:
            # 10vs10 10*10map
            self.left_side_config = [[i - shift_delta, j - shift_delta] for j in range(1, 5) for i in range(1, 4)]
            self.right_side_config = [[i - shift_delta, j - shift_delta] for j in range(8, 4, -1) for i in range(8, 5, -1)]
    
        self.done = False
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {}

        self.step_num = 0
        self.max_step = max_step

        self.group_0_reward = []
        self.group_1_reward = []

    def get_obs(self):
        obs = []
        # 得到活着的agent的id
        agents_id = self.get_group_agent_id(0)
        views, features = self.env.get_observation(self.handles[0])
        obs.append(obs_input(views, features, 0, self.agent_num, have_env_info=self.have_env_info))

        if self.opp_policy_random:
            opp_obs = self.env.get_observation(self.handles[1])
            obs.append(opp_obs)
        else:
            views, features = self.env.get_observation(self.handles[1])
            obs.append(obs_input(views, features, 1, self.agent_num, have_env_info=self.have_env_info))
        return obs

    def get_group_agent_id(self, group_id):
        return self.env.get_agent_id(self.handles[group_id]) - self.agent_num * group_id

    def get_reward(self):
        reward = []
        for i in range(self.n_group):
            group_rewards = np.zeros(self.agent_num)

            real_rewards = self.env.get_reward(self.handles[i])
            # print(real_rewards)

            # 辅助的reward，引导agent占领中间
            pos = self.env.get_pos(self.handles[i])
            # print('group' + str(i + 1) + ' pos: ', pos)
            for (x, y) in pos:
                real_rewards -= ((1.0 * x / self.map_size - 0.5) ** 2 + (1.0 * y / self.map_size - 0.5) ** 2) / 100

            agents_id = self.get_group_agent_id(i)
            group_rewards[agents_id] = real_rewards

            reward.append(group_rewards)

        return reward

    def reset(self, use_random_init=False):
        # print('reset!!!!!')
        # print(len(self.env.get_alive(self.handles[0])), len(self.env.get_alive(self.handles[1])))

        self.done = False

        if use_random_init:
            random_pos_set = np.array(list(itertools.product(range(2, self.map_size-2), range(2, self.map_size-2))))[np.random.choice(range((self.map_size-4)**2), 2 * self.agent_num, replace=False)]
            pos_set = [random_pos_set[:self.agent_num], random_pos_set[self.agent_num:]]
        else:
            pos_set = [self.left_side_config[:self.agent_num],
                       self.right_side_config[:self.agent_num]]

        generate_map(self.env, self.id_set, pos_set, self.handles)

        obs = self.get_obs()

        return obs
    
    def step(self, actions, render=False):
        # print(actions)
        # 需要知道活着的agent的固定id，要不然无法对应next_obs
        assert not self.done, 'you must reset the env'

        self.step_num += 1

        # id_0 = self.env.get_agent_id(self.handles[0])
        # id_1 = self.env.get_agent_id(self.handles[1])

        # print(id_1)

        self.env.set_action(self.handles[0], np.array(actions[0], dtype=np.int32))
        self.env.set_action(self.handles[1], np.array(actions[1], dtype=np.int32))

        self.done = self.env.step()

        if render:
            self.env.render()

        obs = self.get_obs()

        reward = self.get_reward()

        if self.step_num > self.max_step or self.done:
            print('done !!!!!!!!', self.step_num)
            self.step_num = 0
            self.done = True

        self.env.clear_dead()

        return obs, reward, self.done, {'agent_live': self.get_live_agent()}

    def get_live_agent(self):
        live = []

        for i in range(self.n_group):
            group_live = np.zeros(self.agent_num) != 0

            agent_id = self.get_group_agent_id(i)

            real_live = self.env.get_alive(self.handles[i])

            group_live[agent_id] = real_live

            live.append(group_live)

        return live

    @property
    def action_space(self):
        # 8 + 13
        return spaces.Discrete(21)
    
    @property
    def observation_space(self):
        self_info_space = self.env.get_feature_space(self.handles[0])[0] + 3 # 血量, group, minimap size: 34 + 3 = 37
        env_info_space = 0
        if self.have_env_info:
            env_info_space = self.env.get_view_space(self.handles[0])[0]**2
        
        other_agent_info_space = 28 * (self.agent_num * 2 - 1) # size: 28 * 9 = 252
        return spaces.Box(low=0, high=1, shape=(self_info_space + env_info_space + other_agent_info_space, ), dtype=np.float32)


if __name__ == '__main__':
    env = MagentEnv()
    print('action space: ', env.action_space.n)
    print('obs space: ', env.observation_space.shape[0])
    obs = env.reset()
    env.env.render()
    # print('group1 obs: ', obs[0][0])
    # print('group2 obs: ', obs[1])
    # for i in range(len(obs[0])):
    #     print(obs[0][i])
    # obs, reward, done, alive_info = env.step(env.action_space.sample())
    # print(alive_info)