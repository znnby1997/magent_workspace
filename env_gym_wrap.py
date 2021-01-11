import numpy as np
import magent
from magent.builtin.rule_model import RandomActor
import itertools
from gym import spaces
import time

def generate_map(env, pos_sets, handles):
    env.reset()

    for pos, handle in zip(pos_sets, handles):
        if len(pos) > 0:
            env.add_agents(handle, method='custom', pos=pos)

""" 从环境的原始观测中提取信息作为我们的原始观测
"""
def obs_info_extract(views, features, group_id, max_agent, noisy_agent_num, debug=False, have_env_info=False):
    # 这里的views和features表示了一个group的
    # group_id=0表示group1，1表示group2
    # 加了噪声agent之后，每多一个group，就会多3个bit（group, group's hp, group's minimap)
    assert len(views.shape) == 4

    """提取后的观测向量：[agent自身信息，环境信息，观测到的各个对手信息，观测到的各个队友信息]
        agent自身信息：feature包含需要的agent自身信息(id embedding, last action, last reward, relative pos)
                    同时包含自身血量、自身所在group
        环境信息：观测到的wall的13 * 13维向量
        其他agent信息：血量, 所在group, 相对观测者的位置
    """
    self_infos, opp_infos, partner_infos = [], [], []
    env_info = views[0, :, :, 0].flatten() # 环境信息
    # 提取自身观测信息向量
    for view, feature in zip(views, features):
        self_view_vector = view[6][6] # 13*13观测视角中, 自身观测在中间位置
        self_view_info = np.stack([self_view_vector[2], group_id]) # 血量、group minimap(去掉)
        self_info = np.append(feature, self_view_info)
        self_infos.append(self_info)

    # 提取对手以及队友观测信息向量
    for view in views:
        opp_info = np.zeros((max_agent + noisy_agent_num, 2 + view.shape[0] * 2))
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
                
                if noisy_agent_num > 0 and row != 6 and col != 6 and view[row][col][7] == 1:
                    # 噪声agent，作为对手信息，混合在对手中
                    opp_info[opp_index][0] = view[row][col][8]
                    opp_info[opp_index][1] = 2
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
    """
        环境设置中所有agent的开始位置随机
    """
    def __init__(self, agent_num=20, map_size=15, id_set=[0, 1], max_step=100, noisy_agent_num=0,
                    have_env_info=False, opp_policy_random=True, render_url=''):
        magent.utility.init_logger('battle')
        self.env = magent.GridWorld("battle", map_size=map_size, noisy_agent_num=noisy_agent_num)
        self.env.set_render_dir(render_url)
        # self.env.set_render_dir("render_" + time.strftime('%Y%m%d%H%M%S', time.localtime()))
        self.handles = self.env.get_handles()

        self.map_size = map_size
        self.agent_num = agent_num
        self.have_env_info = have_env_info

        self.id_set = id_set
        self.n_group = len(id_set)
        self.opp_policy_random = opp_policy_random

        self.noisy_agent_num = noisy_agent_num
   
        self.done = False

        self.step_num = 0
        self.max_step = max_step

        self.noisy_group = None

    def get_obs(self):
        obs = []
        # 得到活着的agent的id
        agents_id = self.get_group_agent_id(0)

        # handles[0]实际上是最后训练出来的对手的观测,既不需要排序也不需要噪声
        views, features = self.env.get_observation(self.handles[0])
        obs.append(obs_input(views, features, 0, self.agent_num, noisy_agent_num=0, have_env_info=self.have_env_info))

        if self.opp_policy_random:
            opp_obs = self.env.get_observation(self.handles[1])
            obs.append(opp_obs)
        else:
            # 真正的训练模型使用的观测，可以包含排序以及噪声
            views, features = self.env.get_observation(self.handles[1])
            obs.append(obs_input(views, features, 1, self.agent_num, noisy_agent_num=self.noisy_agent_num, have_env_info=self.have_env_info))
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

    def reset(self):
        # print('reset!!!!!')
        # print(len(self.env.get_alive(self.handles[0])), len(self.env.get_alive(self.handles[1])))

        self.done = False

        random_pos_set = np.array(list(itertools.product(range(2, self.map_size-2), range(2, self.map_size-2))))[np.random.choice(range((self.map_size-4)**2), 2 * self.agent_num, replace=False)]
        pos_set = [random_pos_set[:self.agent_num], random_pos_set[self.agent_num:]]

        noisy_agent_pos = [np.random.randint(2, self.map_size - 2, size=2) for _ in range(self.noisy_agent_num)]
        pos_set.append(noisy_agent_pos)

        generate_map(self.env, pos_set, self.handles)

        if self.noisy_agent_num > 0:
            self.noisy_group = RandomActor(self.env, self.handles[2], 'noisy_group')

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

        if self.noisy_agent_num > 0:
            noisy_obs = self.env.get_observation(self.handles[2])
            noisy_ids = self.env.get_agent_id(self.handles[2])
            noisy_actions = self.noisy_group.infer_action(noisy_obs, noisy_ids)
            self.env.set_action(self.handles[2], noisy_actions)

        self.env.set_action(self.handles[0], np.array(actions[0], dtype=np.int32))
        self.env.set_action(self.handles[1], np.array(actions[1], dtype=np.int32))

        self.done = self.env.step()

        if render:
            self.env.render()

        obs = self.get_obs()

        reward = self.get_reward()

        if self.step_num > self.max_step or self.done:
            print('done !!!!!!!!', self.step_num)

            # 加入一个最终胜利的reward
            agent_live_info = self.get_live_agent()

            if not any(agent_live_info[0]):
                # group 0 agent全部死亡, 另一组获胜
                # 存活的agent全部获得一个最终reward
                agent_ids = self.get_group_agent_id(1)
                reward[1][agent_ids] += 10
            
            if not any(agent_live_info[1]):
                # group 1 agent全部死亡
                agent_ids = self.get_group_agent_id(0)
                reward[0][agent_ids] += 10
            
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
        self_info_space = self.env.get_feature_space(self.handles[0])[0] + 2 # 血量, group, minimap size: 34 + 2 = 36
        env_info_space = 0
        if self.have_env_info:
            env_info_space = self.env.get_view_space(self.handles[0])[0]**2
        
        other_agent_info_space = 28 * (self.agent_num * 2 - 1 + self.noisy_agent_num) # size: 28 * 9 = 252
        return spaces.Box(low=0, high=1, shape=(self_info_space + env_info_space + other_agent_info_space, ), dtype=np.float32)


if __name__ == '__main__':
    env = MagentEnv(noisy_agent_num=10)
    print('action space: ', env.action_space.n)
    print('obs space: ', env.observation_space.shape[0])
    obs = env.reset()
    # env.env.render()
    # print('group1 obs: ', obs[0][0])
    # print('group2 obs: ', obs[1])
    # for i in range(len(obs[0])):
    #     print(obs[0][i])
    for i in range(100):
        g1_as = np.random.randint(0, 21, size=20)
        g2_as = np.random.randint(0, 21, size=20)
 
        obs, reward, done, alive_info = env.step([g1_as, g2_as], render=True)
    # print(alive_info)
