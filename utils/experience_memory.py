from collections import namedtuple
import random
import numpy as np

Dynamics = namedtuple('Dynamics', ['state', 'action', 'reward', 'next_state', 'is_end'])


class ExperienceMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, is_end):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Dynamics(state, action, reward, next_state, is_end)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch):
        return random.sample(self.memory, batch)

    def get_all(self):
        return self.memory

    def clear(self):
        # temp = []
        # for d_ in self.memory:
        #     if sum(d_.reward) > 0:
        #         temp.append(d_)
        # self.memory = temp
        # self.position = len(self.memory) % self.capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)


if __name__ == '__main__':
    pool = ExperienceMemory(10)
    pool.push(1,2,[-1,-1,2,3],4,5)
    pool.push(1,2,[1,2,3,4], 4,5)
    pool.push(1,2,[-1,-2,-3,1], 4, 5)
    pool.push(1,2,[-1,2,3,-1], 4, 5)
    pool.push(1,2,[1,2,-3,-4], 4, 5)

    pool.clear()
    print(len(pool))