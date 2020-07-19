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
        return random.sample(self.memory, batch), [], []

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

class PrioritisedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity, ), dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(Dynamics(state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = Dynamics(state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

if __name__ == '__main__':
    pool = ExperienceMemory(10)
    pool.push(1,2,[-1,-1,2,3],4,5)
    pool.push(1,2,[1,2,3,4], 4,5)
    pool.push(1,2,[-1,-2,-3,1], 4, 5)
    pool.push(1,2,[-1,2,3,-1], 4, 5)
    pool.push(1,2,[1,2,-3,-4], 4, 5)

    pool.clear()
    print(len(pool))
