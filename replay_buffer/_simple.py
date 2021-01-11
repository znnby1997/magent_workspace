from collections import deque
import random
import torch

from ._base import BaseReplayBuffer

__all__ = (
    'SimpleReplayBuffer',
)

class SimpleReplayBuffer(BaseReplayBuffer):
    def __init__(self, buffer_limit, device):
        self._buffer = deque(maxlen=buffer_limit)
        self._capacity = int(buffer_limit)
        self.device = device

    @property
    def capacity(self):
        return self._capacity

    def push(self, transitions):
        # self._buffer.append(transition)
        self._buffer.extend(transitions)

    def get(self, batch_size):
        mini_batch = random.sample(self._buffer, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float).to(self.device), torch.tensor(a_lst).to(self.device), \
               torch.tensor(r_lst, dtype=torch.float).to(self.device), torch.tensor(s_prime_lst, dtype=torch.float).to(self.device), \
               torch.tensor(done_mask_lst, dtype=torch.float).to(self.device)

    def clear(self):
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)

    def __bool__(self):
        return bool(len(self))

    def __iter__(self):
        return iter(self._buffer)