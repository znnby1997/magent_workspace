from collections import deque
import random
import torch

from ._base import BaseReplayBuffer

__all__ = (
    'PAReplayBuffer',
)

class PAReplayBuffer(BaseReplayBuffer):
    def __init__(self, buffer_limit, device):
        self._buffer = deque(maxlen=buffer_limit)
        self._capacity = int(buffer_limit)
        self.device = device

    @property
    def capacity(self):
        return self._capacity

    def push(self, transition):
        self._buffer.extend(transition)

    def get(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self._buffer:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_lst.append([done])

        s, a, r, s_prime, done, prob_a = torch.tensor(s_lst, dtype=torch.float).cuda(), torch.tensor(a_lst).to(self.device), \
                                        torch.tensor(r_lst, dtype=torch.float).to(self.device), torch.tensor(s_prime_lst, dtype=torch.float).to(self.device), \
                                            torch.tensor(done_lst, dtype=torch.float).to(self.device), torch.tensor(prob_a_lst, dtype=torch.float).to(self.device)

        return s, a, r, s_prime, done, prob_a

    def clear(self):
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)

    def __bool__(self):
        return bool(len(self))

    def __iter__(self):
        return iter(self._buffer)
    
    
