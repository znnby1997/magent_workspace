import random
import torch
from torch.distributions import Categorical

def sample_a_from_q(q, obs, epsilon):
    out = q(obs.reshape(1, -1))
    n_actions = out[0].shape[1]

    coin = random.random()
    if coin < epsilon:
        return random.randint(0, n_actions - 1), out[1]
    else:
        return out[0].argmax().item(), out[1]

def sample_a_from_pi(model, obs):
    out = model.pi(obs.reshape(1, -1))
    a = Categorical(out[0]).sample().item()
    a_prob = out[0].squeeze()[a].item()
    return a, a_prob, out[1]
