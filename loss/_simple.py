import torch
import torch.nn.functional as F
from torch.distributions import Categorical

def q_one_td_loss(q, q_target, minibatch, gamma):
    s, a, r, s_prime, done_mask = minibatch

    q_out, w_out = q(s)
    q_a = q_out.gather(1, a)
    max_q_prime = q_target(s_prime)[0].max(1)[0].unsqueeze(1)
    target = r + gamma * max_q_prime * (1 - done_mask)
    return F.smooth_l1_loss(q_a, target)

def ac_one_td_loss(model, traj, gamma, entropy_coef):
    s, a, r, s_prime, done_mask = traj

    td_target = r + gamma * model.v(s_prime)[0] * (1 - done_mask)
    advantage = td_target - model.v(s)[0]

    pi, _ = model.pi(s)
    pi_a = pi.gather(1, a)
    entropy = Categorical(pi).entropy()

    actor_loss = torch.log(pi_a + 1e-8) * advantage.detach()
    critic_loss = F.smooth_l1_loss(model.v(s)[0], td_target)
    entropy_loss = entropy * entropy_coef

    # loss = -actor_loss.reshape(-1) + critic_loss - entropy * entropy_coef
    return actor_loss, critic_loss, entropy_loss

def ppo_one_td_loss(model, traj, gamma, entropy_coef, lmbda, eps_clip, device):
    s, a, r, s_prime, done_mask, prob_a = traj

    td_target = r + gamma * model.v(s_prime)[0] * (1 - done_mask)
    delta = (td_target - model.v(s)[0]).detach().cpu().numpy()

    advantage_lst = []
    advantage = 0.0
    for delta_t in delta[::-1]:
        advantage = gamma * lmbda * advantage + delta_t[0]
        advantage_lst.append([advantage])
    advantage_lst.reverse()
    advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)

    pi, _ = model.pi(s)
    pi_a = pi.gather(1, a)
    ratio = torch.exp(torch.log(pi_a + 1e-8) - torch.log(prob_a + 1e-8))
    
    entropy = Categorical(pi).entropy()

    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage

    actor_loss = torch.min(surr1, surr2)
    critic_loss = F.smooth_l1_loss(model.v(s)[0], td_target)
    entropy_loss = entropy * entropy_coef

    return actor_loss, critic_loss, entropy_loss
    