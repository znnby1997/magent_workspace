import torch
import torch.nn.functional as F


def to_one_hot(action, n_actions, device):
    """
        action: (batch, 1)
    """
    one_hot_vector = torch.zeros((action.shape[0], n_actions)).to(device).scatter(1, action, 1) # batch, n_actions
    return one_hot_vector


def train_encoder(obs_encoder, a_encoder, predictor, buffer, optimizer, batch_size, n_actions, r_coef, device):
    s, a, r, s_prime, done_mask = buffer.get(batch_size)

    self_tokens, other_tokens = obs_encoder(s)
    other_tokens = other_tokens.reshape(batch_size, -1)
    action_embeddings = a_encoder(to_one_hot(a, n_actions, device))
    pre_in = torch.cat([self_tokens, other_tokens, action_embeddings], dim=1)

    pre_state, pre_r = predictor(self_tokens, other_tokens, action_embeddings) # (batch, obs_dim), (batch, 1)
    loss = (1-r_coef) * F.mse_loss(pre_state, s_prime) + r_coef * F.mse_loss(pre_r, r)
    last_loss = loss
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()
    return loss.mean(), last_loss.mean()
    
