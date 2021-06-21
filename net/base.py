import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


class BasicNet(nn.Module):
    def __init__(self, obs_dim, output_dim, hidden_dim, **kwargs):
        super(BasicNet, self).__init__()

        self.o2e = nn.Linear(obs_dim, hidden_dim)

        # self.e2b = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, output_dim)
        # )

        self.e2b = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, **kwargs):
        h = f.relu(self.o2e(x))
        return self.e2b(h), h


class LocalNet(nn.Module):
    def __init__(self, input_shape, h_dim, output_dim, hidden_dim, **kwargs):
        super(LocalNet, self).__init__()

        # self.e2l = nn.Sequential(
        #     nn.Linear(input_shape + h_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, output_dim)
        # )

        self.e2l = nn.Linear(input_shape + h_dim, output_dim)

    def forward(self, x, h=None):
        if h is not None:
            x = torch.cat([x, h], dim=2) # shape: nums, batch, 2*(input_shape+h_dim)
        return self.e2l(x)


class Encoder(nn.Module):
    def __init__(self, input_shape, output_dim, **kwargs):
        super(Encoder, self).__init__()

        self.o2e = nn.Linear(input_shape, output_dim)
    
    def forward(self, x):
        return f.relu(self.o2e(x))


class AttnAggV1(nn.Module):
    def __init__(self, obs_dim, output_dim, hidden_dim, **kwargs):
        super(AttnAggV1, self).__init__()

        # self.attn_layer = nn.Sequential(
        #     nn.Linear(obs_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, output_dim)
        # )

        self.attn_layer = nn.Linear(obs_dim, output_dim)

    def forward(self, q_base, q_locals, x):
        """
            x: observations shape=(batch, obs_dim)
            q_base: basic policy distribution shape=(batch, n_actions)
            q_locals: local policy distribution shape=(batch, tokens_nums, n_actions)
        """
        w = f.softmax(self.attn_layer(x), dim=1) # batch, tokens_nums
        local_q = torch.bmm(w.unsqueeze(1), q_locals).squeeze(1) # batch, n_actions
        return (q_base + local_q) / 2., w


class AttnAggV2(nn.Module):
    def __init__(self, output_dim, **kwargs):
        super(AttnAggV2, self).__init__()

        self.output_dim = output_dim
        self.W = nn.Linear(output_dim, output_dim, bias=False)

    def forward(self, q_base, q_locals, *args):
        """
            q_base: basic policy distribution shape=(batch, n_actions)
            q_locals: local policy distribution shape=(batch, tokens_nums, n_actions)
        """
        scores = torch.bmm(q_base.unsqueeze(1), self.W(q_locals).permute(0, 2, 1)) / np.sqrt(self.output_dim) # shape: batch,1, tokens_nums
        w = f.softmax(scores, dim=2)
        local_q = torch.bmm(w, q_locals).squeeze(1) # batch, n_actions
        return (q_base + local_q) / 2., w.squeeze(1)


class TEncoder(nn.Module):
    def __init__(self, input_shape, output_dim, hidden_dim):
        super(SEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class ActionEncoder(nn.Module):
    def __init__(self, input_shape, output_dim, hidden_dim):
        super(ActionEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

class Predictor(nn.Module):
    def __init__(self, input_shape, output_dim, hidden_dim):
        super(Predictor, self).__init__()

        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.out_state = nn.Linear(hidden_dim, output_dim)
        self.out_r = nn.Linear(hidden_dim, 1)

    def forward(self, self_tokens, oth_tokens, action_embedding):
        """
            self_tokens: batch, tokens_dim
            oth_tokens: batch, tokens_dim*(nums-1)
            action_embedding: batch, action_e_dim
        """
        obs_embedding = torch.cat([self_tokens, oth_tokens, action_embedding], dim=1) # batch, tokens_dim*nums+action_e_dim
        predic_state = self.out_state(f.relu(self.fc1(obs_embedding)))
        predic_r = self.out_r(f.relu(self.fc1(obs_embedding)))
        return predic_state, predic_r

class GroupNet(nn.Module):
    def __init__(self, input_shape, hidden_dim, n_groups):
        super().__init__()

        self.tokens_fc = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_groups)
        )

    def forward(self, x, h=None):
        """
            x: [batch, num, tokens],
            h: [batch, hidden_dim]
        """
        if h is not None:
            x = torch.cat([x, h], dim=2) # shape: nums, batch, 2*(input_shape+h_dim)
        return f.softmax(self.tokens_fc(x))



