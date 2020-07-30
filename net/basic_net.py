import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, agent_num, hidden_dim=32, structure_type=0, em_dim=32):
        super(QNet, self).__init__()
        self.structure_type = structure_type  # net_type_flag
        self.att_weight = None
        self.hidden_dim = hidden_dim
        self.agent_num = agent_num

        # embedding and out layer
        self.embedding_layer = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, n_actions)

        # no_attention
        self.no_attention_layer = nn.Linear(obs_dim, hidden_dim)

        # softmax, tanh, sigmoid attention
        self.att_layer = nn.Linear(obs_dim, obs_dim)
        self.trans_layer = nn.Linear(obs_dim, hidden_dim)

        # agent level attention
        self.al_att_layer = nn.Linear(obs_dim, agent_num * 2)
        self.att_output_encoder = nn.Linear(em_dim, hidden_dim)

        # scale dot attention
        self.self_encoder = nn.Linear(37, em_dim)
        self.other_encoder = nn.Linear(28, em_dim)
        self.trans_set = {
            'query_w': nn.Linear(em_dim, hidden_dim, bias=False),
            'key_w': nn.Linear(em_dim, hidden_dim, bias=False),
            'value_w': nn.Linear(em_dim, hidden_dim, bias=False)
        }

        # gru attention
        self.gru_layer = nn.GRU(input_size=em_dim, hidden_size=hidden_dim // 2, bidirectional=True)
        self.w = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.fix_query = nn.Parameter(torch.Tensor(hidden_dim, 1))

        # DyAN网络聚合方式
        self.feature_encoder1 = nn.Linear(28, hidden_dim)
        self.feature_encoder2 = nn.Linear(hidden_dim, hidden_dim)
        self.self_encoder = nn.Linear(37, hidden_dim)
        self.align_embedding = nn.Linear(3 * hidden_dim, hidden_dim)

    def get_attention_layer(self, x):
        if self.structure_type == 0:
            # no attention
            return f.relu(self.no_attention_layer(x))
        elif self.structure_type == 1:
            # softmax attention
            self.att_weight = f.softmax(self.att_layer(x), dim=1)
            att_output = self.att_weight * x
            return f.relu(self.trans_layer(att_output))
        elif self.structure_type == 2:
            # tanh attention
            self.att_weight = torch.tanh(self.att_layer(x))
            att_output = self.att_weight * x
            return f.relu(self.trans_layer(att_output))
        elif self.structure_type == 3:
            # sigmoid attention
            self.att_weight = torch.sigmoid(self.att_layer(x))
            att_output = self.att_weight * x
            return f.relu(self.trans_layer(att_output))
        elif self.structure_type == 4:
            # agent level attention
            # self info: 37bits  opp info: 28bits  partner info: 28bits
            self.att_weight = f.softmax(self.al_att_layer(x), dim=1) # size: [batch, agent_num]
            self_info = x[:, 0:37]
            other_info = x[:, 37:]
            agents_info = other_info.split(28, dim=1)
            encodings = []
            encodings.append(self.self_encoder(self_info))
            for agent_info in agents_info:
                encodings.append(f.relu(self.other_encoder(agent_info)))
            encodings = torch.stack(encodings).permute(1, 0, 2) # size: [batch, agent_num, embedding_dim]
            att_output = torch.bmm(self.att_weight.unsqueeze(1), encodings).squeeze(1) # size: [batch, em_dim]
            return f.relu(self.att_output_encoder(att_output))
        elif self.structure_type == 5:
            # scale dot attention
            self_info = x[:, 0:37]
            other_info = x[:, 37:]
            agents_info = other_info.split(28, dim=1)
            encodings = []
            encodings.append(self.self_encoder(self_info))
            for agent_info in agents_info:
                encodings.append(f.relu(self.other_encoder(agent_info)))
            query = self.trans_set['query_w'](encodings[0]).unsqueeze(1) # size: [batch, 1, hidden]
            keys, values = [], []
            for encoding in encodings:
                keys.append(self.trans_set['key_w'](encoding))
                values.append(self.trans_set['value_w'](encoding))
            
            keys_matrix = torch.stack(keys).permute(1, 2, 0) # size: [batch, hidden, agent_num]
            values_matrix = torch.stack(values).permute(1, 0, 2) # size: [batch, agent_num, hidden]
            self.att_weight = f.softmax((torch.mm(query, keys_matrix) / np.sqrt(self.hidden_dim)), dim=2) # size: [batch, 1, agent_num]
            return torch.mm(self.att_weight, values_matrix).squeeze() # size: [batch, hidden]
        elif self.structure_type == 6:
            # fixed attention generating policy
            pass
        elif self.structure_type == 7:
            # gru attention
            self_info = x[:, 0:37]
            other_info = x[:, 37:]
            agents_info = other_info.split(28, dim=1)
            encodings = []
            encodings.append(self.self_encoder(self_info))
            for agent_info in agents_info:
                encodings.append(f.relu(self.other_encoder(agent_info)))
            gru_input = torch.stack(encodings) # size:[agent_num, batch, em_dim]
            gru_out, _ = self.gru_layer(gru_input)
            x = gru_out.permute(1, 0, 2) # size: [batch, agent_num, hidden]
            u = torch.tanh(torch.matmul(x, self.w)) # size: [batch, agent_num, hidden]
            att = torch.matmul(u, self.fix_query) # size: [batch, agent_num, 1]
            self.att_weight = f.softmax(att, dim=1) # size: [batch, agent_num, 1]
            scored_x = x * self.att_weight # size: [batch, agent_num, hidden]
            att_output = torch.sum(scored_x, dim=1) # size: [batch, hidden]
            return att_output
        elif self.structure_type == 8:
            # DyAN聚合方式, 所有的对手观测聚合, 所有的队友观测聚合, 最后拼接三个信息
            self_info = x[:, 0:37]
            opp_index = 37 + 28 * self.agent_num
            opps_info = x[:, 37:opp_index].split(28, dim=1) # 20 agents
            partners_info = x[:, opp_index:].split(28, dim=1) # 19 agents
            opps_encoder = []
            partners_encoder = []
            for opp_info in opps_info:
                opps_encoder.append(self.feature_encoder2(f.relu(self.feature_encoder1(opp_info))))
            for partner_info in partners_info:
                partners_encoder.append(self.feature_encoder2(f.relu(self.feature_encoder1(partner_info))))
            opps_embedding = torch.stack(opps_encoder) # size: [agent_num, batch, hidden]
            partners_embedding = torch.stack(partners_encoder) # size: [agent_num, batch, hidden]
            aggregate_opp = torch.max(opps_embedding, dim=0)[0] # size: [batch, hidden] sum/mean/max
            aggregate_partner = torch.max(partners_embedding, dim=0)[0] # size: [batch, hidden]
            self_embedding = f.relu(self.feature_encoder2(f.relu(self.self_encoder(self_info)))) # size: [batch, hidden]
            embedding = torch.cat([aggregate_opp, aggregate_partner, self_embedding], dim=1) # size: [batch, 3 * hidden]
            att_output = f.relu(self.align_embedding(embedding)) # size: [batch, hidden]
            return att_output
    def forward(self):
        raise NotImplementedError

    def q(self, obs):
        att_output = self.get_attention_layer(obs)
        embedding = f.relu(self.embedding_layer(att_output))
        return self.out_layer(embedding)

    def get_cur_weight(self):
        return self.att_weight.detach().numpy()


class BasicNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=32, em_dim=32, **kwargs):
        super(BasicNet, self).__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.em_dim = em_dim
        self.att_weight = None

        # 基本的网络中包含一层MLP，用于处理第一层的输出embedding
        self.basic_mlp2 = nn.Linear(self.hidden_dim, self.n_actions)

        # 编码观测信息向量的时候用
        # self info: 37bits  opp info: 28bits  partner info: 28bits
        self.self_encoder = nn.Linear(37, em_dim)
        self.other_encoder = nn.Linear(28, em_dim)

    def forward(self):
        raise NotImplementedError

    def att_layer(self, x):
        # 该部分每个网络继承basic net时需要重写
        return

    def q(self, obs):
        att_output = self.att_layer(obs)
        return self.basic_mlp2(att_output)

    def get_cur_weight(self):
        return self.att_weight.detach().numpy()

