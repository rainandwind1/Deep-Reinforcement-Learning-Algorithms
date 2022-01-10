import torch
import numpy as np
import random
import collections
import torch.nn.functional as F

from collections import deque
from torch import nn, optim


embedding_size = 64
rnn_embedding_size = 32

'''
name: q value net
description: 
'''
class Q_net(nn.Module):
    def __init__(self, args):
        super(Q_net, self).__init__()
        self.input_size, self.output_size = args
        self.q_net = nn.Sequential(
            nn.Linear(self.input_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, self.output_size)
        )

    def forward(self, inputs):
        return self.q_net(inputs)

'''
name: q value net for iqn
description: 
'''
class IQN_qnet(nn.Module):
    def __init__(self, args):
        super(IQN_qnet, self).__init__()
        self.input_size, self.output_size = args
        self.embedding_net = nn.Linear(self.input_size, 32)
        self.phi = nn.Linear(1, 32, bias = False)
        self.phi_bias = nn.Parameter(torch.randn(1, 32), requires_grad = True)
        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )
        

    def forward(self, inputs, toi_num = 64):
        toi = torch.FloatTensor(np.random.uniform(size = (toi_num, 1))).to(inputs.device)
        Num = torch.FloatTensor(range(64)).to(inputs.device)
        cos_op = torch.cos(Num * toi * np.pi).unsqueeze(-1)
        p = self.embedding_net(inputs)
        embedding_op = self.embedding_net(inputs).view(inputs.shape[0], -1).unsqueeze(1)
        phi = F.relu(self.phi(cos_op).mean(1) + self.phi_bias).unsqueeze(0)
        q_val_dis = self.fc(embedding_op * phi).transpose(1, 2)
        return q_val_dis, toi


'''
name: q value net recurrent version for drqn (GRUCell)
description: 
'''
class GRUCell_Q_net(nn.Module):
    def __init__(self, args):
        super(GRUCell_Q_net, self).__init__()
        self.input_size, self.output_size = args
        self.embedding_net = nn.Sequential(
            nn.Linear(self.input_size, embedding_size),
            nn.ReLU()
        )
        self.hidden_state = None
        self.rnn_net = nn.GRUCell(embedding_size, rnn_embedding_size)
        self.q_net = nn.Linear(rnn_embedding_size, self.output_size)

    def init_hidden(self, batch_size = 1):
        self.hidden_state = torch.randn((batch_size, rnn_embedding_size))
        
    def forward(self, inputs):
        embedding_op = self.embedding_net(inputs)
        self.hidden_state = self.hidden_state.reshape(-1, rnn_embedding_size).to(inputs.device)
        self.hidden_state = self.rnn_net(embedding_op, self.hidden_state)
        return self.q_net(self.hidden_state)
        

'''
name: q value net recurrent version for drqn (GRU module)
description: 
'''
class GRU_Q_net(nn.Module):
    def __init__(self, args):
        super(GRU_Q_net, self).__init__()
        self.input_size, self.output_size = args
        self.embedding_net = nn.Sequential(
            nn.Linear(self.input_size, embedding_size),
            nn.ReLU()
        )
        self.num_gru_layer = 2
        self.hidden_state = None
        self.rnn_net = nn.GRU(embedding_size, rnn_embedding_size, self.num_gru_layer, batch_first=True)
        self.q_net = nn.Linear(rnn_embedding_size, self.output_size)

    def init_hidden(self, batch_size = 1):
        self.hidden_state = torch.randn((self.num_gru_layer, batch_size, rnn_embedding_size))
        
    def forward(self, inputs):
        embedding_op = self.embedding_net(inputs)
        self.hidden_state = self.hidden_state.reshape(self.num_gru_layer, -1, rnn_embedding_size).to(inputs.device)
        gru_op, self.hidden_state = self.rnn_net(embedding_op, self.hidden_state)
        return self.q_net(gru_op)


'''
name: policy net
description: 
'''
class Policy_net(nn.Module):
    def __init__(self, args):
        super(Policy_net, self).__init__()
        self.input_size, self.output_size = args
        self.actor = nn.Sequential(
            nn.Linear(self.input_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, self.output_size)
        )

    def forward(self, inputs):
        return self.actor(inputs)


'''
name: replay buffer 
description: for off policy rl algs
'''
class ReplayBuffer(object):
    def __init__(self, args):
        self.mem_len = args
        self.buffer = deque(maxlen=self.mem_len)

    def save_trans(self, transition, episode_data = False, max_len = 0):
        if not episode_data:
            self.buffer.append(transition)
        # for drqn episode data
        elif episode_data:  
            for item in transition:
                if len(item) < max_len:
                    item += [[0 for _ in range(len(item[0]))] for _ in range(max_len - len(item))]
            self.buffer.append(transition)

    def sample_batch(self, batch_size = 32):
        transition_batch = random.sample(self.buffer, batch_size)
        batch_data_ls = [[] for _ in transition_batch[0]]
        for trans in transition_batch:
            for data_id, data in enumerate(trans):
                batch_data_ls[data_id].append(data)
        
        return batch_data_ls

    def sample_all_data(self):
        batch_data_ls = [[] for _ in self.buffer[0]]
        for trans in self.buffer:
            for data_id, data in enumerate(trans):
                batch_data_ls[data_id].append(data)
        
        return batch_data_ls

    def clear(self):
        self.buffer.clear()
