import torch
import numpy as np
import random
import collections

from collections import deque
from torch import nn, optim

embedding_size = 64

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
name: determined policy net
description: for continuous control
'''
class Policy_net_determined(nn.Module):
    def __init__(self, args):
        super(Policy_net_determined, self).__init__()
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

    def save_trans(self, transition):
        self.buffer.append(transition)

    def sample_batch(self, batch_size = 32):
        transition_batch = random.sample(self.buffer, batch_size)
        batch_data_ls = [[] for _ in transition_batch[0]]
        for trans in transition_batch:
            for data_id, data in enumerate(trans):
                batch_data_ls[data_id].append(data)
        
        return batch_data_ls
