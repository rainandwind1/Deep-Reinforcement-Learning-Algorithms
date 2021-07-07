import torch
import numpy as np
import sys
import gym

sys.path.append("./")
from base_net.model import *
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical


class PPO(nn.Module):
    def __init__(self, args):
        super(PPO, self).__init__()
        self.input_size, self.output_size, self.device, self.lr = args
        self.actor = Policy_net(args = (self.input_size, self.output_size))
        self.critic = Q_net(args = (self.input_size, self.output_size))

        self.buffer = ReplayBuffer(args = (10000))
        self.optimizer = optim.Adam([{'params':self.actor.parameters()}, {'params':self.critic.parameters()}], lr = self.lr)

    def get_policy_op(self, inputs):
        policy_op = self.actor(inputs)
        softmax_op = F.softmax(policy_op, -1)
        return softmax_op
    
    def selection_action(self, inputs):
        action_prob = self.get_policy_op(inputs)
        action = Categorical(action_prob)
        action = action.sample().item()
        return action

    def save_trans(self, transition):
        self.buffer.append(transition)

    def to_tensor(self, items):
        s, a, r, s_next, a_prob, done = items
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s_next = torch.FloatTensor(s_next).to(self.device)
        a_prob = torch.FloatTensor(a_prob).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        return s, a.unsqueeze(-1), r.unsqueeze(-1), s_next, a_prob.unsqueeze(-1), done.unsqueeze(-1)

    def train(self, gamma = 0.98, batch_size = 32, k = 3, epsilon_clip = 0.1):
        s, a, r, s_next, done = self.to_tensor(self.buffer.sample_all_data())
        for i in range(k):
        
        self.buffer.clear()