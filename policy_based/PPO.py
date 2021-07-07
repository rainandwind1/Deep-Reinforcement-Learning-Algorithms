import torch
import numpy as np
import sys
import gym

sys.path.append("./")
from base_net.model import *
from torch import nn, optim


class PPO(nn.Module):
    def __init__(self, args):
        super(PPO, self).__init__()
        self.input_size, self.output_size, self.device, self.lr = args
        self.actor = Policy_net(args = (self.input_size, self.output_size))
        self.critic = Q_net(args = (self.input_size, self.output_size))

        self.buffer = []
        self.optimizer = optim.Adam([{'params':self.actor.parameters()}, {'params':self.critic.parameters()}], lr = self.lr)

    def selection_action(self, inputs):
        pass

    def save_trans(self, transition):
        pass

    def train(self, gamma = 0.98, batch_size = 32):
        pass