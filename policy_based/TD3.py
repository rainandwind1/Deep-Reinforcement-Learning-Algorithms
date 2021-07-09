import torch
import numpy as np
import sys
import gym

sys.path.append("./")
from base_net.model import *
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical

class TD3(nn.Module):
    def __init__(self, args):
        super(TD3, args).__init__()
        self.input_size, self.output_size, self.mem_size, self.toi, self.device, self.lr = args
        
        self.actor = Policy_net(args = (self.input_size, self.output_size))
        self.critic1 = Q_net(args = (self.input_size, 1))
        self.critic2 = Q_net(args = (self.input_size, 1))
        
        self.target_actor = Policy_net(args = (self.input_size, self.output_size))
        self.target_critic1 = Q_net(args = (self.input_size, 1))
        self.targete_critic2 = Q_net(args = (self.input_size, 1))

        self.replay_buffer = ReplayBuffer(args = (self.mem_size))
        self.optimizer = optim.Adam([{'params':self.actor.parameters()}, {'params':self.critic1.parameters()}, {'params':self.critic2.parameters()}], lr = self.lr)
        self.update_target_net(initialize = True)

    def update_target_net(self, initialize = False):
        if initialize:
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic1.load_state_dict(self.critic1.state_dict())
            self.targete_critic2.load_state_dict(self.critic2.state_dict())

        elif not initialize:
            for raw, target in zip(self.actor.parameters(), self.target_actor.parameters()):
                target.data.copy_(self.toi * raw.data + (1 - self.toi) * target.data)
            
            for raw, target in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                target.data.copy_(self.toi * raw.data + (1 - self.toi) * target.data)

            for raw, target in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                target.data.copy_(self.toi * raw.data + (1 - self.toi) * target.data)

    def get_policy(self, inputs):
        pass

    def get_target_policy(self, inputs):
        pass

    def select_action(self, inputs):
        pass 


    def train(self, gamma = 0.98, batch_size = 32):
        pass