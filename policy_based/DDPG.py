import torch
import numpy as np
import sys
import gym

sys.path.append("./")
from base_net.model import *
from torch import nn, optim


class DDPG(nn.Module):
    def __init__(self, args):
        super(DDPG, self).__init__()
        self.input_size, self.output_size, self.mem_size, self.clamp, self.action_max, self.action_min, self.device, self.lr = args
        self.toi = 0.01
        
        self.actor = Policy_net(args = (self.input_size, self.output_size))
        self.critic = Q_net(args = (self.input_size, 1))
        
        # target
        self.target_actor = Policy_net(args = (self.input_size, self.output_size))
        self.target_critic = Q_net(args = (self.input_size, 1))

        self.replay_buffer = ReplayBuffer(args = (self.mem_size))
        self.optimizer = optim.Adam([{'param':self.actor.parameters()}, {'param':self.critic.parameters()}], lr = self.lr)
        self.update_target_net(initialize = True)

    def update_target_net(self, initialize = False):
        # initialize param
        if initialize:
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())
        # uopdate param
        if not initialize:
            for raw, target in zip(self.actor.parameters(), self.target_actor.parameters()):
                target.data.copy_(self.toi * raw.data + (1 - self.toi) * target.data)
            for raw, target in zip(self.critic.parameters(), self.target_critic.parameters()):
                target.data.copy_(self.toi * raw.data + (1 - self.toi) * target.data)

    def get_policy_op(self, inputs):
        raw_op = self.actor(inputs)
        if self.clamp:
            raw_op = torch.clamp(raw_op, self.action_min, self.action_max)
        return raw_op

    def get_target_policy_op(self, inputs):
        raw_op = self.target_actor(inputs)
        if self.clamp:
            raw_op = torch.clamp(raw_op, self.action_min, self.action_max)
        return raw_op

    def selection_action(self, inputs, epsilon, eval_mode = False):
        actor_op = self.get_policy_op(inputs)
        noise = torch.randn(actor_op.shape).to(self.device) if not eval_mode else 0.        # Gaussi noise
        actor_op += noise
        return actor_op.detach().cpu().numpy()
    
    def train(self, inputs):
        pass






































if __name__ == "__main__":
    print("yes")