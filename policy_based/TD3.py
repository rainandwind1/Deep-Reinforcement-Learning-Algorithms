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
        super(TD3, self).__init__()
        self.input_size, self.output_size, self.clamp, self.action_max, self.action_min, self.mem_size, self.toi, self.device, self.actor_lr, self.critic_lr = args
        self.critic_update_cnt = 0

        self.actor = Policy_net(args = (self.input_size, self.output_size))
        self.critic1 = Q_net(args = (self.input_size + self.output_size, 1))
        self.critic2 = Q_net(args = (self.input_size + self.output_size, 1))
        
        self.target_actor = Policy_net(args = (self.input_size, self.output_size))
        self.target_critic1 = Q_net(args = (self.input_size + self.output_size, 1))
        self.target_critic2 = Q_net(args = (self.input_size + self.output_size, 1))

        self.replay_buffer = ReplayBuffer(args = (self.mem_size))
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.optimizer_critic = optim.Adam(self.parameters(), lr = self.critic_lr)
        self.update_target_net(initialize = True)

    def update_target_net(self, initialize = False):
        if initialize:
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic1.load_state_dict(self.critic1.state_dict())
            self.target_critic2.load_state_dict(self.critic2.state_dict())

        elif not initialize:
            for raw, target in zip(self.actor.parameters(), self.target_actor.parameters()):
                target.data.copy_(self.toi * raw.data + (1 - self.toi) * target.data)
            
            for raw, target in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                target.data.copy_(self.toi * raw.data + (1 - self.toi) * target.data)

            for raw, target in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                target.data.copy_(self.toi * raw.data + (1 - self.toi) * target.data)

    def get_policy(self, inputs):
        policy_op = self.actor(inputs)
        action_op = torch.tanh(policy_op) * (self.action_max - self.action_min) / 2
        if self.clamp:
            action_op = torch.clamp(action_op, self.action_min, self.action_max)
        return action_op

    def get_target_policy(self, inputs):
        policy_op = self.actor(inputs)
        action_op = torch.tanh(policy_op) * (self.action_max - self.action_min) / 2
        if self.clamp:
            action_op = torch.clamp(action_op, self.action_min, self.action_max)
        return action_op

    def save_trans(self, transition):
        self.replay_buffer.save_trans(transition)

    def to_tensor(self, items):
        s, a, r, s_next, done = items
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s_next = torch.FloatTensor(s_next).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        return s, a, r.unsqueeze(-1), s_next, done.unsqueeze(-1)

    def select_action(self, inputs, epsilon, eval_mode = False):
        action_op = self.get_policy(inputs)
        noise = torch.randn(action_op.shape).to(self.device) if not eval_mode else 0.
        # action_op += noise * epsilon
        if self.clamp:
            action = torch.clamp(action_op, self.action_min, self.action_max)
        return action.detach().cpu().numpy()

    def train(self, gamma = 0.98, batch_size = 32, policy_update_interval = 50):
        s, a, r, s_next, done = self.to_tensor(self.replay_buffer.sample_batch(batch_size = batch_size))
        q_val1 = self.critic1(torch.cat([s, a], -1))
        q_val2 = self.critic2(torch.cat([s, a], -1))
        action_next = self.get_target_policy(s_next).detach()
        target_q_val = r + gamma * torch.min(self.target_critic1(torch.cat([s_next, action_next], -1)), self.target_critic2(torch.cat([s_next, action_next], -1))) * (1 - done)
        loss_critic = ((target_q_val.detach() - q_val1) ** 2 + (target_q_val.detach() - q_val2) ** 2).mean()

        if (self.critic_update_cnt + 1) % policy_update_interval == 0:
            loss_actor = (- self.critic1(torch.cat([s, self.get_policy(s)], -1))).mean()
            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            self.optimizer_actor.step()
            self.update_target_net()


        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        self.critic_update_cnt += 1



'''
TD3 test
'''
if __name__ == "__main__":
    env = gym.make("Pendulum-v0")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TD3(args = (3, 2, True, 2, -2, 30000, 0.01, device, 1e-3, 1e-3)).to(device)
    total_step = 0
    epsilon = 0.3
    for i in range(10000):
        s = env.reset()
        score = 0.
        epsilon = max(0.01, epsilon * 0.999)
        for t in range(200):
            total_step += 1
            action = model.select_action(torch.FloatTensor(s).to(device), epsilon)

            s_next, reward, done, info = env.step(action)
            model.save_trans((s, action, reward, s_next, done))
            score += reward 
            s = s_next
            if len(model.replay_buffer.buffer) > 60:
                model.train()
            if done:
                break
        print("Epoch:{}    Score:{}    epsilon:{:.3}".format(i+1, score, epsilon))