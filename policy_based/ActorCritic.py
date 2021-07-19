import torch
import numpy as np
import sys
import gym

sys.path.append("./")
from base_net.model import *
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, args):
        super(ActorCritic, self).__init__()
        self.input_size, self.output_size, self.device, self.lr = args
        self.actor = Policy_net(args = (self.input_size, self.output_size))
        self.critic = Q_net(args = (self.input_size, 1))

        self.buffer = ReplayBuffer(args = (10000))
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)    
        
    def get_policy(self, inputs):
        return F.softmax(self.actor(inputs))

    def save_trans(self, transition):
        self.buffer.save_trans(transition)

    def to_tensor(self, items):
        s, a, r, s_next, done = items
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s_next = torch.FloatTensor(s_next).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        return s, a.unsqueeze(-1), r.unsqueeze(-1), s_next, done.unsqueeze(-1)

    def select_action(self, inputs):
        action_prob = self.get_policy(inputs)
        action = Categorical(action_prob)
        action = action.sample().item()
        return action

    def train(self, gamma = 0.98):
        s, a, r, s_next, done = self.to_tensor(self.buffer.sample_all_data())
        
        q_val = self.critic(s)
        target_q_val = r + gamma * self.critic(s_next) * (1 - done)
        advantage = target_q_val.detach() - q_val
        loss_critic = advantage ** 2

        loss_actor = -torch.log(self.get_policy(s).gather(-1, a)) * advantage.detach()
        loss = loss_actor.mean() + loss_critic.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.buffer.clear()

    def train_for_a3c(self, replay_buffer, gamma = 0.98):
        s, a, r, s_next, done = self.to_tensor(replay_buffer.sample_all_data())
        
        q_val = self.critic(s)
        target_q_val = r + gamma * self.critic(s_next) * (1 - done)
        advantage = target_q_val.detach() - q_val
        loss_critic = advantage ** 2

        loss_actor = -torch.log(self.get_policy(s).gather(-1, a)) * advantage.detach()
        loss = loss_actor.mean() + loss_critic.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        replay_buffer.clear()



'''
ActorCritic test
'''
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ActorCritic(args = (4, 2,  device, 1e-3)).to(device)
    for i in range(10000):
        s = env.reset()
        score = 0.
        for t in range(200):
            action = model.select_action(torch.FloatTensor(s).to(device))

            s_next, reward, done, info = env.step(action)
            model.save_trans((s, action, reward, s_next, done))
            score += reward 
            s = s_next
            if done:
                break
        model.train()
        print("Epoch:{}    Score:{}".format(i+1, score))