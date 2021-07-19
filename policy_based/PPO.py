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
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def get_policy_op(self, inputs):
        policy_op = self.actor(inputs)
        softmax_op = F.softmax(policy_op, -1)
        return softmax_op
    
    def select_action(self, inputs):
        action_prob = self.get_policy_op(inputs)
        action = Categorical(action_prob)
        action = action.sample().item()
        return action, action_prob.detach().cpu().numpy()[action]

    def save_trans(self, transition):
        self.buffer.save_trans(transition)

    def to_tensor(self, items):
        s, a, r, s_next, a_prob, done = items
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s_next = torch.FloatTensor(s_next).to(self.device)
        a_prob = torch.FloatTensor(a_prob).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        return s, a.unsqueeze(-1), r.unsqueeze(-1), s_next, a_prob.unsqueeze(-1), done.unsqueeze(-1)

    def train(self, gamma = 0.98, batch_size = 32, k_iters = 3, epsilon_clip = 0.1, lmbda = 0.95):
        s, a, r, s_next, a_prob, done = self.to_tensor(self.buffer.sample_all_data())
        for i in range(k_iters):
            td_target = r + gamma * self.critic(s_next) * (1 - done)
            td_error = td_target - self.critic(s)
            td_error = td_error.detach().cpu().numpy()

            advantage_ls = []
            advantage = 0.
            for error in td_error[::-1]:
                advantage = gamma * lmbda * advantage + error[0]
                advantage_ls.append([advantage])
            advantage_ls.reverse()
            advantage = torch.FloatTensor(advantage_ls).to(self.device)

            policy_op = self.get_policy_op(s)
            policy_op = policy_op.gather(-1, a)
            ratio = torch.exp(torch.log(policy_op) - torch.log(a_prob))

            sur1 = ratio * advantage
            sur2 = torch.clamp(ratio, 1 - epsilon_clip, 1 + epsilon_clip) * advantage
            loss = (- torch.min(sur1, sur2) + F.smooth_l1_loss(self.critic(s), td_target.detach())).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.buffer.clear()

'''
PPO test
'''
if __name__ == "__main__":

    # hyper param
    lr = 1e-3
    render = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    env = gym.make("CartPole-v1")
    model = PPO(args = (4, 2, device, lr)).to(device)
    score = 0.
    
    for epo_i in range(10000):
        obs = env.reset()
        score = 0.
        for step in range(200):
            if render:
                env.render()

            a, a_prob = model.select_action(torch.FloatTensor(obs).to(device))
            obs_next, r, done, info = env.step(a)

            model.save_trans((obs, a, r, obs_next, a_prob, done))

            obs = obs_next
            score += r

            if done:
                break
        
        model.train()
        print("Epoch: {}  score: {}".format(epo_i, score))
        
    env.close()