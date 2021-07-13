import torch
import numpy as np
import sys
import gym

sys.path.append("./")
from base_net.model import *
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

class SAC(nn.Module):
    def __init__(self, args):
        super(SAC, self).__init__()
        self.input_size, self.output_size, self.toi, self.device, self.lr = args
        self.min_log_std, self.max_log_std = -20, 2
        # policy
        self.actor = Policy_net(args = (self.input_size, 64))
        self.mu_head = nn.Linear(64, self.output_size)
        self.log_std_head = nn.Linear(64, self.output_size)

        self.q_net1 = Q_net(args = (self.input_size + self.output_size, 1))
        self.q_net2 = Q_net(args = (self.input_size + self.output_size, 1))
        
        self.v_net = Q_net(args = (self.input_size, 1))
        self.v_net_target = Q_net(args = (self.input_size, 1))

        self.replay_buffer = ReplayBuffer(args = (30000))
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)    
        self.update_target_net(initialize = True)

    def update_target_net(self, initialize = False):
        if initialize:
            self.v_net_target.load_state_dict(self.v_net.state_dict())
        elif not initialize:
            for raw, target in zip(self.v_net.parameters(), self.v_net_target.parameters()):
                target.data.copy_(self.toi * raw.data + (1 - self.toi) * target.data)

    def get_policy(self, inputs):
        return self.actor(inputs)

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

    def select_action(self, inputs):
        dist_param_op = self.get_policy(inputs)
        mu = self.mu_head(dist_param_op)
        log_sigma = torch.clamp(self.log_std_head(dist_param_op), self.min_log_std, self.max_log_std)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        z = dist.rsample()
        action = torch.tanh(z).detach().cpu().numpy()
        return action
    
    def get_action_vec(self, inputs):
        dist_param_op = self.get_policy(inputs)
        mu = self.mu_head(dist_param_op)
        log_sigma = torch.clamp(self.log_std_head(dist_param_op), self.min_log_std, self.max_log_std)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        z = dist.rsample()
        action_vec = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action_vec.pow(2) + torch.FloatTensor([1e-7]).to(self.device)).sum(1, keepdim=True)
        return action_vec, log_prob

    def train(self, gamma = 0.98, batch_size = 32, alpha = 0.6):
        s, a, r, s_next, done = self.to_tensor(self.replay_buffer.sample_batch(batch_size = batch_size))
        
        # loss v_net
        rsample_action, log_action_prob = self.get_action_vec(s)
        v_target = torch.min(self.q_net1(torch.cat([s, rsample_action], -1)), self.q_net2(torch.cat([s, rsample_action], -1))) - alpha * log_action_prob
        loss_v_net = (self.v_net(s) - v_target.detach()) ** 2
        # loss actor  gradient ascent
        loss_actor = -(self.q_net1(torch.cat([s, rsample_action], -1)) - alpha * log_action_prob)
        # loss q_net
        q_target = r + gamma * self.v_net_target(s_next) * (1 - done)                     # have problem
        loss_q_net1 = (self.q_net1(torch.cat([s, a], -1)) - q_target.detach()) ** 2
        loss_q_net2 = (self.q_net2(torch.cat([s, a], -1)) - q_target.detach()) ** 2

        loss = loss_actor.mean() + loss_q_net1.mean() + loss_q_net2.mean() + loss_v_net.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target_net()


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low[0]
        high = self.action_space.high[0]

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return [action]

    def reverse_action(self, action):
        low = self.action_space.low[0]
        high = self.action_space.high[0]

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return [action]


'''
SAC test
'''
if __name__ == "__main__":
    # hyper param
    batch_size = 32
    render = False
    lr = 5e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_flag = False

    total_step = 0
    env = NormalizedActions(gym.make("Pendulum-v0"))   
    model = SAC(args = (3, 1, 0.01, device, lr)).to(device)

    for i in range(10000):
        s = env.reset()
        score = 0.
        
        for t in range(200):
            if render:
                env.render()

            total_step += 1
            action = model.select_action(torch.FloatTensor(s).to(device))
                

            s_next, reward, done, info = env.step(action[0])
            model.save_trans((s, action, reward, s_next, done))
            score += reward 
            s = s_next

            if len(model.replay_buffer.buffer) > 60:
                train_flag = True
                model.train(batch_size = batch_size)
            if done:
                break
        print("Epoch:{}    Score:{}    training:{}    total_step:{}".format(i+1, score, train_flag, total_step))