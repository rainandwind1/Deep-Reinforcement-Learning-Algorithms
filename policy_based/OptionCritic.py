import torch
import numpy as np
import sys
import gym

sys.path.append("./")
from base_net.model import *
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli

class OptionCritic(nn.Module):
    def __init__(self, args):
        super(OptionCritic, self).__init__()
        self.input_size, self.option_num, self.output_size, self.device, self.lr = args
        self.macro_policy = Q_net(args = (self.input_size, self.option_num))
        self.terminated_net = nn.Linear(self.input_size, self.option_num)
        self.option_actor = Policy_net(args = (self.input_size, self.output_size))
        # option ls
        self.option_ls = nn.ModuleList([self.option_actor for _ in range(self.option_num)])

        self.replay_buffer = ReplayBuffer(args = (30000))
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)    
        
    def get_option_id(self, inputs, epsilon):
        macro_op = self.macro_policy(inputs)
        coin = np.random.rand()
        if coin > epsilon:
            return torch.argmax(macro_op, -1).detach().cpu().numpy().item()
        else:
            return random.sample(range(self.option_num), 1)[0]
    
    def get_policy(self, inputs, option_id):
        return F.softmax(self.option_ls[option_id](inputs), -1)

    def get_termination_op(self, inputs):
        return self.terminated_net(inputs).sigmoid()

    def save_trans(self, transition):
        self.replay_buffer.save_trans(transition)

    def to_tensor(self, items):
        s, macro_id, a, r, s_next, done = items
        s = torch.FloatTensor(s).to(self.device)
        macro_id = torch.LongTensor(macro_id).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s_next = torch.FloatTensor(s_next).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        return s, macro_id.unsqueeze(-1), a.unsqueeze(-1), r.unsqueeze(-1), s_next, done.unsqueeze(-1)

    def is_terminated(self, inputs, macro_id):
        termination_op = self.get_termination_op(inputs)[macro_id]
        option_termination = Bernoulli(termination_op).sample()
        return bool(option_termination.item())

    def select_action(self, inputs, epsilon, macro_id_pre):
        if self.is_terminated(inputs, macro_id_pre):
            option_id = self.get_option_id(inputs, epsilon)
        else:
            option_id = macro_id_pre
        action_prob = self.get_policy(inputs, option_id)
        action_dis = Categorical(action_prob)
        action = action_dis.sample().item()
        return action, option_id

    def train(self, gamma = 0.98, batch_size = 32, on_off = True):
        if on_off:  
            s, macro_id, a, r, s_next, done = self.to_tensor(self.replay_buffer.sample_all_data())                      # on policy
        else:
            s, macro_id, a, r, s_next, done = self.to_tensor(self.replay_buffer.sample_batch(batch_size = batch_size))  # off policy
        
        # loss critic
        q_val = self.macro_policy(s).gather(-1, macro_id)
        cur_option_terminated_prob = self.get_termination_op(s).gather(-1, macro_id)
        option_terminated_prob = self.get_termination_op(s_next).gather(-1, macro_id).detach()
        q_target_val = self.macro_policy(s_next)
        q_target = r + gamma * ((1 - option_terminated_prob) * q_target_val.gather(-1, macro_id) + option_terminated_prob * torch.max(q_target_val, -1, keepdim=True)[0]) * (1 - done)
        Advantage = q_target.detach() - q_val
        loss_critic = Advantage ** 2
        # loss actor & terminated net
        log_action_prob_ls = []
        for i in range(s.shape[0]):
            log_action_prob = torch.log(self.get_policy(s[i], macro_id[i])[a[i]])
            log_action_prob_ls.append(log_action_prob)
        log_action_prob = torch.stack(log_action_prob_ls, 0)
        policy_loss = - (q_target.detach() - q_val).detach() * log_action_prob
        termination_loss = cur_option_terminated_prob * (self.macro_policy(s_next).gather(-1, macro_id) - self.macro_policy(s_next).max(-1, keepdim=True)[0]).detach()
        loss_actor = policy_loss + termination_loss

        loss = loss_critic.mean() + loss_actor.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if on_off:
            self.replay_buffer.clear()



'''
OptionCritic test
'''
if __name__ == "__main__":
    on_off = True
    env = gym.make("CartPole-v0")
    batch_size = 32
    train_flag = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OptionCritic(args = (4, 4, 2,  device, 1e-3)).to(device)
    epsilon = 0.5

    for i in range(10000):
        s = env.reset()
        score = 0.
        epsilon = max(0.01, epsilon * 0.999)
        macro_id_pre = model.get_option_id(torch.FloatTensor(s).to(device), epsilon)        # initialize
        for t in range(200):
            action, macro_id = model.select_action(torch.FloatTensor(s).to(device), epsilon, macro_id_pre)

            s_next, reward, done, info = env.step(action)
            model.save_trans((s, macro_id, action, reward, s_next, done))
            score += reward 
            s = s_next
            macro_id_pre = macro_id

            # update off policy
            if not on_off:
                if len(model.replay_buffer.buffer) > batch_size:
                    train_flag = True
                    model.train(batch_size = batch_size, on_off = on_off)
            if done:
                break
        # update on policy
        if on_off:
            train_flag = True
            model.train(on_off = on_off)
        print("Epoch:{}    Score:{}    Training:{}    epislon:{}".format(i+1, score, train_flag, epsilon))