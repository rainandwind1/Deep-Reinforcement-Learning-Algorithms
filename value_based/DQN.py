import torch
import numpy as np
import sys
import gym

sys.path.append("./")
from base_net.model import *
from torch import nn, optim

class DQN(nn.Module):
    def __init__(self, args):
        super(DQN, self).__init__()
        self.input_size, self.output_size, self.mem_size, self.device, self.lr = args
        self.q_net = Q_net(args = (self.input_size, self.output_size))
        self.target_q_net = Q_net(args = (self.input_size, self.output_size))
        
        self.replay_buffer = ReplayBuffer(args = (self.mem_size))
        self.optimizer = optim.Adam(self.q_net.parameters(), lr = self.lr)
        self.update_target_net()

    def select_action(self, inputs, epsilon):
        q_val = self.q_net(inputs)
        coin = np.random.rand()
        if coin > epsilon:
            return torch.argmax(q_val).detach().cpu().numpy().item()
        else:
            return random.sample(range(self.output_size), 1)[0]

    def save_trans(self, transition):
        self.replay_buffer.save_trans(transition)

    def update_target_net(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def to_tensor(self, items):
        s, a, r, s_next, done = items
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s_next = torch.FloatTensor(s_next).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        return s, a.unsqueeze(-1), r.unsqueeze(-1), s_next, done.unsqueeze(-1)


    def train(self, gamma = 0.98, batch_size = 32):
        s, a, r, s_next, done = self.to_tensor(self.replay_buffer.sample_batch(batch_size))
        q_val = self.q_net(s).gather(-1, a)
        target_q = r + gamma * torch.max(self.target_q_net(s_next).detach(), -1, keepdim=True)[0] * (1 - done)

        td_error = (q_val - target_q.detach()) ** 2
        loss = td_error.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



'''
DQN test
'''
if __name__ == "__main__":

    # hyper param
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_flag = False
    render = False
    batch_size = 32
    gamma = 0.98
    mem_size = 20000
    update_target_interval = 200
    lr = 1e-3
    total_step = 0

    env = gym.make("CartPole-v1")

    model = DQN(args = (4, 2, mem_size, device, lr)).to(device)
    epsilon = 0.8
    for ep_i in range(10000):
        epsilon = max(0.01, epsilon * 0.999)
        s = env.reset()
        score = 0.
        
        for i in range(200):
            if render:
                env.render()

            a = model.select_action(torch.FloatTensor(s).to(device), epsilon = epsilon)

            s_next, reward, done, info = env.step(a)

            model.save_trans((s, a, reward, s_next, done))
            
            total_step += 1
            score += reward
            s = s_next

            if len(model.replay_buffer.buffer) >= batch_size:
                train_flag = True
                model.train(gamma = gamma, batch_size = batch_size)

            if total_step % update_target_interval == 0 and train_flag:
                model.update_target_net()
            
            if done:
                break
        
        print("{} epoch score: {}  training: {}  epsilon:{:.3}".format(ep_i+1, score, train_flag, epsilon))