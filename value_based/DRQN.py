import torch
import numpy as np
import sys
import gym

sys.path.append("./")
from base_net.model import *
from torch import nn, optim


#  rnn module: GRUCell
class DRQN_GRUCell(nn.Module):
    def __init__(self, args):
        super(DRQN_GRUCell, self).__init__()
        self.input_size, self.output_size, self.max_ep_len, self.mem_size, self.device, self.lr = args
        self.q_net = GRUCell_Q_net(args = (self.input_size, self.output_size))
        self.target_q_net = GRUCell_Q_net(args = (self.input_size, self.output_size))
        
        self.replay_buffer = ReplayBuffer(args = (self.mem_size))
        self.optimizer = optim.Adam(self.q_net.parameters(), lr = self.lr)
        self.update_target_net()

    def select_action(self, inputs, epsilon):
        q_val = self.q_net(inputs)
        coin = np.random.rand()
        if coin > epsilon:
            action = torch.argmax(q_val).detach().cpu().numpy().item()
        else:
            action = random.sample(range(self.output_size), 1)[0]
        return action

    def save_trans(self, transition):
        self.replay_buffer.save_trans(transition, episode_data = True, max_len = self.max_ep_len)

    def update_target_net(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def to_tensor(self, items):
        s, a, r, s_next, done = items
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s_next = torch.FloatTensor(s_next).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        return s, a, r, s_next, done

    def init_hidden(self, batch_size = 1):
        self.q_net.init_hidden(batch_size = batch_size)
        self.target_q_net.init_hidden(batch_size = batch_size)

    def train(self, gamma = 0.98, batch_size = 32, update_time = 2):
        for i in range(update_time):
            s, a, r, s_next, done = self.to_tensor(self.replay_buffer.sample_batch(batch_size))
            q_val_op = []
            target_q_op = []
            self.q_net.init_hidden(batch_size = batch_size)
            self.target_q_net.init_hidden(batch_size = batch_size)
            for t in range(self.max_ep_len):
                q_val = self.q_net(s[:, t]).gather(-1, a[:, t])
                target_q = r[:, t] + gamma * torch.max(self.target_q_net(s_next[:, t]).detach(), -1, keepdim=True)[0] * (1 - done[:, t])
                q_val_op.append(q_val)
                target_q_op.append(target_q)

            # stack on step
            q_val_op = torch.stack(q_val_op, dim = 1)
            target_q_op = torch.stack(target_q_op, dim = 1)

            td_error = (q_val_op - target_q_op.detach()) ** 2
            loss = td_error.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


#  rnn module: GRU
class DRQN_GRU(nn.Module):
    def __init__(self, args):
        super(DRQN_GRU, self).__init__()
        self.input_size, self.output_size, self.max_ep_len, self.mem_size, self.device, self.lr = args
        self.q_net = GRU_Q_net(args = (self.input_size, self.output_size))
        self.target_q_net = GRU_Q_net(args = (self.input_size, self.output_size))
        
        self.replay_buffer = ReplayBuffer(args = (self.mem_size))
        self.optimizer = optim.Adam(self.q_net.parameters(), lr = self.lr)
        self.update_target_net()

    def select_action(self, inputs, epsilon):
        q_val = self.q_net(inputs).squeeze(0)
        coin = np.random.rand()
        if coin > epsilon:
            action = torch.argmax(q_val).detach().cpu().numpy().item()
        else:
            action = random.sample(range(self.output_size), 1)[0]
        return action

    def save_trans(self, transition):
        self.replay_buffer.save_trans(transition, episode_data = True, max_len = self.max_ep_len)

    def update_target_net(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def to_tensor(self, items):
        s, a, r, s_next, done = items
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s_next = torch.FloatTensor(s_next).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        return s, a, r, s_next, done

    def init_hidden(self, batch_size = 1):
        self.q_net.init_hidden(batch_size = batch_size)
        self.target_q_net.init_hidden(batch_size = batch_size)

    def train(self, gamma = 0.98, batch_size = 32, update_time = 2):
        for i in range(update_time):
            s, a, r, s_next, done = self.to_tensor(self.replay_buffer.sample_batch(batch_size))
            self.q_net.init_hidden(batch_size = batch_size)
            self.target_q_net.init_hidden(batch_size = batch_size)
            q_val = self.q_net(s).gather(-1, a)
            target_q = r + gamma * torch.max(self.target_q_net(s_next).detach(), -1, keepdim=True)[0] * (1 - done)

            td_error = (q_val - target_q.detach()) ** 2
            loss = td_error.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def test_GRUCell_DRQN():
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

    model = DRQN_GRUCell(args = (4, 2, 200, mem_size, device, lr)).to(device)
    epsilon = 0.8
    for ep_i in range(10000):
        epsilon = max(0.01, epsilon * 0.999)
        s = env.reset()
        score = 0.
        
        s_ls = []
        a_ls = []
        r_ls = []
        s_next_ls = []
        done_ls = []
        model.init_hidden()

        for i in range(200):
            if render:
                env.render()

            a = model.select_action(torch.FloatTensor(s).unsqueeze(0).to(device), epsilon = epsilon)

            s_next, reward, done, info = env.step(a)

            # episode data save
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([reward])
            s_next_ls.append(s_next)
            done_ls.append([done])

            # cnt update
            total_step += 1
            score += reward
            s = s_next

            if done:
                break
        
        model.save_trans((s_ls, a_ls, r_ls, s_next_ls, done_ls))

        # episode end update 
        if len(model.replay_buffer.buffer) >= batch_size:
            train_flag = True
            model.train(gamma = gamma, batch_size = batch_size, update_time = 2)

        if (ep_i +1) % update_target_interval == 0 and train_flag:
            model.update_target_net()
            
        print("{} epoch score: {}  training: {}  epsilon:{:.3}".format(ep_i+1, score, train_flag, epsilon))



def test_GRU_DRQN():
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

    model = DRQN_GRU(args = (4, 2, 200, mem_size, device, lr)).to(device)
    epsilon = 0.8
    for ep_i in range(10000):
        epsilon = max(0.01, epsilon * 0.999)
        s = env.reset()
        score = 0.
        
        s_ls = []
        a_ls = []
        r_ls = []
        s_next_ls = []
        done_ls = []
        model.init_hidden()

        for i in range(200):
            if render:
                env.render()

            a = model.select_action(torch.FloatTensor(s).unsqueeze(0).unsqueeze(0).to(device), epsilon = epsilon)

            s_next, reward, done, info = env.step(a)

            # episode data save
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([reward])
            s_next_ls.append(s_next)
            done_ls.append([done])

            # cnt update
            total_step += 1
            score += reward
            s = s_next

            if done:
                break
        
        model.save_trans((s_ls, a_ls, r_ls, s_next_ls, done_ls))

        # episode end update 
        if len(model.replay_buffer.buffer) >= batch_size:
            train_flag = True
            model.train(gamma = gamma, batch_size = batch_size, update_time = 2)

        if (ep_i +1) % update_target_interval == 0 and train_flag:
            model.update_target_net()
            
        print("{} epoch score: {}  training: {}  epsilon:{:.3}".format(ep_i+1, score, train_flag, epsilon))


'''
DRQN test
'''
if __name__ == "__main__":
    test_GRU_DRQN()
    # test_GRUCell_DRQN()
