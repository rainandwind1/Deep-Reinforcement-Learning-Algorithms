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
        self.critic = Q_net(args = (self.input_size + self.output_size, 1))
        
        # activate func
        self.activate_function = "tanh"
        
        # target
        self.target_actor = Policy_net(args = (self.input_size, self.output_size))
        self.target_critic = Q_net(args = (self.input_size + self.output_size, 1))

        self.replay_buffer = ReplayBuffer(args = (self.mem_size))
        self.optimizer_actor = optim.Adam(self.actor.parameters(), self.lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), self.lr)
        # self.optimizer = optim.Adam([{'params':self.actor.parameters()}, {'params':self.critic.parameters()}], lr = self.lr)
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
        if self.activate_function == "tanh":
            op = torch.tanh(raw_op) * (self.action_max - self.action_min) / 2
        elif self.activate_function == "sigmoid":
            op = torch.sigmoid(raw_op) * (self.action_max - self.action_min) + self.action_min
        if self.clamp:
            op = torch.clamp(op, self.action_min, self.action_max)
        return op

    def get_target_policy_op(self, inputs):
        raw_op = self.target_actor(inputs)
        if self.activate_function == "tanh":
            op = torch.tanh(raw_op) * (self.action_max - self.action_min) / 2
        elif self.activate_function == "sigmoid":
            op = torch.sigmoid(raw_op) * (self.action_max - self.action_min) + self.action_min
        if self.clamp:
            op = torch.clamp(op, self.action_min, self.action_max)
        return op

    def select_action(self, inputs, epsilon, eval_mode = False):
        actor_op = self.get_policy_op(inputs)
        noise = torch.randn(actor_op.shape).to(self.device) if not eval_mode else 0.        # Gaussi noise
        # print(actor_op, noise * epsilon)
        actor_op += noise * epsilon
        if self.clamp:
            actor_op = torch.clamp(actor_op, self.action_min, self.action_max)
        return actor_op.detach().cpu().numpy()

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

    def train(self, gamma = 0.98, batch_size = 32):
        s, a, r, s_next, done = self.to_tensor(self.replay_buffer.sample_batch(batch_size=batch_size))
        # cal policy loss
        actor_loss = self.critic(torch.cat([s, self.get_policy_op(s)], -1))
        actor_loss = -actor_loss.mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()


        # cal critic loss
        q_val = self.critic(torch.cat([s, a] ,-1))
        target_q_val = r + gamma * self.target_critic(torch.cat([s_next, self.get_target_policy_op(s_next).detach()], -1)) * (1 - done)
        critic_loss = (q_val - target_q_val.detach()) ** 2
        critic_loss = critic_loss.mean()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        
        loss = actor_loss + critic_loss
        self.update_target_net()

        return loss.detach().cpu().numpy()

'''
DDPG test
''' 
if __name__ == "__main__":
    
    # hyper param
    epsilon = 0.3
    batch_size = 32
    render = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    total_step = 0
    env = gym.make("Pendulum-v0")   
    model = DDPG(args = (3, 2, 30000, True, 2, -2, device, 1e-3)).to(device)

    for i in range(10000):
        s = env.reset()
        score = 0.
        epsilon = max(0.01, epsilon * 0.999)
        
        for t in range(200):
            if render:
                env.render()

            total_step += 1
            action = model.select_action(torch.FloatTensor(s).to(device), epsilon)

            s_next, reward, done, info = env.step(action)
            model.save_trans((s, action, reward, s_next, done))
            score += reward 
            s = s_next
            if len(model.replay_buffer.buffer) > 60:
                model.train(batch_size = batch_size)
            if done:
                break
        print("Epoch:{}    Score:{}    epsilon:{:.3}    total_step:{}".format(i+1, score, epsilon, total_step))
