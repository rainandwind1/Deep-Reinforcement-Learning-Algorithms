import torch
import numpy as np
import sys
import gym

sys.path.append("./")
from base_net.model import *
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical
import threading
from ActorCritic import *


class actor_critic_threading(threading.Thread):
    def __init__(self, args):
        super(actor_critic_threading, self).__init__()
        self.model, self.env, self.threading_id = args

    def run(self):
        threadLock.acquire()
        self.runprocess() 
        threadLock.release()

    def runprocess(self):
        global episode_cnt
        episode_cnt += 1
        s = self.env.reset()
        score = 0.
        self.replay_buffer = ReplayBuffer(args = (10000))
        for t in range(200):
            action = self.model.select_action(torch.FloatTensor(s).to(device))

            s_next, reward, done, info = self.env.step(action)
            self.replay_buffer.save_trans((s, action, reward, s_next, done))
            score += reward 
            s = s_next
            if done:
                break
        self.model.train_for_a3c(self.replay_buffer)
        print("Epoch:{}    Score:{}".format(episode_cnt+1, score))
        return         

def make_envs(nums):
    envs = []
    for i in range(nums):
        envs.append(gym.make("CartPole-v1"))
        print("Initializing {} env ...".format(i))
    return envs


if __name__ == "__main__":

    # hyper param
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 1e-3
    thread_num = 4
    episode_cnt = 0

    envs = make_envs(thread_num)
    model = ActorCritic(args = (4, 2, device, lr)).to(device)

    while True:
        thread_list = []
        threadLock = threading.Lock()
        for i in range(thread_num):
            thread_list.append(actor_critic_threading(args = (model, envs[i], i)))

        for i in range(thread_num):
            thread_list[i].start()

        for i in range(thread_num):
            thread_list[i].join()