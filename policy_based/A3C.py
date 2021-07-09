import torch
import numpy as np
import sys
import gym

sys.path.append("./")
from base_net.model import *
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical


class A3C(nn.Module):
    def __init__(self, args):
        super(A3C, args).__init__()
        