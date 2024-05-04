import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

# add folder to python path for relative imports
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

from MLP import MultiLayerPerceptron

class Critic(MultiLayerPerceptron):
    """A critic network that is a function from the state action pair to a value.

    Args:
        lr (float): Learning rate
        obs_size (int): Observation size
        action_size (int): Actions size
        layer_sizes (tuple:int): Sizes of the dense network layers
    """
    def __init__(self, lr, obs_size, act_size, layer_sizes = (256, 256), optim_eps = 1e-4):
        super().__init__(lr = lr,
                         input_size = obs_size, 
                         output_size = act_size, 
                         layer_sizes = layer_sizes,
                         optim_eps = optim_eps)
        
    def forward(self, obs):
        out = super().forward(obs)[0]
                
        return out