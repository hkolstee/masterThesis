import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

from MLP import MultiLayerPerceptron

class Actor(MultiLayerPerceptron):
    """An actor network that is a policy function from the state action pair to an action.

    Args:
        lr (float): Learning rate
        obs_dim (int): Observation dimensions
        actions_dim (int): Actions dimensions
        action_max (int): Scaling of the constrained policy samping which is restricted to [+1, -1]
            such that we get the appropriate action value
        layer_sizes (tuple:int): Sizes of the dense network layers
    """
    def __init__(self, lr, obs_dim, actions_dim, action_max, layer_sizes = (256, 256)):
        super().__init__(lr = lr,
                         input_dim = obs_dim, 
                         output_dim = (actions_dim, actions_dim), 
                         layer_sizes = layer_sizes)
        # action scaling
        self.action_max = action_max
        # noise
        self.noise = 1e-6
        
    def forward(self, obs):
        # output is probability distribution
        [mean, std] = super().forward(obs)
        
        # we clamp the std (reason: we do not want an arbitrary large std)
        # the original SAC paper clamps in the range of [-20, 2], but we can't
        # use a range that encompasses std == 0, therefore, we take a small 
        # min and max = 1.
        std = torch.clamp(std, min = 1e-6, max = 1)
        
        return mean, std

# print(actor.last_layers)


