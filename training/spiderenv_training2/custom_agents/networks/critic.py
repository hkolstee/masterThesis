import os
import sys

from gymnasium import spaces

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
    """A critic network that is a function from the state action pair to a Q value.

    Args:
        lr (float): Learning rate
        obs_space (gymnasium.spaces): Observation space
        action_space (gymnasium.spaces): Action space
        layer_sizes (tuple:int): Sizes of the dense network layers
    """
    def __init__(self, lr, obs_size, act_size, layer_sizes = (256, 256), discrete = False):
        self.discrete = discrete
        # different in/output for discrete / continuous action space
        if discrete:
            input_size = obs_size
            output_size = act_size
        else:
            input_size = obs_size + act_size
            output_size = 1
        # initialize MLP
        super().__init__(lr = lr,
                         input_size = input_size, 
                         output_size = output_size, 
                         layer_sizes = layer_sizes)        
        
    def forward(self, obs, action = None):
        """
        Forward through the critic network. Works for only state, or state + action.
        """
        if self.discrete:
            input = obs
        else: 
            input = torch.cat([obs, action], dim = 1).float()
        out = super().forward(input)[0]
                
        return out.squeeze(-1)