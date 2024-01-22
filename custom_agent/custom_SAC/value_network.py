import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

from MLP import MultiLayerPerceptron

class ValueNetwork(MultiLayerPerceptron):
    """A value network that is a function from the state to a value.

    Args:
        lr (float): Learning rate
        obs_dim (int): Observation dimensions
        layer_sizes (tuple:int): Sizes of the dense network layers
    """
    def __init__(self, lr, obs_dim, layer_sizes = (256, 256)):
        super().__init__(lr = lr,
                         input_dim = obs_dim, 
                         output_dim = 1, 
                         layer_sizes = layer_sizes)
        
    def forward(self, obs):
        out = super().forward(obs)
        
        return out[0]