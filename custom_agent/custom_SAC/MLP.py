import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

class MultiLayerPerceptron(nn.Module):
    """A multilayer perceptron.

    Args:
        lr (float): Learning rate
        input_dim (int): Input dimensions
        output_dim (int, [int]): Output dimensions
        layer_sizes (tuple:int): Sizes of the dense network layers
    """
    def __init__(self, lr, input_dim, output_dim, layer_sizes):
        super(MultiLayerPerceptron, self).__init__()
        self.lr = lr
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_sizes = layer_sizes
        
        # initialize layers
        self.nr_layers = len(layer_sizes)
        self.layers = nn.ModuleList()
        # loop through network shape 
        for idx in range(self.nr_layers): # (256, 128) -> (10, 256), (256, 128)
            # first layer
            if (idx == 0):
                layer = nn.Linear(self.input_dim, self.layer_sizes[idx])

            # layers inbetween first and last
            else:
                layer = nn.Linear(self.layer_sizes[idx - 1], self.layer_sizes[idx])
            
            # add layer
            self.layers.append(layer) 
            
        # last layer (optional multiple last layers)
        self.last_layers = nn.ModuleList()
        if isinstance(self.output_dim, list) or isinstance(self.output_dim, tuple):
            for dim in self.output_dim:
                layer = nn.Linear(self.layer_sizes[-1], dim)
                self.last_layers.append(layer)
        else: 
            self.last_layers.append(nn.Linear(self.layer_sizes[-1], self.output_dim))  
            
        # init optim
        self.optimizer = optim.Adam(self.parameters(), lr = lr)

        # init device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, input):
        """Forward through the MLP

        Args:
            input (torch.tensor): The input Tensor
            
        Returns:
            output ([torch.tensor]): Output of last layer(s)
        """
        data = input
        for idx in range(self.nr_layers):
            # pass through first and middle layers
            data = self.layers[idx](data)    
            data = functional.relu(data)
        
        # last layer(s)
        out = []
        for layer in self.last_layers:
            out.append(layer(data))
            
        return out