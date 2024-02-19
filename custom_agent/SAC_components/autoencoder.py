import torch
import torch.nn as nn
import torch.functional as functional

import numpy as np

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, code_dim):
        super().__init__()

        # encoder
        self.encoder = nn.Sequential()
        # we add dense layers until the given size
        node_sizes = [input_dim]
        while (node_sizes[-1] > code_dim):
            # reduction in nodes
            reduced = int(node_sizes[-1] * (1/2))
            if reduced < code_dim:
                reduced = code_dim

            # add linear and relu activation
            self.encoder.append(nn.Linear(node_sizes[-1], reduced))
            self.encoder.append(nn.ReLU())
            
            # append to list
            node_sizes.append(reduced)
        
        # hidden to code layer if necessary
        if node_sizes[-1] != code_dim:
            self.encoder.append(nn.Linear(node_sizes[-1], code_dim))
            self.encoder.append(nn.ReLU())
            node_sizes.append(code_dim)

        # decoder
        self.decoder = nn.Sequential()
        # add layers according to encoder layer sizes
        for (size_in, size_out) in zip(reversed(node_sizes), reversed(node_sizes[:-1])): 
            self.decoder.append(nn.Linear(size_in, size_out))
            self.decoder.append(nn.ReLU())

        # optimzer
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.01)

    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)

        return encoded, decoded

