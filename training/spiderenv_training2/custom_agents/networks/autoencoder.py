import torch
import torch.nn as nn
import torch.nn.functional as functional

import numpy as np

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, reduc_scale):
        super().__init__()

        # we take a reduction in ratio of 1:reduc_scale
        self.code_dim = input_dim // reduc_scale

        # encoder
        self.encoder = nn.Sequential()
        # we add dense layers until the given size
        node_sizes = [input_dim]
        while (node_sizes[-1] > self.code_dim):
            # reduction in nodes
            reduced = int(node_sizes[-1] * (1/2))
            if reduced < self.code_dim:
                reduced = self.code_dim

            # add linear and leaky relu activation
            self.encoder.append(nn.Linear(node_sizes[-1], reduced))
            self.encoder.append(nn.Sigmoid())
            
            # append to list
            node_sizes.append(reduced)
        
        # hidden to code layer if necessary
        if node_sizes[-1] != self.code_dim:
            self.encoder.append(nn.Linear(node_sizes[-1], self.code_dim))
            self.encoder.append(nn.sigmoid())
            node_sizes.append(self.code_dim)

        # decoder
        self.decoder = nn.Sequential()
        # add layers according to encoder layer sizes
        for (size_in, size_out) in zip(reversed(node_sizes), reversed(node_sizes[:-1])): 
            self.decoder.append(nn.Linear(size_in, size_out))
            self.decoder.append(nn.LeakyReLU())

        # optimzer
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.01)

    def forward(self, input):
        # this forward function automatically also takes a step down the gradient (convenient)
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        
        # calc loss
        loss = functional.mse_loss(decoded, input)

        # gradient step if not in with no_grad() zone
        if torch.is_grad_enabled():
            # zero the gradients
            self.optimizer.zero_grad()
            # backward step
            loss.backward()
            self.optimizer.step()

        return encoded.detach(), decoded.detach(), loss

