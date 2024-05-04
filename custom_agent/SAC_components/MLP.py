import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

class MultiLayerPerceptron(nn.Module):
    """A multilayer perceptron. Allows for multiple output heads.

    Args:
        lr (float): Learning rate
        input_size (int): Input size
        output_size (int, tuple(int)): Output size
        layer_sizes (tuple:int): Sizes of the dense network layers
    """
    def __init__(self, lr, input_size, output_size, layer_sizes, optim_eps = 1e-08, weight_decay = 0):
        super(MultiLayerPerceptron, self).__init__()
        self.lr = lr
        self.input_size = input_size
        self.output_size = output_size
        self.layer_sizes = layer_sizes
        
        # initialize layers
        self.nr_layers = len(layer_sizes)
        self.layers = nn.ModuleList()

        # loop through network shape 
        for idx in range(self.nr_layers): # (256, 128) -> (10, 256), (256, 128)
            # first layer
            if (idx == 0):
                layer = nn.Linear(self.input_size, self.layer_sizes[idx])
            # layers inbetween first and last
            else:
                layer = nn.Linear(self.layer_sizes[idx - 1], self.layer_sizes[idx])
            
            # add layer
            self.layers.append(layer) 
            
        # last layer(s) (optionally multiple as different heads)
        # if multiple last layers
        if isinstance(self.output_size, list) or isinstance(self.output_size, tuple):
            # create new modulelist
            last_layers = nn.ModuleList()
            for dim in self.output_size:
                layer = nn.Linear(self.layer_sizes[-1], dim)
                last_layers.append(layer)
            # add modulelist to modulelist
            self.layers.append(last_layers)
        # single last layer
        else: 
            layer = nn.Linear(self.layer_sizes[-1], self.output_size)
            # add single layer to modulelist
            self.layers.append(layer)  

        # init optim
        self.optimizer = optim.Adam(self.parameters(), lr = lr, eps=optim_eps, weight_decay=weight_decay)

        # init device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        # explicitly set to float32
        self.float()
        
    def forward(self, input):
        """Forward through the MLP

        Args:
            input (torch.tensor): The input Tensor
            
        Returns:
            output ([torch.tensor]): Output of last layer(s)
        """
        data = input
        out = []
        for idx in range(self.nr_layers + 1):
            if (idx < self.nr_layers):
                # pass through first and middle layers
                data = self.layers[idx](data)    
                data = functional.relu(data)
            else:
                # last layer(s)
                # if multiple last layers
                if isinstance(self.layers[idx], nn.ModuleList):
                    for layer in self.layers[idx]:
                        # print("DASFS")
                        out.append(layer(data))
                    # print(out[0].shape, out[1].shape)
                else:
                    out.append(self.layers[idx](data))
        
        return out
    
    def save(self, save_dir, file_name):
        """Save the models weights.

        Args:
            save_dir (str): Directory to save to.
        """
        os.makedirs(save_dir, exist_ok = True)
        torch.save(self.state_dict(), os.path.join(save_dir, file_name))

    def save_checkpoint(self, save_dir, file_name, loss, step):
        """Save the model weights along with checkpoint information.

        Args:
            save_dir (str): Directory to save to.
            file_name (str): File name to save.
            step (int): Checkpoint step in training.
            optimizer (torch.optim): The optimizer to save along with the model.
            loss (float): Checkpoint loss.
        """
        # create checkpoint dict
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "step": step,
        }
        # save model weights as file
        os.makedirs(save_dir, exist_ok = True)
        torch.save(checkpoint, os.path.join(save_dir, file_name))

    def load(self, file_path):
        """Load the models weights.

        Args:
            file_path (str): Directory of file to load.
        """
        self.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))

    def load_checkpoint(self, checkpoint_file_path):
        """Load model weights along with checkpoint information.

        Args:
            checkpoint_file_path (str): Checkpoint file path.

        Returns:
            loss (float): Loss at checkpoint.
            step (int): Step at checkpoint.
        """
        # load checkpoint dict
        checkpoint = torch.load(checkpoint_file_path, map_location=torch.device('cpu'))
        # load model
        self.load_state_dict(checkpoint["model_state_dict"])
        # load optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return checkpoint["loss"], checkpoint["step"]