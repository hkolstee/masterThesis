import os
import sys

import numpy as np

import torch
import torch.nn.functional as F

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

# from ..networks.MLP import MultiLayerPerceptron
from MLP import MultiLayerPerceptron

class DiscreteActor(MultiLayerPerceptron):
    """An actor network that is a policy function from the state action pair to a discrete action.
    Output is a softmax over actions, of which actions are sampled, entropy is entropy over softmax. 

    Args:
        lr (float): Learning rate
        obs_size (int): Observation size
        action_size (int): Actions size
        action_max (int): Scaling of the constrained policy samping which is restricted to [+1, -1]
            such that we get the appropriate action value
        layer_sizes (tuple:int): Sizes of the dense network layers
    """
    def __init__(self, lr, obs_size, action_size, layer_sizes = (256, 256)):
        super().__init__(lr = lr,
                         input_size = obs_size, 
                         output_size = action_size, 
                         layer_sizes = layer_sizes)
        # gumbel temperature (tau)
        self.gumbel_temperature = 1.0

        # action list to take dot product with onehot action vector
        self.action_categories = torch.tensor([i for i in range(action_size)], dtype = torch.float32)

    def set_gumbel_temperature(self, temperature):
        self.gumbel_temperature = temperature
        
    def forward(self, obs):
        # output is logits
        action_logits = super().forward(obs)[0]

        # return prob_distr, log_std
        return action_logits

    def action_distr_sample(self, obs, deterministic = False):
        action_logits = self.forward(obs)

        if deterministic:
            actions = torch.argmax(action_logits, 1).float()
            # avoid instability
            log_probs = F.log_softmax(action_logits).gather(1, actions.long().unsqueeze(1)).squeeze()
        else:
            # reperameterization sampling through gumbel softmax
            one_hot_actions = F.gumbel_softmax(action_logits, hard = True)
            # get action by multiplication while keeping gradients
            actions = torch.matmul(one_hot_actions, self.action_categories)
            # actions = torch.matmul(one_hot_actions, self.action_categories).unsqueeze(1)

            # log probabilities
            log_probs = -torch.sum(-one_hot_actions * F.log_softmax(action_logits, 1), 1)


        return actions, log_probs