import os

import numpy as np

import torch
import torch.nn.functional as functional

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
    def __init__(self, lr, obs_size, act_size, layer_sizes = (256, 256), eps = 1e-4):
        super().__init__(lr = lr,
                         input_size = obs_size, 
                         output_size = act_size, 
                         layer_sizes = layer_sizes,
                         optim_eps = eps)
        # gumbel temperature (tau)
        self.gumbel_temperature = 1.0

        # action list to take dot product with onehot action vector
        self.action_categories = torch.tensor([i for i in range(act_size)], dtype = torch.float32)

    def set_gumbel_temperature(self, temperature):
        self.gumbel_temperature = temperature
        
    def forward(self, obs):
        # output is logits
        action_logits = super().forward(obs)[0]

        # return prob_distr, log_std
        return action_logits

    def action_distr_sample(self, obs, reparameterize = True, deterministic = False):
        action_logits = self.forward(obs)

        prob_distr = torch.distributions.OneHotCategorical(logits = action_logits)

        action = prob_distr.sample()

        action_probs = prob_distr.probs

        # avoid instability
        z = (action_probs == 0.0).float() * 1e-8
        log_prob = torch.log(action_probs + z)

        # final action
        action = action.to(self.device) # should be on device

        return action, log_prob, action_probs
