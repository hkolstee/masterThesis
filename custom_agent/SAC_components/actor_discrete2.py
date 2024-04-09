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

    def action_distr_sample(self, obs, reparameterize = True, deterministic = False):
        action_logits = self.forward(obs)


        # action = prob_distr.sample()

        # action_probs = prob_distr.probs

        log_prob = functional.log_softmax(action_logits, dim = 1).squeeze()

        # the normal prob distribution
        # action_softmax = functional.softmax(action_logits, dim = 1)
        # categorical distribution
        # prob_distr = torch.distributions.OneHotCategorical(action_softmax)

        if reparameterize:
        #     # to use the reparameterization trick instead
        #     # of the stochastic sampling process that stops
        #     # gradients, we can use the gumbel softmax
        #     # which introduces noise as a linear combination
        #     # to sample from the distribution.
            action = functional.gumbel_softmax(action_logits, hard = True, tau = self.gumbel_temperature, dim = 1)
            # one_hot_action = functional.gumbel_softmax(action_logits, hard = True, tau = self.gumbel_temperature, dim = 1)
            
        #     # convert one_hot_action into integer
            # action = torch.matmul(one_hot_action, self.action_categories)
        else:            
            prob_distr = torch.distributions.OneHotCategorical(logits = action_logits)
            if deterministic:
                action = torch.argmax(prob_distr.probs)
            else:
                action = prob_distr.sample()

        # log of the prob_distr function evaluated at sample value
        # log_prob = prob_distr.log_prob(action)
        # log_prob = log_prob.sum(axis = -1)

        # final action
        action = action.to(self.device) # should be on device

        # return action, log_prob, action_probs
        return action, log_prob
    
# a = DiscreteActor(0.0003, 4, 5)

# action = a.action_distr_sample(torch.tensor([[0.3, 0.3, 0.3, 0.3]]))

# print(action)