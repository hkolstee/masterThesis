import os

import numpy as np

import torch
import torch.nn.functional as F

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

        probs = F.softmax(action_logits)

        prob_distr = torch.distributions.Categorical(probs)

        actions = prob_distr.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (probs == 0.0).float() * 1e-8
        log_probs = torch.log(probs + z)

        return actions, log_probs, probs

        # prob_distr = torch.distributions.OneHotCategorical(logits = action_logits)

        # log_prob = F.log_softmax(action_logits, dim = 1).squeeze()

        # if reparameterize:
        #     # to use the reparameterization trick instead
        #     # of the stochastic sampling process that stops
        #     # gradients, we can use the gumbel softmax
        #     # which introduces noise as a linear combination
        #     # to sample from the distribution.
        #     action = functional.gumbel_softmax(action_logits, hard = True, tau = self.gumbel_temperature, dim = 1)
            
        #     # convert one_hot_action into integer
        #     # action = torch.matmul(one_hot_action, self.action_categories)
        # else:            
        #     prob_distr = torch.distributions.OneHotCategorical(logits = action_logits)
        #     if deterministic:
        #         action = torch.argmax(prob_distr.probs)
        #     else:
        #         action = prob_distr.sample()

        # action_clone = action.clone().detach()
        # # get log prob associated with current action
        # log_prob_action = torch.masked_select(log_prob, action_clone.bool())
        # # log_prob_action = prob_distr.log_prob(action)
        # # print("LOGp", log_prob_action.shape)

        # # final action
        # action = action.to(self.device) # should be on device

        # # return action, log_prob, action_probs
        # return action, log_prob_action, prob_distr.probs
    
# a = DiscreteActor(0.0003, 4, 5)

# action = a.action_distr_sample(torch.tensor([[0.3, 0.3, 0.3, 0.3]]))

# print(action)