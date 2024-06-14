import os
import sys

import torch

# add folder to python path for relative imports
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

from MLP import MultiLayerPerceptron

class MultiDiscreteActor(MultiLayerPerceptron):
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
    def __init__(self, lr, obs_size, action_size, layer_sizes = (256, 256), eps = 1e-4):
        super().__init__(lr = lr,
                         input_size = obs_size, 
                         output_size = action_size, 
                         layer_sizes = layer_sizes,
                         optim_eps = eps)
        
    def forward(self, obs):
        # output is list of logits
        action_logits_list = super().forward(obs)

        return action_logits_list

    def action_distr_sample(self, obs):
        action_logits_list = self.forward(obs)

        actions = []
        log_probs = []
        action_probs = []

        for logits in action_logits_list:
            prob_distr = torch.distributions.OneHotCategorical(logits = logits)

            action = prob_distr.sample()

            action_prob = prob_distr.probs

            # avoid instability
            z = (action_prob == 0.0).float() * 1e-8
            log_prob = torch.log(action_prob + z)

            # add to lists
            actions.append(action)
            log_probs.append(log_prob)
            action_probs.append(action_prob)

        return actions, log_probs, action_probs

# act = MultiDiscreteActor(0.0003, 4, [3,3,3])

# input = torch.tensor([0.5, 0.5, 0.5, 0.5])

# print(act.forward(input))