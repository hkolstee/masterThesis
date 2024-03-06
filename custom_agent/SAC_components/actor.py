import os

import numpy as np

import torch
import torch.nn.functional as functional

from MLP import MultiLayerPerceptron

class Actor(MultiLayerPerceptron):
    """An actor network that is a policy function from the state action pair to an action.

    Args:
        lr (float): Learning rate
        obs_size (int): Observation size
        action_size (int): Actions size
        action_max (int): Scaling of the constrained policy samping which is restricted to [+1, -1]
            such that we get the appropriate action value
        layer_sizes (tuple:int): Sizes of the dense network layers
    """
    def __init__(self, lr, obs_size, action_size, action_low, action_high, layer_sizes = (256, 256)):
        super().__init__(lr = lr,
                         input_size = obs_size, 
                         output_size = (action_size, action_size), 
                         layer_sizes = layer_sizes)
        # action scaling
        self.action_high = torch.tensor(action_high, dtype = torch.float32).to(self.device)
        self.action_low = torch.tensor(action_low, dtype = torch.float32).to(self.device)
        self.std_correction = torch.tensor((action_high - action_low) / 2, dtype = torch.float32).to(self.device)

        # clamping region, values taken from paper
        self.clamp_log_min = -20  # -5 also used
        self.clamp_log_max = 2
        
    def forward(self, obs):
        # output is probability distribution
        # log_std depends on the state, unlike in PPO
        mean, log_std = super().forward(obs)

        # we clamp the std (reason: we do not want an arbitrary large std)
        #   the original SAC paper clamps in the range of [-20, 2]
        log_std = torch.clamp(log_std, min = self.clamp_log_min, max = self.clamp_log_max)
        # log_std = torch.tanh(log_std)
        # LOG_MIN = -5
        # LOG_MAX = 2
        # log_std = LOG_MIN + 0.5 * (LOG_MAX - LOG_MIN) * (log_std + 1)

        # convert from log_std to normal std
        std = log_std.exp()

        return mean, std
    
    def scale_action(self, action):
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)
        """
        return 2 * ((action - self.action_low) / (self.action_high - self.action_low)) - 1

    def unscale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        return self.action_low + (0.5 * (scaled_action + 1.0) * (self.action_high - self.action_low))

    def normal_distr_sample(self, obs, reparameterize = True, deterministic = False):
        mean, std = self.forward(obs)
        
        # get prob distribution
        prob_distr = torch.distributions.Normal(mean, std)

        # if we need to evaluate the policy
        if deterministic:
            sample = mean
        # add noise to values (reparameterize trick; exploration)
        #   (mean + std * N(0, I))
        elif reparameterize:
            sample = prob_distr.rsample()
        # no noise
        else:
            sample = prob_distr.sample()

        # tanh of action sample
        tanh_action = torch.tanh(sample) 

        # log of the prob_distr function evaluated at sample value
        log_prob = prob_distr.log_prob(sample)
        # correct change in prob density due to tanh 
        #   (function = magic, taken from openAI spinning up)
        log_prob -= (2 * (np.log(2) - sample - functional.softplus(-2 * sample)))
        log_prob = log_prob.sum(dim = 1, keepdim = True)
        # other function from stable baselines
        # log_prob -= torch.log(self.std_correction * (1 - tanh_action.pow(2)) + 1e-6)
        # log_prob = log_prob.sum(dim = 1, keepdim = True)

        # final action
        #   constrain action within [-1, 1] with tanh,
        #   unscale for regular action range.
        action = self.unscale_action(tanh_action)
        action = action.to(self.device) # should be on device

        return action, log_prob
    
    