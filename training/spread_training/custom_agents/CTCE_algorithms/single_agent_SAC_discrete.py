import os
import sys

import numpy as np

import torch
import torch.nn.functional as F

from gymnasium import spaces

from copy import deepcopy

# add folder to python path for relative imports
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

from ..networks.actor_discrete import DiscreteActor
from ..networks.critic import Critic
from ..core.SAC_single_agent_base import SoftActorCriticCore

class SAC(SoftActorCriticCore):
    """
    Single agent Soft Actor-Critic. Both discrete and continuous action space
    compatible.
    """
    def __init__(self, 
                 env, 
                 lr_actor = 0.0003,
                 lr_critic = 0.0003,
                 gamma=0.99, 
                 polyak=0.995, 
                 buffer_max_size=1000000, 
                 batch_size=256,
                 layer_sizes=(256, 256), 
                 use_AE = False,
                 AE_reduc_scale = 6,
                 log_dir="tensorboard_logs", 
                 ):
        """
        Super class initializes all but the actor and critic networks and 
        entropy target.
        """        
        super().__init__(env = env, 
                         gamma = gamma, 
                         alpha_lr = lr_critic, # shares critic lr
                         polyak = polyak, 
                         buffer_max_size = buffer_max_size, 
                         batch_size = batch_size,
                         use_AE = use_AE,
                         AE_reduc_scale = AE_reduc_scale,
                         log_dir = log_dir)
        
        # different input size if autoencoder is used
        if use_AE:
            obs_size = self.autoencoder.code_dim
        else:
            obs_size = self.env.observation_space.shape[0]

        # initialize networks 
        self._actor = DiscreteActor(lr = lr_actor,
                                   obs_size = obs_size,
                                   action_size = self.env.action_space.n,
                                   layer_sizes = layer_sizes)
        # double clipped Q learning
        self._critic1 = Critic(lr = lr_critic, 
                              obs_size = obs_size,
                              act_size = self.env.action_space.n, 
                              discrete = True,
                              layer_sizes = layer_sizes)
        self._critic2 = Critic(lr = lr_critic, 
                              obs_size = obs_size,
                              act_size = self.env.action_space.n, 
                              discrete = True,
                              layer_sizes = layer_sizes)
        
        # target networks
        self._critic1_targ = deepcopy(self.critic1)
        self._critic2_targ = deepcopy(self.critic2)
        # freeze parameter gradient calculation as it is not used
        self.freeze_network_grads(self.critic1_targ)
        self.freeze_network_grads(self.critic2_targ)

        # entropy target
        # self._entropy_targ = -0.98 * torch.log(1 / torch.tensor(self.env.action_space.n))
        self._entropy_targ = -env.action_space.n

    """
    Manage abstract parent properties/attributes. 
    """
    @property
    def actor(self):
        return self._actor

    @property
    def critic1(self):
        return self._critic1

    @property
    def critic2(self):
        return self._critic2

    @property
    def critic1_targ(self):
        return self._critic1_targ

    @property
    def critic2_targ(self):
        return self._critic2_targ
    
    @property
    def entropy_targ(self):
        return self._entropy_targ

    """
    Overriding abstract parent methods.
    """
    def critic_loss(self, obs, next_obs, replay_act, rewards, dones):
        """
        Returns the critic loss
        """
        # These Q values are the left hand side of the loss function
        # discrete critic estimates Q-values for all discrete actions
        q1_buffer = self.critic1.forward(obs)
        q2_buffer = self.critic2.forward(obs)
        # gather Q-values for chosen action
        q1_buffer = q1_buffer.gather(1, replay_act.long().unsqueeze(1)).squeeze()
        q2_buffer = q2_buffer.gather(1, replay_act.long().unsqueeze(1)).squeeze()

        # For the RHS of the loss function (Approximation of Bellman equation with (1 - d) factor):
        with torch.no_grad():
            # targets from current policy (old policy = buffer)
            _, log_probs, probs = self.actor.action_distr_sample(next_obs)

            # target q values
            q1_policy_targ = self.critic1_targ.forward(next_obs)
            q2_policy_targ = self.critic2_targ.forward(next_obs)
                
            # Clipped double Q trick
            min_q_targ = torch.minimum(q1_policy_targ, q2_policy_targ)

            # Action probabilities can be used to estimate the expectation (cleanRL)
            q_targ =  (probs * (min_q_targ - self.alpha.unsqueeze(1) * log_probs)).sum(dim = 1)
                
            # Bellman approximation
            bellman = rewards + self.gamma * (1 - dones) * q_targ

        # loss is MSEloss over Bellman error (MSBE = mean squared bellman error)
        loss_critic1 = F.mse_loss(q1_buffer, bellman)
        loss_critic2 = F.mse_loss(q2_buffer, bellman)
        loss_critic = loss_critic1 + loss_critic2 

        return loss_critic
    
    def actor_and_alpha_loss(self, obs):
        """
        Returns the actor loss and entropy temperature tuning loss.
        """
        # compute current policy action for pre-transition observation
        _, log_probs, probs = self.actor.action_distr_sample(obs)

        # Q values estimated by critic
        q1_policy = self.critic1.forward(obs)
        q2_policy = self.critic2.forward(obs)
        # take min of these two 
        #   = clipped Q-value for stable learning, reduces overestimation
        q_policy = torch.minimum(q1_policy, q2_policy)
        # entropy regularized loss
        loss_policy = (probs * (self.alpha.unsqueeze(1) * log_probs - q_policy)).sum(1).mean()

        loss_alpha = (probs.detach() * (-self.log_alpha.exp() * (log_probs.detach() + self.entropy_targ))).mean()

        return loss_policy, loss_alpha
    
    def get_action(self, obs):
        # make tensor and send to device
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype = torch.float32).unsqueeze(0).to(self.device)

        # get actor action
        with torch.no_grad():
            action, _, _ = self.actor.action_distr_sample(obs)

        return np.argmax(action.cpu().detach().numpy()[0])