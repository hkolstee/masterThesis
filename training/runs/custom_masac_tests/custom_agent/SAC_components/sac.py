import torch
import torch.nn as nn
import torch.functional as functional

import numpy as np

class SAC():
    """
    Base class with all gradient functions for Soft Actor-Critic
    """
    def __init__(self, gamma):
        self.gamma = gamma

    def criticLoss(self, actor, critic1, critic2, critic1_targ, critic2_targ, alpha, obs, replay_act, policy_act_next_obs, log_prob_next_obs, next_obs, dones, rewards):
        # These Q values are the left hand side of the loss function
        q1_buffer = critic1.forward(obs, replay_act)
        q2_buffer = critic2.forward(obs, replay_act)
        
        # For the RHS of the loss function (Approximation of Bellman equation with (1 - d) factor):
        with torch.no_grad():
            # target q values
            q1_policy_targ = critic1_targ.forward(next_obs, policy_act_next_obs)
            q2_policy_targ = critic2_targ.forward(next_obs, policy_act_next_obs)
            # clipped double Q trick
            q_targ = torch.min(q1_policy_targ, q2_policy_targ)
            # Bellman approximation
            bellman = rewards + self.gamma * (1 - dones) * (q_targ - alpha.detach() * log_prob_next_obs)
        
        # loss is MSEloss over Bellman error (MSBE = mean squared bellman error)
        loss_critic1 = torch.pow((q1_buffer - bellman), 2).mean()
        loss_critic2 = torch.pow((q2_buffer - bellman), 2).mean()
        loss = 0.5 * (loss_critic1 + loss_critic2)
        
        return loss
    
    def actorLoss(self, critic1, critic2, alpha, obs, policy_act_prev_obs, log_prob_prev_obs):
        # compute Q-values
        q1_policy = critic1.forward(obs, policy_act_prev_obs)
        q2_policy = critic2.forward(obs, policy_act_prev_obs)
        # take min of these two 
        #   = clipped Q-value for stable learning, reduces overestimation
        q_policy = torch.min(q1_policy, q2_policy)
        # entropy regularized loss
        loss = (alpha.detach() * log_prob_prev_obs - q_policy).mean()
        
        return loss

    def alphaLoss(self, alpha, log_prob_prev_obs, entropy_targ):
        # Alpha gradient calculation:
        #   optimal alpha_t = arg min(alpha_t) E[-alpha_t * log policy(a_t|s_t; alpha_t) - alpha_t * entropy_target]
        # we detach because otherwise we backward through the graph of previous calculations using log_prob
        #   which also raises an error fortunately, otherwise I would have missed this
        return (-alpha * log_prob_prev_obs.detach() - alpha * entropy_targ).mean()
