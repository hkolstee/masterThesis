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

from ..networks.actor import Actor
from ..networks.critic import Critic
from ..core.SAC_base import SoftActorCriticCore

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
                 log_dir="tensorboard_logs", 
                 ):
        """
        Super class initializes all but the actor and critic networks.
        """        
        super().__init__(env, 
                         gamma, 
                         polyak, 
                         buffer_max_size, 
                         batch_size, 
                         layer_sizes, 
                         log_dir, 
                         multi_agent = False)
        
        # we check for continuous / discrete action space
        if isinstance(self.env.action_space, spaces.Box):
            self.discrete = True
        else:
            self.discrete = False

        # initialize networks 
        self.actor = Actor(lr_actor,
                           self.env.observation_space.shape[0],
                           self.env.action_space.shape[0],
                           self.env.action_space.low,
                           self.env.action_space.high,
                           layer_sizes)
        # double clipped Q learning
        self.critic1 = Critic(lr = lr_critic, 
                              obs_size = self.env.observation_space.shape[0],
                              act_size = self.env.action_space.shape[0], 
                              discrete = self.discrete,
                              layer_sizes = layer_sizes)
        self.critic2 = Critic(lr = lr_critic, 
                              obs_size = self.env.observation_space.shape[0],
                              act_size = self.env.action_space.shape[0], 
                              discrete = self.discrete,
                              layer_sizes = layer_sizes)
        
        # target networks
        self.critic1_targ = deepcopy(self.critic1)
        self.critic2_targ = deepcopy(self.critic2)
        # freeze parameter gradient calculation as it is not used
        self.freeze_gradients(self.critic1_targ)
        self.freeze_gradients(self.critic2_targ)
            
        # coef to be optimized
        self.log_alpha = torch.ones(1, requires_grad = True, device = self.device)  # device like this otherwise leaf tensor
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr = lr_critic)   # shares critic lr
        
        
    def learn(self):
        """Learn the policy by backpropagation over the critics, and actor network.

        next actions come from the policy, previous actions and states from buffer
        Q values are the previous states and policy actions 
            Q^pi(a,s) = r + gamma * (Q^pi(a',s') - alpha * log pi(a', s')), where a' ~ pi(*|'s)
        with a Q network mean squared bellman error loss function of
            L(params, D) = E[(Q_params (s, a) - y(r, s', d))^2]
        where y target for Q is:
            y(r, s', done) = r + gamma * (1-done)(min(Q_1(s',a'), Q_2(s',a')) - alpha * log pi(a', s'))
        
        To get the policy loss we optimize:
            max(params) E[min(Q_1(s',a'), Q_2(s',a')) - alpha * log * pi(a^reparam(s, noise)|s)]
        which means we optimize for the loss:
            loss = (Q(a,s) - (r + gamma * (1-done)(min(Q_1(s',a'), Q_2(s',a')) - alpha * log pi(a', s')))

        The entropy coefficient alpha is automatically adjusted towards the optimal value by solving for:
            alpha*_t = arg min(alpha_t) E[-alpha_t * log policy(a_t|s_t; alpha_t) - alpha_t * entropy_target]
        """ 
        # buffer not full enough
        if self.replay_buffer.buffer_index < self.batch_size:
            # return status 0
            return 0, None, None, None, None, None
        
        # sample from buffer
        obs, replay_act, rewards, next_obs, dones = self.replay_buffer.sample()

        # prepare tensors
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        replay_act = torch.tensor(replay_act, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.int32).to(self.device)

        # compute current policy action for pre-transition observation
        policy_act_prev_obs, log_prob_prev_obs = self.actor.normal_distr_sample(obs)

        # FIRST GRADIENT: automatic entropy coefficient tuning (alpha)
        #   optimal alpha_t = arg min(alpha_t) E[-alpha_t * (log policy(a_t|s_t; alpha_t) - alpha_t * entropy_target)]
        # we detach because otherwise we backward through the graph of previous calculations using log_prob
        #   which also raises an error fortunately, otherwise I would have missed this
        # alpha_loss = (-self.log_alpha * log_prob_prev_obs.detach() - self.log_alpha * self.entropy_targ).mean()
        alpha_loss = (-self.log_alpha.exp() * (log_prob_prev_obs.detach() + self.entropy_targ)).mean()
        
        # backward prop + gradient step
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()   

        # get next alpha
        self.alpha = torch.exp(self.log_alpha.detach())

        # CRITIC GRADIENT
        # These Q values are the left hand side of the loss function
        q1_buffer = self.critic1.forward(obs, replay_act)
        q2_buffer = self.critic2.forward(obs, replay_act)
        
        # For the RHS of the loss function (Approximation of Bellman equation with (1 - d) factor):
        with torch.no_grad():
            # targets from current policy (old policy = buffer)
            policy_act_next_obs, log_prob_next_obs = self.actor.normal_distr_sample(next_obs)

            # target q values
            q1_policy_targ = self.critic1_targ.forward(next_obs, policy_act_next_obs)
            q2_policy_targ = self.critic2_targ.forward(next_obs, policy_act_next_obs)
            # clipped double Q trick
            q_targ = torch.minimum(q1_policy_targ, q2_policy_targ)
            # Bellman approximation
            bellman = rewards + self.gamma * (1 - dones) * (q_targ - self.alpha * log_prob_next_obs)

        # loss is MSEloss over Bellman error (MSBE = mean squared bellman error)
        loss_critic1 = F.mse_loss(q1_buffer, bellman)
        loss_critic2 = F.mse_loss(q2_buffer, bellman)
        loss_critic = loss_critic1 + loss_critic2 # factor of 0.5 also used

        # backward prop + gradient step
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        loss_critic.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        # ACTOR GRADIENT
        # first freeze critic gradient calculation to save computation
        self.freeze_gradients(self.critic1)        
        self.freeze_gradients(self.critic2)        

        # compute Q-values
        q1_policy = self.critic1.forward(obs, policy_act_prev_obs)
        q2_policy = self.critic2.forward(obs, policy_act_prev_obs)
        # take min of these two 
        #   = clipped Q-value for stable learning, reduces overestimation
        q_policy = torch.minimum(q1_policy, q2_policy)
        # entropy regularized loss
        loss_policy = (self.alpha * log_prob_prev_obs - q_policy).mean()

        # backward prop + gradient step
        self.actor.optimizer.zero_grad()
        loss_policy.backward()
        self.actor.optimizer.step()

        # unfreeze critic gradients
        self.unfreeze_gradients(self.critic1)        
        self.unfreeze_gradients(self.critic2)        

        # Polyak averaging update
        self.polyak_update(self.critic1, self.critic1_targ)
        self.polyak_update(self.critic2, self.critic2_targ)
        
        # reutrns policy loss, critic loss, policy entropy, alpha, alpha loss
        return 1, \
               loss_policy.cpu().detach().numpy(), \
               loss_critic.cpu().detach().numpy(), \
               log_prob_prev_obs.cpu().detach().numpy().mean(), \
               self.alpha.cpu().detach().numpy()[0], \
               alpha_loss.cpu().detach().numpy()
               
    def get_action(self, obs, reparameterize = True, deterministic = False):
        # make tensor and send to device
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype = torch.float32).unsqueeze(0).to(self.device)

        # get actor action
        with torch.no_grad():
            actions, _ = self.actor.normal_distr_sample(obs, reparameterize, deterministic)

        return actions.cpu().detach().numpy()[0]
    
    def get_action(self, obs, reparameterize = True, deterministic = False):
        # make tensor and send to device
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype = torch.float32).unsqueeze(0).to(self.device)

        # get actor action
        with torch.no_grad():
            action, _, _ = self.actor.action_distr_sample(obs, reparameterize, deterministic)
    
        return np.argmax(action.cpu().detach().numpy()[0])
    