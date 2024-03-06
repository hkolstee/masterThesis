import os
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as functional

# temporary needed
from custom_agent.CTCE.citylearn_wrapper import CityLearnWrapper
from custom_reward.custom_reward import CustomReward

# add folder to python path for relative imports
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

from ..SAC_components.replay_buffer import ReplayBuffer
from ..SAC_components.critic import Critic
from ..SAC_components.actor import Actor
from ..SAC_components.logger import Logger
from ..SAC_components.autoencoder import AutoEncoder

from copy import deepcopy

from tqdm import tqdm

class Agent:
    """
    An Soft Actor-Critic agent.

    Args:
        env (gym.environment): The environment the agent acts within
        lr_actor (float): Actor network learning rate
        lr_critic (float): Critic network learning rate
        gamma (float): Next reward estimation discount
        polyak (float): Factor of polyak averaging
        buffer_max_size (int): The maximal size of the replay buffer
        batch_size (int): Sample size when sampling from the replay buffer
        layer_sizes (tuple:int): The sizes of the hidden multi-layer-perceptron
            layers within the function estimators (value, actor, critic)
        reward_scaling (int): Scaling of the reward with regard to the entropy
    """
    def __init__(self, 
                 env, 
                 lr_actor = 0.0003, 
                 lr_critic = 0.0003, 
                 gamma = 0.99, 
                 polyak = 0.995,
                 buffer_max_size = 1000000,
                 batch_size = 256,
                 layer_sizes = (256, 256),
                 ):
        self.env = env
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size

        # for now done like this: check if citylearn env with custom reward function for 
        #   additional logging
        self.citylearn = isinstance(self.env.reward_function, CustomReward) if isinstance(self.env, CityLearnWrapper) else False

        # initialize tensorboard logger
        self.logger = Logger(self.env)
        
        # initialize device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_max_size, env.observation_space.shape, env.action_space.shape, batch_size)

        # initialize networks
        self.actor = Actor(lr_actor, env.observation_space.shape[0], env.action_space.shape[0], env.action_space.low, env.action_space.high, layer_sizes)
        self.critic1 = Critic(lr_critic, env.observation_space.shape[0], env.action_space.shape[0], layer_sizes)
        self.critic2 = Critic(lr_critic, env.observation_space.shape[0], env.action_space.shape[0], layer_sizes)

        # make copy target critic networks which only get updated using polyak averaging
        self.critic1_targ = deepcopy(self.critic1)
        self.critic2_targ = deepcopy(self.critic2)
        # freeze parameter gradient calculation as it is not used
        for params in self.critic1_targ.parameters():
            params.requires_grad = False
        for params in self.critic2_targ.parameters():
            params.requires_grad = False

        # target entropy for automatic entropy coefficient adjustment
        self.entropy_targ = torch.tensor(-np.prod(self.env.action_space.shape), dtype=torch.float32).to(self.device)
        # the entropy coef alpha which is to be optimized
        self.log_alpha = torch.ones(1, requires_grad = True, device = self.device)  # adding to device this way makes the tensor not a leaf tensor
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr = lr_critic)   # shares critic lr
        self.alpha = torch.exp(self.log_alpha.detach())

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

        # get next alpha
        self.alpha = torch.exp(self.log_alpha.detach())

        # FIRST GRADIENT: automatic entropy coefficient tuning (alpha)
        #   optimal alpha_t = arg min(alpha_t) E[-alpha_t * (log policy(a_t|s_t; alpha_t) - alpha_t * entropy_target)]
        # we detach because otherwise we backward through the graph of previous calculations using log_prob
        #   which also raises an error fortunately, otherwise I would have missed this
        alpha_loss = -(self.log_alpha * (log_prob_prev_obs + self.entropy_targ).detach()).mean()
        
        # backward prop + gradient step
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()   

        # CRITIC GRADIENT
        # These Q values are the left hand side of the loss function
        q1_buffer = self.critic1.forward(obs, replay_act)
        q2_buffer = self.critic2.forward(obs, replay_act)
        
        # For the RHS of the loss function (Approximation of Bellman equation with (1 - d) factor):
        with torch.no_grad():
            # targets from current policy (old policy = buffer)
            policy_act_next_obs, log_prob_next_obs = self.actor.normal_distr_sample(next_obs)
            # policy_act_next_obs, log_prob_next_obs = self.actor.normal_distr_sample(next_obs)

            # target q values
            q1_policy_targ = self.critic1_targ.forward(next_obs, policy_act_next_obs)
            q2_policy_targ = self.critic2_targ.forward(next_obs, policy_act_next_obs)
            # clipped double Q trick
            q_targ = torch.min(q1_policy_targ, q2_policy_targ)
            # Bellman approximation
            bellman = rewards + self.gamma * (1 - dones) * (q_targ - self.alpha * log_prob_next_obs)

        
        # loss is MSEloss over Bellman error (MSBE = mean squared bellman error)
        # loss_critic1 = torch.pow((q1_buffer - bellman), 2).mean()
        # loss_critic2 = torch.pow((q2_buffer - bellman), 2).mean()
        loss_critic1 = functional.mse_loss(q1_buffer, bellman)
        loss_critic2 = functional.mse_loss(q2_buffer, bellman)
        loss_critic = 0.5 * (loss_critic1 + loss_critic2)

        # backward prop + gradient step
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        loss_critic.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        # ACTOR GRADIENT
        # first freeze critic gradient calculation to save computation
        for params in self.critic1.parameters():
            params.requires_grad = False
        for params in self.critic2.parameters():
            params.requires_grad = False

        # compute Q-values
        q1_policy = self.critic1.forward(obs, policy_act_prev_obs)
        q2_policy = self.critic2.forward(obs, policy_act_prev_obs)
        # take min of these two 
        #   = clipped Q-value for stable learning, reduces overestimation
        q_policy = torch.min(q1_policy, q2_policy)
        # entropy regularized loss
        loss_policy = (self.alpha * log_prob_prev_obs - q_policy).mean()

        # backward prop + gradient step
        self.actor.optimizer.zero_grad()
        loss_policy.backward()
        self.actor.optimizer.step()

        # unfreeze critic gradients
        for params in self.critic1.parameters():
            params.requires_grad = True
        for params in self.critic2.parameters():
            params.requires_grad = True         

        # Polyak averaging update
        with torch.no_grad():
            for (p1, p2, p1_targ, p2_targ) in zip(self.critic1.parameters(),
                                                  self.critic2.parameters(),
                                                  self.critic1_targ.parameters(),
                                                  self.critic2_targ.parameters()):
                # critic1
                p1_targ.data.mul_(self.polyak)
                p1_targ.data.add_((1 - self.polyak) * p1.data)
                # critic2
                p2_targ.data.mul_(self.polyak)
                p2_targ.data.add_((1 - self.polyak) * p2.data)
                # # critic1
                # p1_targ.data *= self.polyak
                # p1_targ.data += ((1 - self.polyak) * p1.data)
                # # critic2
                # p2_targ.data *= self.polyak
                # p2_targ.data += ((1 - self.polyak) * p2.data)
        
        # reutrns policy loss, critic loss, policy entropy, alpha, alpha loss
        return 1, \
               loss_policy.cpu().detach().numpy(), \
               loss_critic.cpu().detach().numpy(), \
               log_prob_prev_obs.cpu().detach().numpy().mean(), \
               self.alpha.cpu().detach().numpy()[0], \
               alpha_loss.cpu().detach().numpy()

    def get_action(self, obs, reparameterize = False, deterministic = False):
        # make tensor and send to device
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype = torch.float32).unsqueeze(0).to(self.device)

        # get actor action
        with torch.no_grad():
            actions, _ = self.actor.normal_distr_sample(obs, reparameterize, deterministic)

        return actions.cpu().detach().numpy()[0]
    
    def train(self, nr_steps, max_episode_len = -1, warmup_steps = 10000, learn_delay = 1000, learn_freq = 50, learn_weight = 50, checkpoint = 100000):
        """Train the SAC agent.

        Args:
            nr_steps (int): The number steps to train the agent
            max_episode_len (int, optional): The max episode length. Defaults to: run environment untill done signal is given.
            warmup_steps (int, optional): Amount of steps the actions are drawn from uniform distribution at the start of training. 
                Defaults to 10000.
            learn_delay (int, optional): Amount of steps before the agent starts learning. Defaults to 1000.
            learn_freq (int, optional): The frequency in steps that the agent undergoes one learning iteration. Defaults to 50.
            learn_weight (int, optional): The amount of gradient descent steps per learning iteration. Defaults to 50.
        """
        # reset env
        obs, info = self.env.reset()

        # episode and epsiode len count
        ep = 0
        ep_steps = 0
        # steps learned per episode count (for avg)
        ep_learn_steps = 0
        # sum of log values for each ep
        ep_rew_sum = 0
        ep_aloss_sum = 0
        ep_closs_sum = 0
        ep_alpha_sum = 0
        ep_alphaloss_sum = 0
        ep_entr_sum = 0
        
        for step in range(nr_steps):
            # sample action (uniform sample for warmup)
            if step < warmup_steps:
                action = self.env.action_space.sample()
            else: 
                # get action
                action = self.get_action(obs)

            # transition
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # step increment 
            ep_steps += 1
            # reward addition to total sum
            ep_rew_sum += reward

            # add transition to buffer
            self.replay_buffer.add_transition(obs, action, reward, next_obs, done)

            # observation update
            obs = next_obs

            # done or max steps
            if (done or truncated or ep_steps == max_episode_len):
                ep += 1

                # avg losses and entropy
                if (ep_learn_steps > 0):
                    avg_actor_loss = ep_aloss_sum / ep_learn_steps
                    avg_critic_loss = ep_closs_sum / ep_learn_steps
                    avg_policy_entr = ep_entr_sum / ep_learn_steps
                    avg_alpha = ep_alpha_sum / ep_learn_steps
                    avg_alpha_loss = ep_alphaloss_sum / ep_learn_steps
                    # save training logs: 
                    logs = {"avg_actor_loss": avg_actor_loss,
                            "avg_critic_loss": avg_critic_loss,
                            "avg_alpha_loss": avg_alpha_loss,
                            "avg_alpha": avg_alpha,
                            "avg_policy_entr": avg_policy_entr}
                    self.logger.log(logs, step, group = "train")
                # log reward seperately
                reward_log = {"reward_sum": ep_rew_sum}
                self.logger.log(reward_log, step, "Reward")

                # NOTE: for now like this for citylearn additional logging, should be in wrapper or something
                if self.citylearn:
                    self.logger.log_custom_reward_values(step)

                # add info to progress bar
                if (ep % 50 == 0):
                    print("[Episode {:d} total reward: {:0.3f}] ~ ".format(ep, ep_rew_sum))
                
                # reset
                obs, info = self.env.reset()
                # reset logging info
                ep_steps = 0
                ep_rew_sum = 0
                ep_aloss_sum = 0
                ep_closs_sum = 0 
                ep_entr_sum = 0
                ep_learn_steps = 0
                ep_alpha_sum = 0
                ep_alphaloss_sum = 0

            # learn
            if step > learn_delay and step % learn_freq == 0:
                for _ in range(learn_weight):
                    # learning step
                    status, loss_actor, loss_critic, policy_entropy, alpha, loss_alpha = self.learn()

                    if status:
                        # keep track for logs
                        ep_learn_steps += 1
                        ep_aloss_sum += loss_actor
                        ep_closs_sum += loss_critic
                        ep_entr_sum += policy_entropy
                        ep_alpha_sum += alpha
                        ep_alphaloss_sum += loss_alpha
                        
            # checkpoint
            if (step % checkpoint == 0):
                self.actor.save("models", "actor" + "_" + str(step))
                self.critic1.save("models", "critic1" + "_" + str(step))
                self.critic2.save("models", "critic2" + "_" + str(step))
                self.critic1_targ.save("models", "critic1_targ" + "_" + str(step))
                self.critic2_targ.save("models", "critic2_targ" + "_" + str(step))
