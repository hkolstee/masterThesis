import sys
import os

from gymnasium import spaces

import torch
import torch.nn.functional as F

import numpy as np

# add folder to python path for relative imports
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

from ..networks.actor import Actor
from ..networks.actor_discrete import DiscreteActor
from ..networks.critic import Critic
from ..networks.autoencoder import AutoEncoder
from ..replay_buffers.replay_buffer import ReplayBuffer
from ..replay_buffers.ma_replay_buffer import MultiAgentReplayBuffer
from ..utils.logger import Logger
# from ..SAC_components.autoencoder import AutoEncoder

# base abstract class
from .RL_algo_base import RLbase
from abc import abstractmethod

class SoftActorCriticCore(RLbase):
    """
    Abstract class.
        
    Serves as the base core class for all SAC implementations. Includes some
    shared methods and shared attributes among all SAC implementations. 
    """
    def __init__(self,
                 env,
                 gamma = 0.99,
                 alpha_lr = 0.0003,
                 polyak = 0.995,
                 buffer_max_size = 1000000,
                 batch_size = 256,
                 use_AE = False,
                 AE_reduc_scale = 6,
                 log_dir = "tensorboard_logs",
                 global_obs = False
                 ):
        self.env = env
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        # use of autoencoder
        self.use_AE = use_AE
        # if env provides global obs to each agent or not
        self.global_obs = global_obs
        
        # initialize tensorboard logger
        self.logger = Logger(env, log_dir)
        
        # initialize device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # discrete or continuous action space check (assuming homogeneous agents)
        self.discrete_actions = isinstance(env.action_space, spaces.Box)

        # use an autoencoder to transform input into latent space
        if self.use_AE:
            self.autoencoder = AutoEncoder(env.observation_space.shape[0], AE_reduc_scale)
        
        # continuous or discrete actions
        if self.discrete_actions:
            act_size = (1,)
        else:
            act_size = env.action_space.shape
        self.replay_buffer = ReplayBuffer(max_size = buffer_max_size, 
                                          observation_size = env.observation_space.shape,
                                          action_size = act_size,
                                          batch_size = batch_size)
            
        # initialize alpha(s) and optimizers
        self.log_alpha = torch.ones(1, requires_grad = True, device = self.device) 
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr = alpha_lr)   

    """
    Abstract properties for attributes the algorithm should have.
    """
    @property
    @abstractmethod
    def actor(self):
        pass

    @property
    @abstractmethod
    def critic1(self):
        pass

    @property
    @abstractmethod
    def critic2(self):
        pass

    @property
    @abstractmethod
    def critic1_targ(self):
        pass

    @property
    @abstractmethod
    def critic2_targ(self):
        pass
    
    @property
    @abstractmethod
    def entropy_targ(self):
        pass

    """
    SAC required methods.
    """
    @abstractmethod
    def actor_and_alpha_loss(self):
        pass

    @abstractmethod
    def critic_loss(self):
        pass

    """
    Abstract parent class abstract methods.
    """
    @abstractmethod
    def get_action(self):
        pass
    
    """
    Universal methods
    """
    def save_networks_parameters(self, save_dir, step):
        self.actor.save(save_dir, "actor" + "_" + str(step))
        self.critic1.save(save_dir, "critic1" + "_" + str(step))
        self.critic2.save(save_dir, "critic2" + "_" + str(step))
        self.critic1_targ.save(save_dir, "critic1_targ" + "_" + str(step))
        self.critic2_targ.save(save_dir, "critic2_targ" + "_" + str(step))

    def sample_batch(self):
        # sample from buffer
        obs, replay_act, rewards, next_obs, dones = self.replay_buffer.sample()

        # prepare tensors
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        replay_act = torch.tensor(replay_act, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.int32).to(self.device)

        # latent representations if autoencoder is used
        if self.use_AE:
            obs, _, _ = self.autoencoder(obs)
            next_obs, _, _ = self.autoencoder(next_obs)

        return obs, next_obs, replay_act, rewards, dones

    def learn(self):
        """
        One step of learning, meaning one step of gradient descend over critic and actor.
        """
        # buffer not full enough
        if self.replay_buffer.buffer_index < self.batch_size:
            # return status 0
            return 0, None, None, None, None

        obs, next_obs, replay_act, rewards, dones = self.sample_batch()

        # get new alpha
        self.alpha = torch.exp(self.log_alpha.detach())
        
        # critic loss
        loss_critic = self.critic_loss(obs, next_obs, replay_act, rewards, dones)

        # backward prop + gradient step
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        loss_critic.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()
        
        # actor loss
        # first freeze critic gradient calculation to save computation
        self.freeze_network_grads(self.critic1)        
        self.freeze_network_grads(self.critic2)

        loss_policy, loss_alpha = self.actor_and_alpha_loss(obs)

        # backward prop + gradient step
        self.actor.optimizer.zero_grad()
        loss_policy.backward()
        self.actor.optimizer.step()

        # unfreeze critic gradient calculation to save computation
        self.unfreeze_network_grads(self.critic1)        
        self.unfreeze_network_grads(self.critic2)

        # backward prop + gradient step
        self.alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self.alpha_optimizer.step()   

        # polyak update of target networks
        self.polyak_update(self.critic1, self.critic1_targ, self.polyak)
        self.polyak_update(self.critic2, self.critic2_targ, self.polyak)

        return 1, \
               loss_policy.cpu().detach().numpy(), \
               loss_critic.cpu().detach().numpy(), \
               self.alpha.cpu().detach().numpy()[0], \
               loss_alpha.cpu().detach().numpy()
    
    def train(self, 
              nr_steps, 
              max_episode_len = -1, 
              warmup_steps = 10000, 
              learn_delay = 100, 
              learn_freq = 50, 
              learn_weight = 50, 
              checkpoint = 100000, 
              save_dir = "models"):
        """
        One training run.
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
        ep_aeloss_sum = 0
        
        for step in range(nr_steps):
            # sample action (uniform sample for warmup)
            if step < warmup_steps:
                action = self.env.action_space.sample()
            else: 
                # autoencoder
                if self.use_AE:
                    with torch.no_grad():
                        latent_obs, decoded_obs, ae_loss = self.autoencoder(torch.tensor(obs, dtype = torch.float32).unsqueeze(0))
                        ep_aeloss_sum += ae_loss
                    # get action
                    action = self.get_action(latent_obs)
                    # action = self.get_action(latent_obs.unsqueeze(0))
                else:
                    action = self.get_action(obs)

            # transition
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # step increment 
            ep_steps += 1
            # reward addition to total sum
            ep_rew_sum += reward

            # set done to false if signal is because of time horizon (spinning up)
            if ep_steps == max_episode_len:
                done = False

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
                    avg_alpha = ep_alpha_sum / ep_learn_steps
                    avg_alpha_loss = ep_alphaloss_sum / ep_learn_steps
                    avg_ae_loss = ep_aeloss_sum / ep_steps
                    # save training logs: 
                    logs = {"avg_actor_loss": avg_actor_loss,
                            "avg_critic_loss": avg_critic_loss,
                            "avg_alpha_loss": avg_alpha_loss,
                            "avg_ae_loss": avg_ae_loss,
                            "avg_alpha": avg_alpha}
                    self.logger.log(logs, step, group = "train")
                # log reward seperately
                reward_log = {"reward_sum": ep_rew_sum}
                self.logger.log(reward_log, step, "reward")

                # NOTE: for now like this for citylearn additional logging, should be in wrapper or something
                # if self.citylearn:
                #     self.logger.log_custom_reward_values(step)

                # add info to progress bar
                if (ep % 50 == 0):
                    print("[Episode {:d} total reward: {:0.3f}] ~ ".format(ep, ep_rew_sum))
                    # print(obs)
                    # print(latent_obs)
                    # print(decoded_obs)
                
                # reset
                obs, info = self.env.reset()
                # reset logging info
                ep_steps = 0
                ep_learn_steps = 0

                ep_rew_sum = 0
                ep_aloss_sum = 0
                ep_closs_sum = 0
                ep_alpha_sum = 0
                ep_alphaloss_sum = 0
                ep_aeloss_sum = 0

            # learn
            if step > learn_delay and step % learn_freq == 0:
                for _ in range(learn_weight):
                    # learning 
                    status, loss_actor, loss_critic, alpha, loss_alpha = self.learn()

                    if status:
                        # keep track for logs
                        ep_learn_steps += 1
                        ep_aloss_sum += loss_actor
                        ep_closs_sum += loss_critic
                        ep_alpha_sum += alpha
                        ep_alphaloss_sum += loss_alpha
                        
            # checkpoint
            if (step % checkpoint == 0 and step != 0):
                self.save_networks_parameters(save_dir, step)