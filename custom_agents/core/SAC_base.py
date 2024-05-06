import sys
import os

from gymnasium import spaces

import torch

import numpy as np

# add folder to python path for relative imports
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

from ..replay_buffers.replay_buffer import ReplayBuffer
from ..replay_buffers.ma_replay_buffer import MultiAgentReplayBuffer
from ..utils.logger import Logger
# from ..SAC_components.autoencoder import AutoEncoder

# base abstract class
from ..core.RL_algo_base import RLbase
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
                 log_dir = "tensorboard_logs",
                #  multi_agent = False,
                 ):
        self.env = env
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        
        # initialize tensorboard logger
        self.logger = Logger(env, log_dir)
        
        # initialize device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # multi-agent check 
        multi_agent = isinstance(env.action_space, np.ndarray) or isinstance(env.action_space, list)

        # discrete or continuous action space check (assuming homogeneous agents)
        if multi_agent:
            discrete_actions = isinstance(env.action_space[0], spaces.Box)
        else:
            discrete_actions = isinstance(env.action_space, spaces.Box)
        
        # initalize replay buffer (NOTE: MA-REPLAY BUFFER CAN BE OPTIMIZER FOR GLOBAL OBS)
        if multi_agent:
            obs_size_list = [obs_space.shape for obs_space in env.observation_space]
            # differentiate between discrete action space or continuous
            if discrete_actions:
                # Multi-discrete not supported
                act_size_list = [(1,) for _ in env.action_space]
            else:
                act_size_list = [act_space.shape for act_space in env.action_space]
            self.replay_buffer = MultiAgentReplayBuffer(buffer_max_size = buffer_max_size, 
                                                        observation_sizes = obs_size_list,
                                                        action_sizes = act_size_list,
                                                        batch_size = batch_size)
            
        # single agent
        else:
            # continuous or discrete actions
            if discrete_actions:
                act_size = (1,)
            else:
                act_size = env.action_space.shape
            self.replay_buffer = ReplayBuffer(buffer_max_size = buffer_max_size, 
                                              observation_size = env.observation_space.shape, 
                                              action_size = act_size,
                                              batch_size = batch_size)
            
        # initialize alpha(s) and optimizers
        if multi_agent:
            self.log_alphas = []
            self.alpha_optimizers = []
            for _ in env.action_space:
                    # add device like this to make tensor not leaf tensor
                self.log_alphas.append(torch.ones(1, requires_grad = True, device = self.device)) 
                self.alpha_optimizers.append(torch.optim.Adam([self.log_alphas[-1]], lr = alpha_lr))
        else:
            self.log_alpha = torch.ones(1, requires_grad = True, device = self.device) 
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr = alpha_lr)   

    """
    Abstract parent class abstract methods.
    """
    @abstractmethod
    def learn(self):
        pass
    
    @abstractmethod
    def get_action(self):
        pass
    
    @abstractmethod
    def train(self):
        pass
    