import os
import sys

from gym import Wrapper
from gymnasium.spaces import Box

import numpy as np

# add folder to python path for relative imports
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

from ..envs.grid_MA_pettingzoo import SpiderFlyEnvMA

class NormalizedObservationSpace(Box):
    def __init__(self, obs_space, mean, std):
        self.obs_space = obs_space
        self.mean = mean
        self.std = std
        super().__init__(low = (obs_space.low - mean) / std, 
                         high = (obs_space.high - mean) / std, 
                         dtype = obs_space.dtype)

    def sample(self):
        sample = self.obs_space.sample()
        return (sample - self.mean) / (self.std + 1e-8)

class NormalizeObsWrapper(Wrapper):
    """
    A wrapper class to correct the output of the pettingzoo env to a gymnasium-like env
    using lists instead of dictionaries.
    """
    def __init__(self, env: SpiderFlyEnvMA):
        self.env = env
        # initialize super
        super().__init__(self.env)
        self.mean, self.std = self.calculate_mean_std()
        
        self.temp_observation_space = [NormalizedObservationSpace(obs_space) for obs_space in self.observation_space]
        self.observation_space = self.temp_observation_space
        
    @property
    def observation_space(self):
        return [self.env.observation_space[idx] for idx in len(self.env.possible_agents)]

    def calculate_mean_std(self):        
        observations = []
        
        # reset
        obs, _ = self.env.reset()
        obs = obs[0]
        observations.append(obs)
        # we take 10000 samples
        for _ in range(10000):
            actions = []
            for act_space in self.env.action_space:
                actions.append(act_space.sample())
    
            # take random action
            next_obs, _, done, trunc, _ = self.env.step(actions)
            
            # transition states
            obs = next_obs[0]
            
            # add to list
            observations.append(obs)
            
            if done[0] or trunc[0]:
                obs, _ = self.env.reset()
                obs = obs[0]
         
        observations = np.array(observations)
        
        return np.mean(observations, axis = 0), np.std(observations, axis = 0)
    
    def normalize(self, obs):
        return (obs - self.mean) / (self.std + 1e-8)
    
    def reset(self):
        obs = self.env.reset()
        return self.normalize(obs)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        return self.normalize(obs), reward, done, trunc, info

            
        
