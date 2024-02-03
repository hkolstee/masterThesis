import os
import sys

from gym import Wrapper

from citylearn.citylearn import CityLearnEnv

class CityLearnWrapper(Wrapper):
    """
    A wrapper class to correct the output of the CityLearn env to a more 
    updated gymnasium-like env.
    """
    def __init__(self, env: CityLearnEnv):
        super().__init__(env)
        self.env = env

    def reset(self):
        """
        Normal citylearn env returns only observation, causing problems with 
        unpacking.
        """
        observations = self.env.reset()

        return observations[0], {}
    
    def step(self, action):
        """
        Citylearn expects a list of actions, even when using a central agent.
        Also, does not provide truncation information
        """
        next_obs, reward, done, info = self.env.step([action])
        truncated = 0

        return next_obs[0], reward[0], done, truncated, info
    
    @property
    def observation_space(self):
        """
        Citylearn returns a list for action_space, even when using a central 
        agent.
        """
        return self.env.observation_space[0]

    @property
    def action_space(self):
        """
        Citylearn returns a list for observation_space, even when using a 
        central agent.
        """
        return self.env.action_space[0]

    
