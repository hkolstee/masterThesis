import os
import sys

from gym import Wrapper

import numpy as np

from supersuit import normalize_obs_v0, dtype_v0

# add folder to python path for relative imports
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

from .grid_MA_pettingzoo import SpiderFlyEnvMA

class PettingZooWrapper(Wrapper):
    """
    A wrapper class to correct the output of the pettingzoo env to a gymnasium-like env
    using lists instead of dictionaries.
    """
    def __init__(self, env: SpiderFlyEnvMA, normalize = False):
        if normalize:
            self.env = normalize_obs_v0(dtype_v0(env, np.float32))
        else:
            self.env = env
        super().__init__(self.env)
        self.env = env

    def reset(self):
        """
        Pettingzoo returns dictionaries, we discard the keys and keep the values.
        Assumption: indexing dictionary is order consistent.
        """
        observations, _ = self.env.reset()

        return list(observations.values()), {}
        # return np.array(observations.values()), {}

    
    def step(self, actions):
        """
        Pettingzoo expects a dict of actions.
        Also returns dictionaries.

        We convert actions from list to dict, return values for dicts to lists.

        Actions: list()
        """
        new_actions = {}
        # agent keys
        for (agent, action) in zip(self.env.possible_agents, actions):
            new_actions[agent] = action

        # check 
        assert len(new_actions) > 0

        next_obs, rewards, dones, truncations, infos = self.env.step(new_actions)

        return list(next_obs.values()), \
               list(rewards.values()), \
               list(dones.values()), \
               list(truncations.values()), \
               list(infos.values())

    @property
    def observation_space(self):
        """
        Pettingzoo returns a dict for observation_space, 
        we convert to list of action spaces.
        """
        return list(self.env.observation_spaces.values())

    @property
    def action_space(self):
        """
        Pettingzoo returns a dict for action_space, 
        we convert to list of action spaces.
        """
        return list(self.env.action_spaces.values())

    
