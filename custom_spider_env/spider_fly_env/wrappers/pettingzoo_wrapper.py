import os
import sys

from gym import Wrapper

import numpy as np

# companion package of pettingzoo for wrappers
from supersuit import normalize_obs_v0, dtype_v0

# add folder to python path for relative imports
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

class PettingZooWrapper(Wrapper):
    """
    A wrapper class to correct the output of the pettingzoo env to a gymnasium-like env
    using lists instead of dictionaries.
    """
    def __init__(self, env, normalize = False):
        if normalize:
            # normalization wrapper
            self.env = normalize_obs_v0(dtype_v0(env, np.float32))
        else:
            self.env = env
        # initialize super
        super().__init__(self.env)

    def reset(self, seed = None, options = None):
        """
        Pettingzoo returns dictionaries, we discard the keys and keep the values.
        Assumption: indexing dictionary is order consistent.
        """
        observations, _ = self.env.reset(seed, options)

        return list(observations.values()), {}

    
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
        return [self.env.observation_space(agent) for agent in self.env.possible_agents]

    @property
    def action_space(self):
        """
        Pettingzoo returns a dict for action_space, 
        we convert to list of action spaces.
        """
        return [self.env.action_space(agent) for agent in self.env.possible_agents]
