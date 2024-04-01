import numpy as np
import pygame
import copy

import gymnasium as gym
from gymnasium import spaces

class GridEnv(gym.Env):
    """
    This environment is a 2-dimensional grid, modelling the spider-and-fly 
    problem illustrated in the paper:
    
    "Multiagent Reinforcement Learning: Rollout and Policy Iteration", 
    Dimitri Bertsekas, Feb 2021.
    
    The problem involbes a grid space with a set number of spiders and one fly.
    The spiders move with perffect knowledge about the location of the other
    spiders and the fly. The actions the spiders can perform is to stay in its
    current location or move to one neighbouring location (not diagonal). The 
    fly moves randomly, without regard of spider location. The spider is 
    caught when it can not move becuase it is enclosed by 4 spiders, one on
    either side. The goal is to catch the fly at minimal costs, where each 
    transition to the next state will cost 1, until the fly is caught, then the
    cost becomes 0. 
    """
    def __init__(self, size = 6, spiders = 3):
        super().__init__()
        self.size = size
        self.nr_spiders = spiders

        # grid locations are integers [x,y], x,y in [0,..,size - 1].
        spider_observations = spaces.Dict({
            "fly": spaces.Box(0, self.size - 1, (2,), dtype = np.int32),
        })
        for idx in range(self.nr_spiders):
            spider_observations["spider_" + str(idx)] = spaces.Box(0, self.size - 1, (2,), dtype = np.int32)

        # multi-agent observations, agent (spider) gets all spider locs + fly loc
        observations = []
        for _ in range(self.nr_spiders):
            observations.append(copy.deepcopy(spider_observations))
                
        
        
GridEnv()