import numpy as np 
import pandas as pd

import math
import sys
import os

import gymnasium as gym
# gym.__version__

import torch
from torch import nn
import torch.optim as optim

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper

# change working directory when running this file for testing
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append("../custom_reward")
from custom_reward import CustomReward

from citylearn.agents.base import Agent

class SAC_agent(Agent):
    def __init__(self, env):
        super().__init__(env)

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        self.reset()
        return self.predict(observations)

    def predict(self, observations):
        """ Just a passthrough, can implement any custom logic as needed """
        return super().predict(observations) 

class SAC():
    """
    Soft actor-critic model.

    Parameters
    ----------

    """