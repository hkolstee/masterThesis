import os
import sys

import numpy as np

from torch.utils.tensorboard import SummaryWriter
from citylearn.citylearn import CityLearnEnv

from custom_reward.custom_reward import CustomReward

class Logger():
    """
    Custom tensorboard logging class for tidy logging from within agent class.
    
    args:
        env (CityLearnEnv): The envionment used while training. From here 
            additional logging information is taken.
        log_dir (str): Logging directory
    """
    def __init__(self, env: CityLearnEnv, log_dir: str = "tensorboard_logs"):
        # citylearn env to take logs from
        self.env = env
        # tensorboard summary writer
        self.writer = SummaryWriter(log_dir = log_dir)

    def log(self, logs: dict, step: int, group: str = ""):
        """Log values given in the dictionary.

        Args:
            logs (dict): Dictionary of values to log.
        """
        for item in logs:
            self.writer.add_scalar(group + "/" + item, logs[item], step)


    def log_custom_reward_values(self, step):
        """
        Log the key performance indicator values from the custom reward function.
        As such, can only be used when the env given to the logger is initialized
        with this custom reward function.
        """
        assert isinstance(self.env.unwrapped.reward_function, CustomReward), \
                "The environment is not initialized with the custom reward function"

        # take KPIs from reward function class and log.
        self.writer.add_scalar("Reward_components/Comfort", -self.env.unwrapped.reward_function.comfort[0], step)
        self.writer.add_scalar("Reward_components/Emissions", -self.env.unwrapped.reward_function.emissions[0], step)
        self.writer.add_scalar("Reward_components/Grid", -self.env.unwrapped.reward_function.grid[0], step)
        self.writer.add_scalar("Reward_components/Resilience", -self.env.unwrapped.reward_function.resilience[0], step)
        self.writer.add_scalar("Reward_KPIs/unmet hours of thermal comfort (u)", -self.env.unwrapped.reward_function.u[0], step)
        self.writer.add_scalar("Reward_KPIs/carbon emissions (g)", -self.env.unwrapped.reward_function.g[0], step)
        self.writer.add_scalar("Reward_KPIs/ramping (r)", -self.env.unwrapped.reward_function.r[0], step)
        self.writer.add_scalar("Reward_KPIs/daily peak (d)", -self.env.unwrapped.reward_function.d[0], step)
        self.writer.add_scalar("Reward_KPIs/load factor (l)", -self.env.unwrapped.reward_function.l[0], step)
        self.writer.add_scalar("Reward_KPIs/all-time peak (a)", -self.env.unwrapped.reward_function.a[0], step)
        self.writer.add_scalar("Reward_KPIs/thermal resilience (m)", -self.env.unwrapped.reward_function.m[0], step)
        self.writer.add_scalar("Reward_KPIs/normalized unserved energy (s)", -self.env.unwrapped.reward_function.s[0], step)

