import os
import sys

import numpy as np

from torch.utils.tensorboard import SummaryWriter

# from custom_reward.custom_reward import CustomReward

class Logger():
    """
    Custom tensorboard logging class for tidy logging from within agent class.
    
    args:
        env (CityLearnEnv): The envionment used while training. From here 
            additional logging information is taken.
        log_dir (str): Logging directory
    """
    def __init__(self, env, log_dir):
        # citylearn env to take logs from
        self.env = env
        # tensorboard summary writer
        self.writer = SummaryWriter(log_dir = log_dir)

    def log(self, logs: dict, step: int, group: str = ""):
        """Log values given in the dictionary.

        Args:
            logs (dict): Dictionary of values to log.
        """
        # check for group
        if group:
            group = group + "/" 
        
        # add to logs
        for item in logs:
            # multiple agents
            if isinstance(logs[item], list) or isinstance(logs[item], np.ndarray):
                dict = {}
                for idx, scalar in enumerate(logs[item]):
                    dict["agent" + str(idx)] = scalar
                self.writer.add_scalars(group + item, dict, step)
            # single agent or single scalar
            else:
                self.writer.add_scalar(group + item, logs[item], step)


    # def log_custom_reward_values(self, step):
    #     """
    #     Log the key performance indicator values from the custom reward function.
    #     As such, can only be used when the env given to the logger is initialized
    #     with this custom reward function.
    #     """
    #     assert isinstance(self.env.unwrapped.reward_function, CustomReward), \
    #             "The environment is not initialized with the custom reward function"
        
    #     # check if multi-agent 
    #     # if isinstance(self.env.unwrapped.reward_function.comfort, list) or isinstance(self.env.unwrapped.reward_function.comfort, np.ndarray):
    #     if not self.env.central_agent:
    #         # create dicts
    #         reward_components = {"Comfort": -self.env.unwrapped.reward_function.comfort,
    #                              "Emissions": -self.env.unwrapped.reward_function.emissions,
    #                              "Grid": -self.env.unwrapped.reward_function.grid,
    #                              "Resilience": -self.env.unwrapped.reward_function.resilience}
    #         reward_KPIs = {"unmet_hours_of_thermal_comfort_(u)": -self.env.unwrapped.reward_function.u,
    #                        "carbon_emissions_(g)": -self.env.unwrapped.reward_function.g,
    #                        "ramping_(r)": -self.env.unwrapped.reward_function.r,
    #                        "daily_peak_(d)": -self.env.unwrapped.reward_function.d,
    #                        "load_factor_(l)": -self.env.unwrapped.reward_function.l,
    #                        "all-time_peak_(a)": -self.env.unwrapped.reward_function.a,
    #                        "thermal_resilience_(m)": -self.env.unwrapped.reward_function.m,
    #                        "normalized_unserved_energy_(s)": -self.env.unwrapped.reward_function.s}
    #         # log
    #         self.log(reward_components, step, "Reward_components")
    #         self.log(reward_KPIs, step, "Reward_KPIs")
    #     else:
    #         # create dicts
    #         reward_components = {"Comfort": -self.env.unwrapped.reward_function.comfort[0],
    #                              "Emissions": -self.env.unwrapped.reward_function.emissions[0],
    #                              "Grid": -self.env.unwrapped.reward_function.grid[0],
    #                              "Resilience": -self.env.unwrapped.reward_function.resilience[0]}
    #         reward_KPIs = {"unmet_hours_of_thermal_comfort_(u)": -self.env.unwrapped.reward_function.u[0],
    #                        "carbon_emissions_(g)": -self.env.unwrapped.reward_function.g[0],
    #                        "ramping_(r)": -self.env.unwrapped.reward_function.r[0],
    #                        "daily_peak_(d)": -self.env.unwrapped.reward_function.d[0],
    #                        "load_factor_(l)": -self.env.unwrapped.reward_function.l[0],
    #                        "all-time_peak_(a)": -self.env.unwrapped.reward_function.a[0],
    #                        "thermal_resilience_(m)": -self.env.unwrapped.reward_function.m[0],
    #                        "normalized_unserved_energy_(s)": -self.env.unwrapped.reward_function.s[0]}
    #         # log
    #         self.log(reward_components, step, "Reward_components")
    #         self.log(reward_KPIs, step, "Reward_KPIs")