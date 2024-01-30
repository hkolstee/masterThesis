import numpy as np
import pandas as pd

import math
import sys
import os

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper
from citylearn.reward_function import RewardFunction, ComfortReward, SolarPenaltyAndComfortReward, SolarPenaltyReward 

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback


from custom_reward import CustomReward

class CustomCallback(BaseCallback):
    """
    Custom callback for plotting additional reward values in tensorboard
    """
    def __init__(self, verbose = 0):
        super().__init__(verbose)

    def _on_rollout_end(self) -> None:
        self.logger.record("Comfort", -self.training_env.get_attr("reward_function")[0].comfort[0])
        self.logger.record("Emissions", -self.training_env.get_attr("reward_function")[0].emissions[0])
        self.logger.record("Grid", -self.training_env.get_attr("reward_function")[0].grid[0])
        self.logger.record("Resilience", -self.training_env.get_attr("reward_function")[0].resilience[0])
        self.logger.record("unmet hours of thermal comfort (u)", -self.training_env.get_attr("reward_function")[0].u[0])
        self.logger.record("carbon emissions (g)", -self.training_env.get_attr("reward_function")[0].g[0])
        self.logger.record("ramping (r)", -self.training_env.get_attr("reward_function")[0].r[0])
        self.logger.record("daily peak (d)", -self.training_env.get_attr("reward_function")[0].d[0])
        self.logger.record("load factor (l)", -self.training_env.get_attr("reward_function")[0].l[0])
        self.logger.record("all-time peak (a)", -self.training_env.get_attr("reward_function")[0].a[0])
        self.logger.record("thermal resilience (m)", -self.training_env.get_attr("reward_function")[0].m[0])
        self.logger.record("normalized unserved energy (s)", -self.training_env.get_attr("reward_function")[0].s[0])

    def _on_step(self) -> bool:
        return True

def main():
    # schema path
    schema_path = os.path.join("./data/", "schema_edited.json")

    # create environment
    env = CityLearnEnv(schema=schema_path, reward_function=CustomReward, central_agent=True)

    # wrap environment for use in stablebaselines3
    env = NormalizedObservationWrapper(env)
    env = StableBaselines3Wrapper(env)
    
    # create SAC model
    model = SAC("MlpPolicy", env, tensorboard_log="./tensorboard_logs/", device = "cuda")
    
    # load model parameters
    # model.load("custom_reward_SAC1.zip")
    model.set_env(env)
    
    model.learn(total_timesteps = env.get_metadata()["simulation_time_steps"] * 5500, 
                log_interval = 1,
                callback = CustomCallback())
    model.save("models/custom_reward_SAC")

if __name__ == "__main__":
    main()
    print("Done")
    
