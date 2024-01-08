import numpy as np
import pandas as pd

import math
import sys
import os

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from custom_reward import CustomReward

class CustomCallback(BaseCallback):
    """
    Custom callback for plotting additional reward values in tensorboard
    """
    def __init__(self, verbose = 0):
        super().__init__(verbose)
        self.reset()
        
    def reset(self):
        self.comfort = 0.
        self.emissions = 0.
        self.grid = 0.
        self.resilience = 0.
        self.u = 0.
        self.g = 0.
        self.r = 0.
        self.d = 0.
        self.l = 0.
        self.a = 0.
        self.m = 0.
        self.s = 0.

    def _on_rollout_end(self) -> None:
        self.logger.record("comfort", -self.comfort)
        self.logger.record("emissions", -self.emissions)
        self.logger.record("grid", -self.grid)
        self.logger.record("resilience", -self.resilience)
        self.logger.record("u", -self.u)
        self.logger.record("g", -self.g)
        self.logger.record("r", -self.r)
        self.logger.record("d", -self.d)
        self.logger.record("l", -self.l)
        self.logger.record("a", -self.a)
        self.logger.record("m", -self.m)
        self.logger.record("s", -self.s)
        self.reset()

    def _on_step(self) -> bool:
        # print(self.training_env.get_attr("reward_function")[0])
        self.comfort += self.training_env.get_attr("reward_function")[0].comfort[0]
        self.emissions += self.training_env.get_attr("reward_function")[0].emissions[0]
        self.grid += self.training_env.get_attr("reward_function")[0].grid[0]
        self.resilience += self.training_env.get_attr("reward_function")[0].resilience[0]
        self.u += self.training_env.get_attr("reward_function")[0].u[0]
        self.g += self.training_env.get_attr("reward_function")[0].g[0]
        self.r += self.training_env.get_attr("reward_function")[0].r[0]
        self.d += self.training_env.get_attr("reward_function")[0].d[0]
        self.l += self.training_env.get_attr("reward_function")[0].l[0]
        self.a += self.training_env.get_attr("reward_function")[0].a[0]
        self.m += self.training_env.get_attr("reward_function")[0].m[0]
        self.s += self.training_env.get_attr("reward_function")[0].s[0]

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
    model = PPO("MlpPolicy", env, tensorboard_log="./tensorboard_logs/", device = "cuda")
    
    # load model parameters
    # model.load("custom_reward_SAC1.zip")
    model.set_env(env)
    
    model.learn(total_timesteps = env.get_metadata()["simulation_time_steps"] * 3500, 
                log_interval = 1,
                callback = CustomCallback())
    model.save("models/custom_reward_SAC_test")

if __name__ == "__main__":
    main()
    print("Done")
    