import numpy as np
import pandas as pd

import math
import sys
import os

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper

from stable_baselines3 import SAC

from custom_reward import CustomReward

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
    
    model.learn(total_timesteps = env.get_metadata()["simulation_time_steps"] * 3500, 
            log_interval = 1)
    model.save("models/custom_reward_SAC_3500")

if __name__ == "__main__":
    main()
    print("Done")
    
