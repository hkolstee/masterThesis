from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

import math
import sys
import os

import gym

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import VecEnv

# from custom_reward import CustomReward
from citylearn.reward_function import RewardFunction, ComfortReward, SolarPenaltyAndComfortReward, SolarPenaltyReward

def main():
    # schema path
    schema_path = "../data/schema.json"

    # create environment
    env = CityLearnEnv(schema=schema_path, reward_function=SolarPenaltyAndComfortReward, central_agent=True, random_seed = 1)
    eval_env = CityLearnEnv(schema=schema_path, reward_function=SolarPenaltyAndComfortReward, central_agent=True, random_seed = 1)

    # wrap environment for use in stablebaselines3
    env = NormalizedObservationWrapper(env)
    env = StableBaselines3Wrapper(env)
    eval_env = NormalizedObservationWrapper(eval_env)
    eval_env = StableBaselines3Wrapper(eval_env)

    # create callback
    eval_callback = EvalCallback(eval_env = eval_env, 
                                 best_model_save_path = "models/",
                                 log_path = "logs/",
                                 eval_freq = 720 * 25)
    
    # create SAC model
    model = SAC("MlpPolicy", env, tensorboard_log="./tensorboard_logs/", device = "cuda", seed = 8, learning_rate = 0.0001)
    # set env
    model.set_env(env)
    # learn 3.950.000 steps
    model.learn(total_timesteps = env.get_metadata()["simulation_time_steps"] * 5500, 
                log_interval = 5,
                callback = eval_callback)
    # save final model
    model.save("models/final_model")

if __name__ == "__main__":
    main()
    print("Done")
    
