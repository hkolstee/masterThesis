import sys
from os import path

# append path to import from parent folder
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# custom imports
from custom_agents.CTCE_algorithms.centralized_DQN import DQN

# custom spider fly environment
from spider_fly_env.envs.grid_MA_pettingzoo import SpiderFlyEnvMA
from custom_spider_env.spider_fly_env.wrappers.pettingzoo_wrapper import PettingZooWrapper

def main():
    env = SpiderFlyEnvMA(size = 4, spiders = 3, max_timesteps = 100)

    # pettingzoo conversion + normalization wrappper
    env = PettingZooWrapper(env, normalize = True)
    
    # create model
    seqdqn = DQN(env, batch_size = 256, layer_sizes = (128, 128), eps_steps = 1000000, global_observations = True, buffer_max_size = 1000000, tau = 0.0025) 
    
    # train agent (ep = 720)
    seqdqn.train(num_episodes = 30000000)

if __name__ == "__main__":
    main()
    
