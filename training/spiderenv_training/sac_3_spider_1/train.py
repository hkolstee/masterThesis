import sys
from os import path

# append path to import from parent folder
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# custom imports
from custom_agents.CTCE_algorithms.centralized_SAC import SAC

# custom spider fly environment
from custom_spider_env.spider_fly_env.envs.grid_MA_pettingzoo2 import SpiderFlyEnvMA
from custom_spider_env.spider_fly_env.wrappers.pettingzoo_wrapper import PettingZooWrapper

def main():
    env = SpiderFlyEnvMA(size = 4, spiders = 3, max_timesteps = 100)

    # pettingzoo conversion + normalization wrappper
    env = PettingZooWrapper(env, normalize = True)
    
    # create model
    sac_agents = SAC(env, batch_size = 256, layer_sizes = (128, 128), global_obs = True) 
    
    # train agent (ep = 720)
    sac_agents.train(nr_eps = 300000000, warmup_steps = 10000)

if __name__ == "__main__":
    main()
    
