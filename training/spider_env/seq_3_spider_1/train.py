import sys
from os import path

# append path to import from parent folder
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# custom imports
from custom_agents.CTCE_algorithms.ma_sac_agents_seq_discrete import Agents

# custom spider fly environment
from custom_spider_env.spider_fly_env.envs.grid_MA_pettingzoo import SpiderFlyEnvMA
from custom_spider_env.spider_fly_env.wrappers.pettingzoo_wrapper import PettingZooWrapper

def main():
    env = SpiderFlyEnvMA(size = 4, spiders = 3, max_timesteps = 100)

    # pettingzoo conversion + normalization wrappper
    env = PettingZooWrapper(env, normalize = True)
    
    # create model
    sac_agents = Agents(env, batch_size = 256, layer_sizes = (128, 128), global_observations = True) 
    
    # train agent (ep = 720)
    sac_agents.train(nr_steps = 720 * 5500,
                     warmup_steps = 5000,
                     learn_delay = 1000,
                     learn_freq = 1,
                     learn_weight = 1)

if __name__ == "__main__":
    main()
    
