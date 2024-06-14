import sys
from os import path

# append path to import from parent folder
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# custom imports
from custom_agents.CTCE_algorithms.centralized_SAC_continuous import SAC

# custom spider fly environment
from pettingzoo.mpe import simple_spread_v3
from custom_spider_env.spider_fly_env.wrappers.pettingzoo_wrapper import PettingZooWrapper

def main():
    env = simple_spread_v3.parallel_env(continuous_actions = True, N = 3)

    # pettingzoo conversion + normalization wrappper
    env = PettingZooWrapper(env)
    
    # create model
    sac_agents = SAC(env, lr_actor = 0.003, lr_critic = 0.003, batch_size = 256, layer_sizes = (128, 128)) 
    
    # train agent (ep = 720)
    sac_agents.train(nr_eps = 300000000, warmup_steps = 10000)

if __name__ == "__main__":
    main()
