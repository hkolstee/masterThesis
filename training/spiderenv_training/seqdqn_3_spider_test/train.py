import sys
from os import path

# append path to import from parent folder
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# custom imports
from custom_agents.CTCE_algorithms.shared_seq_DQN import seqDQN

# custom spider fly environment
from custom_spider_env.spider_fly_env.envs.grid_MA_pettingzoo2 import SpiderFlyEnvMA
from custom_spider_env.spider_fly_env.wrappers.pettingzoo_wrapper import PettingZooWrapper

def main():
    env = SpiderFlyEnvMA(size = 4, spiders = 3, max_timesteps = 100)

    # pettingzoo conversion + normalization wrappper
    env = PettingZooWrapper(env, normalize = True)
    
    # create model
    dqn = seqDQN(env, lr = 1e-3, batch_size = 256, layer_sizes = (128, 128), eps_steps = 1000000, global_observations = True, buffer_max_size = 1000000, tau = 0.001) 
    
    # train agent (ep = 720)
    dqn.train(num_episodes = 300000000)

if __name__ == "__main__":
    main()
    
