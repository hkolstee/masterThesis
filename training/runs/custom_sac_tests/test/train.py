import sys
from os import path

# append path to import from parent folder
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# custom imports
from custom_agent.CTCE.sac_agent import Agent
from custom_agent.CTCE.citylearn_wrapper import CityLearnWrapper
from custom_reward.custom_reward import CustomReward

# citylearn environment
from citylearn.citylearn import CityLearnEnv
# citylearn normalization wrapper
from citylearn.wrappers import NormalizedSpaceWrapper

def main():
    # environment schema 
    schema_path = "../data/schema.json"
    
    # create env
    env = CityLearnEnv(schema = schema_path, 
                       reward_function = CustomReward, 
                       central_agent = True, 
                       random_seed = 1)
    # normalization wrappper
    env = NormalizedSpaceWrapper(env)
    # custom citylearn wrapper for universal input output
    env = CityLearnWrapper(env)
    
    # create model
    sac_agent = Agent(env)
    
    # train agent (ep = 720)
    sac_agent.train(nr_steps = 720 * 5500,
                    warmup_steps = 100,
                    learn_delay = 100,
                    learn_freq = 1,
                    learn_weight = 1)

if __name__ == "__main__":
    main()
    
