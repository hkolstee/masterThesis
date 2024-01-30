from sac_agent import Agent
import gymnasium as gym

env = gym.make("Ant-v4")

agent = Agent(env, env.observation_space.shape[0], env.action_space.shape[0])
