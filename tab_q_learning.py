import os, sys

import numpy as np

import gymnasium as gym

from custom_spider_env.spider_fly_env.envs.grid_MA_pettingzoo import SpiderFlyEnvMA
from custom_spider_env.spider_fly_env.envs.pettingzoo_wrapper import PettingZooWrapper

class TabularQLearning:
    def __init__(self, env, lr = 0.1, gamma = 0.9, eps = 0.1):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.eps = eps

        self.nr_agents = len(env.action_space)

        # Q tables
        self.q_tables = []
        for idx in range(self.nr_agents):
            # Q table dims
            state_space_size = (self.env.observation_space[idx].high + 1) - self.env.observation_space[idx].low
            action_state_size = self.env.action_space[idx].n

            # create
            self.q_tables.append(np.zeros(np.append(state_space_size, action_state_size)))

    def get_action(self, observation, agent_idx, deterministic = False):
        # epsilon non-greedy
        if np.random.uniform(0, 1) < self.eps and not deterministic:
            return self.env.action_space[agent_idx].sample()
        # epsilon greedy
        else:
            return np.argmax(self.q_tables[agent_idx][tuple(observation)])
        
    def update_table(self, obs, action, reward, next_obs, agent_idx):
        # Max Q val of possible actions on next state
        maxQ_next_obs = np.max(self.q_tables[agent_idx][tuple(next_obs)])
        # Q val of action taken on current state
        Q_taken_action = self.q_tables[agent_idx][tuple(obs)][action]

        td_target = reward + self.gamma * maxQ_next_obs
        td_error = td_target - Q_taken_action
        # update
        self.q_tables[agent_idx][tuple(obs)][action] += self.lr * td_error

    def train(self, num_episodes):
        reward_log = []

        for eps in range(num_episodes):
            obs, _ = self.env.reset()
            terminals = [False]
            truncations = [False]
            rew_sum = [0 for _ in range(self.nr_agents)]

            while not (any(terminals) or all(truncations)):
                # get actions
                actions = []
                for agent_idx in range(self.nr_agents):
                    actions.append(self.get_action(obs[agent_idx], agent_idx))
                # take action
                next_obs, rewards, terminals, truncations, _ = self.env.step(actions) 

                # update Q-tables
                for agent_idx in range(self.nr_agents):
                    self.update_table(obs[agent_idx], 
                                      actions[agent_idx], 
                                      rewards[agent_idx], 
                                      next_obs[agent_idx],
                                      agent_idx)
                # update state
                obs = next_obs
                # add to reward sum
                rew_sum = np.add(rew_sum, rewards)

            if eps % (num_episodes // 20) == 0:
                print("Episode: " + str(eps) + " - Reward:" + str(rew_sum))
            reward_log.append(rew_sum)

        return reward_log
