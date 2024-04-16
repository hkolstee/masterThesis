import os, sys

import numpy as np

import gymnasium as gym

from custom_spider_env.spider_fly_env.envs.grid_MA_pettingzoo import SpiderFlyEnvMA
from custom_spider_env.spider_fly_env.envs.pettingzoo_wrapper import PettingZooWrapper

class SequentialTabularQLearning:
    def __init__(self, env, lr = 0.1, gamma = 0.9, eps = 0.1):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.eps = eps

        self.nr_agents = len(env.action_space)

        # Q tables
        self.q_tables = []
        action_state_sizes =[]
        for idx in range(self.nr_agents):
            # Q table dims
            state_space_size = (self.env.observation_space[idx].high + 1) - self.env.observation_space[idx].low
            # sequentially, actions of previous agent is added as input into next agent in sequence
            action_state_sizes.append(self.env.action_space[idx].n)

            # Create
            self.q_tables.append(np.zeros(np.append(state_space_size, action_state_sizes)))

    def get_action(self, observation, agent_idx, deterministic = False):
        # epsilon non-greedy
        if np.random.uniform(0, 1) < self.eps and not deterministic:
            return self.env.action_space[agent_idx].sample()
        # epsilon greedy
        else:
            return np.argmax(self.q_tables[agent_idx][tuple(observation)])
        
    def update_table(self, obs, actions, reward, next_obs, agent_idx):
        # For all agents, except the last we take the difference with the Q-value of the next agent
        # in sequence, instead of the Q-value of the next state and action. We learn from the error
        # between agents judgement of the state plus previous agent decisions. The last agent learns
        # from the TD-error between its own Q-value and the Q-value of the first agent in the 
        # sequence on the next state and best action.  
        # print(obs, action)
        # seq_obs = obs + action[: agent_idx + 1]
        # print(seq_obs)
        # print(tuple(obs))
        # sys.exit()
        if agent_idx < (self.nr_agents - 1):
            # Max Q val of possible actions of next agent in sequence given action of current agent in sequence
            maxQ_next_agent = np.max(self.q_tables[agent_idx + 1][tuple(obs)][tuple(actions[:agent_idx + 1])])
            # Q val of action taken of current agent in sequence
            Q_taken_action = self.q_tables[agent_idx][tuple(obs)][tuple(actions[:agent_idx + 1])]

            # target is just Q-val of next agent (no reward as comparison is not with next state, nor discount)
            td_target = reward + self.gamma * maxQ_next_agent
            td_error = td_target - Q_taken_action
            # update
            self.q_tables[agent_idx][tuple(obs)][tuple(actions[:agent_idx + 1])] += self.lr * td_error
        else:
            # Max Q val of possible actions on next state of first agent in sequence
            maxQ_next_obs = np.max(self.q_tables[0][tuple(next_obs)])
            # Q val of action taken on current state of last agent in sequence
            # print(self.q_tables[agent_idx].shape)
            # print(agent_idx, tuple(obs), tuple(actions))
            Q_taken_action = self.q_tables[agent_idx][tuple(obs)][tuple(actions)]

            # Here target is taken as the normal temporal difference target with next state,
            # so we use transition reward and discount
            td_target = reward + self.gamma * maxQ_next_obs
            td_error = td_target - Q_taken_action
            # update
            self.q_tables[agent_idx][tuple(obs)][tuple(actions)] += self.lr * td_error

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
                # sequentially add actions
                for agent_idx in range(self.nr_agents):
                    # use observations + actions of previous agents as input
                    seq_obs = obs[agent_idx] + actions
                    # get action
                    actions.append(self.get_action(seq_obs, agent_idx))

                # take action
                next_obs, rewards, terminals, truncations, _ = self.env.step(actions) 

                # update Q-tables
                for agent_idx in range(self.nr_agents):
                    self.update_table(obs[agent_idx], 
                                      actions, 
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
