import os, sys

import numpy as np

import gymnasium as gym

import torch
import torch.nn.functional as F

from custom_agent.SAC_components.ma_replay_buffer import MultiAgentReplayBuffer
from custom_agent.SAC_components.critic_discrete import Critic
from custom_agent.SAC_components.logger import Logger

from custom_spider_env.spider_fly_env.envs.grid_MA_pettingzoo import SpiderFlyEnvMA
from custom_spider_env.spider_fly_env.envs.pettingzoo_wrapper import PettingZooWrapper



class IndependentDQN:
    def __init__(self, env, 
                 lr = 1e-3, 
                 gamma = 0.99, 
                 eps_start = 0.9, 
                 eps_end = 0.05, 
                 eps_steps = 1000, 
                 tau = 0.005, 
                 batch_size = 8, 
                 buffer_max_size = 10000, 
                 log_dir = "tensorboard_logs",
                 layer_sizes = (256, 256)):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_max_size = buffer_max_size

        # initialize device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # starting eps
        self.eps = eps_start
        self.global_steps = 0

        # number of agents
        self.nr_agents = len(env.action_space)

        # one-hot identity per agent
        self.agent_ids = F.one_hot(torch.tensor(range(self.nr_agents)))
        # create batched size tensor to cat to input
        self.agent_id_tensors = [self.agent_ids[agent_idx].unsqueeze(0).expand(batch_size, -1) for agent_idx in range(self.nr_agents)]

        # Q networks share parameters among agents, with additional one_hot id as input
        self.shared_DQN = Critic(lr, env.observation_space[0].shape[0] + self.agent_ids.shape[0], env.action_space[0].n, layer_sizes)
        self.shared_target_DQN = Critic(lr, env.observation_space[0].shape[0] + self.agent_ids.shape[0], env.action_space[0].n, layer_sizes)
        # copy parameters
        self.shared_target_DQN.load_state_dict(self.shared_DQN.state_dict())

        # Replay buffer
        obs_size_list = [obs.shape for obs in env.observation_space]
        act_size_list = [(1,) for _ in env.action_space]
        self.replay_buffer = MultiAgentReplayBuffer(buffer_max_size, obs_size_list, act_size_list, batch_size)

    def get_action(self, observation, agent_idx, deterministic = False):
        if not torch.is_tensor(observation):
            observation = torch.tensor(observation, dtype = torch.float64)

        # get epsilon
        current_eps = self.eps_end + (self.eps_start - self.eps_end) * ((self.eps_steps - self.global_steps) / self.eps_steps)
        # epsilon non-greedy
        if np.random.uniform(0, 1) < current_eps and not deterministic:
            return self.env.action_space[agent_idx].sample()
        # epsilon greedy
        else:
            with torch.no_grad():
                one_hot_id = self.agent_ids[agent_idx]
                # clear potentially left-over gradients
                one_hot_id.grad = None
                return self.shared_DQN(torch.cat([observation, one_hot_id])).argmax().item()
        
    def learn(self):
        # buffer not full enough
        if self.replay_buffer.buffer_index < self.batch_size:
            # return status 0
            return 0, None
        
        # logging list for return of the function
        loss_Q_list = []

        # sample from buffer 
        #   list of batches (one for each agent, same indices out of buffer and therefore same multi-agent transition)
        obs_list, replay_act_list, rewards_list, next_obs_list, dones_list = self.replay_buffer.sample()

        # prepare tensors
        obs = [torch.tensor(obs, dtype=torch.float32).to(self.device) for obs in obs_list]
        next_obs = [torch.tensor(next_obs, dtype=torch.float32).to(self.device) for next_obs in next_obs_list]
        replay_act = [torch.tensor(actions, dtype=torch.float32).to(self.device) for actions in replay_act_list]
        # rewards = [torch.tensor(rewards, dtype=torch.float32).to(self.device) for rewards in rewards_list]
        rewards = torch.tensor(np.array(rewards_list), dtype=torch.float32).mean(0)
        dones = [torch.tensor(dones, dtype=torch.float32).to(self.device) for dones in dones_list]

        for agent_idx in range(self.nr_agents):
            # max_a' Q*(s', a')
            with torch.no_grad():
                maxQ_next_obs = self.shared_target_DQN(torch.cat([next_obs[agent_idx], self.agent_id_tensors[agent_idx]], dim = 1)).max(1).values
                
            # Q(s, a)
            Q_taken_action = self.shared_DQN(torch.cat([obs[agent_idx], self.agent_id_tensors[agent_idx]], dim = 1)).gather(1, replay_act[agent_idx].long())
            # Q_target = rewards + (1 - dones[agent_idx]) * self.gamma * maxQ_next_obs
            Q_target = rewards + self.gamma * maxQ_next_obs
            
            # loss
            loss = F.huber_loss(Q_taken_action, Q_target.unsqueeze(1))
            # backward prop + gradient step
            self.shared_DQN.optimizer.zero_grad()
            loss.backward()
            self.shared_DQN.optimizer.step()

            # log losses
            loss_Q_list.append(loss.detach().numpy())

        # return status 1
        return 1, loss_Q_list

    def train(self, num_episodes):
        reward_log = []
        loss_log = []

        for eps in range(num_episodes):
            obs, _ = self.env.reset()
            terminals = [False]
            truncations = [False]
            rew_sum = [0 for _ in range(self.nr_agents)]
            loss_sum = [0 for _ in range(self.nr_agents)]

            episode_steps = 0
            while not (any(terminals) or all(truncations)):
                # get actions
                actions = []
                for agent_idx in range(self.nr_agents):
                    actions.append(self.get_action(obs[agent_idx], agent_idx))
                # take action
                next_obs, rewards, terminals, truncations, _ = self.env.step(actions)

                # add transition to replay buffer
                self.replay_buffer.add_transition(obs, actions, rewards, next_obs, terminals)

                # learning step
                status, losses = self.learn()

                # if step % 1000 == 0:
                #     self.shared_target_DQN.load_state_dict(self.shared_DQN.state_dict())

                if status:
                    # add to loss sum
                    loss_sum = np.add(loss_sum, losses)
                    # soft update / polyak update
                    target_state_dict = self.shared_target_DQN.state_dict()
                    policy_state_dict = self.shared_DQN.state_dict()
                    for params in policy_state_dict:
                        target_state_dict[params] = policy_state_dict[params] * self.tau + target_state_dict[params] * (1 - self.tau)
                    self.shared_target_DQN.load_state_dict(target_state_dict)

                    

                # add to reward sum
                rew_sum = np.add(rew_sum, rewards)

                # update state
                obs = next_obs

                # keep track of steps
                episode_steps += 1
                if self.global_steps < self.eps_steps:
                    self.global_steps += 1

            # log
            reward_log.append(rew_sum)
            loss_log.append(np.divide(loss_sum, episode_steps))

            if eps % (num_episodes // 20) == 0:
                print("Episode: " + str(eps) + " - Reward:" + str(rew_sum) + " - Avg loss (last ep):", loss_log[-1])

        return reward_log, loss_log
