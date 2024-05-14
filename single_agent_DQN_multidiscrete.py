import os, sys

import numpy as np

import gymnasium as gym

import torch
import torch.nn.functional as F

from custom_agents.replay_buffers.replay_buffer import ReplayBuffer
from custom_agents.networks.critic_multidiscrete import MultiDiscreteCritic
from custom_agents.utils.logger import Logger

from custom_spider_env.spider_fly_env.envs.grid_MA_pettingzoo import SpiderFlyEnvMA
from custom_spider_env.spider_fly_env.envs.pettingzoo_wrapper import PettingZooWrapper

class MultiDiscreteDQN:
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

        # init logger
        self.logger = Logger(self.env, log_dir)

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

        # Q networks output multiple discrete actions per forward pass
        action_sizes = [act_space.n for act_space in env.action_space]
        self.shared_DQN = MultiDiscreteCritic(lr, env.observation_space[0].shape[0], action_sizes, layer_sizes)
        self.shared_target_DQN = MultiDiscreteCritic(lr, env.observation_space[0].shape[0], action_sizes, layer_sizes)
        # copy parameters
        self.shared_target_DQN.load_state_dict(self.shared_DQN.state_dict())

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_max_size, env.observation_space[0].shape, (len(env.action_space),), batch_size)

    def polyak_update(self, base_network, target_network, polyak):
        """ 
        Polyak/soft update of target networks.
        """
        with torch.no_grad():
            for (base_params, target_params) in zip(base_network.parameters(), target_network.parameters()):
                target_params.data *= polyak
                target_params.data += ((1 - polyak) * base_params.data)

    def get_action(self, observation, deterministic = False):
        if not torch.is_tensor(observation):
            observation = torch.tensor(observation, dtype = torch.float32)

        # get epsilon
        current_eps = self.eps_end + (self.eps_start - self.eps_end) * ((self.eps_steps - self.global_steps) / self.eps_steps)
        # epsilon non-greedy
        if np.random.uniform(0, 1) < current_eps and not deterministic:
            return [act_space.sample() for act_space in self.env.action_space]
        # epsilon greedy
        else:
            with torch.no_grad():
                return [action_logits.argmax().item() for action_logits in self.shared_DQN(observation)]
        
    def learn(self):
        # buffer not full enough
        if self.replay_buffer.buffer_index < self.batch_size:
            # return status 0
            return 0, None
        
        # logging list for return of the function
        loss_Q_list = []

        # sample from buffer 
        obs, replay_act, rewards, next_obs, dones = self.replay_buffer.sample()

        # prepare tensors
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        replay_act = torch.tensor(replay_act, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.int32).to(self.device)

        # max_a' Q*(s', a')
        with torch.no_grad():
            # maxQ_next_obs = self.shared_target_DQN(torch.cat([next_obs[agent_idx], self.agent_id_tensors[agent_idx]], dim = 1)).max(1).values
            Q_next_obs = self.shared_target_DQN(next_obs)
            maxQ_next_obs = [qvals.max(1).values for qvals in Q_next_obs]
            
        # Q(s, a)
        # Q_taken_action = self.shared_DQN(torch.cat([obs[agent_idx], self.agent_id_tensors[agent_idx]], dim = 1)).gather(1, replay_act[agent_idx].long())
        Q_obs = self.shared_DQN(obs)
        Q_taken_action = [qvals.gather(1, replay_act[:, idx].unsqueeze(1).long()) for (idx, qvals) in enumerate(Q_obs)]
        
        # Q_target = rewards + (1 - dones[agent_idx]) * self.gamma * maxQ_next_obs
        Q_target = [rewards + self.gamma * maxQ for maxQ in maxQ_next_obs]
        
        # loss
        # we just sum the loss of the action heads
        loss = sum([F.huber_loss(Q_taken, Q_targ.unsqueeze(1)) for (Q_taken, Q_targ) in zip(Q_taken_action, Q_target)])
        # backward prop + gradient step
        self.shared_DQN.optimizer.zero_grad()
        loss.backward()
        self.shared_DQN.optimizer.step()

        # return status 1
        return 1, loss.item()

    def train(self, num_episodes):
        for eps in range(num_episodes):
            obs, _ = self.env.reset()
            obs = obs[0]
            terminals = [False]
            truncations = [False]
            rew_sum = 0
            loss_sum = 0

            learn_steps = 0
            while not (any(terminals) or all(truncations)):
                # get actions
                actions = self.get_action(obs)
                # take action
                next_obs, rewards, terminals, truncations, _ = self.env.step(actions)

                # add transition to replay buffer
                self.replay_buffer.add_transition(obs, actions, np.mean(rewards), next_obs[0], terminals[0])

                # learning step
                status, loss = self.learn()

                if status:
                    # add to learn steps
                    learn_steps += 1
                    # add to loss sum
                    loss_sum += loss
                    # soft update / polyak update
                    self.polyak_update(self.shared_DQN, self.shared_target_DQN, 1 - self.tau)
                    
                # add to reward sum
                rew_sum += np.mean(rewards)

                # update state
                obs = next_obs[0]

                # keep track of steps
                if self.global_steps < self.eps_steps:
                    self.global_steps += 1

            # log
            self.logger.log({"avg_Q_loss": loss_sum / learn_steps}, eps, "train")
            self.logger.log({"reward_sum": rew_sum}, eps, "reward")

            if eps % (num_episodes // 20) == 0:
                print("Episode: " + str(eps) + " - Reward:" + str(rew_sum) + " - Avg loss (last ep):", loss_sum / learn_steps)