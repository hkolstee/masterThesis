import os, sys

import numpy as np

import gymnasium as gym

import torch
import torch.optim as optim
import torch.nn.functional as F

from custom_agent.SAC_components.ma_replay_buffer import MultiAgentReplayBuffer
from custom_agent.SAC_components.critic_discrete import Critic
from custom_agent.SAC_components.logger import Logger

from custom_spider_env.spider_fly_env.envs.grid_MA_pettingzoo import SpiderFlyEnvMA
from custom_spider_env.spider_fly_env.envs.pettingzoo_wrapper import PettingZooWrapper

from copy import deepcopy

class seqDoubleDQN:
    def __init__(self, env, 
                 lr = 1e-3, 
                 gamma = 0.99, 
                 eps_start = 0.9, 
                 eps_end = 0.05, 
                 eps_steps = 1000, 
                 tau = 0.005, 
                 batch_size = 256, 
                 buffer_max_size = 10000, 
                 log_dir = "tensorboard_logs",
                 layer_sizes = (256, 256),
                 global_observations = False):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_max_size = buffer_max_size
        self.global_observations = global_observations

        # initialize device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # logging
        self.logger = Logger(env, log_dir)

        # starting epsilon
        self.eps = eps_start
        # count global steps for epsilon 
        self.global_steps = 0

        # number of agents
        self.nr_agents = len(env.action_space)

        # one-hot identity per agent
        self.agent_ids = F.one_hot(torch.tensor(range(self.nr_agents)))
        # create batched size tensor to add to input
        self.agent_id_tensors = [self.agent_ids[agent_idx].unsqueeze(0).expand(batch_size, -1) for agent_idx in range(self.nr_agents)]
        # print(self.agent_id_tensors[0], self.agent_id_tensors[1])

        # if not all agents receive global observations from env
        if not self.global_observations:
            obs_size_global = sum([obs.shape[0] for obs in env.observation_space])
        # if they do
        else:
            obs_size_global = env.observation_space[0].shape[0]

        # Q networks share parameters among agents, with additional one_hot id as input, actions are calculated in sequence, where
        #   the next agent in sequence gets the actions of the previous in sequence additionally as input. Therefore, the total input
        #   size is the obs size, all agent actions except for the last in sequence and the one hot id vector.
        # Agent homogeneuosity is assumed for equal observation/action size.
        self.DQN_input_size = obs_size_global + (self.nr_agents - 1) * env.action_space[0].n + self.agent_ids.shape[0]
        # We use clipped double Q learning
        self.shared_DQN1 = Critic(lr, self.DQN_input_size, env.action_space[0].n, layer_sizes)
        self.shared_DQN2 = Critic(lr, self.DQN_input_size, env.action_space[0].n, layer_sizes)
        self.shared_target_DQN1 = deepcopy(self.shared_DQN1)
        self.shared_target_DQN2 = deepcopy(self.shared_DQN2)

        # we need different optimizers for the different agents (mostly because of lower learning rate)
        self.optimizers_dqn1 = []
        self.optimizers_dqn2 = []
        for agent_idx in range(self.nr_agents):
            self.optimizers_dqn1.append(optim.Adam(self.shared_DQN1.parameters(), lr = ((agent_idx + 1) / self.nr_agents) * lr, eps = 1e-08))
            self.optimizers_dqn2.append(optim.Adam(self.shared_DQN2.parameters(), lr = ((agent_idx + 1) / self.nr_agents) * lr, eps = 1e-08))

        # Replay buffer
        obs_size_list = [obs.shape for obs in env.observation_space]
        act_size_list = [(1,) for _ in env.action_space]
        self.replay_buffer = MultiAgentReplayBuffer(buffer_max_size, obs_size_list, act_size_list, batch_size)

    def learn(self):
        # buffer not full enough
        if self.replay_buffer.buffer_index < self.batch_size:
            # return status 0
            return 0, None
        
        # logging list for return of the function
        loss_Q_list = []
        Q_list = []
        Q_target_list = []

        # sample from buffer 
        #   list of batches (one for each agent, same indices out of buffer and therefore same multi-agent transition)
        obs_list, replay_act_list, rewards_list, next_obs_list, dones_list = self.replay_buffer.sample()

        # prepare tensors
        obs = [torch.tensor(obs, dtype=torch.float32).to(self.device) for obs in obs_list]
        next_obs = [torch.tensor(next_obs, dtype=torch.float32).to(self.device) for next_obs in next_obs_list]
        replay_act = [torch.tensor(actions, dtype=torch.int64).to(self.device) for actions in replay_act_list]
        # rewards = [torch.tensor(rewards, dtype=torch.float32).to(self.device) for rewards in rewards_list]
        rewards = torch.tensor(np.array(rewards_list), dtype=torch.float32).mean(0)
        dones = [torch.tensor(dones, dtype=torch.float32).to(self.device) for dones in dones_list]

        # if we are not given the global observation, for each agent, and we need to construct it
        with torch.no_grad():
            if not self.global_observations:
                obs = torch.cat(obs, dim = 1)
                next_obs = torch.cat(next_obs, dim = 1)
            else:
                # all agents are given global observations
                obs = obs[0]
                next_obs = next_obs[0]

        # we keep track of index of point where to add sequential actions to input tensor
        seq_action_index = obs.shape[1]

        with torch.no_grad():
            # prepare input
            input_tensor = torch.zeros((self.batch_size, self.shared_DQN1.input_size))
            # observations first
            input_tensor[:, 0 : obs.shape[1]] = obs
            # one_hot id (of current agent)
            input_tensor[:, -self.agent_id_tensors[0].shape[1] :] = self.agent_id_tensors[0]

        with torch.autograd.set_detect_anomaly(True):
            for agent_idx in range(self.nr_agents):
                # we can use last used tensor in target Q as current Q input
                if agent_idx > 0:
                    # last target Q input is same as next normal Q input (even onehot id)
                    input_tensor = targ_input_tensor.clone().detach()

                # Q_i(s, a_1, ..., a_i)
                Q1_taken_action = self.shared_DQN1(input_tensor)
                Q2_taken_action = self.shared_DQN2(input_tensor)
                Q_taken_action = torch.minimum(Q1_taken_action, Q2_taken_action).gather(1, replay_act[agent_idx].long())

                # TARGET Q values
                with torch.no_grad():
                    # for each agent we compare the Q values to the next agent Qvalue in the sequence on the current state, except for the last agent.
                    if agent_idx < (self.nr_agents - 1):
                        # prepare input for: max_a_i+1 Q(s, a_1, ..., a_i+1))
                        # so input consists of [s, a_1, ..., a_i]
                        targ_input_tensor = input_tensor.clone().detach()

                        # add additional actions
                        current_action = F.one_hot(replay_act[agent_idx], num_classes = self.shared_DQN1.output_size).squeeze(1)
                        targ_input_tensor[:, seq_action_index : seq_action_index + current_action.shape[1]] = current_action
                        # move index
                        seq_action_index += current_action.shape[1]

                        # we do need to change the onehot id vector to next in sequence
                        #   one_hot id (of NEXT agent)
                        targ_input_tensor[:, -self.agent_id_tensors[agent_idx + 1].shape[1] :] = self.agent_id_tensors[agent_idx + 1]

                        # max_a_i+1 Q*_i+1(s, a_1, ..., a_i+1)
                        Q1_next_agent = self.shared_target_DQN1(targ_input_tensor)
                        Q2_next_agent = self.shared_target_DQN2(targ_input_tensor)
                        Q_next_agent = torch.minimum(Q1_next_agent, Q2_next_agent).max(1).values

                        # we learn from the next agent Qvals only, with diminished learning rate
                        Q_target = Q_next_agent

                    # for the last agent in the sequence, we compare with the temporal difference Q-val of the first agent.
                    else:
                        # prepare input
                        targ_input_tensor = torch.zeros((self.batch_size, self.DQN_input_size))
                        # observations first
                        targ_input_tensor[:, 0 : next_obs.shape[1]] = next_obs
                        # no sequential action values to add, we take max action of the first agent
                        # one_hot id (of first agent !!)
                        targ_input_tensor[:, -self.agent_id_tensors[0].shape[1] :] = self.agent_id_tensors[0]

                        # max_a' Q*_1(s', a')
                        Q1_next_obs = self.shared_target_DQN1(targ_input_tensor)
                        Q2_next_obs = self.shared_target_DQN2(targ_input_tensor)
                        Q_next_obs = torch.minimum(Q1_next_obs, Q2_next_obs).max(1).values

                        # normal temporal difference target
                        # Q_target = rewards + self.gamma * Q_next_obs
                        Q_target = rewards + (1 - dones[agent_idx]) * self.gamma * Q_next_obs
                
                # loss
                loss = F.huber_loss(Q_taken_action, Q_target.unsqueeze(1))
                # backward prop + gradient step
                self.optimizers_dqn1[agent_idx].zero_grad()
                self.optimizers_dqn2[agent_idx].zero_grad()
                loss.backward()
                self.optimizers_dqn1[agent_idx].step()
                self.optimizers_dqn2[agent_idx].step()

                # soft update / polyak update
                for (p1, p2, p1_targ, p2_targ) in zip(self.shared_DQN1.parameters(),
                                                        self.shared_DQN2.parameters(),
                                                        self.shared_target_DQN1.parameters(),
                                                        self.shared_target_DQN2.parameters()):
                    # critic1
                    p1_targ.data *= (1 - self.tau)
                    p1_targ.data += (self.tau * p1.data)
                    # critic2
                    p2_targ.data *= (1 - self.tau)
                    p2_targ.data += (self.tau * p2.data)

                # log losses
                loss_Q_list.append(loss.detach().item())
                Q_list.append(torch.mean(Q_taken_action).detach().item())
                Q_target_list.append(torch.mean(Q_target).detach().item())

        # return status 1
        return 1, loss_Q_list


    def get_actions(self, observations, deterministic = False):
        # action list
        actions = []

        # get epsilon
        current_eps = self.eps_end + (self.eps_start - self.eps_end) * (np.max((self.eps_steps - self.global_steps), 0) / self.eps_steps)
        self.logger.log({"epsilon", current_eps}, self.global_steps, "train")

        with torch.no_grad():
            # make tensor if needed
            if not torch.is_tensor(observations[0]):
                # if global observations are given to each agent, we do not need to create it
                if self.global_observations:
                    global_obs = torch.tensor(observations[0], dtype = torch.float32)
                # we need to construct the global observation
                else:
                    global_obs = torch.tensor([o for observation in observations for o in observation], dtype = torch.float32)
            else:
                if self.global_observations:
                    global_obs = observations[0].to(torch.float32)
                else:
                    global_obs = torch.cat(observations)

            # create input tensor in case of epsilon greedy
            # which is [observations, seqential_acts, onehot id]
            input_tensor = torch.zeros((self.DQN_input_size))
            # inplace does not matter because of no_grad
            # observations first
            input_tensor[0 : global_obs.shape[0]] = global_obs

            # index in input tensor where to add additional sequential actions
            seq_action_index = global_obs.shape[0]
        
        for agent_idx in range(self.nr_agents):
            # epsilon non-greedy
            if np.random.uniform(0, 1) < current_eps and not deterministic:
                # add to actions, concat to sequential actions
                actions.append(self.env.action_space[agent_idx].sample())
            # epsilon greedy
            else:
                with torch.no_grad():
                    if agent_idx > 0:
                        # add previous sequential action
                        current_action = F.one_hot(torch.tensor(actions[-1], dtype = torch.int64), num_classes = self.shared_DQN1.output_size)
                        # add to tensor on the right indices
                        input_tensor[seq_action_index : seq_action_index + current_action.shape[0]] = current_action
                        # move sequential index
                        seq_action_index += current_action.shape[0]

                    # add new one_hot id at the end of the tensor
                    input_tensor[-self.agent_ids[agent_idx].shape[0] :] = self.agent_ids[agent_idx]

                    # forward through DQNs, clip Q vals
                    Q1_vals = self.shared_DQN1(input_tensor.unsqueeze(0))
                    Q2_vals = self.shared_DQN2(input_tensor.unsqueeze(0))
                    Q_vals = torch.minimum(Q1_vals, Q2_vals)

                    # take argmax from clipped Qvals
                    actions.append(Q_vals.argmax().item())
            
        return actions


    def train(self, num_episodes):
        reward_log = []
        loss_log = []

        for eps in range(num_episodes):
            obs, _ = self.env.reset()
            terminals = [False]
            truncations = [False]
            rew_sum = [0 for _ in range(self.nr_agents)]
            loss_sum = [0 for _ in range(self.nr_agents)]
            
            # keep track of number of learning steps
            learn_steps = 0
            ep_steps = 0
            while not (any(terminals) or all(truncations)):
                # get actions
                actions = self.get_actions(obs)
                # print("CHOSEN ACTIONS", actions)
                # take action
                next_obs, rewards, terminals, truncations, _ = self.env.step(actions)

                # add transition to replay buffer
                self.replay_buffer.add_transition(obs, actions, rewards, next_obs, terminals)

                # learning step
                status, losses = self.learn()

                if status:
                    # add to loss sum
                    learn_steps += 1
                    loss_sum = np.add(loss_sum, losses)

                # add to reward sum
                rew_sum = np.add(rew_sum, rewards)

                # update state
                obs = next_obs

                # keep track of steps
                self.global_steps += 1
                ep_steps += 1

            # log
            avg_loss = loss_sum / learn_steps
            reward_log.append(rew_sum)
            loss_log.append(loss_sum / learn_steps)
            # tensorboard
            rollout_log = {"reward_sum": rew_sum,
                           "ep_len": ep_steps}
            self.logger.log(rollout_log, self.global_steps, group = "rollout")
            if learn_steps > 0:
                logs = {"average_loss": avg_loss}
                self.logger.log(logs, self.global_steps, group = "train")


            if eps % (num_episodes // 20) == 0:
                print("Episode: " + str(eps) + " - Reward:" + str(rew_sum) + " - Avg loss (last ep):", loss_log[-1])
                print("eps: ", self.eps_end + (self.eps_start - self.eps_end) * ((self.eps_steps - self.global_steps) / self.eps_steps))

        return reward_log, loss_log
