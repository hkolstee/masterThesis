import os, sys

import numpy as np

import gymnasium as gym
from gymnasium import spaces

import torch
import torch.optim as optim
import torch.nn.functional as F

# add folder to python path for relative imports
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

from ..replay_buffers.ma_replay_buffer import MultiAgentReplayBuffer
from ..networks.critic_discrete import Critic
from ..utils.logger import Logger

from copy import deepcopy

class seqDQN:
    def __init__(self, 
                 env, 
                 lr = 1e-3, 
                 gamma = 0.99, 
                 eps_start = 0.9, 
                 eps_end = 0.05, 
                 eps_steps = 1000, 
                 tau = 0.005, 
                 batch_size = 256, 
                 buffer_max_size = 10000, 
                 layer_sizes = (256, 256),
                 global_observations = False,
                 log_dir = "tensorboard_logs",
                 save_dir = "models",
                 eval_every = 25,
                 ):
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
        self.save_dir = save_dir
        self.eval_every = eval_every

        # initialize device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # logging utility
        self.logger = Logger(env, log_dir)
        # logging best eval model
        self.best_eval = -np.inf
        
        if isinstance(self.env.action_space[0], spaces.Box):
            # pad by min - 1
            self.padding_val = min([act_space.low.item for act_space in self.env.action_space]) - 1
        else:
            # padding values we take as - 1, 0 is lowest action possible with discrete action space
            self.padding_val = - 1.0

        # starting epsilon
        self.eps = eps_start
        # count global steps for epsilon 
        self.global_steps = 0

        # number of agents
        self.nr_agents = len(env.action_space)

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
        self.DQN_input_size = obs_size_global + (self.nr_agents - 1) * env.action_space[0].n
        self.shared_DQN = Critic(lr, self.DQN_input_size, env.action_space[0].n, layer_sizes)
        self.shared_target_DQN = deepcopy(self.shared_DQN)

        # we need different optimizers for the different agents (mostly because of lower learning rate)
        self.optimizers = []
        for agent_idx in range(self.nr_agents):
            self.optimizers.append(optim.Adam(self.shared_DQN.parameters(), lr = ((agent_idx + 1) / self.nr_agents) * lr, eps = 1e-8))

        # Replay buffer
        obs_size_list = [obs.shape for obs in env.observation_space]
        act_size_list = [(1,) for _ in env.action_space]
        self.replay_buffer = MultiAgentReplayBuffer(buffer_max_size, obs_size_list, act_size_list, batch_size)

    def polyak_update(self, base_network, target_network, polyak):
        """ 
        Polyak/soft update of target networks.
        """
        with torch.no_grad():
            for (base_params, target_params) in zip(base_network.parameters(), target_network.parameters()):
                target_params.data *= polyak
                target_params.data += ((1 - polyak) * base_params.data)

    def learn(self):
        # buffer not full enough
        if self.replay_buffer.buffer_index < self.batch_size:
            # return status 0
            return 0, None
        
        # logging list for return of the function
        loss_list = []

        # sample from buffer 
        #   list of batches (one for each agent, same indices out of buffer and therefore same multi-agent transition)
        obs_list, replay_act_list, rewards_list, next_obs_list, dones_list = self.replay_buffer.sample()

        # prepare tensors
        obs = [torch.tensor(obs, dtype=torch.float32).to(self.device) for obs in obs_list]
        next_obs = [torch.tensor(next_obs, dtype=torch.float32).to(self.device) for next_obs in next_obs_list]
        replay_act = [torch.tensor(actions, dtype=torch.int64).to(self.device) for actions in replay_act_list]
        # rewards = [torch.tensor(rewards, dtype=torch.float32).to(self.device) for rewards in rewards_list]
        rewards = torch.tensor(np.array(rewards_list), dtype=torch.float32).mean(0).to(self.device)
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
            input_tensor = torch.full((self.batch_size, self.shared_target_DQN.input_size), self.padding_val, dtype = torch.float32)
            # observations first
            input_tensor[:, 0 : obs.shape[1]] = obs

        with torch.autograd.set_detect_anomaly(True):
            for agent_idx in range(self.nr_agents):
                # we can use last used tensor in target Q as current Q input
                if agent_idx > 0:
                    # last target Q input is same as next normal Q input (even onehot id)
                    input_tensor = targ_input_tensor.clone().detach()

                Q_vals = self.shared_DQN(input_tensor.to(self.device))
                Q_taken_action = Q_vals.gather(1, replay_act[agent_idx].long())

                # TARGET Q values
                with torch.no_grad():
                    # for each agent we compare the Q values to the next agent Qvalue in the sequence on the current state, except for the last agent.
                    if agent_idx < (self.nr_agents - 1):
                        # prepare input for: max_a_i+1 Q(s, a_1, ..., a_i+1))
                        # so input consists of [s, a_1, ..., a_i]
                        targ_input_tensor = input_tensor.clone().detach()

                        # add additional actions
                        current_action = F.one_hot(replay_act[agent_idx], num_classes = self.shared_DQN.output_size).squeeze(1)
                        targ_input_tensor[:, seq_action_index : seq_action_index + current_action.shape[1]] = current_action
                        # move index
                        seq_action_index += current_action.shape[1]

                        # max_a_i+1 Q*_i+1(s, a_1, ..., a_i+1)
                        Q_vals = self.shared_target_DQN(targ_input_tensor.to(self.device))
                        max_Q_next_agent = Q_vals.max(1).values

                        # we learn from the next agent Qvals only, with diminished learning rate
                        Q_target = max_Q_next_agent

                    # for the last agent in the sequence, we compare with the temporal difference Q-val of the first agent.
                    else:
                        # prepare input
                        targ_input_tensor = torch.full((self.batch_size, self.DQN_input_size), self.padding_val, dtype = torch.float32)
                        # observations first
                        targ_input_tensor[:, 0 : next_obs.shape[1]] = next_obs
                        # no sequential action values to add, we take max action of the first agent

                        Q_vals = self.shared_target_DQN(targ_input_tensor.to(self.device))
                        max_Q_next_obs = Q_vals.max(1).values

                        # normal temporal difference target
                        Q_target = rewards + (1 - dones[agent_idx]) * self.gamma * max_Q_next_obs
                
                # loss
                loss = F.huber_loss(Q_taken_action, Q_target.unsqueeze(1))
                # backward prop + gradient step
                self.shared_DQN.optimizer.zero_grad()
                # self.optimizers[agent_idx].zero_grad()
                loss.backward()
                self.shared_DQN.optimizer.step()
                # self.optimizers[agent_idx].step()

                # log losses
                loss_list.append(loss.detach().item())

        # return status 1
        return 1, loss_list
    
    def evaluate(self, eps):
        """
        Evaluate the current policy.
        """
        self.shared_DQN.eval()
        
        obs, _ = self.env.reset()
        
        terminals = [False]
        truncations = [False]
        rew_sum = 0
        ep_steps = 0
        
        while not (any(terminals) or all(truncations)):
            # get action
            with torch.no_grad():
                act = self.get_action(obs, deterministic = True)
            # execute action
            next_obs, rewards, terminals, truncations, _ = self.env.step(act)
            
            obs = next_obs

            # keep track of steps
            ep_steps += 1
            
            # add to reward sum
            rew_sum += np.mean(rewards)
        
        # save if best
        if rew_sum > self.best_eval:
            self.best_eval = rew_sum
            self.shared_DQN.save(self.save_dir, "DQN_eval")
            self.shared_target_DQN.save(self.save_dir, "target_DQN_eval")
            
        # log rewards
        self.logger.log({"eval_reward_sum": rew_sum}, eps, "rollout")
        
        # turn off eval mode
        self.shared_DQN.train()

    def get_Q(self, observations):
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
            # which is [observations, seqential_acts]
            input_tensor = torch.full((self.DQN_input_size,), self.padding_val, dtype = torch.float32)
            # inplace does not matter because of no_grad
            # observations first
            input_tensor[0 : global_obs.shape[0]] = global_obs
            
            self.shared_DQN.eval()
            
            with torch.no_grad():
                Q_val = self.shared_DQN(input_tensor.to(self.device)).max().item()
            
            self.shared_DQN.train()
            
        return Q_val
        
    def get_action(self, observations, deterministic = False):
        # action list
        actions = []

        # get epsilon
        self.current_eps = self.eps_end + (self.eps_start - self.eps_end) * (np.max((self.eps_steps - self.global_steps, 0)) / self.eps_steps)

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
            # which is [observations, seqential_acts]
            input_tensor = torch.full((self.DQN_input_size,), self.padding_val, dtype = torch.float32)
            # inplace does not matter because of no_grad
            # observations first
            input_tensor[0 : global_obs.shape[0]] = global_obs

            # index in input tensor where to add additional sequential actions
            seq_action_index = global_obs.shape[0]
        
        for agent_idx in range(self.nr_agents):
            # epsilon non-greedy
            if np.random.uniform(0, 1) < self.current_eps and not deterministic:
                # add to actions
                actions.append(self.env.action_space[agent_idx].sample())
            # epsilon greedy
            else:
                with torch.no_grad():
                    # forward through DQN, take argmax for max action
                    actions.append(self.shared_DQN(input_tensor.unsqueeze(0).to(self.device)).argmax().item())

            # for all agents except the last we add the action to the input of the next
            if agent_idx < (self.nr_agents - 1):
                # add selected action to tensor but first convert to onehot
                current_action = F.one_hot(torch.tensor(actions[-1], dtype = torch.int64), num_classes = self.shared_DQN.output_size)
                # add to tensor on the correct indices
                input_tensor[seq_action_index : seq_action_index + current_action.shape[0]] = current_action
                # move sequential index for the next action
                seq_action_index += current_action.shape[0]
            
        return actions

    def train(self, num_episodes):
        current_best = 0

        for eps in range(num_episodes):
            obs, _ = self.env.reset()
            terminals = [False]
            truncations = [False]
            rew_sum = np.zeros(self.nr_agents)
            loss_sum = np.zeros(self.nr_agents)

            learn_steps = 0
            ep_steps = 0
            
            # get initial Q-val for logging
            initial_Q = self.get_Q(obs)
            
            while not (any(terminals) or all(truncations)):
                # get actions
                actions = self.get_action(obs)
                # take action
                next_obs, rewards, terminals, truncations, _ = self.env.step(actions)

                # add transition to replay buffer
                # print("Transition added: ", obs, actions, rewards, next_obs, terminals)
                self.replay_buffer.add_transition(obs, actions, rewards, next_obs, terminals)

                # learning step
                status, losses = self.learn()

                # update state
                obs = next_obs

                if status:
                    # learn step
                    learn_steps += 1
                    # add to loss sum
                    np.add(loss_sum, losses, out = loss_sum)

                    # soft update / polyak update
                    self.polyak_update(self.shared_DQN, self.shared_target_DQN, 1 - self.tau)

                # add to reward sum
                rew_sum = np.add(rew_sum, rewards)

                # keep track of steps
                if self.global_steps < self.eps_steps:
                    self.global_steps += 1
                ep_steps += 1
            
            # eval
            if eps % self.eval_every == 0:
                self.evaluate(eps)

            # save if best
            current_rew = np.mean(rew_sum)
            if current_rew > current_best:
                current_best = current_rew
                self.shared_DQN.save_checkpoint(self.save_dir, "seqDQN", losses, eps)
                self.shared_target_DQN.save_checkpoint(self.save_dir, "target_seqDQN", losses, eps)

            # tensorboard logs
            if learn_steps:
                avg_loss = loss_sum / learn_steps
                self.logger.log({"average_loss": avg_loss,
                                 "start_state_Q": initial_Q,
                                 "epsilon": self.current_eps}, eps, group = "train")
            else:
                avg_loss = None
            rollout_log = {"reward_sum": rew_sum,
                           "ep_len": ep_steps}
            self.logger.log(rollout_log, eps, group = "rollout")

            # command line info print
            if eps % (num_episodes // 20) == 0:
                print("Episode: " + str(eps) + " - Reward:" + str(rew_sum) + " - Avg loss (last ep):", avg_loss)
