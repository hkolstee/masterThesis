import os, sys
import itertools

import numpy as np

import gymnasium as gym

import torch
import torch.nn.functional as F

# add folder to python path for relative imports
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

from ..replay_buffers.ma_replay_buffer import MultiAgentReplayBuffer
from ..replay_buffers.replay_buffer import ReplayBuffer
from ..networks.critic_discrete import Critic
from ..utils.logger import Logger

class DQN:
    def __init__(self, env, 
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
        self.global_obs = global_observations
        self.save_dir = save_dir
        self.eval_every = eval_every

        # evaluation performance meter for saving best model
        self.best_eval = -np.inf

        # init logger
        self.logger = Logger(env, log_dir)

        # initialize device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # starting eps
        self.eps = eps_start
        self.global_steps = 0

        # number of agents
        self.nr_agents = len(env.action_space)

        # Q networks output all combinations of actions as action space, over which we can choose the best estimated combination 
        # all possible action combinations
        action_ranges = [range(act_space.n) for act_space in env.action_space]
        self.index_to_act_combi = list(itertools.product(*action_ranges))
        self.act_combi_to_index = {combo: index for index, combo in enumerate(self.index_to_act_combi)}
        # concat obs if not global for each agent
        if self.global_obs:
            obs_size = env.observation_space[0].shape[0]
        else:
            obs_size = sum([obs_space.shape[0] for obs_space in env.observation_space])

        self.DQN = Critic(lr, obs_size, len(self.index_to_act_combi), layer_sizes)
        self.target_DQN = Critic(lr, obs_size, len(self.index_to_act_combi), layer_sizes)
        # copy parameters
        self.target_DQN.load_state_dict(self.DQN.state_dict())

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_max_size, (obs_size,), (1,), batch_size)

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
            return 0, None, None, None
        
        # sample from buffer 
        obs, replay_act, rewards, next_obs, done = self.replay_buffer.sample()

        # prepare tensors
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        replay_act = torch.tensor(replay_act, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.int32).to(self.device)

        # max_a' Q*(s', a')
        with torch.no_grad():
            maxQ_next_obs = self.target_DQN(next_obs).max(1).values
            
        # Q(s, a)
        Q_taken_action = self.DQN(obs).gather(1, replay_act.long())
        Q_target = rewards + (1 - done) * self.gamma * maxQ_next_obs
        
        # loss
        loss = F.huber_loss(Q_taken_action, Q_target.unsqueeze(1))
        # backward prop + gradient step
        self.DQN.optimizer.zero_grad()
        loss.backward()
        self.DQN.optimizer.step()

        # return status 1
        return 1, loss.detach().item(), torch.mean(Q_taken_action).detach().item(), torch.mean(Q_target.detach()).item()
    
    def get_action(self, observation, deterministic = False):
        if not torch.is_tensor(observation):
            observation = torch.tensor(np.array(observation), dtype = torch.float32).to(self.device)

        # get epsilon
        self.current_eps = self.eps_end + (self.eps_start - self.eps_end) * ((self.eps_steps - self.global_steps) / self.eps_steps)
        # epsilon non-greedy
        if np.random.uniform(0, 1) < self.current_eps and not deterministic:
            act = [act_space.sample() for act_space in self.env.action_space]
            idx = self.act_combi_to_index[tuple(act)]
            return act, idx
        # epsilon greedy
        else:
            with torch.no_grad():
                # get action index corresponding to certain combi of actions
                idx = self.DQN(observation).argmax().item()
                # return combi of actions
                act = list(self.index_to_act_combi[idx]) 
                return act, idx
            
    def evaluate(self, eps):
        """
        Evaluate the current policy.
        """
        self.DQN.eval()
        
        obs, _ = self.env.reset()
        if self.global_obs:
            obs = obs[0]
        else:
            obs = [item for o in obs for item in o]
        
        terminals = [False]
        truncations = [False]
        rew_sum = 0
        ep_steps = 0
        
        while not (any(terminals) or all(truncations)):
            # get action
            act, idx = self.get_action(obs, deterministic = True)
            # execute action
            next_obs, rewards, terminals, truncations, _ = self.env.step(act)
            
            # next state
            if self.global_obs:
                obs = next_obs[0]
            else:
                # concat list of lists
                obs = [item for o in next_obs for item in o]

            # keep track of steps
            ep_steps += 1
            
            # add to reward sum
            rew_sum += np.mean(rewards)
        
        # save if best
        if rew_sum > self.best_eval:
            self.best_eval = rew_sum
            self.DQN.save(self.save_dir, "DQN_eval")
            self.target_DQN.save(self.save_dir, "target_DQN_eval")
            
        # log rewards
        self.logger.log({"eval_reward_sum": rew_sum}, eps, "rollout")
        
        # turn off eval mode
        self.DQN.train()
                
    def train(self, num_episodes):
        best_train = 0

        for eps in range(num_episodes):
            obs, _ = self.env.reset()
            if self.global_obs:
                obs = obs[0]
            else:
                # concat list of lists
                obs = [item for o in obs for item in o]

            terminals = [False]
            truncations = [False]
            rew_sum = 0
            Q_sum = 0
            Q_target_sum = 0
            loss_sum = 0

            learn_steps = 0
            ep_steps = 0
            while not (any(terminals) or all(truncations)):
                # get actions
                act, idx = self.get_action(obs)

                # take action
                next_obs, rewards, terminals, truncations, _ = self.env.step(act)

                # add transition to replay buffer
                if self.global_obs:
                    next_obs = next_obs[0]
                else:
                    # concat list of lists
                    next_obs = [item for o in next_obs for item in o]
                # we save index of combis as this is used in gradient calculations
                act = idx
                self.replay_buffer.add_transition(obs, act, np.mean(rewards), next_obs, terminals[0])

                # learning step
                status, loss, Q_val, Q_target = self.learn()

                if status:
                    # add to learn steps
                    learn_steps += 1
                    # add to loss sum
                    loss_sum += loss
                    Q_sum += Q_val
                    Q_target_sum += Q_target
                    # soft update / polyak update
                    self.polyak_update(self.DQN, self.target_DQN, 1 - self.tau)
                    
                # add to reward sum
                rew_sum += np.mean(rewards)

                # update state
                obs = next_obs

                # keep track of steps
                if self.global_steps < self.eps_steps:
                    self.global_steps += 1
                ep_steps += 1
                
            # eval
            if eps % self.eval_every == 0:
                self.evaluate(eps)

            # save if best
            if rew_sum > best_train:
                best_train = rew_sum
                self.DQN.save_checkpoint(self.save_dir, "DQN", loss, eps)
                self.target_DQN.save_checkpoint(self.save_dir, "target_DQN", loss, eps)

            # log
            if learn_steps:
                avg_loss = loss_sum / learn_steps
                avg_Q = Q_sum / learn_steps
                avg_target_Q = Q_target_sum / learn_steps
                self.logger.log({"avg_Q_loss":  avg_loss, 
                                 "avg_Q_value": avg_Q,
                                 "avg_Q_target_value": avg_target_Q,
                                 "epsilon": self.current_eps}, eps, "train")
            else:
                avg_loss = None
            rollout_log = {"reward_sum": rew_sum,
                           "ep_len": ep_steps}
            self.logger.log(rollout_log, eps, group = "rollout")

            if eps % (num_episodes // 20) == 0:
                print("Episode: " + str(eps) + " - Reward:" + str(rew_sum) + " - Avg loss (last ep):", avg_loss)
                
