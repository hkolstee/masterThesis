import os
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

# temporary needed
from custom_agent.CTCE.citylearn_wrapper import CityLearnWrapper
from custom_reward.custom_reward import CustomReward

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedSpaceWrapper

from gymnasium import spaces

# add folder to python path for relative imports
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

from ..SAC_components.ma_replay_buffer import MultiAgentReplayBuffer
from ..SAC_components.critic_discrete import Critic
from ..SAC_components.actor_discrete import DiscreteActor
from ..SAC_components.logger import Logger

from copy import deepcopy

from tqdm import tqdm

class Agents:
    """
    Multi-agent Soft Actor-Critic centralized training decentralized execution agents.
    The critics are centralized, while the actors are decentralized.

    Args:
        env (gym.environment): The environment the agent acts within
        lr_actor (float): Actor network learning rate
        lr_critic (float): Critic network learning rate
        gamma (float): Next reward estimation discount
        polyak (float): Factor of polyak averaging
        buffer_max_size (int): The maximal size of the replay buffer
        batch_size (int): Sample size when sampling from the replay buffer
        layer_sizes (tuple:int): The sizes of the hidden multi-layer-perceptron
            layers within the function estimators (value, actor, critic)
        reward_scaling (int): Scaling of the reward with regard to the entropy
    """
    def __init__(self, 
                 env, 
                 lr_actor = 0.0003, 
                 lr_critic = 0.0003, 
                 gamma = 0.99, 
                 polyak = 0.995,
                 buffer_max_size = 1000000,
                 batch_size = 256,
                 layer_sizes = (256, 256),
                 log_dir = "tensorboard_logs"):
        self.env = env
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.nr_agents = len(env.action_space)
        
        # initialize device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # initialize logger
        self.logger = Logger(self.env, log_dir)
        
        # for now done like this: check if citylearn env with custom reward function for 
        #   additional logging
        self.citylearn = isinstance(self.env.reward_function, CustomReward) \
                            if (isinstance(self.env, CityLearnWrapper) \
                            or isinstance(self.env, CityLearnEnv) \
                            or isinstance(self.env, NormalizedSpaceWrapper)) \
                            else False

        # initialize replay buffer
        obs_size_list = [obs.shape for obs in self.env.observation_space]
        act_size_list = [(1,) for _ in self.env.action_space]
        self.replay_buffer = MultiAgentReplayBuffer(buffer_max_size, obs_size_list, act_size_list, batch_size)
        
        # one-hot identity per agent
        self.agent_ids = functional.one_hot(torch.tensor(range(self.nr_agents))).float()

        # global obs and action for centralized critics
        obs_size_global = sum([obs.shape[0] for obs in self.env.observation_space])
        act_size_global = sum([act.n for act in self.env.action_space])

        # policies according to method as described in: https://arxiv.org/pdf/1910.00120.pdf
        #   Simply: sequentially calculate policies, while conditioning next policy calculation on action result of last.
        #           Done sequentially per agent, where we do not care about order. One actor in total for all agents, with parameter sharing.
        #           Requires homogenous agent action spaces.
        # Two centralized critics (two for stable learning), also sequential for update rules as in https://arxiv.org/pdf/1910.00120.pdf
        self.global_act_plus_id_size = act_size_global + self.agent_ids.shape[0]

        # discrete or not discrete actors
        self.actor = DiscreteActor(lr_actor,
                                    obs_size_global + self.global_act_plus_id_size,
                                    self.env.action_space[0].n,
                                    layer_sizes)
        self.critic1 = Critic(lr_critic,
                              obs_size_global + self.global_act_plus_id_size,
                              env.action_space[0].n,
                              layer_sizes)
        self.critic2 = Critic(lr_critic,
                              obs_size_global + self.global_act_plus_id_size,
                              env.action_space[0].n,
                              layer_sizes)

        # optimizers
        self.actor_optimizers = []
        self.critic1_optimizers = []
        self.critic2_optimizers = []
        for agent_idx in range(self.nr_agents):
            # learning rate should anneal when going through more depth
            self.actor_optimizers.append(optim.Adam(self.actor.parameters(), lr = lr_actor - 0.75 * (((self.nr_agents - 1) - agent_idx) / (self.nr_agents - 1))))
            self.critic1_optimizers.append(optim.Adam(self.critic1.parameters(), lr = lr_critic - 0.75 * (((self.nr_agents - 1) - agent_idx) / (self.nr_agents - 1))))
            self.critic2_optimizers.append(optim.Adam(self.critic2.parameters(), lr = lr_critic - 0.75 * (((self.nr_agents - 1) - agent_idx) / (self.nr_agents - 1))))

        # make copy target critic networks which only get updated using polyak averaging
        self.critic1_targ = deepcopy(self.critic1)
        self.critic2_targ = deepcopy(self.critic2)
        # freeze parameter gradient calculation as it is not used
        for params in self.critic1_targ.parameters():
            params.requires_grad = False
        for params in self.critic2_targ.parameters():
            params.requires_grad = False

        # target entropy for automatic entropy coefficient adjustment, one per actor
                # not pytorch modules, so a normal list
        self.entropy_targs = []
        self.log_alphas = []
        self.alpha_optimizers = []
        for act_space in self.env.action_space:
            # from cleanRL
            # self.entropy_targs.append(-0.89 * torch.log(1 / torch.tensor(act_space.n)))
            self.entropy_targs.append(-torch.prod(torch.Tensor(act_space.shape).to(self.device)).item())
            # the entropy coef alpha which is to be optimized
            self.log_alphas.append(torch.ones(1, requires_grad = True, device = self.device))   # device this way otherwise leaf tensor
            self.alpha_optimizers.append(torch.optim.Adam([self.log_alphas[-1]], lr = lr_critic))   # shares critic lr
            
    def createPaddedActionInput(self, actions, agent_idx):
        """
        Creates appropriate input actions for parameter sharing networks.
        Adds an one-hot ID vector and pads the remaining empty input with 0s.

        returns:
            [A_1, ..., A_i-1, A_i, ..., A_m, one_hot_id],
            where [A_i, ..., A_m] = 0
        """
        one_hot_id = self.agent_ids[agent_idx]
    
        padding_len = self.global_act_plus_id_size - actions.shape[1] - one_hot_id.shape[0]
        # pad input
        padded_input = functional.pad(actions, (0, padding_len), value = 0)
        # reshape id vector into tensor we can cat
        one_hot_id = one_hot_id.unsqueeze(0).expand(padded_input.shape[0], -1)
        # add id vector
        padded_input = torch.cat([padded_input, one_hot_id], dim = 1)

        return padded_input

    def learn(self):
        """Learn the policy by backpropagation over the critics, and actor network.

        next actions come from the policy, previous actions and states from buffer
        Q values are the previous states and policy actions 
            Q^pi(a,s) = r + gamma * (Q^pi(a',s') - alpha * log pi(a', s')), where a' ~ pi(*|'s)
        with a Q network mean squared bellman error loss function of
            L(params, D) = E[(Q_params (s, a) - y(r, s', d))^2]
        where y target for Q is:
            y(r, s', done) = r + gamma * (1-done)(min(Q_1(s',a'), Q_2(s',a')) - alpha * log pi(a', s'))
        
        To get the policy loss we optimize:
            max(params) E[min(Q_1(s',a'), Q_2(s',a')) - alpha * log * pi(a^reparam(s, noise)|s)]
        which means we optimize for the loss:
            loss = (Q(a,s) - (r + gamma * (1-done)(min(Q_1(s',a'), Q_2(s',a')) - alpha * log pi(a', s')))

        The entropy coefficient alpha is automatically adjusted towards the optimal value by solving for:
            alpha*_t = arg min(alpha_t) E[-alpha_t * log policy(a_t|s_t; alpha_t) - alpha_t * entropy_target]
        """ 
        # buffer not full enough
        if self.replay_buffer.buffer_index < self.batch_size:
            # return status 0
            return 0, None, None, None, None, None
        
        # logging lists for return of the function
        loss_pi_list = []
        loss_Q_list = []
        logp_list = []
        alpha_list = []
        alpha_loss_list = []
        
        # sample from buffer 
        #   list of batches (one for each agent, same indices out of buffer and therefore same multi-agent transition)
        obs_list, replay_act_list, rewards_list, next_obs_list, dones_list = self.replay_buffer.sample()

        # prepare tensors
        obs = [torch.tensor(obs, dtype=torch.float32).to(self.device) for obs in obs_list]
        next_obs = [torch.tensor(next_obs, dtype=torch.float32).to(self.device) for next_obs in next_obs_list]
        replay_act = [torch.tensor(actions, dtype=torch.float32).to(self.device) for actions in replay_act_list]
        rewards = [torch.tensor(rewards, dtype=torch.float32).to(self.device) for rewards in rewards_list]
        dones = [torch.tensor(dones, dtype=torch.int32).to(self.device) for dones in dones_list]
        
        with torch.autograd.set_detect_anomaly(True):
            # combi set of states and actions from all agents for the critics (shape = (batch, total_obs)
            obs_set = torch.cat(obs, dim = 1)
            next_obs_set = torch.cat(next_obs, dim = 1)
            replay_act_set = torch.cat(replay_act, dim = 1)

            # We need, for agent i:
            # seq_a_im1    = tensor[(A_1, ... , A_i-1)]
            # seq_a_i      = tensor[(A_1, ... , A_i)]
            # seq_a_ip1    = tensor[(A_1, ... , A_i+1)]
            seq_a_im1 = torch.empty((replay_act[0].shape[0], 0), dtype = torch.float32)

            # SEQUENTIALLY update shared critics/actor for each agent
            for agent_idx in range(self.nr_agents):
                # get current obs policy action
                # input is [S, A_1, ..., A_i-1] 
                seq_pi_input = torch.cat([obs_set, self.createPaddedActionInput(seq_a_im1, agent_idx)], dim = 1)
                # get current obs action and logp
                a_i, logp_i, probs_i = self.actor.action_distr_sample(seq_pi_input)

                with torch.no_grad():
                    # for all in sequence except for the last we need the next in sequence action
                    if agent_idx < (self.nr_agents - 1):
                        # action of next actor
                        seq_a_i = torch.column_stack((seq_a_im1.detach(), a_i))
                        # add S to seqential A, input = [S, padded_seqA, one_hot_id]
                        seq_pi_input = torch.cat([obs_set, self.createPaddedActionInput(seq_a_i, agent_idx)], dim = 1)
                        a_ip1, logp_ip1, probs_ip1 = self.actor.action_distr_sample(seq_pi_input)
                        
                        # create sequential action tensor
                        # seq_a_ip1 = torch.column_stack((seq_a_i.detach(), a_ip1))

                    # for the last in sequence we need the action the first actor predicts on the next state
                    else:
                        # action of first actor on new state
                        seq_pi_input = torch.cat([next_obs_set, self.createPaddedActionInput(torch.empty((replay_act[0].shape[0], 0)), 0)], dim = 1)
                        # get action
                        a_1next, logp_1next, probs_1next = self.actor.action_distr_sample(seq_pi_input)


                # ALPHA GRADIENT
                alpha_loss = (probs_i.detach() * -(self.log_alphas[agent_idx].exp() * (logp_i.detach() + self.entropy_targs[agent_idx]))).mean()
                # backprop + grad descent step
                self.alpha_optimizers[agent_idx].zero_grad()        
                alpha_loss.backward()
                self.alpha_optimizers[agent_idx].step()
                # get current/next alpha
                alpha_i = torch.exp(self.log_alphas[agent_idx]).detach()
                if agent_idx < (self.nr_agents - 1):
                    alpha_ip1 = torch.exp(self.log_alphas[agent_idx + 1]).detach()
                else:
                    alpha_0 = torch.exp(self.log_alphas[0]).detach()

                # CRITIC GRADIENT
                # add action to sequential actions for seq_input to Q-func
                seq_Q_input = self.createPaddedActionInput(seq_a_im1, agent_idx)
                q1 = self.critic1.forward(torch.column_stack((obs_set, seq_Q_input)))
                q2 = self.critic2.forward(torch.column_stack((obs_set, seq_Q_input)))

                # For the bellman target, we do not need grad
                with torch.no_grad():
                    # target q values are from the next agent in sequence, so we use input actions [A_1, ..., A_i+1]
                    #   with one hot id of the next critic in sequence
                    if agent_idx < (self.nr_agents - 1):
                        seq_Q_targ_input = self.createPaddedActionInput(seq_a_i, agent_idx + 1)
                        q1_targ = self.critic1_targ.forward(torch.column_stack((obs_set, seq_Q_targ_input)))
                        q2_targ = self.critic2_targ.forward(torch.column_stack((obs_set, seq_Q_targ_input)))

                        # Clipped double Q-learning trick
                        # We can use the action probs to estimate the expectation for discrete action spaces (from cleanRL)
                        q_targ =  (probs_ip1 * (torch.minimum(q1_targ, q2_targ) - alpha_ip1.unsqueeze(1) * logp_ip1)).sum(dim = 1)
                        # adapted bellman target for discrete actions
                        bellman = rewards[agent_idx + 1] + self.gamma * (1 - dones[agent_idx + 1]) * q_targ
                        
                    # The target for the final critic in the sequence is the normal temporal difference (next state + next state action)
                    # But, we take the first critic in the sequence to calculate this target.
                    else:
                        seq_Q_targ_input = self.createPaddedActionInput(torch.empty((replay_act[0].shape[0], 0)), 0)
                        # So: Q target for the last critic in sequence is on the next state policy action Q-value, predicted by the first actor and critic.
                        q1_targ = self.critic1_targ.forward(torch.column_stack((next_obs_set, seq_Q_targ_input)))
                        q2_targ = self.critic2_targ.forward(torch.column_stack((next_obs_set, seq_Q_targ_input)))

                        # Clipped double Q-learning trick
                        # We can use the action probs to estimate the expectation for discrete action spaces (from cleanRL)
                        q_targ =  (probs_1next * (torch.minimum(q1_targ, q2_targ) - alpha_0.unsqueeze(1) * logp_1next)).sum(dim = 1)
                        # adapted bellman target for discrete actions
                        bellman = rewards[0] + self.gamma * (1 - dones[0]) * q_targ
                
                # we need to only use Q vals of actions we chose/sampled
                q1 = q1.gather(1, replay_act[agent_idx].long()).view(-1)
                q2 = q2.gather(1, replay_act[agent_idx].long()).view(-1)

                # calculate loss
                loss_Q1 = functional.mse_loss(q1, bellman)
                loss_Q2 = functional.mse_loss(q2, bellman)
                loss_Q = loss_Q1 + loss_Q2

                # backward prop + gradient step
                self.critic1_optimizers[agent_idx].zero_grad()
                self.critic2_optimizers[agent_idx].zero_grad()
                loss_Q.backward()
                self.critic1_optimizers[agent_idx].step()
                self.critic2_optimizers[agent_idx].step()

                # ACTOR GRADIENT
                # Q-values already computed (same as q1, and q2 in critic gradient)
                q_pi = torch.minimum(q1.detach(), q2.detach())
                # loss policy
                # no need for reparameterization, expectation can be calculated for discrete actions (cleanRL)
                loss_pi = (probs_i * (alpha_i.unsqueeze(1) * logp_i - q_pi.unsqueeze(1))).mean()

                # backward prop + gradient step
                self.actor_optimizers[agent_idx].zero_grad()
                loss_pi.backward()
                self.actor_optimizers[agent_idx].step()

                # set [A_1, ..., A_i-1]
                seq_a_im1 = seq_a_i.detach()

                # log values
                loss_Q_list.append(loss_Q.cpu().detach().numpy())
                loss_pi_list.append(loss_pi.cpu().detach().numpy())
                logp_list.append(logp_i.cpu().detach().numpy().mean())
                alpha_list.append(alpha_i.cpu().detach().numpy()[0])
                alpha_loss_list.append(alpha_loss.cpu().detach().numpy())


        # Polyak averaging update
        with torch.no_grad():
            for (p1, p2, p1_targ, p2_targ) in zip(self.critic1.parameters(),
                                                  self.critic2.parameters(),
                                                  self.critic1_targ.parameters(),
                                                  self.critic2_targ.parameters()):
                # critic1
                p1_targ.data *= self.polyak
                p1_targ.data += ((1 - self.polyak) * p1.data)
                # critic2
                p2_targ.data *= self.polyak
                p2_targ.data += ((1 - self.polyak) * p2.data)

            
        # reutrns policy loss, critic loss, policy entropy, alpha, alpha loss
        return 1, np.array(loss_pi_list), np.array(loss_Q_list), np.array(logp_list), np.array(alpha_list), np.array(alpha_loss_list)

    def get_action(self, observations, reparameterize = True, deterministic = False):
        # get actor action
        action_list = []
        with torch.no_grad():
            # get global observation
            global_obs = torch.cat([torch.tensor(obs, dtype = torch.float32).unsqueeze(0).to(self.device) for obs in observations], dim = 1)

            # sequential actions used as seq_input in succeeding actors
            seq_acts = torch.empty((1, 0))
            for agent_idx in range(self.nr_agents):
                
                # seq_input is observations plus preceding actor actions
                seq_input = torch.cat([global_obs, self.createPaddedActionInput(seq_acts, agent_idx)], dim = 1)
                # sample action from policy
                actions, _, _ = self.actor.action_distr_sample(seq_input, reparameterize, deterministic)
                    
                # add to next seq_input
                seq_acts = torch.cat([seq_acts, actions], dim = 1)

                # add to list
                # convert from one hot to integer if needed
                action_list.append(torch.argmax(actions).item())

        return action_list
    
    def train(self, nr_steps, max_episode_len = -1, warmup_steps = 10000, learn_delay = 1000, learn_freq = 50, learn_weight = 50, 
              checkpoint = 100000, save_dir = "models"):
        """Train the SAC agent.

        Args:
            nr_steps (int): The number steps to train the agent
            max_episode_len (int, optional): The max episode length. Defaults to: run environment untill done signal is given.
            warmup_steps (int, optional): Amount of steps the actions are drawn from uniform distribution at the start of training. 
                Defaults to 10000.
            learn_delay (int, optional): Amount of steps before the agent starts learning. Defaults to 1000.
            learn_freq (int, optional): The frequency in steps that the agent undergoes one learning iteration. Defaults to 50.
            learn_weight (int, optional): The amount of gradient descent steps per learning iteration. Defaults to 50.
        """
        # reset env
        obs, _ = self.env.reset()

        # episode and epsiode len count
        ep = 0
        ep_steps = 0
        # steps learned per episode count (for avg)
        ep_learn_steps = 0
        # sum of log values for each ep
        ep_rew_sum = np.zeros(self.nr_agents)
        ep_aloss_sum = np.zeros(self.nr_agents)
        ep_closs_sum = np.zeros(self.nr_agents)
        ep_alpha_sum = np.zeros(self.nr_agents)
        ep_alphaloss_sum = np.zeros(self.nr_agents)
        ep_entr_sum = np.zeros(self.nr_agents)

        for step in range(nr_steps):
            # sample action (uniform sample for warmup)
            if step < warmup_steps:
                action = [act_space.sample() for act_space in self.env.action_space]
            else: 
                # get action
                action = self.get_action(obs)

            # transition
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # reward addition to total sum
            np.add(ep_rew_sum, reward, out = ep_rew_sum)

            # set done to false if signal is because of time horizon (spinning up)
            if ep_steps == max_episode_len:
                done = False

            # add transition to replay buffer
            self.replay_buffer.add_transition(obs, action, reward, next_obs, done)

            # observation update
            obs = next_obs

            # done or max 
            if (any(done) or all(truncated) or ep_steps == max_episode_len):
                ep += 1

                # avg losses and entropy
                if (ep_learn_steps > 0):
                    avg_actor_loss = ep_aloss_sum / ep_learn_steps
                    avg_critic_loss = ep_closs_sum / ep_learn_steps
                    avg_policy_entr = ep_entr_sum / ep_learn_steps
                    avg_alpha = ep_alpha_sum / ep_learn_steps
                    avg_alpha_loss = ep_alphaloss_sum / ep_learn_steps
                    # save logs 
                    logs = {"avg_actor_loss": avg_actor_loss,
                            "avg_critic_loss": avg_critic_loss,
                            "avg_alpha_loss": avg_alpha_loss,
                            "avg_alpha": avg_alpha,
                            "avg_policy_entr": avg_policy_entr}
                    self.logger.log(logs, step, group = "train")
                # log reward seperately
                rollout_log = {"reward_sum": ep_rew_sum,
                               "ep_steps": ep_steps}
                self.logger.log(rollout_log, step, "rollout")
                
                # NOTE: for now like this for citylearn additional logging, should be in wrapper or something
                if self.citylearn:
                    self.logger.log_custom_reward_values(step)

                # add info to progress bar
                if (ep % 50 == 0):
                    print("[Episode {:d} total reward: ".format(ep) + str(ep_rew_sum) + "] ~ ")
                # pbar.set_description("[Episode {:d} mean reward: {:0.3f}] ~ ".format(ep, ', '.join(avg_rew)))
                
                # reset
                obs, _ = self.env.reset()
                # reset logging info
                ep_steps = 0
                ep_learn_steps = 0
                ep_rew_sum = np.zeros(self.nr_agents)
                ep_aloss_sum = np.zeros(self.nr_agents)
                ep_closs_sum = np.zeros(self.nr_agents)
                ep_entr_sum = np.zeros(self.nr_agents)
                ep_alpha_sum = np.zeros(self.nr_agents)
                ep_alphaloss_sum = np.zeros(self.nr_agents)

            # learn
            if step > learn_delay and step % learn_freq == 0:
                for _ in range(learn_weight):
                    # learning step
                    status, loss_actor, loss_critic, policy_entropy, alpha, loss_alpha = self.learn()

                    # if buffer full enough for a batch
                    if status:
                        # keep track for logs
                        ep_learn_steps += 1
                        np.add(ep_aloss_sum, loss_actor, out = ep_aloss_sum)
                        np.add(ep_closs_sum, loss_critic, out = ep_closs_sum)
                        np.add(ep_entr_sum, policy_entropy, out = ep_entr_sum)
                        np.add(ep_alpha_sum, alpha, out = ep_alpha_sum)
                        np.add(ep_alphaloss_sum, loss_alpha, out = ep_alphaloss_sum)
                        
            # checkpoint
            if (step % checkpoint == 0):
                self.actor.save(save_dir, "actor" + "_" + str(step))
                self.critic1.save(save_dir, "critic1" + "_" + str(step))
                self.critic2.save(save_dir, "critic2" + "_" + str(step))
                self.critic1_targ.save(save_dir, "critic1_targ" + "_" + str(step))
                self.critic2_targ.save(save_dir, "critic2_targ" + "_" + str(step))

            # finally, step increment
            ep_steps += 1
