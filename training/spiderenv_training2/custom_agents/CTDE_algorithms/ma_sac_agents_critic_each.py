import os
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as functional

# temporary needed
from custom_agent.CTCE.citylearn_wrapper import CityLearnWrapper
from custom_reward.custom_reward import CustomReward

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedSpaceWrapper

# add folder to python path for relative imports
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

from ..SAC_components.ma_replay_buffer import MultiAgentReplayBuffer
from ..SAC_components.critic import Critic
from ..SAC_components.actor import Actor
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
                 log_dir = "tensorboard_logs"
                 ):
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
        act_size_list = [act.shape for act in self.env.action_space]
        self.replay_buffer = MultiAgentReplayBuffer(buffer_max_size, obs_size_list, act_size_list, batch_size)
        
        # one local actor per agent, gets local observations only
        self.actors = nn.ModuleList()
        for (obs_space, act_space) in zip(self.env.observation_space, self.env.action_space):
            self.actors.append(Actor(lr_actor, obs_space.shape[0], act_space.shape[0], act_space.low, act_space.high, layer_sizes))
        
        # two centralized critics per agent (two for stable learning), 
        #   gets combination set of all obs and actions of all agents, but is only used while training
        self.critics1 = nn.ModuleList()
        self.critics2 = nn.ModuleList()
        # get global obs and action size
        obs_size_global = sum([obs.shape[0] for obs in self.env.observation_space])
        act_size_global = sum([act.shape[0] for act in self.env.action_space])
        for act_space in self.env.action_space: 
            self.critics1.append(Critic(lr_critic, obs_size_global, act_size_global, layer_sizes))
            self.critics2.append(Critic(lr_critic, obs_size_global, act_size_global, layer_sizes))

        # make copy target critic networks which only get updated using polyak averaging
        self.critics1_targ = deepcopy(self.critics1)
        self.critics2_targ = deepcopy(self.critics2)
        # freeze parameter gradient calculation as it is not used
        for critic in self.critics1_targ:
            for params in critic.parameters():
                params.requires_grad = False
        for critic in self.critics2_targ:
            for params in critic.parameters():
                params.requires_grad = False

        # target entropy for automatic entropy coefficient adjustment, one per actor
                # not pytorch modules, so a normal list
        self.entropy_targs = []
        self.log_alphas = []
        self.alpha_optimizers = []
        for act_space in self.env.action_space:
            self.entropy_targs.append(torch.tensor(-np.prod(act_space.shape[0]), dtype=torch.float32).to(self.device))
            # the entropy coef alpha which is to be optimized
            self.log_alphas.append(torch.ones(1, requires_grad = True, device = self.device))   # device this way otherwise leaf tensor
            self.alpha_optimizers.append(torch.optim.Adam([self.log_alphas[-1]], lr = lr_critic))   # shares critic lr

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
        loss_policy_list = []
        loss_critic_list = []
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

        # combi set of states and actions from all agents for the critics (shape = (batch, total_obs)
        obs_set = torch.cat(obs, dim = 1)
        next_obs_set = torch.cat(next_obs, dim = 1)
        replay_act_set = torch.cat(replay_act, dim = 1)
        # we also need a combi set of the actions of the policy (on both pre-transition observations and next observations)
        # when sampling the next observations we don't track the gradient as it is for the critics gradient updates
        with torch.no_grad():
            policy_act_next_obs, log_prob_next_obs = \
                zip(*[actor.normal_distr_sample(next_obs) for (actor, next_obs) in zip(self.actors, next_obs)])
            policy_act_next_obs_set = torch.cat(policy_act_next_obs, axis = 1)  
        
        # with grad 
        policy_act_prev_obs, log_prob_prev_obs = \
            zip(*[actor.normal_distr_sample(obs) for (actor, obs) in zip(self.actors, obs)])
        # create entire set, without grad
        policy_act_prev_obs_nograd = [act.detach() for act in policy_act_prev_obs]
        
        for agent_idx in range(self.nr_agents):
            # FIRST GRADIENT: automatic entropy coefficient tuning (alpha)
            #   optimal alpha_t = arg min(alpha_t) E[-alpha_t * log policy(a_t|s_t; alpha_t) - alpha_t * entropy_target]
            # we detach because otherwise we backward through the graph of previous calculations using log_prob
            #   which also raises an error fortunately, otherwise I would have missed this
            alpha_loss = -(self.log_alphas[agent_idx].exp() * (log_prob_prev_obs[agent_idx].detach() + self.entropy_targs[agent_idx])).mean()

            # backward prop + gradient step
            self.alpha_optimizers[agent_idx].zero_grad()        
            alpha_loss.backward()
            self.alpha_optimizers[agent_idx].step()

            # get current alpha
            alpha = torch.exp(self.log_alphas[agent_idx].detach())

            # CRITIC GRADIENT            
            # These Q values are the left hand side of the loss function
            q1_buffer = self.critics1[agent_idx].forward(obs_set, replay_act_set)
            q2_buffer = self.critics2[agent_idx].forward(obs_set, replay_act_set)
            
            # For the RHS of the loss function (Approximation of Bellman equation with (1 - d) factor):
            with torch.no_grad():
                # target q values
                q1_policy_targ = self.critics1_targ[agent_idx].forward(next_obs_set, policy_act_next_obs_set)
                q2_policy_targ = self.critics2_targ[agent_idx].forward(next_obs_set, policy_act_next_obs_set)
                # clipped double Q trick
                q_targ = torch.minimum(q1_policy_targ, q2_policy_targ)
                # Bellman approximation
                bellman = rewards[agent_idx] + self.gamma * (1 - dones[agent_idx]) * (q_targ - alpha * log_prob_next_obs[agent_idx])
            
            # loss is MSEloss over Bellman error (MSBE = mean squared bellman error)
            loss_critic1 = functional.mse_loss(q1_buffer, bellman)
            loss_critic2 = functional.mse_loss(q2_buffer, bellman)
            loss_critic = loss_critic1 + loss_critic2

            # backward prop
            self.critics1[agent_idx].optimizer.zero_grad()
            self.critics2[agent_idx].optimizer.zero_grad()
            loss_critic.backward()
            # step down gradient
            self.critics1[agent_idx].optimizer.step()
            self.critics2[agent_idx].optimizer.step()

            # ACTOR GRADIENT
            # first freeze critic gradient calculation to save computation
            for params in self.critics1[agent_idx].parameters():
                params.requires_grad = False
            for params in self.critics2[agent_idx].parameters():
                params.requires_grad = False

            # compute Q-values
            # for this we need set of actions of all agents, where the gradient graph only
            #   persists of the action of the current actor
            policy_action_set = torch.cat([act if idx == agent_idx else act_nograd for idx, (act, act_nograd) 
                            in enumerate(zip(policy_act_prev_obs, policy_act_prev_obs_nograd))], axis = 1)
            q1_policy = self.critics1[agent_idx].forward(obs_set, policy_action_set)
            q2_policy = self.critics2[agent_idx].forward(obs_set, policy_action_set)
            # take min of these two 
            #   = clipped Q-value for stable learning, reduces overestimation
            q_policy = torch.minimum(q1_policy, q2_policy)
            # entropy regularized loss
            loss_policy = (alpha * log_prob_prev_obs[agent_idx] - q_policy).mean()

            # zero grad
            self.actors[agent_idx].optimizer.zero_grad()
            # backward prop
            loss_policy.backward()
            # step down gradient
            self.actors[agent_idx].optimizer.step()

            # unfreeze critic gradients
            for params in self.critics1[agent_idx].parameters():
                params.requires_grad = True
            for params in self.critics2[agent_idx].parameters():
                params.requires_grad = True

            # Polyak averaging update
            with torch.no_grad():
                for (p1, p2, p1_targ, p2_targ) in zip(self.critics1[agent_idx].parameters(),
                                                    self.critics2[agent_idx].parameters(),
                                                    self.critics1_targ[agent_idx].parameters(),
                                                    self.critics2_targ[agent_idx].parameters()):
                    # critic1
                    p1_targ.data *= self.polyak
                    p1_targ.data += ((1 - self.polyak) * p1.data)
                    # critic2
                    p2_targ.data *= self.polyak
                    p2_targ.data += ((1 - self.polyak) * p2.data)
            
            # log each agent's values
            loss_policy_list.append(loss_policy.cpu().detach().numpy())
            loss_critic_list.append(loss_critic.cpu().detach().numpy())
            alpha_list.append(alpha.cpu().detach().numpy()[0])
            alpha_loss_list.append(alpha_loss.cpu().detach().numpy())
            
        # reutrns policy loss, critic loss, policy entropy, alpha, alpha loss
        return 1, np.array(loss_policy_list), np.array(loss_critic_list), np.array(log_prob_list), np.array(alpha_list), np.array(alpha_loss_list)

    def get_action(self, observations, reparameterize = True, deterministic = False):
        # get actor action
        action_list = []
        with torch.no_grad():
            for actor, obs in zip(self.actors, observations):
                # make tensor and send to device
                obs = torch.tensor(obs, dtype = torch.float32).unsqueeze(0).to(self.device)
                # sample action from policy
                actions, _ = actor.action_distr_sample(obs, reparameterize, deterministic)
                # add to list
                action_list.append(actions.cpu().detach().numpy()[0])

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
        obs = self.env.reset()

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
            next_obs, reward, done, info = self.env.step(action)
            
            # step increment 
            ep_steps += 1
            # reward addition to total sum
            np.add(ep_rew_sum, reward, out = ep_rew_sum)

            # set done to false if signal is because of time horizon (spinning up)
            if ep_steps == max_episode_len:
                done = False

            # add transition to buffer
            self.replay_buffer.add_transition(obs, action, reward, next_obs, done)

            # observation update
            obs = next_obs

            # done or max steps
            if (done or ep_steps == max_episode_len):
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
                reward_log = {"reward_sum": ep_rew_sum}
                self.logger.log(reward_log, step, "Reward")
                
                # NOTE: for now like this for citylearn additional logging, should be in wrapper or something
                if self.citylearn:
                    self.logger.log_custom_reward_values(step)

                # add info to progress bar
                if (ep % 50 == 0):
                    print("[Episode {:d} total reward: ".format(ep) + str(ep_rew_sum) + "] ~ ")
                # pbar.set_description("[Episode {:d} mean reward: {:0.3f}] ~ ".format(ep, ', '.join(avg_rew)))
                
                # reset
                obs = self.env.reset()
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
                for actor_idx in range(len(self.actors)):
                    self.actors[actor_idx].save(save_dir, "actor" + str(actor_idx) + "_" + str(step))
                for critic_idx in range(len(self.critics1)):
                    self.critics1[critic_idx].save(save_dir, "critic1" + str(critic_idx) + "_" + str(step))
                    self.critics2[critic_idx].save(save_dir, "critic2" + str(critic_idx) + "_" + str(step))
                for critic_targ_idx in range(len(self.critics2_targ)):
                    self.critics1_targ[critic_targ_idx].save(save_dir, "critic1_targ" + str(critic_targ_idx) + "_" + str(step))
                    self.critics2_targ[critic_targ_idx].save(save_dir, "critic2_targ" + str(critic_targ_idx) + "_" + str(step))


