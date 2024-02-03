import os
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as functional

# add folder to python path for relative imports
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

from ..SAC_components.ma_replay_buffer import MultiAgentReplayBuffer
from ..SAC_components.critic import Critic
from ..SAC_components.actor import Actor

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
                 ):
        self.env = env
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.nr_agents = len(env.action_space)
        
        # initialize device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # initialize replay buffer
        obs_size_list = [obs.shape for obs in self.env.observation_space]
        act_size_list = [act.shape for act in self.env.action_space]
        self.replay_buffer = MultiAgentReplayBuffer(buffer_max_size, obs_size_list, act_size_list, batch_size)
        
        # one local actor per agent, gets local observations only
        self.actors = nn.ModuleList()
        for (obs_space, act_space) in zip(self.env.observation_space, self.env.action_space):
            self.actors.append(Actor(lr_actor, obs_space.shape[0], act_space.shape[0], act_space.low, act_space.high, layer_sizes))
        
        # two global critics per agent (two for stable learning), 
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
        self.alphas = []
        self.alpha_optimizers = []
        for act_space in self.env.action_space:
            self.entropy_targs.append(torch.tensor(-np.prod(act_space.shape[0]), dtype=torch.float32).to(self.device))
            # the entropy coef alpha which is to be optimized
            self.alphas.append(torch.ones(1, requires_grad = True, device = self.device))
            self.alpha_optimizers.append(torch.optim.Adam([self.alphas[-1]], lr = lr_critic))   # shares critic lr

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
            return
        
        # logging lists for return of the function
        loss_policy_list = []
        loss_critic_list = []
        log_prob_list = []
        alpha_list = []
        alpha_loss_list = []
        
        # sample from buffer 
        #   list of batches (one for each agent, same indices out of buffer and therefore same multi-agent transition)
        obs_list, replay_actions_list, rewards_list, next_obs_list, dones_list = self.replay_buffer.sample()

        # prepare tensors
        observations = [torch.tensor(obs, dtype=torch.float32).to(self.device) for obs in obs_list]
        next_observations = [torch.tensor(next_obs, dtype=torch.float32).to(self.device) for next_obs in next_obs_list]
        replay_actions = [torch.tensor(actions, dtype=torch.float32).to(self.device) for actions in replay_actions_list]
        rewards = [torch.tensor(rewards, dtype=torch.float32).to(self.device) for rewards in rewards_list]
        dones = [torch.tensor(dones, dtype=torch.int32).to(self.device) for dones in dones_list]

        # combi set of states and actions from all agents for the critics (shape = (batch, total_obs)
        obs_set = torch.concat(observations, dim = 1)
        next_obs_set = torch.concat(next_observations, dim = 1)
        replay_act_set = torch.concat(replay_actions, dim = 1)
        # we also need a combi set of the actions of the policy (on both pre-transition observations and next observations)
        # when sampling the next observations we don't track the gradient as it is for the critics gradient updates
        with torch.no_grad():
            policy_act_next_observations, log_prob_next_observations = \
                zip(*[actor.normal_distr_sample(next_obs) for (actor, next_obs) in zip(self.actors, next_observations)])
            policy_act_prev_observations, log_prob_prev_observations = \
                zip(*[actor.normal_distr_sample(obs) for (actor, obs) in zip(self.actors, observations)])

            # make last sets
            policy_action_set_next_obs = torch.concat(policy_act_next_observations, dim = 1)
            policy_action_set_prev_obs = torch.concat(policy_act_prev_observations, dim = 1)
        
        # zip together all needed for one agent gradient updates
        # NOTE: Perhaps test efficiency of this method, 
        for (actor, critic1, critic2, critic1_targ, critic2_targ, obs, rew, done, alpha, alpha_optim, entropy_targ, 
            log_prob_prev_obs_temp, log_prob_next_obs) \
            in zip(self.actors, self.critics1, self.critics2, self.critics1_targ, self.critics2_targ, observations,
                rewards, dones, self.alphas, self.alpha_optimizers, self.entropy_targs, log_prob_prev_observations, 
                log_prob_next_observations):
            
            # CRITIC GRADIENT
            # first reset gradients
            critic1.optimizer.zero_grad()
            critic2.optimizer.zero_grad()

            # These Q values are the left hand side of the loss function
            q1_buffer = critic1.forward(obs_set, replay_act_set)
            q2_buffer = critic2.forward(obs_set, replay_act_set)

            # For the RHS of the loss function (Approximation of Bellman equation with (1 - d) factor):
            with torch.no_grad():
                # target q values
                q1_policy_targ = critic1_targ.forward(next_obs_set, policy_action_set_next_obs)
                q2_policy_targ = critic2_targ.forward(next_obs_set, policy_action_set_next_obs)
                # clipped double Q trick
                q_targ = torch.min(q1_policy_targ, q2_policy_targ)
                # Bellman approximation
                bellman = rew + self.gamma * (1 - done) * (q_targ - alpha * log_prob_next_obs)
            
            # loss is MSEloss over Bellman error (MSBE = mean squared bellman error)
                # NOTE: SOME IMPLEMENTATIONS USE "0.5 *" FOR EACH, IDK WHAT IS BEST 
            loss_critic1 = torch.pow((q1_buffer - bellman), 2).mean()
            loss_critic2 = torch.pow((q2_buffer - bellman), 2).mean()
            loss_critic = loss_critic1 + loss_critic2

            # backward prop
            loss_critic.backward()
            # step down gradient
            critic1.optimizer.step()
            critic2.optimizer.step()

            # ACTOR GRADIENT
            # first freeze critic gradient calculation to save computation
            for params in critic1.parameters():
                params.requires_grad = False
            for params in critic2.parameters():
                params.requires_grad = False

            # reset actor gradient
            actor.optimizer.zero_grad()

            # compute current policy action for pre-transition observation
            # policy_actions_prev_obs, log_prob_prev_obs = actor.normal_distr_sample(obs)
            _, log_prob_prev_obs = actor.normal_distr_sample(obs)
            # compute Q-values
            q1_policy = critic1.forward(obs_set, policy_action_set_prev_obs)
            q2_policy = critic2.forward(obs_set, policy_action_set_prev_obs)
            # take min of these two 
            #   = clipped Q-value for stable learning, reduces overestimation
            q_policy = torch.min(q1_policy, q2_policy)
            # entropy regularized loss
            loss_policy = (alpha * log_prob_prev_obs - q_policy).mean()

            # backward prop
            loss_policy.backward(retain_graph = True)
            # step down gradient
            actor.optimizer.step()

            # unfreeze critic gradients
            for params in critic1.parameters():
                params.requires_grad = True
            for params in critic2.parameters():
                params.requires_grad = True

            # LAST GRADIENT: automatic entropy coefficient tuning (alpha)
            #   optimal alpha_t = arg min(alpha_t) E[-alpha_t * log policy(a_t|s_t; alpha_t) - alpha_t * entropy_target]
            alpha_optim.zero_grad()
            # we detach because otherwise we backward through the graph of previous calculations using log_prob
            #   which also raises an error fortunately, otherwise I would have missed this
            alpha_loss = (-alpha * log_prob_prev_obs.detach() - alpha * entropy_targ).mean()
            alpha_loss.backward()
            alpha_optim.step()

            # Polyak averaging update
            with torch.no_grad():
                for (p1, p2, p1_targ, p2_targ) in zip(critic1.parameters(),
                                                    critic2.parameters(),
                                                    critic1_targ.parameters(),
                                                    critic2_targ.parameters()):
                    # critic1
                    p1_targ.data *= self.polyak
                    p1_targ.data += ((1 - self.polyak) * p1.data)
                    # critic2
                    p2_targ.data *= self.polyak
                    p2_targ.data += ((1 - self.polyak) * p2.data)
            
            # log each agent's values
            loss_policy_list.append(loss_policy.cpu().detach().numpy())
            loss_critic_list.append(loss_critic.cpu().detach().numpy())
            log_prob_list.append(log_prob_prev_obs.cpu().detach().numpy().mean())
            alpha_list.append(alpha.cpu().detach().numpy()[0])
            alpha_loss_list.append(alpha_loss.cpu().detach().numpy())
            
        # reutrns policy loss, critic loss, policy entropy, alpha, alpha loss
        return np.array(loss_policy_list), np.array(loss_critic_list), np.array(log_prob_list), np.array(alpha_list), np.array(alpha_loss_list)

    def get_action(self, observations, reparameterize = False, deterministic = False):
        # get actor action
        action_list = []
        with torch.no_grad():
            for actor, obs in zip(self.actors, observations):
                # make tensor and send to device
                obs = torch.tensor(obs, dtype = torch.float32).unsqueeze(0).to(self.device)
                # sample action from policy
                actions, _ = actor.normal_distr_sample(obs, reparameterize, deterministic)
                # add to list
                action_list.append(actions.cpu().detach().numpy()[0])

        return action_list
    
    def train(self, nr_steps, max_episode_len = -1, warmup_steps = 10000, learn_delay = 1000, learn_freq = 50, learn_weight = 50):
        """Train the SAC agent.

        Args:
            nr_steps (int): The number steps to train the agent
            max_episode_len (int, optional): The max episode length. Defaults to: run environment untill done signal is given.
            warmup_steps (int, optional): Amount of steps the actions are drawn from uniform distribution at the start of training. 
                Defaults to 10000.
            learn_delay (int, optional): Amount of steps before the agent starts learning. Defaults to 1000.
            learn_freq (int, optional): The frequency in steps that the agent undergoes one learning iteration. Defaults to 50.
            learn_weight (int, optional): The amount of gradient descent steps per learning iteration. Defaults to 50.

        Returns:
            logs (list([float, float, float, float, int])): List of episode average reward, episode average actor loss, 
                episode average critic loss, episode average policy entropy, and total steps.
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

        # logging values
        logs = pd.DataFrame(index = None, columns = ["step", 
                                                     "avg_reward", 
                                                     "avg_actor_loss", 
                                                     "avg_critic_loss", 
                                                     "avg_policy_entr",
                                                     "avg_alpha",
                                                     "avg_alpha_loss"])
        
        for step in (pbar := tqdm(range(nr_steps))):
            # sample action (uniform sample for warmup)
            if step < warmup_steps:
                action = [act_space.sample() for act_space in self.env.action_space]
            else: 
                # obs from env is not a tensor
                # obs = torch.tensor(obs, dtype = torch.float32)
                # get action
                action = self.get_action(obs)

            # transition
            next_obs, reward, done, info = self.env.step(action)
            # print(reward)
            
            # step increment 
            ep_steps += 1
            # reward addition to total sum
            np.add(ep_rew_sum, reward, out = ep_rew_sum)

            # add transition to buffer
            self.replay_buffer.add_transition(obs, action, reward, next_obs, done)

            # observation update
            obs = next_obs

            # done or max steps
            if (done or ep_steps == max_episode_len):
                ep += 1

                # avg reward
                avg_rew = ep_rew_sum / ep_steps
                # avg losses and entropy
                if (ep_learn_steps > 0):
                    avg_actor_loss = ep_aloss_sum / ep_learn_steps
                    avg_critic_loss = ep_closs_sum / ep_learn_steps
                    avg_policy_entr = ep_entr_sum / ep_learn_steps
                    avg_alpha = ep_alpha_sum / ep_learn_steps
                    avg_alpha_loss = ep_alphaloss_sum / ep_learn_steps
                    # save logs: 
                    logs = logs.append({"step": step, 
                                        "avg_reward": avg_rew,
                                        "avg_actor_loss": avg_actor_loss,
                                        "avg_critic_loss": avg_critic_loss,
                                        "avg_alpha_loss": avg_alpha_loss,
                                        "avg_alpha": avg_alpha,
                                        "avg_policy_entr": avg_policy_entr}, ignore_index = True)
                else:
                    logs = logs.append({"step": step, "avg_reward": avg_rew}, ignore_index = True)

                # add info to progress bar
                pbar.set_description("[Episode {:d} mean reward: ".format(ep) + str(avg_rew) + "] ~ ")
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
                    loss_actor, loss_critic, policy_entropy, alpha, loss_alpha = self.learn()

                    # keep track for logs
                    ep_learn_steps += 1
                    np.add(ep_aloss_sum, loss_actor, out = ep_aloss_sum)
                    np.add(ep_closs_sum, loss_critic, out = ep_closs_sum)
                    np.add(ep_entr_sum, policy_entropy, out = ep_entr_sum)
                    np.add(ep_alpha_sum, alpha, out = ep_alpha_sum)
                    np.add(ep_alphaloss_sum, loss_alpha, out = ep_alphaloss_sum)

        return logs




