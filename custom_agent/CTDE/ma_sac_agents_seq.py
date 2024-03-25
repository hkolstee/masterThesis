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
        
        # one-hot identity per agent
        self.agent_ids = functional.one_hot(torch.tensor(range(self.nr_agents))).float()

        # global obs and action for centralized critics
        obs_size_global = sum([obs.shape[0] for obs in self.env.observation_space])
        act_size_global = sum([act.shape[0] for act in self.env.action_space])

        # initialize actor, critics and critic targs
        self.actors = nn.ModuleList()
        self.critics1 = nn.ModuleList()
        self.critics2 = nn.ModuleList()
        self.critics1_targ = nn.ModuleList()
        self.critics2_targ = nn.ModuleList()

        # policies according to method as described in: https://arxiv.org/pdf/1910.00120.pdf
        #   Simply: sequentially calculate policies, while conditioning next policy calculation on action result of last.
        #           Done sequentially per agent, where we do not care about order.
        # two centralized critics per agent (two for stable learning), also sequential for update rules as in https://arxiv.org/pdf/1910.00120.pdf
        #   gets combination set of all obs and actions of all agents, but is only used while training.
        seq_act = 0
        for (obs_space, act_space) in zip(self.env.observation_space, self.env.action_space):
            # actor
            self.actors.append(Actor(lr_actor, obs_space.shape[0] + seq_act, act_space.shape[0], act_space.low, act_space.high, layer_sizes))
            # global centralized critics
            self.critics1.append(Critic(lr_critic, obs_size_global, act_space.shape[0] + seq_act, layer_sizes))
            self.critics2.append(Critic(lr_critic, obs_size_global, act_space.shape[0] + seq_act, layer_sizes))
            seq_act += act_space.shape[0]

        # make copy target critic networks which only get updated using polyak averaging
        self.critics1_targ = deepcopy(self.critics1)
        self.critics2_targ = deepcopy(self.critics2)
        # freeze parameter gradient calculation as it is not used
        for agent_idx in range(self.nr_agents):
            for params in self.critics1_targ[agent_idx].parameters():
                params.requires_grad = False
            for params in self.critics2_targ[agent_idx].parameters():
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

    def getSequentialAct(self, observations):
        """Sequentially get values by introducing the last output as input into the next network forward"""
        # initialize as empty(batch, 0)
        seq_acts = torch.empty((observations[0].shape[0], 0))
        seq_act_list = []
        logps_list = []
        for agent_idx, obs in enumerate(observations):
            # get action and logp
            stacked = torch.cat([obs, seq_acts], dim = 1)
            act, logp = self.actors[agent_idx].normal_distr_sample(stacked)
            # prepare for next input
            #   detach from graph such that we do not backward through all 
            #   previous actors on gradient update
            seq_acts = torch.cat([seq_acts, act.detach()], dim = 1)
            # append
            seq_act_list.append(act)
            logps_list.append(logp)

        return seq_act_list, logps_list
    
    def getSequentialQs(self, obs, acts, next_obs, next_acts):
        """
        Get Q-values by iteratively introducing agent actions sequentially.

        For m-agents:
        Q_i(S, A_1, ... , A_i),             for i = 1, ... , m

        Qtarg_1(S', A'_1)
        Qtarg_i(S, A_1, ... , A_i)       for i = 2, ... , m

        As Q_m gets trained with Qtarg_1 with next state information as target,
        but Q_1 , ... ,Q_m-1 get trained with Qtarg_2, ... , Qtarg_m as target.  
        
        """
        seq_q1, seq_q2, seq_q1_targ, seq_q2_targ = [],[],[],[]
        
        # current sequential actions
        seq_acts = torch.empty((acts[0].shape[0], 0))
        
        for agent_idx in range(self.nr_agents):
            # add action for input
            seq_acts = torch.column_stack((seq_acts, acts[agent_idx]))
            
            # targ no grad
            with torch.no_grad():
                if agent_idx == 0:
                    q1_targ = self.critics1_targ[agent_idx].forward(next_obs, next_acts[0])
                    q2_targ = self.critics2_targ[agent_idx].forward(next_obs, next_acts[0])
                else:
                    q1_targ = self.critics1_targ[agent_idx].forward(obs, seq_acts)
                    q2_targ = self.critics2_targ[agent_idx].forward(obs, seq_acts)
            # with grad
            q1 = self.critics1[agent_idx].forward(obs, seq_acts)
            q2 = self.critics2[agent_idx].forward(obs, seq_acts)

            # add to list
            seq_q1.append(q1)
            seq_q2.append(q2)
            seq_q1_targ.append(q1_targ)
            seq_q2_targ.append(q2_targ)

        return seq_q1, seq_q2, seq_q1_targ, seq_q2_targ

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
        log_prob_list = []
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
        # get set of current policy actions for next observation 
        with torch.no_grad():
            pi_next_actions, logp_next_obs = self.getSequentialAct(next_obs)
        
        # set of current policy actions for current observation
        # with grad 
        pi_actions, logp = self.getSequentialAct(obs)
        # create entire set, without grad
        pi_act_nograd = [act.detach() for act in pi_actions]
        
        alphas = []
        for agent_idx in range(self.nr_agents):
            # FIRST GRADIENT: automatic entropy coefficient tuning (alpha)
            #   optimal alpha_t = arg min(alpha_t) E[-alpha_t * log policy(a_t|s_t; alpha_t) - alpha_t * entropy_target]
            # we detach because otherwise we backward through the graph of previous calculations using log_prob
            #   which also raises an error fortunately, otherwise I would have missed this
            alpha_loss = -(self.log_alphas[agent_idx].exp() * (logp[agent_idx].detach() + self.entropy_targs[agent_idx])).mean()

            # backward prop + gradient step
            self.alpha_optimizers[agent_idx].zero_grad()        
            alpha_loss.backward()
            self.alpha_optimizers[agent_idx].step()

            # get current alpha
            alphas.append(torch.exp(self.log_alphas[agent_idx].detach()))

            # log values
            alpha_loss_list.append(alpha_loss.cpu().detach().numpy())

        # CRITIC GRADIENT            
        # get Q values
        q1, q2, q1_targ, q2_targ = self.getSequentialQs(obs_set, replay_act, next_obs_set, pi_next_actions)
        
        with torch.no_grad():
            # get min q_targ network values
            q_targ = []
            for targ_val1, targ_val2 in zip(q1_targ, q2_targ):
                q_targ.append(torch.minimum(targ_val1, targ_val2))

            # gradients for all but last critic in sequence is the normal target but substituted by next critic in sequence
            #   with current state and actions
            bellman = []
            for agent_idx in range(self.nr_agents - 1):
                # we take the next in sequence
                idx = agent_idx + 1
                # bellman target on next in sequence q_targ network for agents 1, ..., m-1
                bellman_targ = rewards[idx] + self.gamma * (1 - dones[idx]) * (q_targ[idx] - alphas[idx] * logp_next_obs[idx])
                bellman.append(bellman_targ)
            # last critic is compared to first in sequence, with next state (which is used in Q value calculation, see getSequentialQs())
            bellman_targ = rewards[0] + self.gamma * (1 - dones[0]) * (q_targ[0] - alphas[0] * logp_next_obs[0])
            bellman.append(bellman_targ)
        
        # loss for each critic (except last) based on comparison to next in sequence
        for agent_idx in range(self.nr_agents):
            loss_critic1 = functional.mse_loss(q1[agent_idx], bellman[agent_idx])
            loss_critic2 = functional.mse_loss(q2[agent_idx], bellman[agent_idx])
            loss_critic = loss_critic1 + loss_critic2

            # zero gradient
            self.critics1[agent_idx].optimizer.zero_grad()
            self.critics2[agent_idx].optimizer.zero_grad()
            # backward prop
            loss_critic.backward()
            # step
            self.critics1[agent_idx].optimizer.step()
            self.critics2[agent_idx].optimizer.step()

            # log loss
            loss_critic_list.append(loss_critic.cpu().detach().numpy())    

        for agent_idx in range(self.nr_agents):
            # ACTOR GRADIENT
            # first freeze critic gradient calculation to save computation
            for params in self.critics1[agent_idx].parameters():
                params.requires_grad = False
            for params in self.critics2[agent_idx].parameters():
                params.requires_grad = False

            # compute Q-values
            #   Sequence of actions in sequential critic, where only the gradient graph of the current policy persists
            if agent_idx == 0:
                seq_actions = pi_actions[0]
            else:
                seq_actions = torch.column_stack((torch.column_stack(pi_act_nograd[:agent_idx]), pi_actions[agent_idx]))
            # forward
            q1_policy = self.critics1[agent_idx].forward(obs_set, seq_actions)
            q2_policy = self.critics2[agent_idx].forward(obs_set, seq_actions)
            
            # take min of these two 
            #   = clipped Q-value for stable learning, reduces overestimation
            q_policy = torch.minimum(q1_policy, q2_policy)
            # entropy regularized loss
            loss_policy = (alphas[agent_idx] * logp[agent_idx] - q_policy).mean()

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

            # log policy loss
            loss_policy_list.append(loss_policy.cpu().detach().numpy())

        # log last values
        for agent_idx in range(self.nr_agents):
            log_prob_list.append(logp[agent_idx].cpu().detach().numpy().mean())
            alpha_list.append(alphas[agent_idx].cpu().detach().numpy()[0])
            
        # reutrns policy loss, critic loss, policy entropy, alpha, alpha loss
        return 1, np.array(loss_policy_list), np.array(loss_critic_list), np.array(log_prob_list), np.array(alpha_list), np.array(alpha_loss_list)

    def get_action(self, observations, reparameterize = True, deterministic = False):
        # get actor action
        action_list = []
        seq_acts = torch.empty((observations[0].shape[0], 0))
        with torch.no_grad():
            for actor, obs in zip(self.actors, observations):
                # make tensor and send to device
                obs = torch.tensor(obs, dtype = torch.float32).unsqueeze(0).to(self.device)
                # input is observations plus preceding actor actions
                stacked = torch.cat([obs, seq_acts], dim = 1)
                # sample action from policy
                actions, _ = actor.normal_distr_sample(stacked, reparameterize, deterministic)
                # add to next input
                seq_acts = torch.cat([seq_acts, actions], dim = 1)

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



