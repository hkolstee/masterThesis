import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# temporary needed
# from custom_agents.utils.citylearn_wrapper import CityLearnWrapper
# from custom_reward.custom_reward import CustomReward

# from citylearn.citylearn import CityLearnEnv
# from citylearn.wrappers import NormalizedSpaceWrapper

from gymnasium import spaces

# add folder to python path for relative imports
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

from ..replay_buffers.ma_replay_buffer import MultiAgentReplayBuffer
from ..networks.critic import Critic
from ..networks.actor_discrete import DiscreteActor
from ..utils.logger import Logger

from copy import deepcopy

from tqdm import tqdm

class Agents:
    """Sequential multi-agent soft actor-critic.

    Args:
        env (_type_): _description_
        lr_actor (float, optional): _description_. Defaults to 0.0003.
        lr_critic (float, optional): _description_. Defaults to 0.0003.
        gamma (float, optional): _description_. Defaults to 0.99.
        polyak (float, optional): _description_. Defaults to 0.995.
        buffer_max_size (int, optional): _description_. Defaults to 1000000.
        batch_size (int, optional): _description_. Defaults to 256.
        layer_sizes (tuple, optional): _description_. Defaults to (256, 256).
        log_dir (str, optional): _description_. Defaults to "tensorboard_logs".
        global_observations (bool, optional): Whether the environment is modelled to return 
            the global observation for each agent, or a local observation per agent. Defaults
            to False.
    """
    def __init__(self, 
                 env, 
                 lr_actor = 0.0003, 
                 lr_critic = 0.0003, 
                 gamma = 0.99, 
                 polyak = 0.995,
                 alpha_temp_multiplier = 1,
                 buffer_max_size = 1000000,
                 batch_size = 256,
                 layer_sizes = (256, 256),
                 log_dir = "tensorboard_logs",
                 global_observations = False,
                 save_dir = "models",
                 eval_every = 25,
                 ):
        self.env = env
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.global_observations = global_observations
        self.eval_every = eval_every
        self.best_eval = -np.inf
        self.nr_agents = len(env.action_space)
        self.save_dir = save_dir
        
        # initialize device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # initialize logger
        self.logger = Logger(self.env, log_dir)
        
        # for now done like this: check if citylearn env with custom reward function for 
        #   additional logging
        # self.citylearn = isinstance(self.env.reward_function, CustomReward) \
        #                     if (isinstance(self.env, CityLearnWrapper) \
        #                     or isinstance(self.env, CityLearnEnv) \
        #                     or isinstance(self.env, NormalizedSpaceWrapper)) \
        #                     else False
        
        if isinstance(self.env.action_space[0], spaces.Box):
            # pad by min - 1
            self.padding_val = min([act_space.low.item for act_space in self.env.action_space]) - 1
        else:
            # padding values we take as - 1, 0 is lowest action possible with discrete action space
            self.padding_val = - 1.0

        # initialize replay buffer
        obs_size_list = [obs.shape for obs in self.env.observation_space]
        act_size_list = [(1,) for _ in self.env.action_space]
        self.replay_buffer = MultiAgentReplayBuffer(buffer_max_size, obs_size_list, act_size_list, batch_size)
        
        # global obs size, action sizes
        obs_sizes = [obs.shape[0] for obs in self.env.observation_space]
        self.action_sizes = [act.n for act in self.env.action_space]

        # if not all agents receive global observations from env
        if not self.global_observations:
            obs_size_global = sum(obs_sizes)
        # if they do
        else:
            obs_size_global = obs_sizes[0]

        # policies according to method as described in: https://arxiv.org/pdf/1910.00120.pdf
        #   Simply: sequentially calculate policies, while conditioning next policy calculation on action result of last.
        #           Done sequentially per agent, where we do not care about order. One actor in total for all agents, with parameter sharing.
        #           Requires homogenous agent action spaces.
        # Two centralized critics (Double Q learning), also sequential for update rules as in https://arxiv.org/pdf/1910.00120.pdf
        self.actor = DiscreteActor(lr = lr_actor,
                                   obs_size = obs_size_global + sum(self.action_sizes[:-1]),
                                   action_size = self.env.action_space[0].n,
                                   layer_sizes = layer_sizes)
        self.critic1 = Critic(lr = lr_critic,
                              obs_size = obs_size_global + sum(self.action_sizes[:-1]),
                              act_size = self.env.action_space[0].n,
                              discrete = True,
                              layer_sizes = layer_sizes)
        self.critic2 = Critic(lr = lr_critic,
                              obs_size = obs_size_global + sum(self.action_sizes[:-1]),
                              act_size = self.env.action_space[0].n,
                              discrete = True,
                              layer_sizes = layer_sizes)
        
        # actor outputs onehot vectors for the actions, we can multiply it with this vector using torch.matmul to get the indices, while keeping grad
        self.action_categories = torch.tensor([i for i in range(self.actor.output_size)], dtype = torch.float32)

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
            self.entropy_targs.append(alpha_temp_multiplier * -act_space.n)
            # self.entropy_targs.append(-torch.prod(torch.Tensor(act_space.shape).to(self.device)).item())
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
        replay_act = [torch.tensor(actions, dtype=torch.int64).to(self.device) for actions in replay_act_list]
        # rewards = [torch.tensor(rewards, dtype=torch.float32).to(self.device) for rewards in rewards_list]
        rewards = torch.tensor(np.array(rewards_list), dtype=torch.float32).mean(0).to(self.device)
        # rewards = torch.tensor(np.array(rewards_list), dtype=torch.float32).mean(0).to(self.device)
        dones = [torch.tensor(dones, dtype=torch.int32).to(self.device) for dones in dones_list]

        # if we are not given the global observation, for each agent, and we need to construct it
        with torch.no_grad():
            if not self.global_observations:
                obs = torch.cat(obs, dim = 1)
                next_obs = torch.cat(next_obs, dim = 1)
            else:
                # all agents are given global observations
                obs = obs[0]
                next_obs = next_obs[0]
        
        # keep track of index of point where to add sequential actions to input tensor
        seq_action_index = obs.shape[1]

        # preconstruct input tensors to which the sequential actions will be added
        with torch.no_grad():
            # prepare input
            input_tensor = torch.full((self.batch_size, self.actor.input_size), self.padding_val, dtype = torch.float32).to(self.device)
            # observations first
            input_tensor[:, 0 : obs.shape[1]] = obs

        with torch.autograd.set_detect_anomaly(True):
            # SEQUENTIALLY update shared critics/actor for each agent
            for agent_idx in range(self.nr_agents):
                if agent_idx > 0:
                    # last target Q input is same as next normal Q input 
                    input_tensor = targ_input_tensor.clone().detach()

                # get alpha for remainder of this agent loop
                alpha_i = torch.exp(self.log_alphas[agent_idx].detach()).to(self.device)

                # --- CRITIC GRADIENT ---
                # Q values to measure against target
                # print("INPUT AGENT ", agent_idx)
                # print(input_tensor)
                q1 = self.critic1(input_tensor)
                q2 = self.critic2(input_tensor)

                # NOTE: WE CAN REUSE THE OUTPUT OF TARGET NETWORK AS WELL, INSTEAD OF ONLY INPUT OF TARGET NETWORK, JUST TAKE
                #   THE CHOSEN ACTION Q-VAL INSTEAD OF MAX
                with torch.no_grad():
                    if agent_idx < (self.nr_agents - 1):
                        """
                        FOR ALL AGENTS IN SEQUENCE i = (1, ..., m - 1):
                            We compare against only the Q-value of the next in sequence.
                        """
                        # alpha of next in sequence
                        alpha_ip1 = torch.exp(self.log_alphas[agent_idx + 1].detach())

                        # first we have to add the actions to the input tensor, and change the onehot id
                        targ_input_tensor = input_tensor.clone().detach()

                        # add additional actions
                        current_action = F.one_hot(replay_act[agent_idx], num_classes = self.actor.output_size).squeeze(1)
                        targ_input_tensor[:, seq_action_index : seq_action_index + current_action.shape[1]] = current_action
                        # move index 
                        seq_action_index += current_action.shape[1]

                        # get actor output of next stage in sequence
                        act_ip1, logp_ip1, probs_ip1 = self.actor.action_distr_sample(targ_input_tensor)

                        # get critics output of next stage in sequence
                        q1_targ = self.critic1_targ(targ_input_tensor)
                        q2_targ = self.critic2_targ(targ_input_tensor)

                        # Clipped double Q trick
                        min_q_targ = torch.minimum(q1_targ, q2_targ)
                        # Action probabilities can be used to estimate the expectation (cleanRL)
                        # we do not compare to the temporal difference target for all agents in sequence except the last.
                        # we just compare to the next in sequence Q val as target (no reward/discount)
                        target = (probs_ip1 * (min_q_targ - alpha_ip1.unsqueeze(1) * logp_ip1)).sum(dim = 1)
                    else:
                        """
                        FOR AGENT i = m:
                            We compare against only the temporal difference target of the first in sequence.
                        """
                        # alpha of first in sequence
                        alpha_0 = torch.exp(self.log_alphas[0].detach())

                        # create new target tensors
                        targ_input_tensor = torch.full((self.batch_size, self.actor.input_size), self.padding_val, dtype = torch.float32).to(self.device)

                        # add next observation
                        targ_input_tensor[:, 0 : next_obs.shape[1]] = next_obs

                        # get actor output of first stage/agent in sequence
                        act_0_nextobs, logp_0_nextobs, probs_0_nextobs = self.actor.action_distr_sample(targ_input_tensor)

                        # get critics output of first stage in sequence
                        q1_targ = self.critic1_targ.forward(targ_input_tensor)
                        q2_targ = self.critic2_targ.forward(targ_input_tensor)

                        # Clipped double Q trick
                        min_q_targ = torch.minimum(q1_targ, q2_targ)
                        # Action probabilities can be used to estimate the expectation (cleanRL)
                        q_targ = (probs_0_nextobs * (min_q_targ - alpha_0.unsqueeze(1) * logp_0_nextobs)).sum(dim = 1)

                        # for the last we do compare with the temporal difference target, so we use reward (mean) and discount
                        target = rewards + self.gamma * (1 - dones[agent_idx]) * q_targ

                # gather Q values from actions taken in replay (gather not inplace operation)
                q1 = q1.gather(1, replay_act[agent_idx]).squeeze()
                q2 = q2.gather(1, replay_act[agent_idx]).squeeze()

                # loss is MSEloss
                loss_critic1 = F.mse_loss(q1, target)
                loss_critic2 = F.mse_loss(q2, target)
                loss_critic = loss_critic1 + loss_critic2 # factor of 0.5 also used

                # backward prop + gradient step
                self.critic1.optimizer.zero_grad()
                self.critic2.optimizer.zero_grad()
                loss_critic.backward()
                self.critic1.optimizer.step()
                self.critic2.optimizer.step()

                # --- ACTOR GRADIENT ---
                # first we compute the actor output for the current stage in the sequence
                act_i, logp_i, probs_i = self.actor.action_distr_sample(input_tensor)
                # first freeze critic gradient calculation to save computation
                for params in self.critic1.parameters():
                    params.requires_grad = False
                for params in self.critic2.parameters():
                    params.requires_grad = False

                # compute Q-values (NOTE: SAME AS ABOVE? TEST IF MATTERS)
                q1_policy = self.critic1.forward(input_tensor)
                q2_policy = self.critic2.forward(input_tensor)
                # Clipped double Q-trick
                q_policy = torch.minimum(q1_policy, q2_policy)
                # entropy regularized loss
                loss_policy = (probs_i * (alpha_i.unsqueeze(1) * logp_i - q_policy)).sum(dim = 1).mean()

                # step along gradient
                self.actor.optimizer.zero_grad()
                loss_policy.backward()
                self.actor.optimizer.step()

                # unfreeze critic gradient
                for params in self.critic1.parameters():
                    params.requires_grad = True
                for params in self.critic2.parameters():
                    params.requires_grad = True

                # --- AUTOMATIC ENTROPY TEMPERATURE TUNING ---
                #   optimal alpha_t = arg min(alpha_t) E[-alpha_t * (log policy(a_t|s_t; alpha_t) - alpha_t * entropy_target)]
                # we detach because otherwise we backward through the graph of previous calculations using log_prob
                # Action probabilities can be used to estimate the expectation (cleanRL)
                alpha_loss = (probs_i.detach() * (-self.log_alphas[agent_idx].exp() * logp_i.detach() - self.log_alphas[agent_idx].exp() * self.entropy_targs[agent_idx])).mean()

                # backward prop + gradient step
                self.alpha_optimizers[agent_idx].zero_grad()
                alpha_loss.backward()
                self.alpha_optimizers[agent_idx].step()   

                # log values
                loss_Q_list.append(loss_critic.cpu().detach().numpy())
                loss_pi_list.append(loss_policy.cpu().detach().numpy())
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

    def get_action(self, observations, deterministic = False):
        # get actor action
        action_list = []

        with torch.no_grad():
            # make tensor if needed
            if not torch.is_tensor(observations[0]):
                # if global observations are given to each agent, we do not need to create it
                if self.global_observations:
                    global_obs = torch.tensor(observations[0], dtype = torch.float32).to(self.device)
                # we need to construct the global observation
                else:
                    global_obs = torch.tensor([o for observation in observations for o in observation], dtype = torch.float32).to(self.device)
            else:
                if self.global_observations:
                    global_obs = torch.cat(observations).to(self.device)
                else:
                    global_obs = observations[0].to(torch.float32).to(self.device)
            
            # create input tensor
            input_tensor = torch.full((self.actor.input_size,), self.padding_val, dtype = torch.float32)
            # inplace does not matter because of no_grad()
            # observations first
            input_tensor[0 : global_obs.shape[0]] = global_obs

            # index in input tensor where to add additional sequential actions
            seq_action_index = global_obs.shape[0]

            for agent_idx in range(self.nr_agents):
                if agent_idx > 0:
                    # add previous sequential action
                    current_action = F.one_hot(torch.tensor(action_list[-1], dtype = torch.int64), num_classes = self.actor.output_size)
                    # add to tensor on the right indices
                    input_tensor[seq_action_index : seq_action_index + current_action.shape[0]] = current_action
                    # move sequential index
                    seq_action_index += current_action.shape[0]

                # sample action from policy
                actions, _, _ = self.actor.action_distr_sample(input_tensor.unsqueeze(0).to(self.device), deterministic)
                    
                # add to list, convert from one hot to integer
                action_list.append(torch.argmax(actions).item())

        return action_list
    
    def evaluate(self, eps):
        """
        Evaluate the current policy.
        """
        self.actor.eval()
        
        obs, _ = self.env.reset()
        
        terminals = [False]
        truncations = [False]
        rew_sum = 0
        ep_steps = 0
        
        while not (any(terminals) or all(truncations)):
            # get action
            with torch.no_grad():
                act = self.get_action(obs)
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
            self.actor.save(self.save_dir, "actor_eval")
            
        # log rewards
        self.logger.log({"eval_reward_sum": rew_sum}, eps, "rollout")
        
        # turn off eval mode
        self.actor.train()
    
    def train(self, nr_steps, max_episode_len = -1, warmup_steps = 10000, learn_delay = 1000, learn_freq = 50, learn_weight = 50, 
              checkpoint = 250000, save_dir = "models"):
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
        current_best = 0

        # reset env
        obs, _ = self.env.reset()

        # episode and epsiode len count
        ep = 0
        ep_steps = 0
        # steps learned per episode count (for avg)
        ep_learn_steps = 0
        # sum of log values for each ep
        ep_rew_sum = np.zeros(self.nr_agents)
        ep_sharedrew_sum = 0
        ep_aloss_sum = np.zeros(self.nr_agents)
        ep_closs_sum = np.zeros(self.nr_agents)
        ep_alpha_sum = np.zeros(self.nr_agents)
        ep_alphaloss_sum = np.zeros(self.nr_agents)
        ep_entr_sum = np.zeros(self.nr_agents)

        for step in range(nr_steps):
            # step increment
            ep_steps += 1
            
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
            ep_sharedrew_sum += np.mean(reward)

            # set done to false if signal is because of time horizon (spinning up)
            # if ep_steps == max_episode_len:
            #     done = False

            # add transition to replay buffer
            self.replay_buffer.add_transition(obs, action, reward, next_obs, done)

            # observation update
            obs = next_obs

            # done or max 
            if (any(done) or any(truncated) or ep_steps == max_episode_len):
                ep += 1

                # avg losses and entropy
                if (ep_learn_steps > 0):
                    avg_actor_loss = ep_aloss_sum / ep_learn_steps
                    avg_critic_loss = ep_closs_sum / ep_learn_steps
                    avg_policy_entr = ep_entr_sum / ep_learn_steps
                    avg_alpha = ep_alpha_sum / ep_learn_steps
                    avg_alpha_loss = ep_alphaloss_sum / ep_learn_steps
                    # save logs (self. because we can use it to save checkpoints)
                    self.logs = {"avg_actor_loss": avg_actor_loss,
                                 "avg_critic_loss": avg_critic_loss,
                                 "avg_alpha_loss": avg_alpha_loss,
                                 "avg_alpha": avg_alpha,
                                 "avg_policy_entr": avg_policy_entr}
                    self.logger.log(self.logs, ep, group = "train")
                # log reward seperately
                self.rollout_log = {"reward_sum": ep_rew_sum,
                                    "shared_rew_sum": ep_sharedrew_sum,
                                    "ep_steps": ep_steps}
                self.logger.log(self.rollout_log, ep, "rollout")

                # eval
                if ep % self.eval_every == 0:
                    self.evaluate(ep)
                    # pass

                # add info to progress bar
                if step % (nr_steps // 20) == 0:
                    print("[Episode {:d} total reward: ".format(ep) + str(ep_rew_sum) + "] ~ ")
                    # pbar.set_description("[Episode {:d} mean reward: {:0.3f}] ~ ".format(ep, ', '.join(avg_rew)))
                
                # checkpoint
                if np.mean(ep_rew_sum) > current_best:
                    current_best = np.mean(ep_rew_sum)
                    self.actor.save(save_dir, "actor_best")
                    self.critic1.save(save_dir, "critic1_best")
                    self.critic2.save(save_dir, "critic2_best")
                    self.critic1_targ.save(save_dir, "critic1_targ_best")
                    self.critic2_targ.save(save_dir, "critic2_targ_best")

                # reset
                obs, _ = self.env.reset()
                # reset logging info
                ep_steps = 0
                ep_learn_steps = 0
                ep_sharedrew_sum = 0
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

            if step % checkpoint == 0:
                self.actor.save(save_dir, "actor_" + str(step))
                self.critic1.save(save_dir, "critic1_" + str(step))
                self.critic2.save(save_dir, "critic2_" + str(step))
                self.critic1_targ.save(save_dir, "critic1_targ_" + str(step))
                self.critic2_targ.save(save_dir, "critic2_targ_" + str(step))
