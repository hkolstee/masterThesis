import os
import sys

import numpy as np

import torch
import torch.nn.functional as F

from gymnasium import spaces

from copy import deepcopy

import itertools

# add folder to python path for relative imports
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

from ..networks.actor_discrete import DiscreteActor
from ..networks.critic import Critic
from ..core.SAC_single_agent_base import SoftActorCriticCore
from ..replay_buffers.replay_buffer import ReplayBuffer
from ..utils.logger import Logger

class SAC:
    """
    Single agent Soft Actor-Critic. Both discrete and continuous action space
    compatible.
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
                 use_AE = False,
                 AE_reduc_scale = 6,
                 log_dir = "tensorboard_logs", 
                 global_obs = False
                 ):
        self.env = env
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        # use of autoencoder
        self.use_AE = use_AE
        # if env provides global obs to each agent or not
        self.global_obs = global_obs
        
        # initialize tensorboard logger
        self.logger = Logger(env, log_dir)
        
        # initialize device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
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
        
        # NOTE: should be multiagent
        self.replay_buffer = ReplayBuffer(max_size = buffer_max_size, 
                                          observation_size = (obs_size,),
                                          action_size = (1,),
                                          batch_size = batch_size)

        # initialize networks 
        self.actor = DiscreteActor(lr = lr_actor,
                                   obs_size = obs_size,
                                   action_size = len(self.index_to_act_combi),
                                   layer_sizes = layer_sizes)
        # double clipped Q learning
        self.critic1 = Critic(lr = lr_critic, 
                              obs_size = obs_size,
                              act_size = len(self.index_to_act_combi), 
                              discrete = True,
                              layer_sizes = layer_sizes)
        self.critic2 = Critic(lr = lr_critic, 
                              obs_size = obs_size,
                              act_size = len(self.index_to_act_combi), 
                              discrete = True,
                              layer_sizes = layer_sizes)
        
        # target networks
        self.critic1_targ = deepcopy(self.critic1)
        self.critic2_targ = deepcopy(self.critic2)
        # freeze parameter gradient calculation as it is not used
        self.freeze_network_grads(self.critic1_targ)
        self.freeze_network_grads(self.critic2_targ)

        # initialize alpha(s) and optimizers
        self.log_alpha = torch.ones(1, requires_grad = True, device = self.device) 
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr = lr_critic) \
            
        # entropy target
        # self._entropy_targ = -0.98 * torch.log(1 / torch.tensor(self.env.action_space.n))
        self.entropy_targ = -len(self.index_to_act_combi)
        
    def polyak_update(self, base_network, target_network, polyak):
        """ 
        Polyak/soft update of target networks.
        """
        with torch.no_grad():
            for (base_params, target_params) in zip(base_network.parameters(), target_network.parameters()):
                target_params.data *= polyak
                target_params.data += ((1 - polyak) * base_params.data)
    
    def freeze_network_grads(self, network):
        """
        Freeze parameter gradient calculation.
        """
        for param in network.parameters():
            param.requires_grad = False
        
    def unfreeze_network_grads(self, network):
        """
        Freeze parameter gradient calculation.
        """
        for param in network.parameters():
            param.requires_grad = True
    
    def save_networks_parameters(self, save_dir, step):
        self.actor.save(save_dir, "actor" + "_" + str(step))
        self.critic1.save(save_dir, "critic1" + "_" + str(step))
        self.critic2.save(save_dir, "critic2" + "_" + str(step))
        self.critic1_targ.save(save_dir, "critic1_targ" + "_" + str(step))
        self.critic2_targ.save(save_dir, "critic2_targ" + "_" + str(step))

    def sample_batch(self):
        # sample from buffer
        obs, replay_act, rewards, next_obs, dones = self.replay_buffer.sample()

        # prepare tensors
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        replay_act = torch.tensor(replay_act, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.int32).to(self.device)

        return obs, next_obs, replay_act, rewards, dones

    def learn(self):
        """
        One step of learning, meaning one step of gradient descend over critic and actor.
        """
        # buffer not full enough
        if self.replay_buffer.buffer_index < self.batch_size:
            # return status 0
            return 0, None, None, None, None

        obs, next_obs, replay_act, rewards, dones = self.sample_batch()

        # get new alpha
        self.alpha = torch.exp(self.log_alpha.detach())
        
        # critic loss
        loss_critic = self.critic_loss(obs, next_obs, replay_act, rewards, dones)

        # backward prop + gradient step
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        loss_critic.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()
        
        # actor loss
        # first freeze critic gradient calculation to save computation
        self.freeze_network_grads(self.critic1)        
        self.freeze_network_grads(self.critic2)

        loss_policy, loss_alpha = self.actor_and_alpha_loss(obs)

        # backward prop + gradient step
        self.actor.optimizer.zero_grad()
        loss_policy.backward()
        self.actor.optimizer.step()

        # unfreeze critic gradient calculation to save computation
        self.unfreeze_network_grads(self.critic1)        
        self.unfreeze_network_grads(self.critic2)

        # backward prop + gradient step
        self.alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self.alpha_optimizer.step()   

        # polyak update of target networks
        self.polyak_update(self.critic1, self.critic1_targ, self.polyak)
        self.polyak_update(self.critic2, self.critic2_targ, self.polyak)

        return 1, \
               loss_policy.cpu().detach().numpy(), \
               loss_critic.cpu().detach().numpy(), \
               self.alpha.cpu().detach().numpy()[0], \
               loss_alpha.cpu().detach().numpy()

    """
    Overriding abstract parent methods.
    """
    def critic_loss(self, obs, next_obs, replay_act, rewards, dones):
        """
        Returns the critic loss
        """
        # These Q values are the left hand side of the loss function
        # discrete critic estimates Q-values for all discrete actions
        q1_buffer = self.critic1.forward(obs)
        q2_buffer = self.critic2.forward(obs)
        # gather Q-values for chosen action
        q1_buffer = q1_buffer.gather(1, replay_act.long()).squeeze()
        q2_buffer = q2_buffer.gather(1, replay_act.long()).squeeze()

        # For the RHS of the loss function (Approximation of Bellman equation with (1 - d) factor):
        with torch.no_grad():
            # targets from current policy (old policy = buffer)
            _, log_probs, probs = self.actor.action_distr_sample(next_obs)

            # target q values
            q1_policy_targ = self.critic1_targ.forward(next_obs)
            q2_policy_targ = self.critic2_targ.forward(next_obs)
                
            # Clipped double Q trick
            min_q_targ = torch.minimum(q1_policy_targ, q2_policy_targ)

            # Action probabilities can be used to estimate the expectation (cleanRL)
            q_targ =  (probs * (min_q_targ - self.alpha.unsqueeze(1) * log_probs)).sum(dim = 1)
                
            # Bellman approximation
            bellman = rewards + self.gamma * (1 - dones) * q_targ

        # loss is MSEloss over Bellman error (MSBE = mean squared bellman error)
        loss_critic1 = F.mse_loss(q1_buffer, bellman)
        loss_critic2 = F.mse_loss(q2_buffer, bellman)
        loss_critic = loss_critic1 + loss_critic2 

        return loss_critic
    
    def actor_and_alpha_loss(self, obs):
        """
        Returns the actor loss and entropy temperature tuning loss.
        """
        # compute current policy action for pre-transition observation
        _, log_probs, probs = self.actor.action_distr_sample(obs)

        # Q values estimated by critic
        q1_policy = self.critic1.forward(obs)
        q2_policy = self.critic2.forward(obs)
        # take min of these two 
        #   = clipped Q-value for stable learning, reduces overestimation
        q_policy = torch.minimum(q1_policy, q2_policy)
        # entropy regularized loss
        loss_policy = (probs * (self.alpha.unsqueeze(1) * log_probs - q_policy)).sum(1).mean()

        loss_alpha = (probs.detach() * (-self.log_alpha.exp() * (log_probs.detach() + self.entropy_targ))).mean()

        return loss_policy, loss_alpha
    
    def get_action(self, obs):
        # make tensor and send to device
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype = torch.float32).unsqueeze(0).to(self.device)

        # get actor action
        with torch.no_grad():
            action, _, _ = self.actor.action_distr_sample(obs)

        # argmax for largest logit 
        index = np.argmax(action.cpu().detach().numpy()[0])
        
        # index gives combination of actions
        actions = list(self.index_to_act_combi[index])

        return actions, index
    
    def train(self, 
              nr_eps, 
              max_episode_len = -1, 
              warmup_steps = 10000, 
              learn_delay = 100, 
              learn_freq = 50, 
              learn_weight = 50, 
              checkpoint = 100000, 
              save_dir = "models"):
        """
        One training run.
        """
        current_best = 0
        step = 0
        for ep in range(nr_eps):
            # reset env
            obs, info = self.env.reset()
            if self.global_obs:
                obs = obs[0]
            else:
                next_obs = [item for o in obs for item in o]

            # episode and epsiode len count
            ep_steps = 0
            # steps learned per episode count (for avg)
            ep_learn_steps = 0
            # sum of log values for each ep
            ep_rew_sum = 0
            ep_aloss_sum = 0
            ep_closs_sum = 0
            ep_alpha_sum = 0
            ep_alphaloss_sum = 0
            ep_aeloss_sum = 0
            
            done = False
            truncated = False
            while not done and not truncated:
            # sample action (uniform sample for warmup)
                if step < warmup_steps:
                    actions = [act_space.sample() for act_space in self.env.action_space]
                    index = self.act_combi_to_index[tuple(actions)]
                else: 
                    actions, index = self.get_action(obs)
                
                # transition
                next_obs, reward, done, truncated, info = self.env.step(actions)
                done = done[0]
                truncated = truncated[0]
                if self.global_obs:
                    next_obs = next_obs[0]
                else:
                    next_obs = [item for o in next_obs for item in o]
                
                # NOTE: for our env we take mean of agent rewards
                reward = np.mean(reward)
                
                # step increment 
                ep_steps += 1
                step += 1
                # reward addition to total sum
                ep_rew_sum += reward

                # set done to false if signal is because of time horizon (spinning up)
                if ep_steps == max_episode_len:
                    done = False

                # add transition to buffer
                self.replay_buffer.add_transition(obs, index, reward, next_obs, done)

                # observation update
                obs = next_obs
                
                # learn
                if step > learn_delay and step % learn_freq == 0:
                    for _ in range(learn_weight):
                        # learning 
                        status, loss_actor, loss_critic, alpha, loss_alpha = self.learn()

                        if status:
                            # keep track for logs
                            ep_learn_steps += 1
                            ep_aloss_sum += loss_actor
                            ep_closs_sum += loss_critic
                            ep_alpha_sum += alpha
                            ep_alphaloss_sum += loss_alpha
                            
            # avg losses and entropy
            if (ep_learn_steps > 0):
                avg_actor_loss = ep_aloss_sum / ep_learn_steps
                avg_critic_loss = ep_closs_sum / ep_learn_steps
                avg_alpha = ep_alpha_sum / ep_learn_steps
                avg_alpha_loss = ep_alphaloss_sum / ep_learn_steps
                avg_ae_loss = ep_aeloss_sum / ep_steps
                # save training logs: 
                logs = {"avg_actor_loss": avg_actor_loss,
                        "avg_critic_loss": avg_critic_loss,
                        "avg_alpha_loss": avg_alpha_loss,
                        "avg_ae_loss": avg_ae_loss,
                        "avg_alpha": avg_alpha}
                self.logger.log(logs, ep, group = "train")
            # log reward seperately
            reward_log = {"reward_sum": ep_rew_sum}
            self.logger.log(reward_log, ep, "reward")

            # NOTE: for now like this for citylearn additional logging, should be in wrapper or something
            # if self.citylearn:
            #     self.logger.log_custom_reward_values(step)

            # add info to progress bar
            if ep % (nr_eps // 20) == 0:
                print("[Episode {:d} total reward: {:0.3f}] ~ ".format(ep, ep_rew_sum))
                
            # checkpoint save
            if ep % checkpoint == 0:
                self.save_networks_parameters(save_dir, ep)
            
            # save if best
            if ep_rew_sum > current_best:
                current_best = ep_rew_sum
                self.save_networks_parameters(save_dir, "best")