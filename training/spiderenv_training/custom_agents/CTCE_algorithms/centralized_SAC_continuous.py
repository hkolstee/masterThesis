import os
import sys

import numpy as np

import torch
import torch.nn.functional as F

from copy import deepcopy

# add folder to python path for relative imports
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

from ..networks.actor import Actor
from ..networks.critic import Critic
from ..core.SAC_single_agent_base import SoftActorCriticCore
from ..replay_buffers.replay_buffer import ReplayBuffer

class SAC(SoftActorCriticCore):
    """
    Single agent Soft Actor-Critic. Both discrete and continuous action space
    compatible.
    """
    def __init__(self, 
                 env, 
                 lr_actor = 0.0003,
                 lr_critic = 0.0003,
                 gamma=0.99, 
                 polyak=0.995, 
                 buffer_max_size=1000000, 
                 batch_size=256,
                 layer_sizes=(256, 256), 
                 use_AE = False,
                 AE_reduc_scale = 6,
                 log_dir="tensorboard_logs", 
                 save_dir = "models",
                 global_observations = False,
                 eval_every = 25
                 ):
        """
        Super class initializes all but the actor and critic networks and 
        entropy target.
        """        
        super().__init__(env = env, 
                         gamma = gamma, 
                         alpha_lr = lr_critic, # shares critic lr
                         polyak = polyak, 
                         buffer_max_size = buffer_max_size, 
                         batch_size = batch_size,
                         use_AE = use_AE,
                         AE_reduc_scale = AE_reduc_scale,
                         log_dir = log_dir)
        self.global_obs = global_observations
        self.eval_every = eval_every
        self.best_eval = -np.inf
        self.save_dir = save_dir
        
        # nr of agents in env
        self.nr_agents = len(env.action_space)
        
        # concat obs if not global for each agent
        if self.global_obs:
            obs_size = env.observation_space[0].shape[0]
        else:
            obs_size = sum([obs_space.shape[0] for obs_space in env.observation_space])
        # get action size
        agent_action_sizes = [act_space.shape[0] for act_space in env.action_space]
        act_size = sum(agent_action_sizes)
        # used to split concatenated action into agent actions
        self.action_indices = np.cumsum([0] + agent_action_sizes)

        # act low and high
        act_low = np.concatenate([act_space.low for act_space in env.action_space])
        act_high = np.concatenate([act_space.high for act_space in env.action_space])
        
        # init replay buffer
        self.replay_buffer = ReplayBuffer(buffer_max_size, (obs_size,), (act_size,), batch_size)
        
        # initialize networks 
        self._actor = Actor(lr = lr_actor,
                            obs_size = obs_size,
                            action_size = act_size,
                            action_high = act_high,
                            action_low = act_low,
                            layer_sizes = layer_sizes)
        # double clipped Q learning
        self._critic1 = Critic(lr = lr_critic, 
                               obs_size = obs_size,
                               act_size = act_size, 
                               discrete = False,
                               layer_sizes = layer_sizes)
        self._critic2 = Critic(lr = lr_critic, 
                               obs_size = obs_size,
                               act_size = act_size, 
                               discrete = False,
                               layer_sizes = layer_sizes)
        
        # target networks
        self._critic1_targ = deepcopy(self.critic1)
        self._critic2_targ = deepcopy(self.critic2)
        # freeze parameter gradient calculation as it is not used
        self.freeze_network_grads(self.critic1_targ)
        self.freeze_network_grads(self.critic2_targ)

        # entropy target
        self._entropy_targ = torch.tensor(-np.prod(act_size), dtype=torch.float32).to(self.device)

    """
    Manage abstract parent properties/attributes. 
    """
    @property
    def actor(self):
        return self._actor

    @property
    def critic1(self):
        return self._critic1

    @property
    def critic2(self):
        return self._critic2

    @property
    def critic1_targ(self):
        return self._critic1_targ

    @property
    def critic2_targ(self):
        return self._critic2_targ
    
    @property
    def entropy_targ(self):
        return self._entropy_targ

    """
    Overriding abstract methods
    """
    def critic_loss(self, obs, next_obs, replay_act, rewards, dones):
        """
        Returns the critic loss
        """
        # These Q values are the left hand side of the loss function
        q1_buffer = self.critic1.forward(obs, replay_act)
        q2_buffer = self.critic2.forward(obs, replay_act)

        # For the RHS of the loss function (Approximation of Bellman equation with (1 - d) factor):
        with torch.no_grad():
            # targets from current policy (old policy = buffer)
            actions, log_probs = self.actor.action_distr_sample(next_obs)

            # target q values
            q1_policy_targ = self.critic1_targ.forward(next_obs, actions)
            q2_policy_targ = self.critic2_targ.forward(next_obs, actions)
                
            # Clipped double Q 
            min_q_targ = torch.minimum(q1_policy_targ, q2_policy_targ)

            # final 
            q_targ = (min_q_targ - self.alpha * log_probs)
                
            # Bellman approximation
            bellman = rewards + self.gamma * (1 - dones) * q_targ

        # loss is MSEloss over Bellman error (MSBE = mean squared bellman error)
        loss_critic1 = F.mse_loss(q1_buffer, bellman)
        loss_critic2 = F.mse_loss(q2_buffer, bellman)
        loss_critic = loss_critic1 + loss_critic2 

        return loss_critic
    
    def actor_and_alpha_loss(self, obs):
        """
        Returns the actor loss.
        """
        # compute current policy action for pre-transition observation
        policy_actions, log_probs = self.actor.action_distr_sample(obs)

        # compute Q-values
        q1_policy = self.critic1.forward(obs, policy_actions)
        q2_policy = self.critic2.forward(obs, policy_actions)
        # take min of these two 
        #   = clipped Q-value for stable learning, reduces overestimation
        q_policy = torch.minimum(q1_policy, q2_policy)
        # entropy regularized loss
        loss_policy = (self.alpha * log_probs - q_policy).mean()

        loss_alpha = (-self.log_alpha.exp() * (log_probs.detach() + self.entropy_targ)).mean()

        return loss_policy, loss_alpha

    def get_action(self, obs, reparameterize = True, deterministic = False):
        # convert if neccessary
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype = torch.float32).unsqueeze(0).to(self.device)

        # get action
        with torch.no_grad():
            actions, _ = self.actor.action_distr_sample(obs, reparameterize, deterministic)

        # to numpy
        actions = actions.cpu().detach().numpy()[0]
        
        # split into agent actions (only works for homogeneous agents)
        agent_actions = [actions[self.action_indices[idx] : self.action_indices[idx + 1]] for idx in range(self.nr_agents)]

        return agent_actions
    
    """
    Override normal methods
    """
    def evaluate(self, eps):
        """
        Evaluate the current policy.
        """
        self.actor.eval()
        
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
            act = self.get_action(obs, deterministic = True)
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
            self.actor.save(self.save_dir, "actor_eval")
            self.critic1.save(self.save_dir, "crit1_eval")
            self.critic1_targ.save(self.save_dir, "crit1_target_eval")
            self.critic2.save(self.save_dir, "crit2_eval")
            self.critic2_targ.save(self.save_dir, "crit2_target_eval")
            
        # log rewards
        self.logger.log({"eval_reward_sum": rew_sum}, eps, "rollout")
        
        # turn off eval mode
        self.actor.train()
    
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
                obs = [item for o in obs for item in o]
                
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
                else: 
                    actions = self.get_action(obs)
                
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
                self.replay_buffer.add_transition(obs, np.concatenate(actions), reward, next_obs, done)

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

            # eval
            if ep % self.eval_every == 0:
                self.evaluate(ep)

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

