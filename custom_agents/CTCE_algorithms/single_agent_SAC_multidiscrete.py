import os
import sys

import numpy as np

import torch
import torch.nn.functional as F

from gymnasium import spaces

from copy import deepcopy

# add folder to python path for relative imports
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

from ..networks.actor_multidiscrete import MultiDiscreteActor
from ..networks.critic_multidiscrete import MultiDiscreteCritic
from ..networks.autoencoder import AutoEncoder
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
                 gamma=0.99, 
                 polyak=0.995, 
                 buffer_max_size=1000000, 
                 batch_size=256,
                 layer_sizes=(256, 256), 
                 AE_reduc_scale = 6,
                 log_dir="tensorboard_logs", 
                 ):
        self.env = env
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        # nr of agents/actions
        self.nr_actions = len(env.action_space)
        
        # initialize tensorboard logger
        self.logger = Logger(env, log_dir)
        
        # initialize device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.replay_buffer = ReplayBuffer(max_size = buffer_max_size, 
                                          observation_size = env.observation_space[0].shape,
                                          action_size = (len(env.action_space),),
                                          batch_size = batch_size)
            
        # initialize alpha(s) and optimizers
        self.log_alphas = []
        self.alpha_optimizers = []
        for idx in range(self.nr_actions):
            self.log_alphas.append(torch.ones(1, requires_grad = True, device = self.device)) 
            self.alpha_optimizers.append(torch.optim.Adam([self.log_alphas[idx]], lr = lr_critic))   #shares critic lr   
        
        # action space output for networks
        act_output_size = [act_space.n for act_space in env.action_space]

        # initialize networks 
        self.actor = MultiDiscreteActor(lr = lr_actor,
                                        obs_size = env.observation_space[0].shape[0],
                                        action_size = act_output_size,
                                        layer_sizes = layer_sizes)
        # double clipped Q learning
        self.critic1 = MultiDiscreteCritic(lr = lr_critic, 
                                           obs_size = env.observation_space[0].shape[0],
                                           act_size = act_output_size, 
                                           layer_sizes = layer_sizes)
        self.critic2 = MultiDiscreteCritic(lr = lr_critic, 
                                           obs_size = env.observation_space[0].shape[0],
                                           act_size = act_output_size, 
                                           layer_sizes = layer_sizes)
        
        # target networks
        self.critic1_targ = deepcopy(self.critic1)
        self.critic2_targ = deepcopy(self.critic2)
        # freeze parameter gradient calculation as it is not used
        self.freeze_network_grads(self.critic1_targ)
        self.freeze_network_grads(self.critic2_targ)

        # entropy target
        # self._entropy_targ = -0.98 * torch.log(1 / torch.tensor(self.env.action_space.n))
        self.entropy_targs = []
        for act_space in env.action_space:
            self.entropy_targs.append(-act_space.n)
    
    def save_networks_parameters(self, save_dir, step):
        self.actor.save(save_dir, "actor" + "_" + str(step))
        self.critic1.save(save_dir, "critic1" + "_" + str(step))
        self.critic2.save(save_dir, "critic2" + "_" + str(step))
        self.critic1_targ.save(save_dir, "critic1_targ" + "_" + str(step))
        self.critic2_targ.save(save_dir, "critic2_targ" + "_" + str(step))

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
    
    def critic_loss(self, obs, next_obs, replay_act, rewards, dones):
        """
        Returns the critic loss
        """
        # These Q values are the left hand side of the loss function
        # discrete critic estimates Q-values for all discrete actions
        q1_buffer = self.critic1.forward(obs)
        q2_buffer = self.critic2.forward(obs)
        # gather Q-values for chosen action
        q1_chosen_actions = []
        q2_chosen_actions = []
        for idx, (q1vals, q2vals) in enumerate(zip(q1_buffer, q2_buffer)):
            q1_chosen_actions.append(q1vals.gather(1, replay_act[:,idx].long().unsqueeze(1)).squeeze())
            q2_chosen_actions.append(q2vals.gather(1, replay_act[:,idx].long().unsqueeze(1)).squeeze())

        # For the RHS of the loss function (Approximation of Bellman equation with (1 - d) factor):
        with torch.no_grad():
            # targets from current policy (old policy = buffer)
            _, log_probs, probs = self.actor.action_distr_sample(next_obs)

            # target q values
            q1_targ = self.critic1_targ.forward(next_obs)
            q2_targ = self.critic2_targ.forward(next_obs)
                
            # Clipped double Q trick
            q_target_vals = []
            for (q1vals, q2vals) in zip(q1_targ, q2_targ):
                q_target_vals.append(torch.minimum(q1vals, q2vals))

            # Action probabilities can be used to estimate the expectation (cleanRL)
            q_targs = []
            for idx in range(self.nr_actions):
                q_targs.append((probs[idx] * (q_target_vals[idx] - self.log_alphas[idx].unsqueeze(1) * log_probs[idx])).sum(dim = 1))
                
            # Bellman approximation
            bellman = []
            for idx in range(self.nr_actions):
                bellman.append(rewards + self.gamma * (1 - dones) * q_targs[idx])

        losses = []
        for (b, q1, q2) in zip(bellman, q1_chosen_actions, q2_chosen_actions):
            # loss is MSEloss over Bellman error (MSBE = mean squared bellman error)
            loss_critic1 = F.mse_loss(q1, b)
            loss_critic2 = F.mse_loss(q2, b)
            losses.append(loss_critic1 + loss_critic2) 

        return sum(losses)
    
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
        q_policy = []
        for q1, q2 in zip(q1_policy, q2_policy):
            q_policy.append(torch.minimum(q1, q2))
        # entropy regularized loss
        loss_policy = []
        loss_alpha = []
        for idx in range(self.nr_actions):
            loss_policy.append((probs[idx] * (self.alphas[idx].unsqueeze(1) * log_probs[idx] - q_policy[idx])).sum(1).mean())
            loss_alpha.append((probs[idx].detach() * (-self.log_alphas[idx].exp() * (log_probs[idx].detach() + self.entropy_targs[idx]))).mean())

        return sum(loss_policy), loss_alpha

    def learn(self):
        """
        One step of learning, meaning one step of gradient descend over critic and actor.
        """
        # buffer not full enough
        if self.replay_buffer.buffer_index < self.batch_size:
            # return status 0
            return 0, None, None, None, None

        obs, next_obs, replay_act, rewards, dones = self.sample_batch()

        # get new alphas
        self.alphas = [] 
        for idx in range(self.nr_actions):
            self.alphas.append(torch.exp(self.log_alphas[idx].detach()))
        
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
        for idx, loss in enumerate(loss_alpha):
            self.alpha_optimizers[idx].zero_grad()
            loss.backward()
            self.alpha_optimizers[idx].step()   

        # polyak update of target networks
        self.polyak_update(self.critic1, self.critic1_targ, self.polyak)
        self.polyak_update(self.critic2, self.critic2_targ, self.polyak)

        return 1, \
               loss_policy.cpu().detach().numpy(), \
               loss_critic.cpu().detach().numpy(), \
               [a.cpu().detach().numpy()[0] for a in self.alphas], \
               [l.cpu().detach().numpy() for l in loss_alpha]
    
    def get_action(self, obs):
        # make tensor and send to device
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(np.array(obs), dtype = torch.float32).unsqueeze(0).to(self.device)

        # get actor action
        with torch.no_grad():
            actions, _, _ = self.actor.action_distr_sample(obs)
        
        return [np.argmax(tensor.cpu().detach().numpy()[0]) for tensor in actions]

    def train(self, 
              nr_steps, 
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
        # reset env
        obs, info = self.env.reset()

        # episode and epsiode len count
        ep = 0
        ep_steps = 0
        # steps learned per episode count (for avg)
        ep_learn_steps = 0
        # sum of log values for each ep
        ep_rew_sum = np.zeros(self.nr_actions)
        ep_aloss_sum = 0
        ep_closs_sum = 0
        ep_alpha_sum = np.zeros(self.nr_actions)
        ep_alphaloss_sum = np.zeros(self.nr_actions)
        
        for step in range(nr_steps):
            # sample action (uniform sample for warmup)
            if step < warmup_steps:
                actions = [act_space.sample() for act_space in self.env.action_space]
            else: 
                actions = self.get_action(obs)

            # transition
            next_obs, reward, done, truncated, info = self.env.step(actions)
            
            # step increment 
            ep_steps += 1
            # reward addition to total sum
            ep_rew_sum += reward

            # set done to false if signal is because of time horizon (spinning up)
            if ep_steps == max_episode_len:
                done = False

            # add transition to buffer
            # NOTE: FOR NOW WE TAKE THE GLOBAL OBS, REWARD MEANS, BECAUSE THE EXPERIMENTAL ENV 
            # IS MULTIAGENT, WITH GLOBAL OBS FOR EACH AGENT, SHOULD BE CHANGED FOR MULTIDISCRETE 
            # SINGLE AGENT ENVS. 
            self.replay_buffer.add_transition(obs[0], actions, np.mean(reward), next_obs[0], done[0])

            # observation update
            obs = next_obs[0]

            # done or max steps
            if (done[0] or truncated[0] or ep_steps == max_episode_len):
                ep += 1

                # avg losses and entropy
                if (ep_learn_steps > 0):
                    avg_actor_loss = ep_aloss_sum / ep_learn_steps
                    avg_critic_loss = ep_closs_sum / ep_learn_steps
                    avg_alpha = ep_alpha_sum / ep_learn_steps
                    avg_alpha_loss = ep_alphaloss_sum / ep_learn_steps
                    # save training logs: 
                    logs = {"avg_actor_loss": avg_actor_loss,
                            "avg_critic_loss": avg_critic_loss,
                            "avg_alpha_loss": avg_alpha_loss,
                            "avg_alpha": avg_alpha}
                    self.logger.log(logs, step, group = "train")
                # log reward seperately
                reward_log = {"reward_sum": ep_rew_sum}
                self.logger.log(reward_log, step, "Reward")

                # NOTE: for now like this for citylearn additional logging, should be in wrapper or something
                # if self.citylearn:
                #     self.logger.log_custom_reward_values(step)

                # add info to progress bar
                # if (ep % 1000 == 0):
                #     print("[Episode {:d} total reward: {:0.3f}] ~ ".format(ep, ep_rew_sum))
                    # print(obs)
                    # print(latent_obs)
                    # print(decoded_obs)
                
                # reset
                obs, info = self.env.reset()
                obs = obs[0]
                # reset logging info
                ep_steps = 0
                ep_learn_steps = 0

                ep_rew_sum = np.zeros(self.nr_actions)
                ep_aloss_sum = 0
                ep_closs_sum = 0
                ep_alpha_sum = np.zeros(self.nr_actions)
                ep_alphaloss_sum = np.zeros(self.nr_actions)

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
                        
            # checkpoint
            if (step % checkpoint == 0 and step != 0):
                self.save_networks_parameters(save_dir, step)