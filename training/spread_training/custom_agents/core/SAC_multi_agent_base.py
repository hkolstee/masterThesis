import sys
import os

from gymnasium import spaces

import torch

import numpy as np

# add folder to python path for relative imports
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

from ..replay_buffers.replay_buffer import ReplayBuffer
from ..replay_buffers.ma_replay_buffer import MultiAgentReplayBuffer
from ..utils.logger import Logger
# from ..SAC_components.autoencoder import AutoEncoder

# base abstract class
from .RL_algo_base import RLbase
from abc import abstractmethod

class SoftActorCriticCore(RLbase):
    """
    Abstract class.
        
    Serves as the base core class for all SAC implementations. Includes some
    shared methods and shared attributes among all SAC implementations. 
    """
    def __init__(self,
                 env,
                 parameter_sharing_actors,
                 parameter_sharing_critics,
                 gamma = 0.99,
                 alpha_lr = 0.0003,
                 polyak = 0.995,
                 buffer_max_size = 1000000,
                 batch_size = 256,
                 log_dir = "tensorboard_logs",
                 global_observations = False,
                 ):
        self.env = env,
        self.parameter_sharing_actors = parameter_sharing_actors,
        self.parameter_sharing_critics = parameter_sharing_critics,
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.global_observations = global_observations

        # get number of agents corresponding to action space list size
        self.nr_agents = len(env.action_space)
        
        # initialize tensorboard logger
        self.logger = Logger(env, log_dir)
        
        # initialize device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # discrete or continuous action space check (assuming homogeneous agents)
        discrete_actions = isinstance(env.action_space[0], spaces.Box)
        
        # initalize replay buffer (NOTE: MA-REPLAY BUFFER CAN BE OPTIMIZER FOR GLOBAL OBS)
        obs_size_list = [obs_space.shape for obs_space in env.observation_space]
        # differentiate between discrete action space or continuous
        if discrete_actions:
            # Multi-discrete not supported
            act_size_list = [(1,) for _ in env.action_space]
        else:
            act_size_list = [act_space.shape for act_space in env.action_space]
        self.replay_buffer = MultiAgentReplayBuffer(buffer_max_size = buffer_max_size, 
                                                    observation_sizes = obs_size_list,
                                                    action_sizes = act_size_list,
                                                    batch_size = batch_size)
        
        # Networks and entropy tuning  
        self.log_alphas = []
        self.alpha_optimizers = []
        for _ in range(self.nr_agents):
            # initialize alpha(s) and optimizers
                # add device like this to make tensor not leaf tensor
            self.log_alphas.append(torch.ones(1, requires_grad = True, device = self.device)) 
            self.alpha_optimizers.append(torch.optim.Adam([self.log_alphas[-1]], lr = alpha_lr))

    """
    Abstract properties for attributes the algorithm should have.
    """
    @property
    @abstractmethod
    def actors(self):
        pass

    @property
    @abstractmethod
    def critics1(self):
        pass

    @property
    @abstractmethod
    def critics2(self):
        pass

    @property
    @abstractmethod
    def critics1_targ(self):
        pass

    @property
    @abstractmethod
    def critics2_targ(self):
        pass
    
    @property
    @abstractmethod
    def entropy_targ(self):
        pass

    """
    SAC required methods.
    """
    @abstractmethod
    def actor_and_alpha_loss(self):
        pass

    @abstractmethod
    def critic_loss(self):
        pass

    """
    Abstract parent class abstract methods.
    """
    @abstractmethod
    def learn(self):
        pass
    
    @abstractmethod
    def get_actions(self):
        pass

    @abstractmethod
    def get_critic_input_tensors(self):
        pass

    @abstractmethod
    def get_actor_input_tensors(self):
        pass
    
    """
    Universal methods
    """
    def save_networks_parameters(self, save_dir, step):
        # actors
        if self.parameter_sharing_actors:
            self.actors.save(save_dir, "actor" + "_" + str(step))
        else:
            for actor_idx in range(len(self.actors)):
                self.actors[actor_idx].save(save_dir, "actor" + str(actor_idx) + "_" + str(step))
        # critics
        if self.parameter_sharing_critics:
            self.critics1.save(save_dir, "critic1" + "_" + str(step))
            self.critics2.save(save_dir, "critic2" + "_" + str(step))
            self.critics1_targ.save(save_dir, "critic1_targ" + "_" + str(step))
            self.critics2_targ.save(save_dir, "critic2_targ" + "_" + str(step))
        else:
            for critic_idx in range(len(self.critics1)):
                self.critics1[critic_idx].save(save_dir, "critic1" + str(critic_idx) + "_" + str(step))
                self.critics2[critic_idx].save(save_dir, "critic2" + str(critic_idx) + "_" + str(step))
                self.critics1_targ[critic_idx].save(save_dir, "critic1_targ" + str(critic_idx) + "_" + str(step))
                self.critics2_targ[critic_idx].save(save_dir, "critic2_targ" + str(critic_idx) + "_" + str(step))

    def learn(self):
                # buffer not full enough
        if self.replay_buffer.buffer_index < self.batch_size:
            # return status 0
            return 0, None, None, None, None
        
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

        # learn step for each agent
        for agent_idx in range(self.nr_agents):
            # get new alpha
            self.alpha = torch.exp(self.log_alphas[agent_idx].detach())
            
            # critic loss
            critic_input, critic_target_input = self.get_critic_input_tensors(obs, next_obs, replay_act, agent_idx)
            loss_critic = self.critic_loss(critic_input, critic_target_input, replay_act, rewards, dones)

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

            # actor_input, critic_input = self.get_actor_input_tensors(obs[agent_idx])
            loss_policy, loss_alpha = self.actor_and_alpha_loss(obs, )

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

        for step in range(nr_steps):
            # sample action (uniform sample for warmup)
            if step < warmup_steps:
                action = [act_space.sample() for act_space in self.env.action_space]
            else: 
                # get action
                action = self.get_actions(obs)

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
                    avg_alpha = ep_alpha_sum / ep_learn_steps
                    avg_alpha_loss = ep_alphaloss_sum / ep_learn_steps
                    # save logs 
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
                ep_alpha_sum = np.zeros(self.nr_agents)
                ep_alphaloss_sum = np.zeros(self.nr_agents)

            # learn
            if step > learn_delay and step % learn_freq == 0:
                for _ in range(learn_weight):
                    # learning step
                    status, loss_actor, loss_critic, alpha, loss_alpha = self.learn()

                    # if buffer full enough for a batch
                    if status:
                        # keep track for logs
                        ep_learn_steps += 1
                        np.add(ep_aloss_sum, loss_actor, out = ep_aloss_sum)
                        np.add(ep_closs_sum, loss_critic, out = ep_closs_sum)
                        np.add(ep_alpha_sum, alpha, out = ep_alpha_sum)
                        np.add(ep_alphaloss_sum, loss_alpha, out = ep_alphaloss_sum)
                        
            # checkpoint
            if (step % checkpoint == 0):
                self.save_networks_parameters(save_dir, step)