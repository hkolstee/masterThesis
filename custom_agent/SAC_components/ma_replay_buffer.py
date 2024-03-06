import numpy as np

class MultiAgentReplayBuffer:
    """A multi-agent capable replay buffer. 
    Here, each agent's observation space and action space can be different.
    
    Args:
        max_shape (int): Maximum size of the buffers in number of transitions
        observation_sizes (list (tuple:int)): List of observation sizes for each agent
        action_sizes (list (tuple:int)): List of action sizes for each agent
        batch_size (int): Number of transitions per batch when calling sample()
    """
    def __init__(self, max_size, observation_sizes, action_sizes, batch_size):
        self.max_size = max_size
        self.observation_sizes = observation_sizes
        self.action_sizes = action_sizes
        self.batch_size = batch_size
        
        # the idea is to create a different replaybuffer for each agent, 
        # and sample using the same indices 

        # set buffer sizes
        self.obs_buffer = []
        self.next_obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        for (obs_size, act_size) in zip(self.observation_sizes, self.action_sizes):
            self.obs_buffer.append(np.zeros((self.max_size, *obs_size)))
            self.next_obs_buffer.append(np.zeros((self.max_size, *obs_size)))
            self.action_buffer.append(np.zeros((self.max_size, *act_size)))
            self.reward_buffer.append(np.zeros((self.max_size, 1)))
            self.done_buffer.append(np.zeros((self.max_size, 1)))
        # keep track of memory index
        self.buffer_index = 0

    def add_transition(self, obs, actions, rewards, next_obs, done):
        """
        Add a transition to the replay buffer

        Args:
            obs (list (np.ndarray)): List of agents' observations of the state the action is taken in
            next_obs (list (np.ndarray)): List of agents' observation from the resulting transition state of the old state and action pair
            actions (list (np.ndarray)): List of agents' action taken in the state
            rewards (list (float)): List of agents' reward as a result of the transition
            dones (list (int)): List of agents' done boolean
        """
        index = self.update_index()
        
        for agent_nr, (o, a, r, n_o) in enumerate(zip(obs, actions, rewards, next_obs)):
            self.obs_buffer[agent_nr][index] = o
            self.next_obs_buffer[agent_nr][index] = n_o
            self.action_buffer[agent_nr][index] = a
            self.reward_buffer[agent_nr][index] = r
            self.done_buffer[agent_nr][index] = done
        
    def sample(self):
        """
        Get a batch of samples from the replay buffer
        """
        random_indices = np.random.choice(min(self.buffer_index, self.max_size), self.batch_size)
        
        # we can use nparray as indices for sampling
        obs = [agent[random_indices] for agent in self.obs_buffer]
        next_obs = [agent[random_indices] for agent in self.next_obs_buffer]
        actions = [agent[random_indices] for agent in self.action_buffer]
        rewards = [agent[random_indices] for agent in self.reward_buffer]
        dones = [agent[random_indices] for agent in self.done_buffer]
        
        return obs, actions, rewards, next_obs, dones

    def update_index(self):
        """
        Update the index to the correct value when taking into account the max buffer size
        """
        # get correct current index
        index = self.buffer_index % self.max_size 
        # increase index
        if self.buffer_index <= self.max_size:
            self.buffer_index += 1

        return index