import numpy as np

class ReplayBuffer():
    """
    A replay buffer class for use in off-policy algorithms.
    
    Args:
        max_shape (int): Maximum size of the buffers in number of transitions
        observation_size (tuple:int): Observation size
        action_size (tuple:int): Action size
        batch_size (int): Number of transitions per batch when calling sample()
    """
    def __init__(self, max_size, observation_size, action_size, batch_size) -> None:
        self.max_size = max_size
        self.observation_size = observation_size
        self.action_size = action_size
        self.batch_size = batch_size
        # set buffer sizes
        self.obs_buffer = np.zeros((self.max_size, *observation_size))
        self.next_obs_buffer = np.zeros((self.max_size, *observation_size))
        self.action_buffer = np.zeros((self.max_size, *action_size))
        self.reward_buffer = np.zeros((self.max_size))
        self.done_buffer = np.zeros((self.max_size))
        # keep track of memory index
        self.buffer_index = 0
        
        
    def add_transition(self, obs, action, reward, next_obs, done):
        """
        Add a transition to the replay buffer

        Args:
            obs (np.ndarray): The observation of the state the action is taken in
            next_obs (np.ndarray): The observation from the resulting transition state of the old state and action pair
            action (np.ndarray): Action taken in the state
            reward (float): Reward as a result of the transition
            done (bool): done boolean
        """
        index = self.update_index()

        self.obs_buffer[index] = obs
        self.next_obs_buffer[index] = next_obs
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.done_buffer[index] = done
        
    def sample(self):
        """
        Get a batch of samples from the replay buffer
        """
        random_indices = np.random.choice(min(self.buffer_index, self.max_size), self.batch_size)
        
        # we can use nparray as indices for sampling
        obs = self.obs_buffer[random_indices]
        next_obs = self.next_obs_buffer[random_indices]
        actions = self.action_buffer[random_indices]
        rewards = self.reward_buffer[random_indices]
        dones = self.done_buffer[random_indices]
        
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