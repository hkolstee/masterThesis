import numpy as np

class ReplayBuffer():
    """
    A replay buffer class for use in off-policy algorithms.
    
    Args:
        max_shape (int): Maximum size of the buffers in number of transitions
        observation_size (tuple:int): Size of the observation space
        action_shape (tuple:int): Size of the action space
        batch_size (int): Number of transitions per batch when calling sample()
    """
    def __init__(self, max_size, observation_shape, action_shape, batch_size) -> None:
        self.max_size = max_size
        self.observation_size = observation_shape
        self.action_size = action_shape
        self.batch_size = batch_size
        # set buffer sizes
        self.obs_buffer = np.zeros((self.max_size, *observation_shape))
        self.next_obs_buffer = np.zeros((self.max_size, *observation_shape))
        self.action_buffer = np.zeros((self.max_size, *action_shape))
        self.reward_buffer = np.zeros((self.max_size))
        self.done_buffer = np.zeros((self.max_size))
        # keep track of memory index
        self.buffer_index = 0
        
        
    def add_transition(self, obs, next_obs, action, reward, done):
        """
        Add a transition to the replay buffer

        Args:
            obs (np.ndarray): The observation of the state the action is taken in
            next_obs (np.ndarray): The observation from the resulting transition state of the old state and action pair
            action (np.ndarray): Action taken in the state
            reward (float): Reward as a result of the transition
            done (bool): done boolean
        """
        self.obs_buffer[self.buffer_index] = obs
        self.next_obs_buffer[self.buffer_index] = next_obs
        self.action_buffer[self.buffer_index] = action
        self.reward_buffer[self.buffer_index] = reward
        self.done_buffer[self.buffer_index] = done
        
        # increase index
        self.update_index()
        
    def sample(self):
        """
        Get a batch of samples from the replay buffer
        """
        random_indices = np.random.choice(self.buffer_index, self.batch_size)
        
        # we can use nparray as indices for sampling
        obs = self.obs_buffer[random_indices]
        next_obs = self.next_obs_buffer[random_indices]
        actions = self.action_buffer[random_indices]
        rewards = self.reward_buffer[random_indices]
        dones = self.done_buffer[random_indices]
        
        return obs, next_obs, actions, rewards, dones

    def update_index(self):
        """
        Update the index to the correct value when taking into account the max buffer size
        """
        self.buffer_index = (self.buffer_index + 1) % self.max_size