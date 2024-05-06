from abc import ABC, abstractmethod

import torch

class RLbase(ABC):
    """
    Abstract class to function as a blueprint for reinforcement learning
    algorithms in this package. Additionally, methods that are used by
    various different RL algorithms are defined.
    """        
    @abstractmethod
    def get_action(self):
        """
        This functions should return actions for (a) given observation(s).
        """
        pass
    
    @abstractmethod
    def learn(self):
        """
        One step of learning.
        """
        pass
        
    @abstractmethod
    def train(self):
        """
        Complete one training run.
        """
        pass
    
    """
    RL helper functions
    """      
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