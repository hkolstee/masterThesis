import torch
import torch.nn as nn
import torch.functional as functional

import numpy as np

class SAC():
    """
    Base class with all gradient functions for Soft Actor-Critic
    """
    def __init__(self):
        pass

    def criticLoss(critic1, critic2, critic1_targ, critic2_targ):
        pass

    def alphaLoss(alpha, log_prob, entropy_targ):
        loss =(-alpha * log_prob.detach() - alpha * entropy_targ).detach().mean()

        return loss