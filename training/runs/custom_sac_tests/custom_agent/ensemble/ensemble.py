import sys, os

# add folder to python path for relative imports
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)

from ..SAC_components.actor import Actor

import torch

class PolicyEnsemble():
    """
    A policy ensemble class for std-weighted policy action sampling from the ensemble.
    """
    def __init__(self, models_dir):
        """
        Args:
            models_dir (str): The directory to find the model weight files to use in the policy ensemble.
        """
        self.load_models(models_dir)

    def load_models(self, models_dir):
        """Load the models to use as ensemble of policies.

        Args:
            models_dir (str): The directory to find the model weight files.
        """
        # load all models 
        self.models = []
        
        for model_weights in models_dir:
            # load model
            model = torch.load(model_weights)
            # add to list
            self.models.append(model)
    
    def sample_action(self, observation):
        """Std-weighted action sampling from ensemble of policies.

        Args:
            observation (torch.tensor): Observations for input into the models.

        Returns:
            Action (np.ndarray): The std-weighted action sampled from the ensemble of policies.
        """
        # sum of weighted actions
        w_act_sum = 0
        # sum of standard deviations
        std_sum = 0

        # get actions and standard deviations
        for model in self.models:
            # sample action
            act, std = model.forward(observation)

            # uncertainty weighting
            w_act_sum += (1 / std) * act
            std_sum += (1 / std) 

        # final action
        action = w_act_sum / std_sum

        return action