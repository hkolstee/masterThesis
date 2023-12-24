from typing import Any, List, Mapping, Union
from citylearn.reward_function import RewardFunction
from KPI_calculation_util import *

###################################################################
#####                Specify your reward here                 #####
###################################################################

class CustomReward(RewardFunction):
    """Calculates our custom user-defined multi-agent reward.
    
    This reward is a stepwise conversion of the final score
    calculation. 

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment
    """
    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)

        def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
            """Calculates the rewards.
            
            Parameters
            ----------
            observations: List[Mapping[str, Union[int, float]]]
                List of all buildings observations at current citylean.citylearn.CityLearnEnv.time_step
                that are gotten from calling citylearn.building.Building.observations.
                
            Returns
            -------
            reward: List[float]
                Ceward for transition to current timestep.
            """
            comfort = calculateComfort(observations)

            reward = 0.3 * comfort

            return reward

        def calculateComfort(self, observations: List[Mapping[str, Union[int, float]]]) -> float:
            """Calculates the comfort component of the reward function. This component of the reward 
            function consists of 1 key performance indicator: Unmet thermal comfort (U).
            
            Parameters
            ----------
            observations: List[Mapping[str, Union[int, float]]]
                List of all buildings observations at current citylean.citylearn.CityLearnEnv.time_step
                that are gotten from calling citylearn.building.Building.observations.
                
            Returns
            -------
            reward: List[float]
                Comfort reward for transition to current timestep.
            """

            return unmetThermalComfort(observations)



            