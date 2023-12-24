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
        # for use in calculations requiring last timestep information (ex. ramping)
        # this works when assumption of linear successive calling through time is satisfied.
        self.previous_observations = None

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
            
            # check if previous observations have been initialized 
            # NOTE: for the first timestep we will see a great reward for ramping (0 ramping)
            if (self.previous_observations != None):
                # for first timestep only
                self.previous_observations = observations
            
            comfort = calculateComfort(observations)
            emissions = calculateEmissions(observations)
            grid = calculateGrid(observations, self.previous_observations)
            # weights based on 2023 citylearn challenge control track score
            # reward = 0.3 * comfort + 0.1 * emissions + 0.3 * Grid + 0.3 * Resilience
            reward = 0.3 * comfort + 0.1 * emissions

            # make this observation the previous observation
            self.previous_observations = observations

            return reward

        def calculateComfort(self, observations: List[Mapping[str, Union[int, float]]]) -> [float]:
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
                Comfort reward for transition to the current timestep.
            """
            return unmetThermalComfort(observations)

        def calculateEmissions(self, observations: List[Mapping[str, Union[int, float]]]) -> [float]:
            """Calculates the emissions component of the reward function. This component of the reward 
            function consists of 1 key performance indicator: Carbon emissions (G).
            
            Parameters
            ----------
            observations: List[Mapping[str, Union[int, float]]]
                List of all buildings observations at current citylean.citylearn.CityLearnEnv.time_step
                that are gotten from calling citylearn.building.Building.observations.
                
            Returns
            -------
            reward: List[float]
                Emissions reward devided by baseline for the transition to the current timestep.
            """
            return carbonEmissions(observations)
            
        def calculateGrid(self, observations: List[Mapping[str, Union[int, float]]]) -> float:
            """Calculates all grid-level components of the reward function. This component of 
            the reward function consists of 4 key performance indicator: ramping (R), 
            1 - load factor (L), daily electricity peak (D), all-time electricity peak (A).
            
            Parameters
            ----------
            observations: List[Mapping[str, Union[int, float]]]
                List of all buildings observations at current citylean.citylearn.CityLearnEnv.time_step
                that are gotten from calling citylearn.building.Building.observations.
                
            Returns
            -------
            reward: List[float]
                The average over the 4 KPIs.
            """
            r = ramping(observations, self.prev_observations)
            l = loadFactor(observations, self.prev_observations)
            # d
            # a
            
            return 