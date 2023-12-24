from typing import Any, List, Mapping, Union
import numpy as np


def unmetThermalComfort(self, observations: List[Mapping[str, Union[int, float]]]) -> [float]:
    """Calculates the proportion of buildings in the environment that does not meet thermal 
    comfort requirements. If no central agent is used, a boolean int list is created where 1
    means thermal comfort not satisfied.

    Parameters
    ----------
    observations: List[Mapping[str, Union[int, float]]]
        List of all buildings observations at current citylean.citylearn.CityLearnEnv.time_step
        that are gotten from calling citylearn.building.Building.observations.
    
    Returns
    -------
    reward_list: [float]
        Either a list of 1s and 0s for each building when no central agent is used where a 1 
        means termal comfort not satisfied, and a 0 if thermal comfort is satisfied, or a list 
        with one float for the proportion of buildings that satisfy thermal comfort in a 
        central agent setting.
    """

    # 1 if building is occupied, 0 otherwise
    occupations = [0 if obs['occupant_count'] == 0 else 1 for obs in observations]

    # temperature difference to setpoint
    indoor_temp_deltas = [abs(obs["indoor_dry_bulb_temperature_set_point"] 
                              - obs["indoor_dry_bulb_temperature"]) for obs in observations]
    
    # 1 if building thermal comfort is not satisfied, 0 if it is satisfied
    # in the citylearn documentation a default temperature band of 2.0 is used
    thermal_comfort_violations = [1 if delta > 2.0 else 0 for delta in indoor_temp_deltas]

    # final rewards
    reward_list = []
    for violation, occupants in zip(thermal_comfort_violations, occupations):
        if (violation and occupants):
            reward_list.append(1)
        else:
            reward_list.append(0)

    # NOTE: negative reward given in citylearn example docs
    if self.central_agent:
        reward = [sum(reward_list) / len(reward_list)]
    else:
        reward = reward_list
        
def carbonEmissions(self, observations: List[Mapping[str, Union[int, float]]]) -> [float]:
    """Calculates the carbon emissions devided by the baseline emissions. In a centralized 
    setting, the sum is calculated. Otherwise, a building wise list is returned.

    Parameters
    ----------
    observations: List[Mapping[str, Union[int, float]]]
        List of all buildings observations at current citylean.citylearn.CityLearnEnv.time_step
        that are gotten from calling citylearn.building.Building.observations.
    
    Returns
    -------
    reward_list: [float]
        Carbon emissions devided by baseline.
    """
    emissions = [max(0, obs["net_electricity_consumption"] * obs["carbon_intensity"]) 
                 for obs in observations]
    
    # NOTE: negative reward given in citylearn example docs
    if self.central_agent:
        reward = [sum(emissions) / len(reward_list)]
    else:
        reward = reward_list