from typing import Any, List, Mapping, Union
import numpy as np


def unmetThermalComfort(self, observations: List[Mapping[str, Union[int, float]]]) -> float:
    """Calculates the proportion of buildings in the environment that does not meet thermal 
    comfort requirements.

    Parameters
    ----------
    observations: List[Mapping[str, Union[int, float]]]
        List of all buildings observations at current citylean.citylearn.CityLearnEnv.time_step
        that are gotten from calling citylearn.building.Building.observations.
    
    Returns
    -------
    proportion: float
        The proportion of occupied buildings of which the indoor dry bulb temperature is 
        currently not in range of the thermal comfort band around the thermal comfort 
        setpoint. 
    """

    # 1 if building is occupied, 0 otherwise
    occupations = [0 if observation['occupant_count'] == 0 else 1 for observation in observations]

    # temperature difference to setpoint
    indoor_temp_deltas = [abs(observation["indoor_dry_bulb_temperature_set_point"] 
                              - observation["indoor_dry_bulb_temperature"]) for observation in observations]
    
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
        reward = 
