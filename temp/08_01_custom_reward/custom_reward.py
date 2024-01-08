from typing import Any, List, Mapping, Union
from citylearn.reward_function import RewardFunction
from citylearn.citylearn import EvaluationCondition

import numpy as np
import math
# from KPI_calculation_util import *

###################################################################
#####                Specify your reward here                 #####
###################################################################

# NOTE: All but U, M, and S KPIs are normalized by their baseline value 
# where the baseline is the result from when none of the distributed 
# energy resources (DHW storage system, battery, and heat pump) is 
# controlled.

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
        self.reset()

    def get_metadata(self):
        return self.env_metadata

    def pre_calc_variables(self, observations: List[Mapping[str, Union[int, float]]]):
        """
        For efficiency purposes. Variables computed in this function are used multiple times in 
        different class methods."""

        # grid level total energy consumption
        self.grid_net_elec_consumption = sum([obs["net_electricity_consumption"] for obs in observations])
        # for use in baseline calculations
        self.net_elec_consumption_baseline = []
        for obs in observations:
            if obs["power_outage"]:
                consumption = 0.
            else: 
                # We base this on demand, as demands are satisfied optimally, either through storage or 
                # device energy. For the baseline, no storage is available, therefore all demand will 
                # supplied through electrical energy from the grid.
                consumption = (obs["cooling_demand_without_control"] 
                                + obs["heating_demand_without_control"]
                                + obs["dhw_demand_without_control"] 
                                + obs["non_shiftable_load_without_control"])
            self.net_elec_consumption_baseline.append(consumption)
            
        self.grid_net_elec_consumption_baseline = sum(self.net_elec_consumption_baseline)

        # 1 if building is occupied, 0 otherwise
        self.occupations = [0 if obs['occupant_count'] == 0 else 1 for obs in observations]

        # temperature difference to setpoint
        indoor_temp_deltas = [abs(obs["indoor_dry_bulb_temperature_set_point"] 
                                    - obs["indoor_dry_bulb_temperature"]) 
                                for obs in observations]
        # 1 if building thermal comfort is not satisfied, 0 if it is satisfied
        # in the citylearn documentation a default temperature band of 2.0 is used
        self.thermal_comfort_violations = [1 if delta > 2.0 else 0 for delta in indoor_temp_deltas]
        
        # daily peak needs to be reset if it is a new day (hour = [1...24])
        if (observations[0]["hour"] == 1):
            self.daily_peak = 0.
            self.daily_peak_baseline = 0.
            self.total_elec_today = 0.
            self.total_elec_today_baseline = 0.
        
        # if a new peak is achieved, we need to know the difference to the last daily peak
        if (self.grid_net_elec_consumption > self.daily_peak):
            self.peak_delta = self.grid_net_elec_consumption - self.daily_peak
            self.daily_peak = self.grid_net_elec_consumption
        else:
            self.peak_delta = 0.
        if (self.grid_net_elec_consumption_baseline > self.daily_peak_baseline):
            self.peak_delta_baseline = self.grid_net_elec_consumption_baseline - self.daily_peak_baseline
            self.daily_peak_baseline = self.grid_net_elec_consumption_baseline
        else: 
            self.peak_delta_baseline = 0.

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Calculates the rewards.
        
        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all buildings observations at current citylean.citylearn.CityLearnEnv.time_step
            that are gotten from calling citylearn.building.Building.observations.
            
        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.
        """

        # pre calculate some instance variables for efficiency purposes
        self.pre_calc_variables(observations)
        
        # check if previous observations have been initialized 
        # NOTE: for the first timestep we will see a great reward for ramping (0 ramping)
        if not self.prev_net_elec_consumption_baseline:     # checks if list is empty
            # for first timestep only
            self.prev_grid_net_elec_consumption = self.grid_net_elec_consumption
            self.prev_grid_net_elec_consumption_baseline = self.grid_net_elec_consumption_baseline
            self.prev_net_elec_consumption_baseline = self.net_elec_consumption_baseline
            
        # the four components of our reward
        # weights based on 2023 citylearn challenge control track score
        self.comfort = 0.3 * self.calculateComfort(observations)
        self.emissions = 0.1 * self.calculateEmissions(observations)
        self.grid = 0.3 * self.calculateGrid(observations)
        self.resilience = 0.3 * self.calculateResilience(observations)

        # weights based on 2023 citylearn challenge control track score
        # reward = 0.3 * comfort + 0.1 * emissions + 0.3 * grid + 0.3 * resilience
        reward = self.comfort + self.emissions + self.grid + self.resilience

        # save this net elec consumption as the previous observation
        self.prev_grid_net_elec_consumption = self.grid_net_elec_consumption
        self.prev_grid_net_elec_consumption_baseline = self.grid_net_elec_consumption_baseline
        self.prev_net_elec_consumption_baseline = self.net_elec_consumption_baseline

        # negative such that algorithms can maximize the reward
        return -reward

    def calculateComfort(self, observations: List[Mapping[str, Union[int, float]]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Calculates the comfort component of the reward function. This component of the reward 
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
        
        self.u = self.unmetThermalComfort(observations)

        return self.u

    def calculateEmissions(self, observations: List[Mapping[str, Union[int, float]]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Calculates the emissions component of the reward function. This component of the reward 
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

        g = self.carbonEmissions(observations)
        
        return g
        
    def calculateGrid(self, observations: List[Mapping[str, Union[int, float]]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Calculates all grid-level components of the reward function. This component of the reward 
        function consists of 4 key performance indicator: ramping (R), 1 - load factor (L), daily 
        electricity peak (D), all-time electricity peak (A).
        
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

        self.r = self.ramping(observations)
        # order of dailyPeak() and loadFactor() important!
        #   the daily peak gets updated which is important for the load factor calculation.
        self.d = self.dailyPeak(observations)
        self.l = self.loadFactor(observations)
        self.a = self.allTimePeak(observations)
        
        # average
        reward = (self.r + self.d + self.l + self.a) / 4

        return reward

    def calculateResilience(self, observations: List[Mapping[str, Union[int, float]]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Calculates the grid resilience reward. This is based on two key performance indicators.
        Namely, thermal resilience (M) and normalized unserved energy (S).
        
        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all buildings observations at current citylean.citylearn.CityLearnEnv.time_step
            that are gotten from calling citylearn.building.Building.observations.
            
        Returns
        -------
        reward: List[float]
            The grid resilience reward.
        """

        m = self.thermalResilience(observations)
        s = self.normalizedUnservedEnergy(observations)

        # print("m", m)
        # print("s", s)

        # average over KPIs
        reward = (m + s) / 2

        return reward

    def unmetThermalComfort(self, observations: List[Mapping[str, Union[int, float]]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Calculates the proportion of buildings in the environment that does not meet thermal 
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
            A list (or ratio in case of central agent) of which buildings violate thermal comfort. 
            Here, 1 means a violation of thermal comfort and a 0 means thermal comfort is achieved.
        """

        # rewards
        reward_list = []
        for violation, occupancy in zip(self.thermal_comfort_violations, self.occupations):
            if (violation and occupancy):
                reward_list.append(1)
            else:
                reward_list.append(0)

        # NOTE: negative reward given in citylearn example docs
        if self.central_agent:
            reward = [sum(reward_list) / len(reward_list)]
        else:
            reward = reward_list
            
        return np.array(reward)

    # NOTE: Baseline
    def carbonEmissions(self, observations: List[Mapping[str, Union[int, float]]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Calculates the carbon emissions devided by the baseline emissions. In a centralized 
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
        
        emissions = 0.0
        emissions_baseline = 0.0
        for idx, obs in enumerate(observations):
            # emission
            emissions += max(0, obs["net_electricity_consumption"] * obs["carbon_intensity"]) 
            
            # baseline                
            emissions_baseline += max(0, self.net_elec_consumption_baseline[idx] * obs["carbon_intensity"])
                        
        # devision by 0.0 check
        if (emissions_baseline < 0.001):
            reward = 0.0
        else: 
            reward = emissions / emissions_baseline
            
        
            
        if (math.isnan(reward) or math.isinf(reward)):
            print("Carbon emissions")
            reward = 0.0
                        
        if self.central_agent:
            reward = [reward]
        else:
            reward = [reward for i in range(len(observations))]
            
        return np.array(reward)
        
    def ramping(self, observations: List[Mapping[str, Union[int, float]]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Calculates the ramping of electricity consumption from last timestep to the
        current over the entire grid.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all buildings observations at current citylean.citylearn.CityLearnEnv.time_step
            that are gotten from calling citylearn.building.Building.observations.

        Returns
        -------
        reward_list: [float]
            Ramping of electrical consumption for this timestep over entire grid.
        """
        
        # absolute difference
        abs_delta = abs(self.grid_net_elec_consumption - self.prev_grid_net_elec_consumption)
        
        # baseline
        abs_delta_baseline = abs(self.grid_net_elec_consumption_baseline
                                    - self.prev_grid_net_elec_consumption_baseline)
        
        # zero devision check
        if (abs_delta_baseline < 0.001):
            reward = 0.0
        else:
            reward = abs_delta / abs_delta_baseline
            
        if (math.isnan(reward) or math.isinf(reward)):
            print("Ramping")
        
        if self.central_agent:
            reward = [reward]
        else:
            reward = [reward for i in range(len(observations))]            
            
        return np.array(reward)
    
    # NOTE: Baseline
    def loadFactor(self, observations: List[Mapping[str, Union[int, float]]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Ratio of daily average and peak consumption. This indicates the efficiency of electricity
        consumption. Here, we give the 1 - load factor, as to minimize the reward.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all buildings observations at current citylean.citylearn.CityLearnEnv.time_step
            that are gotten from calling citylearn.building.Building.observations.

        Returns
        -------
        reward_list: [float]
            1 - load factor of the entire grid. 
        """
        
        # calculate average of today till the current step
        self.total_elec_today += self.grid_net_elec_consumption
        average = self.total_elec_today / observations[0]["hour"]
        # baseline
        self.total_elec_today_baseline += self.grid_net_elec_consumption_baseline
        average_baseline = self.total_elec_today_baseline / observations[0]["hour"]

        # calculate ratio
        load_factor = 1 - (average / self.daily_peak)
        load_factor_baseline = 1 - (average_baseline / self.daily_peak_baseline)
        
  
        # zero devision check
        if (load_factor_baseline < 0.01):
            reward = 0.0
        else:
            # normalize
            reward = load_factor / load_factor_baseline

        if (math.isnan(reward) or math.isinf(reward)):
            print("Load factor")
            reward = 0.0

        if self.central_agent:
            reward = [reward]
        else:
            # NOTE: perhaps len(observations) gives a wrong len
            reward = [reward for i in range(len(observations))]  

        return np.array(reward)

    # NOTE: Baseline
    def dailyPeak(self, observations: List[Mapping[str, Union[int, float]]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Daily peak as escalated reward.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all buildings observations at current citylean.citylearn.CityLearnEnv.time_step
            that are gotten from calling citylearn.building.Building.observations.

        Returns
        -------
        reward_list: [float]
            The difference between the, up until now for that day, max electricity consumption 
            and the current consumption if it is higher. 
        """

        # Score calculation = average over all days
        #   therefore, to reflect the score calculation in this reward
        #   the reward should be devided by the number of days in the 
        #   simulation. We can see the number of days in the metadata.
        
        # zero devision check
        if (self.peak_delta_baseline < 0.01):
            reward = 0.0
        else:
            reward = self.peak_delta / self.peak_delta_baseline   
            
        # Daily peak Score calculation = average over all days
        #   therefore, to reflect the score calculation in this reward
        #   the reward should be devided by the number of days in the 
        #   simulation. We can see the number of days in the metadata.
        reward /= (self.env_metadata["simulation_time_steps"] 
                    * (self.env_metadata["seconds_per_time_step"] 
                        / (60*60*24)))
        
        if (math.isnan(reward) or math.isinf(reward)):
            print("Daily peak")

        if self.central_agent:
            reward = [reward]
        else:
            reward = [reward for i in range(len(observations))]            
            
        return np.array(reward)

    def allTimePeak(self, observations: List[Mapping[str, Union[int, float]]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        All-time peak as escalated reward.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all buildings observations at current citylean.citylearn.CityLearnEnv.time_step
            that are gotten from calling citylearn.building.Building.observations.

        Returns
        -------
        reward_list: [float]
            The difference between the, up until now, max electricity consumption 
            and the current consumption if it is higher. 
        """
        
        delta = 0.0
        delta_baseline = 0.0
        
        # new peak consumption
        if (self.grid_net_elec_consumption > self.all_time_peak):
            # set new peak
            self.all_time_peak = self.grid_net_elec_consumption
            # reward is difference 
            delta = self.grid_net_elec_consumption - self.daily_peak
            
        # baseline
        if (self.grid_net_elec_consumption_baseline > self.all_time_peak):
            delta_baseline = self.grid_net_elec_consumption_baseline - self.all_time_peak
            
        # zero devision check
        if (delta_baseline < 0.001):
            reward = 0.0
        else:
            reward = delta / delta_baseline   
            
        if (math.isnan(reward) or math.isinf(reward)):
            print("All-time peak")

        if self.central_agent:
            reward = [reward]
        else:
            reward = [reward for i in range(len(observations))]             
        
        return np.array(reward)
    
    def thermalResilience(self, observations: List[Mapping[str, Union[int, float]]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Thermal resilience. Same as unmetThermalComfort but with the constraint that there has 
        to be a power outage.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all buildings observations at current citylean.citylearn.CityLearnEnv.time_step
            that are gotten from calling citylearn.building.Building.observations.

        Returns
        -------
        reward_list: [float]
            In case of power outage, a list (or ratio in case of central agent) of which
            buildings violate thermal comfort. Here, 1 means a violation of thermal comfort
            and a 0 means thermal comfort is achieved.
        """

        reward_list = []
        for violation, occupants, observation in zip(self.thermal_comfort_violations, 
                                                     self.occupations, 
                                                     observations):
            # if thermal comfort violated, occupants in the building and there is a power outage
            if (violation and occupants and observation["power_outage"]):
                reward_list.append(1)
            else:
                reward_list.append(0)

        # NOTE: negative reward given in citylearn example docs
        if self.central_agent:
            reward = [sum(reward_list) / len(reward_list)]
        else:
            reward = reward_list            
            
        return np.array(reward)

    def normalizedUnservedEnergy(self, observations: List[Mapping[str, Union[int, float]]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Proportion of unmet demand due to supply shortage in a power outage.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all buildings observations at current citylean.citylearn.CityLearnEnv.time_step
            that are gotten from calling citylearn.building.Building.observations.

        Returns
        -------
        reward_list: [float]
            Ratio of served energy to the total energy demand when in a power outage.
        """
        
        # power outage status per building
        power = [obs["power_outage"] for obs in observations]

        reward_list = []
        for obs in observations:
            # current building power outage
            if obs["power_outage"]:
                # total demand (cooling, dmh, non-shiftable load energy demand):
                expected = obs["cooling_demand"] + obs["heating_demand"]\
                            + obs["dhw_demand"] + obs["non_shiftable_load"]

                # total energy supply (electrical storage)
                # served = obs["cooling_electricity_consumption"] + obs["heating_electricity_consumption"]\
                #          + obs["dhw_storage_electricity_consumption"] + obs[""]

                # NOTE: These are legitimate observations, but not active in the citylearn challenge
                # ..._storage_electricity_consumption = 
                #       Positive values indicate `..._device` electricity consumption to charge `..._storage` 
                #       while negative values indicate avoided `..._device` electricity consumption by 
                #       discharging `cooling_storage` to meet `..._demand`.
                served = abs(obs["cooling_storage_electricity_consumption"]
                                + obs["heating_storage_electricity_consumption"]
                                + obs["dhw_storage_electricity_consumption"]
                                + obs["electrical_storage_electricity_consumption"])
                # OR 
                # ..._electricity_consumption = 
                #       `..._device` net electricity consumption in meeting cooling demand and `..._storage` energy 
                #       demand time series, in [kWh]. 
                # served = abs(obs["cooling_electricity_consumption"]
                #              + obs["heating_electricity_consumption"]
                #              + obs["dhw_electricity_consumption"]
                #              + obs["electrical_electricity_consumption"])
                        
                # in docs cost functions: served_energy = b.energy_from_cooling_device + b.energy_from_cooling_storage\
                                                            # + b.energy_from_heating_device + b.energy_from_heating_storage\
                                                                # + b.energy_from_dhw_device + b.energy_from_dhw_storage\
                                                                    # + b.energy_to_non_shiftable_load
                


                # reward is 1 minus ratio between realized and unserved energy demands.
                # NOTE: different from the docs, there they just take the ratio
                reward = 1 - (served / expected)

            # no power outage
            else:
                reward = 0.0
                
            if (math.isnan(reward) or math.isinf(reward)):
                print("Unserved")
                
            reward_list.append(reward)
        
        if self.central_agent:
            reward = [sum(reward_list) / len(reward_list)]
        else:
            reward = reward_list

        return np.array(reward)

    def unservedEnergyAlternative(self, observations: List[Mapping[str, Union[int, float]]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Alternative to unmet demand due to power outage. This reward is a simplified rough estimation 
        of the unmet demand by just using the current demand if there is a power outage. 
        
        This can be seen as a rough estimate, as when demands can not be met, the following demands will
        (probably) be even greater to meet optimal thermal comfort. If there is a power outage, the 
        storage devices will optimally meet these demands untill they are empty, which keeps the 
        indoor temperature close to the optimal. Once they are empty, the temperature will divert from
        the optimal, resulting in larger demands. Therefore, the thought process is that the higher the
        current demand, the larger the unmet demand has been.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all buildings observations at current citylean.citylearn.CityLearnEnv.time_step
            that are gotten from calling citylearn.building.Building.observations.
        prev_observations: List[Mapping[str, Union[int, float]]]
            Observations from the previous timestep.

        Returns
        -------
        reward_list: [float]
            
        """
        
        reward_list = []
        for obs in observations:
            # current building power outage
            if obs["power_outage"]:
                reward = obs["cooling_demand"] + obs["heating_demand"] + obs["dhw_demand"]
            else: 
                reward = 0.0
            reward_list.append(reward)
            
        if self.central_agent:
            reward = [sum(reward_list) / len(reward_list)]
        else:
            reward = reward_list

        return np.array(reward)


    def reset(self):
        """
        Used to reset variables at the start of an episode.
        """
        # daily peak electricity consumption for this day until the current timestep 
        # (10 am -> 10 timesteps, 4 pm -> 16 timesteps)
        # should be reset every day (hour = 0)
        self.daily_peak = 0.
        self.daily_peak_baseline = 0.
        # all time peak
        self.all_time_peak = 0.
        # used to calculate average of the day until the current timestep
        self.total_elec_today = 0.
        self.total_elec_today_baseline = 0.
        # net elec consumption over grid for current and previous timestep
        self.grid_net_elec_consumption = 0.
        self.prev_grid_net_elec_consumption = 0.
        self.grid_net_elec_consumption_baseline = 0.
        self.prev_net_elec_consumption_baseline = []
        self.prev_grid_net_elec_consumption_baseline = 0.
        # to access KPIs for tensorboard plots
        self.comfort = 0.
        self.emissions = 0.
        self.grid = 0.
        self.resilience = 0.
        self.u = 0.
        self.g = 0.
        self.r = 0.
        self.d = 0.
        self.l = 0.
        self.a = 0.
        self.m = 0.
        self.s = 0.
