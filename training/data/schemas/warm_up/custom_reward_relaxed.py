from typing import Any, List, Mapping, Union
from citylearn.reward_function import RewardFunction
# from KPI_calculation_util import *

###################################################################
#####                Specify your reward here                 #####
###################################################################

# NOTE: U, M, and S all KPIs are normalized by their baseline value 
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
        # daily peak electricity consumption for this day until the current timestep 
        # (10 am -> 10 timesteps, 4 pm -> 16 timesteps)
        # should be reset every day (hour = 0)
        self.daily_peak = 0.0
        # all time peak
        self.all_time_peak = 0.0
        # used to calculate average of the day until the current timestep
        self.total_elec_today = 0.0
        # net elec consumption over grid for current and previous timestep
        self.grid_net_elec_consumption = 0.0
        self.prev_grid_net_elec_consumption = 0.0

        def pre_calc_variables(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
            """For efficiency purposes. Variables in this function are used multiple times in different 
            functions"""

            # grid level total energy consumption
            self.grid_net_elec_consumption = sum([obs["net_electricity_consumption"] for obs in observations])

            # 1 if building is occupied, 0 otherwise
            self.occupations = [0 if obs['occupant_count'] == 0 else 1 for obs in observations]

            # temperature difference to setpoint
            indoor_temp_deltas = [abs(obs["indoor_dry_bulb_temperature_set_point"] 
                                    - obs["indoor_dry_bulb_temperature"]) for obs in observations]
            # 1 if building thermal comfort is not satisfied, 0 if it is satisfied
            # in the citylearn documentation a default temperature band of 2.0 is used
            self.thermal_comfort_violations = [1 if delta > 2.0 else 0 for delta in indoor_temp_deltas]


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
                Reward for transition to current timestep.
            """

            # pre calculate some instance variables for efficiency purposes
            pre_calc_variables(observations)
            
            # check if previous observations have been initialized 
            # NOTE: for the first timestep we will see a great reward for ramping (0 ramping)
            if (self.previous_observations != None):
                # for first timestep only
                self.prev_grid_net_elec_consumption = self.grid_net_elec_consumption

            # the four components of our reward
            comfort = calculateComfort(observations)
            emissions = calculateEmissions(observations)
            # grid = calculateGrid(observations, self.previous_observations)
            resilience = calculateResilience(observations)

            # grid rewards are left out to relax the reward function into a seperable function 
            reward = 0.3 * comfort + 0.1 * emissions + 0.3 * resilience
            # reward = 0.3 * comfort + 0.1 * emissions + 0.0 * grid + 0.3 * resilience

            # save this net elec consumption as the previous observation
            self.prev_grid_net_elec_consumption = self.grid_net_elec_consumption

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
            
            u = unmetThermalComfort(observations)

            return u

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

            g = carbonEmissions(observations)
            
            return g
            
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
            l = loadFactor(observations)
            d = dailyPeak(observations)
            a = allTimePeak(observations)
            
            # average
            reward = (r + l + d + a) / 4
 
            return reward

        def calculateResilience(self, observations: List[Mapping[str, Union[int, float]]]) -> float:
            """Calculates the grid resilience reward. This is based on two key performance indicators.
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

            m = thermalResilience(observations)
            s = normalizedUnservedEnergy(observations)

            # average over KPIs
            reward = (m + s) / 2

            return reward

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
                A list (or ratio in case of central agent) of which buildings violate thermal comfort. 
                Here, 1 means a violation of thermal comfort and a 0 means thermal comfort is achieved.
            """

            # rewards
            reward_list = []
            for violation, occupants in zip(self.thermal_comfort_violations, self.occupations):
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
            # NOTE: WHERE BASELINE?
            if self.central_agent:
                reward = [sum(emissions)]
            else:
                reward = emissions
                
            return reward
                
        def ramping(self, observations: List[Mapping[str, Union[int, float]]]) -> [float]:
            """Calculates the ramping of electricity consumption from last timestep to the
            current over the entire grid.

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
                Ramping of electrical consumption for this timestep over entire grid.
            """
            
            # absolute difference
            abs_delta = abs(self.grid_net_elec_consumption - self.prev_grid_net_elec_consumption)
            
            if self.central_agent:
                reward = [abs_delta]
            else:
                # NOTE: perhaps len(observations) gives a wrong len
                reward = [abs_delta for i in range(len(observations))]            
                
            return reward
            
        def loadFactor(self, observations: List[Mapping[str, Union[int, float]]]) -> [float]:
            """Ratio of daily average and peak consumption. This indicates the efficiency of electricity
            consumption. Here, we give the 1 - load factor, as to minimize the reward.

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
                1 - load factor of the entire grid. 
            """

            # needs to be reset if it is a new day (hour = [1...24])
            if (observations[0]["hour"] == 1):
                self.total_elec_today = 0.0
            
            # calculate average of today till the current step
            self.total_elec_today += self.grid_net_elec_consumption
            average = self.total_elec_today / observations[0]["hour"]

            # calculate ratio
            load_factor = average / self.daily_peak

            # NOTE: WHERE BASELINE?
            if self.central_agent:
                reward = [1 - load_factor]
            else:
                # NOTE: perhaps len(observations) gives a wrong len
                reward = [(1 - load_factor) for i in range(len(observations))]  


        def dailyPeak(self, observations: List[Mapping[str, Union[int, float]]]) -> [float]:
            """Daily peak as escalated reward.

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
                The difference between the, up until now for that day, max electricity consumption 
                and the current consumption if it is higher. 
            """

            # needs to be reset if it is a new day (hour = [1...24])
            if (observations[0]["hour"] == 1):
                self.daily_peak = 0
            
            # new peak consumption
            if (self.grid_net_elec_consumption > self.daily_peak):
                # set new peak
                self.daily_peak = self.grid_net_elec_consumption
                # reward is difference 
                delta = self.grid_net_elec_consumption - self.daily_peak

            # NOTE: WHERE BASELINE?
            if self.central_agent:
                reward = [delta]
            else:
                # NOTE: perhaps len(observations) gives a wrong len
                reward = [delta for i in range(len(observations))]            
                
            return reward
        
        def allTimePeak(self, observations: List[Mapping[str, Union[int, float]]]) -> [float]:
            """All-time peak as escalated reward.

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
                The difference between the, up until now, max electricity consumption 
                and the current consumption if it is higher. 
            """
            
            # new peak consumption
            if (self.grid_net_elec_consumption > self.all_time_peak):
                # set new peak
                self.all_time_peak = self.grid_net_elec_consumption
                # reward is difference 
                delta = self.grid_net_elec_consumption - self.daily_peak

            # NOTE: WHERE BASELINE?
            if self.central_agent:
                reward = [delta]
            else:
                # NOTE: perhaps len(observations) gives a wrong len
                reward = [delta for i in range(len(observations))]            
                
            return reward
        
        def thermalResilience(self, observations: List[Mapping[str, Union[int, float]]]) -> [float]:
            """Thermal resilience. Same as unmetThermalComfort but with the constraint that there has 
            to be a power outage.

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
                In case of power outage, a list (or ratio in case of central agent) of which
                buildings violate thermal comfort. Here, 1 means a violation of thermal comfort
                and a 0 means thermal comfort is achieved.
            """

            reward_list = []
            for violation, occupants, power_outage in zip(self.thermal_comfort_violations, 
                                                          self.occupations, 
                                                          observations["power_outage"]):
                # if thermal comfort violated, occupants in the building and there is a power outage
                if (violation and occupants and power_outage):
                    reward_list.append(1)
                else:
                    reward_list.append(0)

            # NOTE: negative reward given in citylearn example docs
            if self.central_agent:
                reward = [sum(reward_list) / len(reward_list)]
            else:
                reward = reward_list

        def normalizedUnservedEnergy(self, observations: List[Mapping[str, Union[int, float]]]) -> [float]:
            """Proportion of unmet demand due to supply shortage in a power outage.

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
                Ratio of served energy to the total energy demand when in a power outage.
            """
            
            # power outage status per building
            power = [obs["power_outage"] for obs in observations]

            reward_list = []
            for obs in observations:
                # current building power outage
                if obs["power_outage"]:
                    # total demand (cooling, dmh, non-shiftable load energy demand):
                    total_demand = obs["cooling_demand"] + obs["dhw_demand"] + obs["non_shiftable_load"]

                    # total energy supply (electrical storage)


                    # electrical storage
                    
                    # reward is ratio between realized and unserved energy demands.
                    reward = 0

            reward = [0.0]
            return reward

        def reset(self):
            """Used to reset variables at the start of an episode."""

            self.previous_observations = None
            self.daily_peak = 0.0
            self.all_time_peak = 0.0
            self.total_elec_today = 0.0
            self.grid_net_elec_consumption = 0.0
