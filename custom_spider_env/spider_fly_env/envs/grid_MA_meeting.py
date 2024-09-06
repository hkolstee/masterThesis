import numpy as np

from collections import Counter

import copy
import time
import sys
import functools

import gymnasium as gym
from gymnasium import spaces

from pettingzoo import ParallelEnv

AGENT = 1
EMPTY = 0

class grid_meeting_env(ParallelEnv):
    """

    """
    metadata = {"name": "GridMeeting-v0",
                "render_modes": ["ascii"]}

    def __init__(self, size = 5, agents = 4, max_timesteps = 100, render_mode = None):
        super().__init__()
        self.size = size
        self.nr_agents = agents
        self.timestep = 0
        self.max_steps = max_timesteps

        # create observation/action spaces 
        self.observation_spaces = dict()
        self.action_spaces = dict()
        self.agents = list()
        for agent_idx in range(self.nr_agents):
            # agent name string
            agent = "agent_" + str(agent_idx)
            self.agents.append(agent)

            # grid locations are integers [x,y], x,y in [0,..,size - 1].
            self.observation_spaces[agent] = spaces.Box(0, self.size - 1, (2 * self.nr_agents,), dtype = np.int64)
            # action space is 5 (left, right, up, down, nothing) for each AGENT
            self.action_spaces[agent] = spaces.Discrete(5)

        # mapping from action to direction of this action on the grid (=delta (x, y))
        self._action_to_direction = {
            0: np.array([0, 0]),    # nothing        
            1: np.array([1, 0]),    # right
            2: np.array([-1, 0]),   # left
            3: np.array([0, 1]),    # up
            4: np.array([0, -1]),   # down
        }
        self._action_to_all_direction = {
            0: np.array([0, 0]),    # nothing        
            1: np.array([1, 0]),    # right
            2: np.array([-1, 0]),   # left
            3: np.array([0, 1]),    # up
            4: np.array([0, -1]),   # down
            5: np.array([1, -1]),   # down-right
            6: np.array([1, 1]),    # up-right
            7: np.array([-1, -1]),  # down-left
            8: np.array([-1, 1]),   # up-left
        }

        # for use when debugging
        self.action_to_direction_string = {
            0: "nothing",        
            1: "right",
            2: "left",
            3: "up",
            4: "down",
        }

        # render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"], \
                "Render mode \"" + str(render_mode) + "\" not available, choose from: " + str(self.metadata["render_modes"])
        self.render_mode = render_mode

        # mapping from int identity to nice ascii chars for printing
        self._id_to_ascii = {
            AGENT: 'X',
            EMPTY: ' ',
        }

        # possible agent string name list needed for pettingzoo api
        self.possible_agents = [a for a in self.agents]

        # reset
        self.reset()

    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    # Action space defined here.
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def _get_obs(self):
        # flatten
        obs = np.array([coord for agent in self._agent_locations for coord in agent])
        return {a: obs for a in self.possible_agents}

    def _get_agent_locations(self):
        return {a: self._agent_locations[agent_idx] for (agent_idx, a) in enumerate(self.agents)}
    
    def _create_state_matrix(self):
        self._state = np.zeros((self.size, self.size))

        # agents
        if len(self._agent_locations) > 0:
            for agent_loc in self._agent_locations:
                self._state[agent_loc[0], agent_loc[1]] += AGENT

    def _print_state_matrix(self):
        # erase previous print
        # create char array
        content = np.zeros((self.size, self.size), dtype = 'U1')
        for x in range(self.size):
            for y in range(self.size):
                content[x,y] = self._state[x,y]
                # content[x,y] = self._id_to_ascii[self._state[x,y]]
        # print char array
        print(np.rot90(content))

    def _add_agents(self, number):
        # add agents randomly on empty locations
        for _ in range(number):
            loc = self.rng.integers(0, self.size, size = (2,)).tolist()
                
            # also add to state matrix
            self._state[tuple(loc)] = AGENT
            self._agent_locations.append(loc)
        
    def reset(self, seed = None, options = None):
        # set np random seed
        self.rng = np.random.default_rng(seed = seed)

        # reset agents
        self.agents = list()
        for agent_idx in range(self.nr_agents):
            # agent name string
            agent = "agent_" + str(agent_idx)
            self.agents.append(agent)

        # initialize locations
        self._agent_locations = []

        # create state matrix
        self._create_state_matrix()

        # random AGENT/fly locations 
        # random AGENT locations but can't already be a location of another AGENT
        # set() because faster (eventhough its not needed)
        self._add_agents(self.nr_agents)
        # get observations
        observations = self._get_obs()
    
        if self.render_mode == "ascii":
            # print grid, AGENTs, and flies 
            self._print_state_matrix()

        # info is necessary like this
        infos = {a: {} for a in self.possible_agents}

        return observations, infos
    
    def take_action(self, AGENT_idx, action):
        """
        We check if the move is legal (within walls, not on top of other AGENT or fly). 
        If illigal, the AGENT does nothing.
        """
        # AGENT stays in location, therefore always legal
        if action == 0:
            return self._action_to_direction[action]

        # AGENT wants to move to new_loc
        new_loc = self._agent_locations[AGENT_idx] + self._action_to_direction[action]
        # check if move is within walls
        if all((coord >= 0 and coord < self.size) for coord in new_loc):
            # first update state matrix
            self._state[self._agent_locations[AGENT_idx][0], self._agent_locations[AGENT_idx][1]] = EMPTY
            self._state[new_loc[0], new_loc[1]] += AGENT
            # change loc in list
            self._agent_locations[AGENT_idx] = new_loc.tolist()

    def _max_dist(self, AGENT_loc, fly_loc):
        """
        Returns max of x, y distance
        """
        abs_dist = np.abs(np.array(AGENT_loc) - np.array(fly_loc))
        
        return np.max(abs_dist)
    
    def _euclidean_dist(self, loc1, loc2):
        return np.sqrt(np.square(loc1[0] - loc2[0]) + np.square(loc1[1] - loc2[1]))

    def print_info(self):
        print("-------------------------------------")
        self._print_state_matrix()
        print("AGENT_LOCS", self._agent_locations)
        
    def check_score(self):
        # counts the largest group of agents on one spot
        counter = Counter([tuple(loc) for loc in self._agent_locations])
        _, count = counter.most_common(1)[0]
        
        return count
        
    def step(self, actions):
        # add step to counter
        self.timestep += 1

        # we move the fly first
        # self.move_fly()

        # Move each AGENT in a legal manner, if an illigal move is done, the
        # AGENT does nothing. The AGENTs move sequentially, so the new 
        # position of the previous AGENT will be used for deciding illigality.
        assert len(actions) > 0
        for idx, (agent, action) in enumerate(actions.items()):
            self.take_action(idx, action)

        # render
        if self.render_mode == "ascii":
            self._print_state_matrix()

        # check if terminal state has been reached
        agents_rew = self.check_score()

        # get reward, each step 
        if agents_rew == self.nr_agents:
            rewards = {a: 1 for a in self.possible_agents}
            terminals = {a: True for a in self.possible_agents}
        else:
            terminals = {a: False for a in self.possible_agents}
            rewards = {a: 0.001 * (agents_rew - self.nr_agents) for a in self.possible_agents}
        print(rewards)

        # get observation
        observations = self._get_obs()
        # infos
        infos = {a: {} for a in self.possible_agents}
        # terminals
        
        # truncations
        if self.max_steps == self.timestep:
            truncations = {a: True for a in self.possible_agents}
            self.timestep = 0
        else:
            truncations = {a: False for a in self.possible_agents}

        # also needed for pettingzoo api
        if any(terminals.values()) or all(truncations.values()):
            self.agents = []

        # return obs, rew, done, truncated, info
        return observations, rewards, terminals, truncations, infos


