import numpy as np

import copy
import time
import sys
import functools

import gymnasium as gym
from gymnasium import spaces

from pettingzoo import ParallelEnv

FLY = 2
SPIDER = 1
EMPTY = 0

class SpiderFlyEnvMA(ParallelEnv):
    """
    This environment is a 2-dimensional grid, modelling the spider-and-fly 
    problem illustrated in the paper:
    
    "Multiagent Reinforcement Learning: Rollout and Policy Iteration", 
    Dimitri Bertsekas, Feb 2021.
    
    The problem involves a grid space with a set number of spiders and one fly.
    The spiders move with perffect knowledge about the location of the other
    spiders and the fly. The actions the spiders can perform is to stay in its
    current location or move to one neighbouring location (not diagonal). The 
    fly moves randomly, without regard of spider location. The spider is 
    caught when it can not move becuase it is enclosed by 4 spiders, one on
    either side. The goal is to catch the fly at minimal costs, where each 
    transition to the next state will cost 1, until the fly is caught, then the
    cost becomes 0. 
    """
    metadata = {"name": "SpiderFlyGridMA-v0",
                "render_modes": ["ascii"]}

    def __init__(self, size = 5, spiders = 4, flies = 1, max_timesteps = 1000, render_mode = None):
        super().__init__()
        self.size = size
        self.nr_spiders = spiders
        self.nr_flies = flies
        self.timestep = 0
        self.max_steps = max_timesteps

        # create observation/action spaces 
        self.observation_spaces = dict()
        self.action_spaces = dict()
        self.agents = list()
        for agent_idx in range(self.nr_spiders):
            # agent name string
            agent = "spider_" + str(agent_idx)
            self.agents.append(agent)

            # grid locations are integers [x,y], x,y in [0,..,size - 1].
            # each spider sees all spider locations and fly locations (in coordinates) 
            self.observation_spaces[agent] = spaces.Box(0, self.size - 1, (2 * self.nr_spiders + 2 * self.nr_flies,), dtype = np.int64)
            # action space is 5 (left, right, up, down, nothing) for each spider
            self.action_spaces[agent] = spaces.Discrete(5)


        # mapping from action to direction of this action on the grid (=delta (x, y))
        self._action_to_direction = {
            0: np.array([0, 0]),    # nothing        
            1: np.array([1, 0]),    # right
            2: np.array([-1, 0]),   # left
            3: np.array([0, 1]),    # up
            4: np.array([0, -1]),   # down
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
            SPIDER: 'X',
            FLY: 'O',
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
    
    def _euclidean_dist(self, loc1, loc2):
        return np.sqrt(np.sqaure(loc1[0] - loc2[0]) + np.square(loc1[1] - loc2[1]))

    def _get_obs(self):
        # append flattens by itself
        return {a: np.array(np.append(self._spider_locations, self._fly_locations)) for a in self.possible_agents}

    def _get_spider_locations(self):
        return {a: self._spider_locations[agent_idx] for (agent_idx, a) in enumerate(self.agents)}
    
    def _create_state_matrix(self):
        self._state = np.zeros((self.size, self.size))

        # fly
        if len(self._fly_locations) > 0:
            for fly_loc in self._fly_locations:
                self._state[fly_loc[0], fly_loc[1]] = FLY
        # spiders
        if len(self._spider_locations) > 0:
            for spider_loc in self._spider_locations:
                self._state[spider_loc[0], spider_loc[1]] = SPIDER

    def _print_state_matrix(self):
        # erase previous print
        # create char array
        content = np.zeros((self.size, self.size), dtype = 'U1')
        for x in range(self.size):
            for y in range(self.size):
                content[x,y] = self._id_to_ascii[self._state[x,y]]
        # print char array
        print(np.rot90(content))

    def _add_spiders(self, number):
        # add spiders randomly on empty locations
        for _ in range(number):
            loc = self.rng.integers(0, self.size, size = (2,)).tolist()
            while loc in self._spider_locations or loc in self._fly_locations:
                loc = self.rng.integers(0, self.size, size = (2,)).tolist()
                
            # also add to state matrix
            self._state[tuple(loc)] = SPIDER
            self._spider_locations.append(loc)
        
    def _add_flies(self, number):
        # add flies randomly on empty locations
        for _ in range(number):
            loc = self.rng.integers(0, self.size, size = (2,)).tolist()
            while loc in self._spider_locations or loc in self._fly_locations:
                loc = self.rng.integers(0, self.size, size = (2,)).tolist()

            # also add to state matrix
            self._state[tuple(loc)] = FLY
            self._fly_locations.append(loc)

    def reset(self, seed = None, options = None):
        # set np random seed
        self.rng = np.random.default_rng(seed = seed)

        # reset agents
        self.agents = list()
        for agent_idx in range(self.nr_spiders):
            # agent name string
            agent = "spider_" + str(agent_idx)
            self.agents.append(agent)

        # initialize locations
        self._spider_locations = []
        self._fly_locations = []

        # create state matrix
        self._create_state_matrix()

        # random spider/fly locations 
        # random spider locations but can't already be a location of another spider
        # set() because faster (eventhough its not needed)
        self._add_flies(self.nr_flies)
        self._add_spiders(self.nr_spiders)
        # get observations
        observations = self._get_obs()

        if self.render_mode == "ascii":
            # print grid, spiders, and flies 
            self._print_state_matrix()

        # info is necessary like this
        infos = {a: {} for a in self.possible_agents}

        return observations, infos
    
    def take_action(self, spider_idx, action):
        """
        We check if the move is legal (within walls, not on top of other spider or fly). 
        If illigal, the spider does nothing.
        """
        # spider stays in location, therefore always legal
        if action == 0:
            return self._action_to_direction[action]

        # spider wants to move to new_loc
        new_loc = self._spider_locations[spider_idx] + self._action_to_direction[action]
        # check if move is within walls
        if all((coord >= 0 and coord < self.size) for coord in new_loc):
            # check for other spiders and fly, checking own loc is fine as it moves
            if not np.any(np.all(new_loc == self._spider_locations, axis = 1)):
                if not np.any(np.all(new_loc == self._fly_locations, axis = 1)):
                    # wants to move to empty location so we allow
                    # first update state matrix
                    self._state[self._spider_locations[spider_idx][0], self._spider_locations[spider_idx][1]] = EMPTY
                    self._state[new_loc[0], new_loc[1]] = SPIDER
                    # change loc in list
                    self._spider_locations[spider_idx] = new_loc.tolist()

    def move_flies(self):
        for idx, fly_loc in enumerate(self._fly_locations):
            # random direction
            new_loc = fly_loc + self._action_to_direction[self.rng.integers(0, 5)]
            # check within walls
            if all((coord >= 0 and coord < self.size) for coord in new_loc):
                # check not on other spider or fly
                if not np.any(np.all(new_loc == self._spider_locations, axis = 1)):
                    if not np.any(np.all(new_loc == self._fly_locations, axis = 1)):
                        # move
                        # first update state matrix
                        self._state[fly_loc[0], fly_loc[1]] = EMPTY
                        self._state[new_loc[0], new_loc[1]] = FLY
                        # change loc in list
                        self._fly_locations[idx] = new_loc

    def move_fly_random_loc(self, idx):
        loc = self.rng.integers(0, self.size, size = (2,)).tolist()
        while loc in self._spider_locations or loc in self._fly_locations:
            loc = self.rng.integers(0, self.size, size = (2,)).tolist()
        
        # first update state matrix
        self._state[self._fly_locations[idx][0], self._fly_locations[idx][1]] = EMPTY
        self._state[loc[0], loc[1]] = FLY
        # found free spot
        self._fly_locations[idx] = loc

    def check_caught(self):
        flies_caught = [False for _ in range(len(self._fly_locations))]

        for fly_idx, fly_loc in enumerate(self._fly_locations):
            # create set of locations around fly
            sides = [tuple(side) for side in [self._action_to_direction[action] + fly_loc for action in [1, 2, 3, 4]]]
            # check in grid array if spiders are on these possible possitions
            count = 0
            spider_rew = np.repeat(0, self.nr_spiders)
            for side in sides:
                # collides with wall
                if any((coord < 0 or coord >= self.size) for coord in side):
                    count += 1
                # spider
                elif (self._state[side] == SPIDER):
                    count += 1
                    for (spider_idx, loc) in enumerate(self._spider_locations):
                        if tuple(loc) == side:
                            spider_rew[spider_idx] += 2
            # if all sides are occupied
            if count == 4 or sum(spider_rew) == self.nr_spiders:
                flies_caught[fly_idx] = True
                # remove fly and spawn somewhere else
                self.move_fly_random_loc(fly_idx)

            # add eucledian distance to rewards
            # NOTE: BROKEN FOR > 1 FLY, HAVE TO FIND FAST SOLUTION FOR MORE FLIES
            for idx in range(len(spider_rew)):
                spider_rew[idx] -= self._euclidean_dist(self._spider_locations[idx], self._fly_locations[0])
        
        return flies_caught, spider_rew

    def step(self, actions):
        # add step to counter
        self.timestep += 1

        # we move the fly first
        # self.move_fly()

        # Move each spider in a legal manner, if an illigal move is done, the
        # spider does nothing. The spiders move sequentially, so the new 
        # position of the previous spider will be used for deciding illigality.
        assert len(actions) > 0
        for idx, (agent, action) in enumerate(actions.items()):
            self.take_action(idx, action)

        # render
        if self.render_mode == "ascii":
            self._print_state_matrix()

        # check if terminal state has been reached
        flies_caught, spiders_rew = self.check_caught()

        # get reward, each step 
        if any(flies_caught):
            rewards = {a: 1 for a in self.possible_agents}
            # terminals = {a: True for a in self.possible_agents}
        else:
            rewards = {a: 0.001 * (rew) for (rew, a) in zip(spiders_rew, self.possible_agents)}

        # get observation
        observations = self._get_obs()
        # infos
        infos = {a: {} for a in self.possible_agents}
        # terminals
        terminals = {a: False for a in self.possible_agents}
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


