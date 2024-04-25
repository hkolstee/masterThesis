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
    This environment is a very easy testing environment.
    """
    metadata = {"name": "SpiderFlyGridMA-v0",
                "render_modes": ["ascii"]}

    def __init__(self, max_steps = 100, render_mode = None):
        super().__init__()
        self.max_steps = max_steps
        self.size = 7
        self.nr_spiders = 2
        self.nr_flies = 2
        self.timestep = 0

        # observations are vector locations of the spiders and the flies
        # each spider sees all spider locations and fly location
        spider_obs = spaces.Box(0, self.size - 1, (self.nr_spiders + self.nr_flies,), dtype = np.int64)

        # multi-agent observations, agent (spider) gets all spider locs + fly loc
        # action space is 3 (left, right, nothing) for each spider
        self.observation_spaces = dict()
        self.action_spaces = dict()
        self.agents = list()
        for agent_idx in range(self.nr_spiders):
            # agent name string
            agent = "spider_" + str(agent_idx)

            self.observation_spaces[agent] = copy.deepcopy(spider_obs)
            self.action_spaces[agent] = spaces.Discrete(3)
            self.agents.append(agent)

        # mapping from action to direction of this action on the grid (=delta (x, y))
        self._action_to_direction = {
            0: 0,    # nothing        
            1: 1,    # right
            2: -1,   # left
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

        # list for locations
        self._spider_locations = []
        self._fly_locations = []

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
        # append flattens by itself
        return {a: list(np.append(self._spider_locations, self._fly_locations)) for a in self.possible_agents}

    def _get_spider_locations(self):
        return {a: self._spider_locations[agent_idx] for (agent_idx, a) in enumerate(self.agents)}
    
    def _create_state_matrix(self):
        self._state = np.zeros(self.size)

        # fly
        for fly_loc in self._fly_locations:
            self._state[fly_loc] = FLY
        # spiders
        for spider_loc in self._spider_locations:
            self._state[spider_loc] = SPIDER

    def _print_state_matrix(self):
        # erase previous print
        # create char array
        content = np.zeros(self.size, dtype = 'U1')
        for x in range(self.size):
            content[x] = self._id_to_ascii[self._state[x]]
        # print char array
        print(content)

    def _add_spiders(self, number):
        # add spiders randomly on empty locations
        loc = self.rng.integers(0, self.size, size = 1)[0]

        for _ in range(number):
            loc = self.rng.integers(0, self.size, size = 1)[0]
            while loc in self._spider_locations or loc in self._fly_locations:
                loc = self.rng.integers(0, self.size, size = 1)[0]
            # also add to state matrix
            self._state[loc] = SPIDER
            self._spider_locations.append(loc)

    def _add_flies(self, number):
        # add flies randomly on empty locations
        loc = self.rng.integers(0, self.size, size = 1)[0]
        
        for _ in range(number):
            while loc in self._spider_locations or loc in self._fly_locations:
                loc = self.rng.integers(0, self.size, size = 1)[0]
            # also add to state matrix
            self._state[loc] = FLY
            self._fly_locations.append(loc)

    def reset(self, seed = None, options = None):
        # set np random seed
        self.rng = np.random.default_rng(seed = seed)

        # reset spiders and flies
        self._spider_locations = []
        self._fly_locations = []

        # create state matrix
        self._create_state_matrix()

        # random spider/fly locations 
        self._add_spiders(self.nr_spiders)
        self._add_flies(self.nr_flies)

        # get observations
        observations = self._get_obs()

        if self.render_mode == "ascii":
            # print grid, spiders, and flies 
            self._print_state_matrix()

        # info is necessary like this
        infos = {a: {} for a in self.possible_agents}

        # reset agents
        self.agents = [a for a in self.possible_agents]

        return observations, infos
    
    def take_action(self, spider_idx, action):
        """
        We check if the move is legal (within walls, not on top of other spider or fly). 
        If illigal, the spider does nothing.
        """
        # spider stays in location, therefore always legal
        if action == 0:
            return 0
            # return self._action_to_direction[action]

        # spider wants to move to new_loc
        new_loc = self._spider_locations[spider_idx] + self._action_to_direction[action]
        # check if move is within walls
        if (new_loc >= 0 and new_loc < self.size):
            # check for other spiders and fly, checking own loc is fine as it moves
            if not new_loc in self._spider_locations:
                # first update state matrix
                self._state[self._spider_locations[spider_idx]] = EMPTY
                self._state[new_loc] = SPIDER
                # change loc in list
                self._spider_locations[spider_idx] = new_loc
                
                # tries to take action ontop of fly, thus catching it
                if new_loc in self._fly_locations:
                    # "caught" a fly, therefore reward
                    # we also need to spawn a new fly
                    # and remove the caught fly
                    self._fly_locations.remove(new_loc)
                    self._add_flies(1)
                    return 1
        return 0        

    def step(self, actions):
        if len(self._fly_locations) > 2 or len(self._spider_locations) > 2:
            raise Exception("There was an error in the number of spiders and/or flies")
        # we move the fly first
        # self.move_fly()

        # Move each spider in a legal manner, if an illigal move is done, the
        # spider does nothing. The spiders move sequentially, so the new 
        # position of the previous spider will be used for deciding illigality.
        agent_rewards = []
        for idx, (agent, action) in enumerate(actions.items()):
            reward = self.take_action(idx, action)
            agent_rewards.append(reward)

        # render
        if self.render_mode == "ascii":
            self._print_state_matrix()

        # check if terminal state has been reached
        # terminal, occupied_sides = self.check_terminal()

        # get reward, each step 
        # if terminal:
        #     rewards = {a: rew for (a, rew) in zip(self.possible_agents, self.agent_rewards)}
        #     self.reset()
        # else:
        #     rewards = {a: -0.1 for a in self.possible_agents}
        # rewards = {a: np.mean(agent_rewards) for a in self.possible_agents}
        rewards = {a: rew for (a, rew) in zip(self.possible_agents, agent_rewards)}

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

        # add step to counter
        self.timestep += 1

        # return obs, rew, done, truncated, info
        return observations, rewards, terminals, truncations, infos


