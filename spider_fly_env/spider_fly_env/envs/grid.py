import numpy as np
# import pygame
import copy
import time
import sys

import gymnasium as gym
from gymnasium import spaces

FLY = 2
SPIDER = 1
EMPTY = 0

class GridEnv(gym.Env):
    """
    This environment is a 2-dimensional grid, modelling the spider-and-fly 
    problem illustrated in the paper:
    
    "Multiagent Reinforcement Learning: Rollout and Policy Iteration", 
    Dimitri Bertsekas, Feb 2021.
    
    The problem involbes a grid space with a set number of spiders and one fly.
    The spiders move with perffect knowledge about the location of the other
    spiders and the fly. The actions the spiders can perform is to stay in its
    current location or move to one neighbouring location (not diagonal). The 
    fly moves randomly, without regard of spider location. The spider is 
    caught when it can not move becuase it is enclosed by 4 spiders, one on
    either side. The goal is to catch the fly at minimal costs, where each 
    transition to the next state will cost 1, until the fly is caught, then the
    cost becomes 0. 
    """
    metadata = {"render_modes": ["ascii"]}

    def __init__(self, size = 5, spiders = 4, render_mode = None):
        super().__init__()
        self.size = size
        self.nr_spiders = spiders

        # grid locations are integers [x,y], x,y in [0,..,size - 1].
        # each spider sees all spider locations and fly location
        spider_obs = spaces.Dict({
            "fly": spaces.Box(0, self.size - 1, (2,), dtype = np.int32),
        })
        for idx in range(self.nr_spiders):
            spider_obs["spider_" + str(idx)] = spaces.Box(0, self.size - 1, (2,), dtype = np.int32)

        # multi-agent observations, agent (spider) gets all spider locs + fly loc
        # action space is 5 (left, right, up, down, nothing) for each spider
        self.observation_space = []
        self.action_space = []
        for _ in range(self.nr_spiders):
            self.observation_space.append(copy.deepcopy(spider_obs))
            self.action_space.append(spaces.Discrete(5))

        # mapping from action to direction of this action on the grid (=delta (x, y))
        self._action_to_direction = {
            0: np.array([0, 0]),    # nothing        
            1: np.array([1, 0]),    # right
            2: np.array([-1, 0]),   # left
            3: np.array([0, 1]),    # up
            4: np.array([0, -1]),   # down
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

        # reset
        self.reset()

    def _get_obs(self):
        # flattens by itself
        return np.append(self._spider_locations, self._fly_location)
    
    def _create_state_matrix(self):
        self._state = np.zeros((self.size, self.size))

        # fly
        self._state[self._fly_location[0], self._fly_location[1]] = FLY
        # spiders
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
        print(content)
        # time.sleep(0.3)

    def reset(self, seed = None, render = False):
        # seed self.np_random
        super().reset(seed = seed)

        # random spider/fly locations 
        spawn_locs = set()
        self._spider_locations = []
        # random spider locations but can't already be a location of another spider
        # set() because faster (eventhough its not needed)
        for _ in range(self.nr_spiders):
            loc = self.np_random.integers(0, self.size, size = (2,))
            while tuple(loc) in spawn_locs:
                loc = self.np_random.integers(0, self.size, size = (2,))
            spawn_locs.add(tuple(loc))
            self._spider_locations.append(loc)
        # fly also has to spawn on a free space
        loc = self.np_random.integers(0, self.size, size = (2,))
        while tuple(loc) in spawn_locs:
            loc = self.np_random.integers(0, self.size, size = (2,))
        self._fly_location = loc
        
        # get observations
        observations = self._get_obs()

        # create state matrix
        self._create_state_matrix()

        if self.render_mode == "ascii" and render:
            # print grid, spiders, and flies 
            self._print_state_matrix()

        return observations, {}
    
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
                if not np.array_equal(new_loc, self._fly_location):
                    # wants to move to empty location so we allow
                    # first update state matrix
                    self._state[self._spider_locations[spider_idx][0], self._spider_locations[spider_idx][1]] = EMPTY
                    self._state[new_loc[0], new_loc[1]] = SPIDER
                    # change loc in list
                    self._spider_locations[spider_idx] = new_loc

    def move_fly(self):
        new_loc = self._fly_location + self._action_to_direction[np.random.randint(0, 5)]
        # check within walls
        if all((coord >= 0 and coord < self.size) for coord in new_loc):
            # check not on other spider
            if not np.any(np.all(new_loc == self._spider_locations, axis = 1)):
                # move
                # first update state matrix
                self._state[self._fly_location[0], self._fly_location[1]] = EMPTY
                self._state[new_loc[0], new_loc[1]] = FLY
                # change loc in list
                self._fly_location = new_loc

    def check_terminal(self):
        # create set of locations around fly
        sides = [tuple(side) for side in [self._action_to_direction[action] + self._fly_location for action in [1, 2, 3, 4]]]
        # check in grid array if flies are on these possible possitions
        count = 0
        self._print_state_matrix()
        for side in sides:
            # collides with wall
            if any((coord < 0 or coord >= self.size) for coord in side):
                count += 1
            # spider
            elif (self._state[side] == SPIDER):
                count += 1
        if count == 4:
            return True
        else:
            return False

    def step(self, actions):
        # we move the fly first
        self.move_fly()

        # Move each spider in a legal manner, if an illigal move is done, the
        # spider does nothing. The spiders move sequentially, so the new 
        # position of the previous spider will be used for deciding illigality.
        for idx, action in enumerate(actions):
            self.take_action(idx, action)

        # render
        if self.render_mode == "ascii":
            self._print_state_matrix()

        # check if terminal state has been reached
        terminal = self.check_terminal()

        # get reward, each step 
        if terminal:
            reward = 0
            self.reset(render = False)
        else:
            reward = 1

        # get observation
        observations = self._get_obs()

        # return obs, rew, done, truncated, info
        return observations, reward, terminal, False, {}

        
# env = GridEnv(render_mode = "ascii")
# for _ in range(100000):
#     obs, rew, term, _, _ = env.step(np.random.randint(0,5, (env.nr_spiders,)))
#     if term:
#         break


