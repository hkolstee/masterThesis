from grid_MA_pettingzoo import SpiderFlyEnvMA

from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = SpiderFlyEnvMA()

    parallel_api_test(env, num_cycles=1_000_000)
