from gym.envs.registration import register

register(
    id="spider_fly_env/Grid-v0",
    entry_point="spider_fly_env.envs:GridEnv",
)
