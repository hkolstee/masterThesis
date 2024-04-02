from gymnasium.envs.registration import register

register(
    id="SpiderFlyGrid-v0",
    entry_point="spider_fly_env.envs:SpiderFlyEnv",
    max_episode_steps = 5000,
)
