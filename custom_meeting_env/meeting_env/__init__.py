from gymnasium.envs.registration import register


register(
    id="Meeting-v0",
    entry_point="meeting_env.envs:MeetingEnv",
    max_episode_steps = 5000,
)
