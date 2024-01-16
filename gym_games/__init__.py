from gymnasium.envs.registration import register


register(
    id="gym_games/WindyGridWorld-v0",
    entry_point="gym_games.envs:WindyGridWorldEnv",
)

