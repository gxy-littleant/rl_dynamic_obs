from gymnasium.envs.registration import register
register(
    id='MiniGrid-Dynamic-Obstacles-32*32-v0',
    entry_point="gymnasium.envs.classic_control.MiniGrid-Dynamic-Obstacles-32*32:DynamicObstaclesEnv",
)
