"""
Wrapper the windy grid world game to provide external attributes like: last state, state, goal reward...
"""

import gymnasium as gym
from copy import deepcopy


class WindyWrapper(gym.Wrapper):
    def __init__(self, env, goal_reward=0.0):
        super().__init__(env)
        self.goal_reward = goal_reward  # -1, 100
        self.last_state = None
        self.state = None
        self.step_count = 0

    def step(self, action):
        self.last_state = deepcopy(self.state)
        self.step_count += 1
        self.state, reward, terminated, truncated, info = super().step(action)
        if terminated:
            reward = self.goal_reward
        elif self.step_count == self.unwrapped.max_steps:     # not terminated
            truncated = True
        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.state, info = super().reset(seed=seed, options=options)
        self.step_count = 0
        self.last_state = None
        return self.state, info

