import gymnasium as gym
import numpy as np
from gymnasium import spaces


class WindyGridWorldEnv(gym.Env):
    # attributes used out of env should be written in wrapper
    def __init__(self, render_mode=None):
        self.start = np.array([3, 0])
        self.goal = np.array([3, 7])
        self.world_height = 7
        self.world_width = 10
        self.max_steps = self.world_width * self.world_height
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]  # wind strength for each column
        self._agent_location = None
        self.observation_space = spaces.Discrete(self.world_height * self.world_width)
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([-1, 0]),    # up
            1: np.array([0, 1]),    # right
            2: np.array([1, 0]),   # down
            3: np.array([0, -1]),   # left
        }

    def _get_obs(self):
        return self._agent_location[0] * self.world_width + self._agent_location[1]

    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction - np.array([self.wind[self._agent_location[1]], 0]),
            np.array([0, 0]),
            np.array([self.world_height - 1, self.world_width - 1])
        )

        terminated = True if np.array_equal(self._agent_location, self.goal) else False
        reward = -1
        return self._get_obs(), reward, terminated, False, {}  # np.array(self.state).astype(np.int32)

    def reset(self, seed=None, options=None):
        self._agent_location = self.start
        return self._get_obs(), {}   # np.array(self.state).astype(np.int32)

    def render(self):
        pass

    def close(self):
        pass


