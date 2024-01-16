"""
Wrapper the game env to make it can be used as coach training env
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from copy import deepcopy


class CoachWrapper(gym.Wrapper):
    def __init__(self, game_env, student):
        super().__init__(env=game_env)
        self.student = student
        self.student.game_env = game_env
        self.game_env = game_env
        self.observation_space = spaces.MultiDiscrete([
            self.game_env.observation_space.n,
            self.game_env.action_space.n
        ])
        self.action_space = spaces.Discrete(len(self.student.instructions_lib))
        self.policy_state = None
        self.last_policy_state = None

    def step(self, action=None):
        """
        Given instruction to calculate actual action and the reward
        :param action: instruction index, usage is moved outside for consistence with Delayed training
        :return:
        """
        self.last_policy_state = deepcopy(self.policy_state)
        game_obs, reward, terminated, truncated, info = super().step(self.student.actual_action_index)  # after step instruction
        self.student.q_learning_update(
            last_a=self.student.actual_action_index,
            reward=reward, done=terminated, evaluative_reward=False
        )
        action_idea = self.student.predict_action_idea(game_obs)   # next intention based on updated policy

        self.policy_state = [game_obs]
        self.policy_state.append(action_idea)
        return np.array(self.policy_state).astype(np.int32), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        game_obs, info = super().reset(seed=seed)
        self.last_policy_state = None
        action_idea = self.student.predict_action_idea(game_obs)
        self.policy_state = [game_obs]
        self.policy_state.append(action_idea)
        return np.array(self.policy_state).astype(np.int32), {}

    def render(self, mode='console'):
        pass

    def close(self):
        pass

