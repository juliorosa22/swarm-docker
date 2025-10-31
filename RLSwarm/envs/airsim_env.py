import numpy as np
import airsim
import gymnasium as gym
from gymnasium import spaces


class AirSimEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self,render_mode=None):
        self.viewer = None
        self.render_mode=render_mode

    #def __del__(self):
    #    raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _compute_reward(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def render(self):
        return self._get_obs()
