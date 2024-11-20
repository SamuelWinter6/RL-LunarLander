import gym
import numpy as np
from gym_torcs.gym_torcs import TorcsEnv

class RacingEnv(gym.Env):
    def __init__(self):
        super(RacingEnv, self).__init__()
        self.env = TorcsEnv(vision=False, throttle=True, gear_change=False)

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)  # Steering, acceleration, brake
        self.observation_space = gym.spaces.Box(low=-np.
        inf, high=np.inf, shape=(29,), dtype=np.float32)

    def reset(self):
        # Reset the environment
        return self.env.reset(relaunch=True)

    def step(self, action):
        # Take an action in the environment
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def close(self):
        # Close the environment
        self.env.end()
