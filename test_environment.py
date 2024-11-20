import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "gym_torcs")))

from envs.racing_env import RacingEnv

env = RacingEnv()
obs = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Random actions
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
env.close()
print("Environment tested successfully!")
