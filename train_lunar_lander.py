import gymnasium as gym
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import csv
import os

# Configurations
TOTAL_TIMESTEPS = 50000
CHECKPOINT_INTERVAL = 10000
MODEL_PATH = "dqn_lunarlander_v3"
REPLAY_BUFFER_PATH = "replay_buffer.pkl"
LOG_FILE = "training_data.csv"

# Initialize LunarLander-v3 environment with proper render mode
env = gym.make("LunarLander-v3", render_mode="human")  # Adjusted to v2 for compatibility

# Try loading an existing model and replay buffer, else start fresh
if os.path.exists(f"{MODEL_PATH}.zip"):
    model = DQN.load(MODEL_PATH, env=env)
    if os.path.exists(REPLAY_BUFFER_PATH):
        model.load_replay_buffer(REPLAY_BUFFER_PATH)
        print("Loaded existing model and replay buffer.")
    else:
        print("Loaded existing model, no replay buffer found.")
else:
    model = DQN("MlpPolicy", env, verbose=1)
    print("No existing model found. Starting fresh training.")

# Prepare CSV log file
with open(LOG_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Episode", "Total Reward", "Episode Length"])

# Training loop with checkpoints
print("Training started...")
for timestep in range(0, TOTAL_TIMESTEPS, CHECKPOINT_INTERVAL):
    model.learn(total_timesteps=CHECKPOINT_INTERVAL, reset_num_timesteps=False)
    model.save(MODEL_PATH)
    model.save_replay_buffer(REPLAY_BUFFER_PATH)
    print(f"Checkpoint saved at timestep {timestep + CHECKPOINT_INTERVAL}")

print("Training completed!")
model.save(MODEL_PATH)
print(f"Final model saved as '{MODEL_PATH}'.")

# Evaluate the trained model
episodes = 5
rewards = []

for episode in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        env.render()

        done = terminated or truncated

    rewards.append(total_reward)

    # Log episode data
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode + 1, total_reward, step_count])

    print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {step_count}")

env.close()

# Plot rewards
plt.plot(rewards)
plt.title("LunarLander-v3 Performance")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.show()
