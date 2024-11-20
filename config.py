# Configuration for RL training
CONFIG = {
    "total_timesteps": 50000,
    "learning_rate": 0.0003,
    "gamma": 0.99,  # Discount factor
    "batch_size": 64,
    "n_steps": 2048,  # Number of steps per update
    "ent_coef": 0.01,  # Entropy coefficient
    "clip_range": 0.2,  # PPO clipping range
}
