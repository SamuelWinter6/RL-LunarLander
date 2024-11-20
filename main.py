from stable_baselines3 import PPO
from envs.racing_env import RacingEnv

def main():
    # Initialize the custom TORCS environment
    env = RacingEnv()

    # Define the RL model
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    print("Starting training...")
    model.learn(total_timesteps=50000)

    # Save the trained model
    model.save("models/ppo_racing_model")
    print("Model saved!")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
