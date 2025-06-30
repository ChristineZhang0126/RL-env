from RL_env import DualFuelEngineEnv
import numpy as np

# Create environment
env = DualFuelEngineEnv(mat_path='firstDataset.mat')

# Reset environment
obs, _ = env.reset()
print("Initial observation:", obs)

# Run 10 steps with random actions
for i in range(10):
    action = env.action_space.sample()  # Random action
    next_obs, reward, done, truncated, info = env.step(action)
    print(f"Step {i+1}")
    print("  Action:", action)
    print("  Observation:", next_obs)
    print("  Reward:", reward)
    env.render()

env.close()
