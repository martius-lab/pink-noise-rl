"""Comparing pink action noise with the default noise on SAC."""

import gym
import numpy as np
import torch
from pink import PinkNoiseDist
from stable_baselines3 import SAC

# Reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
rng = np.random.default_rng(seed)

# Initialize environment
env = gym.make("MountainCarContinuous-v0")
action_dim = env.action_space.shape[-1]
seq_len = env._max_episode_steps
rng = np.random.default_rng(0)

# Initialize agents
model_default = SAC("MlpPolicy", env, seed=seed)
model_pink = SAC("MlpPolicy", env, seed=seed)

# Set action noise
model_pink.actor.action_dist = PinkNoiseDist(seq_len, action_dim, rng=rng)

# Train agents
model_default.learn(total_timesteps=10_000)
model_pink.learn(total_timesteps=10_000)

# Evaluate learned policies
N = 100
for name, model in zip(["Default noise\n-------------", "Pink noise\n----------"], [model_default, model_pink]):
    solved = 0
    for i in range(N):
        obs = env.reset()
        done = False
        while not done:
            obs, r, done, _ = env.step(model.predict(obs, deterministic=True)[0])
            if r > 0:
                solved += 1
                break

    print(name)
    print(f"Solved: {solved/N * 100:.0f}%\n")


# - Output of this program -
# Default noise
# -------------
# Solved: 0%
#
# Pink noise
# ----------
# Solved: 100%
