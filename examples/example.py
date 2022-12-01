"""Comparing pink action noise with the default noise on SAC."""

import gym
from stable_baselines3 import SAC

from pink import PinkNoiseDist

# Initialize environment
env = gym.make("MountainCarContinuous-v0")
action_dim = env.action_space.shape[-1]
seq_len = env._max_episode_steps

# Initialize agents
model_default = SAC("MlpPolicy", env)
model_pink = SAC("MlpPolicy", env)

# Set action noise
model_pink.actor.action_dist = PinkNoiseDist(action_dim, seq_len)

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
