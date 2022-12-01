# Colored Action Noise for Deep RL

This repository contains easy-to-use implementations of pink noise and general colored noise for use as action noise in deep reinforcement learning. Included are the following classes:
- `ColoredNoiseProcess` and `PinkNoiseProcess` for general use, based on the [colorednoise](https://github.com/felixpatzelt/colorednoise) library
- `ColoredActionNoise` and `PinkActionNoise` to be used with deterministic policy algorithms like DDPG and TD3 in Stable Baselines3, both are subclasses of `stable_baselines3.common.noise.ActionNoise`
- `ColoredNoiseDist`, `PinkNoiseDist` to be used with stochastic policy algorithms like SAC in Stable Baselines3
- `MPO_CN` for using colored noise (incl. pink noise) with MPO using the Tonic RL library.

For more information, please see our paper: [Pink Noise Is All You Need: Colored Noise Exploration in Deep Reinforcement Learning](https://bit.ly/pink-noise-rl).

## Installation
You can install the library via pip:
```
pip install pink-noise-rl
```
Note: In Python, the import statement is simply `import pink`.

## Usage
We provide minimal examples for using pink noise on SAC, TD3 and MPO below. An example comparing pink noise with the default action noise of SAC is included in the `examples` directory.

### Stable Baselines3: SAC, TD3
```python
import gym
from stable_baselines3 import SAC, TD3

# All classes mentioned above can be imported from `pink`
from pink import PinkNoiseDist, PinkActionNoise

# Initialize environment
env = gym.make("MountainCarContinuous-v0")
action_dim = env.action_space.shape[-1]
seq_len = env._max_episode_steps
```

#### SAC
```python
# Initialize agent
model = SAC("MlpPolicy", env)

# Set action noise
model.actor.action_dist = PinkNoiseDist(action_dim, seq_len)

# Train agent
model.learn(total_timesteps=10_000)
```

#### TD3
```python
# Initialize agent
model = TD3("MlpPolicy", env)

# Set action noise
noise_scale = 0.3*np.ones(action_dim)
model.action_noise = PinkActionNoise(noise_scale, seq_len)

# Train agent
model.learn(total_timesteps=10_000)
```

### Tonic: MPO
```python
import gym
from tonic import Trainer
from pink import MPO_CN

# Initialize environment
env = gym.make("MountainCarContinuous-v0")
seq_len = env._max_episode_steps

# Initialize agent with pink noise
beta = 1
model = MPO_CN()
model.initialize(beta, seq_len, env.observation_space, env.action_space)

# Train agent
trainer = tonic.Trainer(steps=10_000)
trainer.initialize(model, env)
trainer.run()
```


## Citing
If you use this code in your research, please cite our paper:
```bibtex
@misc{eberhard-2022-pink,
  title = {Pink {{Noise Is All You Need}}: {{Colored Noise Exploration}} in {{Deep Reinforcement Learning}}},
  author = {Eberhard, Onno and Hollenstein, Jakob and Pinneri, Cristina and Martius, Georg},
  date = {2022},
  howpublished = {NeurIPS Deep RL Workshop 2022}
}
```

If there are any problems, or you have a question, don't hesitate to open an issue here on GitHub.
