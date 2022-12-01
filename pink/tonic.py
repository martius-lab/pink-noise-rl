"""Colored noise implementations for Tonic RL library"""

import numpy as np
import torch as th
from tonic.torch.agents import MPO

from .cnrl import ColoredNoiseProcess


class MPO_CN(MPO):
    """MPO with colored noise exploration"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, beta, seq_len, observation_space, action_space, rng=None, seed=None):
        """For documentation of beta, seq_len, rng see `pink.sb3.ColoredNoiseDist`."""
        super().initialize(observation_space, action_space, seed)
        self.seq_len = seq_len
        self.rng = rng
        self.action_space = action_space
        self.set_beta(beta)

    def set_beta(self, beta):
        if np.isscalar(beta):
            beta = [beta] * self.action_space.shape[0]
        self.cn_processes = [
            ColoredNoiseProcess(beta=b, chunksize=self.seq_len, largest_wavelength=None, rng=self.rng) for b in beta]

    def _step(self, observations):
        observations = th.as_tensor(observations, dtype=th.float32)
        cn_sample = th.tensor([[cnp.sample() for cnp in self.cn_processes]]).float()
        with th.no_grad():
            loc = self.model.actor(observations).loc
            scale = self.model.actor(observations).scale
            return loc + scale*cn_sample
