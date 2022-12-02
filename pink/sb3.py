"""Colored noise implementations for Stable Baselines3"""

import numpy as np
import torch as th
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.noise import ActionNoise

from .cnrl import ColoredNoiseProcess


class ColoredActionNoise(ActionNoise):
    def __init__(self, beta, sigma, seq_len, rng=None):
        """Action noise from a colored noise process.

        Parameters
        ----------
        beta : array_like
            Exponents of colored noise power-law spectra. Should be a list of the same dimensionality as the action
            space (one beta for each action dimension).
        sigma : array_like
            Noise scales of colored noise signals. Should be a list of the same dimensionality as the action
            space (one scale for each action dimension).
        seq_len : int
            Length of sampled colored noise signals. If sampled for longer than `seq_len` steps, a new
            colored noise signal of the same length is sampled. Should usually be set to the episode length
            (horizon) of the RL task.
        rng : np.random.Generator, optional, by default None
            Random number generator (for reproducibility). If None, a new random number generator is created by calling
            `np.random.default_rng()`.
        """
        super().__init__()
        self._beta = beta
        self._sigma = sigma
        self._gen = [ColoredNoiseProcess(beta=b, scale=s, size=seq_len, rng=rng)
                     for b, s in zip(beta, sigma)]

    def __call__(self) -> np.ndarray:
        return np.asarray([g.sample() for g in self._gen])

    def __repr__(self) -> str:
        return f"ColoredActionNoise(beta={self._beta}, sigma={self._sigma})"


class PinkActionNoise(ColoredActionNoise):
    def __init__(self, sigma, seq_len, rng=None):
        """Action noise from a pink noise process.

        Parameters
        ----------
        sigma : array_like
            Noise scales of pink noise signals. Should be a list of the same dimensionality as the action
            space (one scale for each action dimension).
        seq_len : int
            Length of sampled pink noise signals. If sampled for longer than `seq_len` steps, a new
            pink noise signal of the same length is sampled. Should usually be set to the episode length
            (horizon) of the RL task.
        rng : np.random.Generator, optional, by default None
            Random number generator (for reproducibility). If None, a new random number generator is created by calling
            `np.random.default_rng()`.
        """
        super().__init__(np.ones_like(sigma), sigma, seq_len, rng)


class ColoredNoiseDist(SquashedDiagGaussianDistribution):
    def __init__(self, beta, seq_len, rng=None, epsilon=1e-6):
        """
        Gaussian colored noise distribution for using colored action noise with stochastic policies.

        The colored noise is only used for sampling actions. In all other respects, this class acts like its parent
        class (`SquashedDiagGaussianDistribution`).

        Parameters
        ----------
        beta : array_like
            Exponents of colored noise power-law spectra. Should be a list of the same dimensionality as the action
            space (one beta for each action dimension).
        seq_len : int
            Length of sampled colored noise signals. If sampled for longer than `seq_len` steps, a new
            colored noise signal of the same length is sampled. Should usually be set to the episode length
            (horizon) of the RL task.
        rng : np.random.Generator, optional, by default None
            Random number generator (for reproducibility). If None, a new random number generator is created by calling
            `np.random.default_rng()`.
        epsilon : float, optional, by default 1e-6
            A small value to avoid NaN due to numerical imprecision.
        """
        super().__init__(len(beta), epsilon)
        self.cn_processes = [ColoredNoiseProcess(beta=b, size=seq_len, rng=rng)
                             for b in beta]

    def sample(self) -> th.Tensor:
        cn_sample = th.tensor([cnp.sample() for cnp in self.cn_processes]).float()
        self.gaussian_actions = self.distribution.mean + self.distribution.stddev*cn_sample
        return th.tanh(self.gaussian_actions)


class PinkNoiseDist(ColoredNoiseDist):
    def __init__(self, action_dim, seq_len, rng=None, epsilon=1e-6):
        """
        Gaussian pink noise distribution for using pink action noise with stochastic policies.

        The pink noise is only used for sampling actions. In all other respects, this class acts like its parent
        class (`SquashedDiagGaussianDistribution`).

        Parameters
        ----------
        action_dim : int
            Dimension of the action space.
        seq_len : int
            Length of sampled colored noise signals. If sampled for longer than `seq_len` steps, a new
            colored noise signal of the same length is sampled. Should usually be set to the episode length
            (horizon) of the RL task.
        rng : np.random.Generator, optional, by default None
            Random number generator (for reproducibility). If None, a new random number generator is created by calling
            `np.random.default_rng()`.
        epsilon : float, optional, by default 1e-6
            A small value to avoid NaN due to numerical imprecision.
        """
        super().__init__(np.ones(action_dim), seq_len, rng, epsilon)
