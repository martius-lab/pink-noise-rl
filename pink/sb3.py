"""Colored noise implementations for Stable Baselines3"""

import numpy as np
import torch as th
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.noise import ActionNoise

from .cnrl import ColoredNoiseProcess


class ColoredActionNoise(ActionNoise):
    def __init__(self, beta, sigma, seq_len, action_dim=None, rng=None):
        """Action noise from a colored noise process.

        Parameters
        ----------
        beta : float or array_like
            Exponent(s) of colored noise power-law spectra. If it is a single float, then `action_dim` has to be
            specified and the noise will be sampled in a vectorized manner for each action dimension. If it is
            array_like, then it specifies one beta for each action dimension. This allows different betas for different
            action dimensions, but sampling might be slower for large action dimensionalities.
        sigma : float or array_like
            Noise scale(s) of colored noise signals. Either a single float to be used for all action dimensions, or
            an array_like of the same dimensionality as the action space (one scale for each action dimension).
        seq_len : int
            Length of sampled colored noise signals. If sampled for longer than `seq_len` steps, a new
            colored noise signal of the same length is sampled. Should usually be set to the episode length
            (horizon) of the RL task.
        action_dim : int, optional
            Dimensionality of the action space. If passed, `beta` has to be a single float and the noise will be
            sampled in a vectorized manner for each action dimension.
        rng : np.random.Generator, optional
            Random number generator (for reproducibility). If not passed, a new random number generator is created by
            calling `np.random.default_rng()`.
        """
        super().__init__()
        assert (action_dim is not None) ^ np.isscalar(beta), \
            "`action_dim` has to be specified if and only if `beta` is a scalar."
        # TODO: logic not done!
        self._sigma = np.asarray(sigma)
        if action_dim is None:
            self._beta = np.asarray(beta)
            self._gen = [ColoredNoiseProcess(beta=b, scale=s, size=seq_len, rng=rng)
                        for b, s in zip(beta, sigma)]
        else:
            self._beta = np.full(action_dim, beta)

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
