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
            action dimensions, but sampling might be slower for high-dimensional action spaces.
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
        assert (action_dim is not None) == np.isscalar(beta), \
            "`action_dim` has to be specified if and only if `beta` is a scalar."

        self.sigma = np.full(action_dim or len(beta), sigma) if np.isscalar(sigma) else np.asarray(sigma)

        if np.isscalar(beta):
            self.beta = beta
            self.gen = ColoredNoiseProcess(beta=self.beta, scale=self.sigma, size=(action_dim, seq_len), rng=rng)
        else:
            self.beta = np.asarray(beta)
            self.gen = [ColoredNoiseProcess(beta=b, scale=s, size=seq_len, rng=rng)
                        for b, s in zip(self.beta, self.sigma)]

    def __call__(self) -> np.ndarray:
        return self.gen.sample() if np.isscalar(self.beta) else np.asarray([g.sample() for g in self.gen])

    def __repr__(self) -> str:
        return f"ColoredActionNoise(beta={self.beta}, sigma={self.sigma})"


class PinkActionNoise(ColoredActionNoise):
    def __init__(self, sigma, seq_len, action_dim, rng=None):
        """Action noise from a pink noise process.

        Parameters
        ----------
        sigma : float or array_like
            Noise scale(s) of colored noise signals. Either a single float to be used for all action dimensions, or
            an array_like of the same dimensionality as the action space (one scale for each action dimension).
        seq_len : int
            Length of sampled pink noise signals. If sampled for longer than `seq_len` steps, a new
            pink noise signal of the same length is sampled. Should usually be set to the episode length
            (horizon) of the RL task.
        action_dim : int
            Dimensionality of the action space.
        rng : np.random.Generator, optional
            Random number generator (for reproducibility). If not passed, a new random number generator is created by
            calling `np.random.default_rng()`.
        """
        super().__init__(1, sigma, seq_len, action_dim, rng)


class ColoredNoiseDist(SquashedDiagGaussianDistribution):
    def __init__(self, beta, seq_len, action_dim=None, rng=None, epsilon=1e-6):
        """
        Gaussian colored noise distribution for using colored action noise with stochastic policies.

        The colored noise is only used for sampling actions. In all other respects, this class acts like its parent
        class (`SquashedDiagGaussianDistribution`).

        Parameters
        ----------
        beta : float or array_like
            Exponent(s) of colored noise power-law spectra. If it is a single float, then `action_dim` has to be
            specified and the noise will be sampled in a vectorized manner for each action dimension. If it is
            array_like, then it specifies one beta for each action dimension. This allows different betas for different
            action dimensions, but sampling might be slower for high-dimensional action spaces.
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
        epsilon : float, optional, by default 1e-6
            A small value to avoid NaN due to numerical imprecision.
        """
        assert (action_dim is not None) == np.isscalar(beta), \
            "`action_dim` has to be specified if and only if `beta` is a scalar."

        if np.isscalar(beta):
            super().__init__(action_dim, epsilon)
            self.beta = beta
            self.gen = ColoredNoiseProcess(beta=self.beta, size=(action_dim, seq_len), rng=rng)
        else:
            super().__init__(len(beta), epsilon)
            self.beta = np.asarray(beta)
            self.gen = [ColoredNoiseProcess(beta=b, size=seq_len, rng=rng) for b in self.beta]

    def sample(self) -> th.Tensor:
        if np.isscalar(self.beta):
            cn_sample = th.tensor(self.gen.sample()).float()
        else:
            cn_sample = th.tensor([cnp.sample() for cnp in self.gen]).float()
        self.gaussian_actions = self.distribution.mean + self.distribution.stddev*cn_sample
        return th.tanh(self.gaussian_actions)

    def __repr__(self) -> str:
        return f"ColoredNoiseDist(beta={self.beta})"


class PinkNoiseDist(ColoredNoiseDist):
    def __init__(self, seq_len, action_dim, rng=None, epsilon=1e-6):
        """
        Gaussian pink noise distribution for using pink action noise with stochastic policies.

        The pink noise is only used for sampling actions. In all other respects, this class acts like its parent
        class (`SquashedDiagGaussianDistribution`).

        Parameters
        ----------
        seq_len : int
            Length of sampled colored noise signals. If sampled for longer than `seq_len` steps, a new
            colored noise signal of the same length is sampled. Should usually be set to the episode length
            (horizon) of the RL task.
        action_dim : int
            Dimensionality of the action space.
        rng : np.random.Generator, optional
            Random number generator (for reproducibility). If not passed, a new random number generator is created by
            calling `np.random.default_rng()`.
        epsilon : float, optional, by default 1e-6
            A small value to avoid NaN due to numerical imprecision.
        """
        super().__init__(1, seq_len, action_dim, rng, epsilon)
