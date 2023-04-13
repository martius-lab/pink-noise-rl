import numpy as np

from . import colorednoise as cn


class ColoredNoiseProcess():
    """Infinite colored noise process.

    Implemented as a buffer: every `size[-1]` samples, a cut to a new time series starts. As this cut influences the
    PSD of the combined signal, the maximum period (1 / low-frequency cutoff) can be specified.

    Methods
    -------
    sample(T=1)
        Sample `T` timesteps from the colored noise process.
    reset()
        Reset the buffer with a new time series.
    """
    def __init__(self, beta, size, scale=1, max_period=None, rng=None):
        """Infinite colored noise process.

        Implemented as a buffer: every `size[-1]` samples, a cut to a new time series starts. As this cut influences
        the PSD of the combined signal, the maximum period (1 / low-frequency cutoff) can be specified.

        Parameters
        ----------
        beta : float
            Exponent of colored noise power-law spectrum.
        size : int or tuple of int
            Shape of the sampled colored noise signals. The last dimension (`size[-1]`) specifies the time range, and
            is thus ths maximum possible correlation length of the combined signal.
        scale : int, optional, by default 1
            Scale parameter with which samples are multiplied
        max_period : float, optional, by default None
            Maximum correlation length of sampled colored noise singals (1 / low-frequency cutoff). If None, it is
            automatically set to `size[-1]` (the sequence length).
        rng : np.random.Generator, optional
            Random number generator (for reproducibility). If not passed, a new random number generator is created by
            calling `np.random.default_rng()`.
        """
        self.beta = beta
        if max_period is None:
            self.minimum_frequency = 0
        else:
            self.minimum_frequency = 1 / max_period
        self.scale = scale
        self.rng = rng

        # The last component of size is the time index
        try:
            self.size = list(size)
        except TypeError:
            self.size = [size]
        self.time_steps = self.size[-1]

        # Fill buffer and reset index
        self.reset()

    def reset(self):
        """Reset the buffer with a new time series."""
        self.buffer = cn.powerlaw_psd_gaussian(
                exponent=self.beta, size=self.size, fmin=self.minimum_frequency, rng=self.rng)
        self.idx = 0

    def sample(self, T=1):
        """
        Sample `T` timesteps from the colored noise process.

        The buffer is automatically refilled when necessary.

        Parameters
        ----------
        T : int, optional, by default 1
            Number of samples to draw

        Returns
        -------
        array_like
            Sampled vector of shape `(*size[:-1], T)`
        """
        n = 0
        ret = []
        while n < T:
            if self.idx >= self.time_steps:
                self.reset()
            m = min(T - n, self.time_steps - self.idx)
            ret.append(self.buffer[..., self.idx:(self.idx + m)])
            n += m
            self.idx += m

        ret = self.scale * np.concatenate(ret, axis=-1)
        return ret if n > 1 else ret[..., 0]


class PinkNoiseProcess(ColoredNoiseProcess):
    """Infinite pink noise process.

    Implemented as a buffer: every `size[-1]` samples, a cut to a new time series starts. As this cut influences the
    PSD of the combined signal, the maximum period (1 / low-frequency cutoff) can be specified.

    Methods
    -------
    sample(T=1)
        Sample `T` timesteps from the pink noise process.
    reset()
        Reset the buffer with a new time series.
    """
    def __init__(self, size, scale=1, max_period=None, rng=None):
        """Infinite pink noise process.

        Implemented as a buffer: every `size[-1]` samples, a cut to a new time series starts. As this cut influences
        the PSD of the combined signal, the maximum period (1 / low-frequency cutoff) can be specified.

        Parameters
        ----------
        size : int or tuple of int
            Shape of the sampled pink noise signals. The last dimension (`size[-1]`) specifies the time range, and is
            thus ths maximum possible correlation length of the combined signal.
        scale : int, optional, by default 1
            Scale parameter with which samples are multiplied
        max_period : float, optional, by default None
            Maximum correlation length of sampled pink noise singals (1 / low-frequency cutoff). If None, it is
            automatically set to `size[-1]` (the sequence length).
        rng : np.random.Generator, optional
            Random number generator (for reproducibility). If not passed, a new random number generator is created by
            calling `np.random.default_rng()`.
        """
        super().__init__(1, size, scale, max_period, rng)
