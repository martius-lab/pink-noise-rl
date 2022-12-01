from . import colorednoise as cn

class ColoredNoiseProcess():
    def __init__(self, beta, scale=1, chunksize=None, largest_wavelength=None, rng=None):
        """Colored noise implemented as a process that allows subsequent samples.
        Implemented as a buffer; every "chunksize[-1]" items, a cut to a new time series starts.
        """
        self.beta = beta
        if largest_wavelength is None:
            self.minimum_frequency = 0
        else:
            self.minimum_frequency = 1 / largest_wavelength
        self.scale = scale
        self.rng = rng

        # The last component of chunksize is the time index
        try:
            self.chunksize = list(chunksize)
        except TypeError:
            self.chunksize = [chunksize]
        self.time_steps = self.chunksize[-1]

        # Set first time-step such that buffer will be initialized
        self.idx = self.time_steps

    def sample(self):
        self.idx += 1    # Next time step

        # Refill buffer if depleted
        if self.idx >= self.time_steps:
            self.buffer = cn.powerlaw_psd_gaussian(
                exponent=self.beta, size=self.chunksize, fmin=self.minimum_frequency, rng=self.rng)
            self.idx = 0

        return self.scale * self.buffer[..., self.idx]

class PinkNoiseProcess(ColoredNoiseProcess):
    def __init__(self, scale=1, chunksize=None, largest_wavelength=None, rng=None):
        """Colored noise implemented as a process that allows subsequent samples.
        Implemented as a buffer; every "chunksize[-1]" items, a cut to a new time series starts.
        """
        super().__init__(1, scale, chunksize, largest_wavelength, rng)
