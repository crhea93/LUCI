"""
Suite of tests for Luci
"""
import numpy as np
from astropy.io import fits
import keras

from LuciFit import Fit
from LuciFit import Gaussian


class Test:
    def __init__(self):
        self.amp = 1
        self.pos = 15234
        self.sigma = 2.5
        self.model_type = 'gaussian'
        self.lines = ['Halpha']
        self.ML_model = None  # keras.models.load_model('ML/R5000-PREDICTOR-I')
        self.axis = None
        self.spectrum = None
        self.wavenumbers_syn = None
        self.make_spectrum()
        self.read_ref_spec()
        self.LuciFit_ = Fit(self.spectrum, self.axis, self.wavenumbers_syn, self.model_type, self.lines, self.ML_model)

    def make_spectrum(self):
        self.axis = np.linspace(14400, 15800, 10000)
        self.spectrum = Gaussian(self.axis, [self.amp, self.pos, self.sigma]).func
        return None

    def read_ref_spec(self):
        ref_spec = fits.open('ML/Reference-Spectrum-R5000.fits')[1].data
        channel = []
        counts = []
        for chan in ref_spec:  # Only want SN3 region
            channel.append(chan[0])
            counts.append(np.real(chan[1]))
        min_ = np.argmin(np.abs(np.array(channel) - 14700))
        max_ = np.argmin(np.abs(np.array(channel) - 15600))
        self.wavenumbers_syn = np.array(channel[min_:max_], dtype=np.float32)


    def test_spec_length(self):
        assert 'Halpha' in self.lines

    def test_fit(self):
        self.LuciFit_.fit()
        # Check that fit is within 10%
        assert self.LuciFit_.fit_sol[0]-1 < 0.1


def test():
    Test_ = Test()
    Test_.test_spec_length()
    Test_.test_fit()