"""
Suite of tests for Luci
"""
import numpy as np
from astropy.io import fits
import keras
from scipy import interpolate

from LUCI.LuciFit import Fit
from LUCI.LuciFit import Gaussian


class Test:
    def __init__(self):
        self.amp = 1
        self.pos = 15234  # Corresponds to a velocity of 68.55 km/s
        self.sigma = 0.5  # Corresponds to a broadening of 9.85 km/s
        self.model_type = 'gaussian'
        self.lines = ['Halpha']
        self.ML_model = None  # keras.models.load_model('ML/R5000-PREDICTOR-I')
        self.axis = None
        self.spectrum = None
        self.wavenumbers_syn = None
        self.make_spectrum()
        self.read_ref_spec()
        self.transmission_interpolated = None
        self.read_in_transmission()
        self.LuciFit_ = Fit(self.spectrum, self.axis, self.wavenumbers_syn, self.model_type, self.lines, [1], [1], self.ML_model, self.transmission_interpolated)

    def read_in_transmission(self):
        """
        Read in the transmission spectrum for the filter. Then apply interpolation
        on it to make it have the same x-axis as the spectra.
        """
        transmission = np.loadtxt('Data/SN3_filter.dat')  # first column - axis; second column - value
        f = interpolate.interp1d(transmission[:,0], [val/100 for val in transmission[:,1]], kind='slinear', fill_value="extrapolate")
        self.transmission_interpolated = f(self.axis)

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
        # Check that amplitude of the fit is within 10% of the true value which is 1
        assert self.LuciFit_.fit_sol[0]-1 < 0.1
        # Check that velocity of the fit is within 10% of the true value which is 68.55 km/s
        assert np.abs((self.LuciFit_.calculate_vel(0) - 68.55)/68.55) < 0.1
        # Check that broadening of the fit is within 10% of the true value which is 9.85 km/s
        assert np.abs((self.LuciFit_.calculate_broad(0) - 9.85)/9.85) < 0.1




def test():
    Test_ = Test()
    Test_.test_spec_length()
    Test_.test_fit()
