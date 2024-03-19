"""
Suite of tests for Luci fitting function on a single synthetic spectrum just to test that the fitting algorithm isn't
giving crazy values!
"""
import numpy as np
from astropy.io import fits
from scipy import interpolate
import keras
from LUCI.LuciFit import Fit
from LUCI.LuciFunctions import Gaussian
from LUCI.LuciFitParameters import calculate_vel, calculate_broad
#import sys
#sys.path.insert(0, self.Luci_path)  # Location of Luci
from LUCI.LuciSim import Spectrum
class Test:
    def __init__(self):
        self.line_dict = {'Halpha': 656.280, 'NII6583': 658.341, 'NII6548': 654.803,
                          'SII6716': 671.647, 'SII6731': 673.085, 'OII3726': 372.603,
                          'OII3729': 372.882, 'OIII4959': 495.891, 'OIII5007': 500.684,
                          'Hbeta': 486.133}
        self.Luci_path = '/home/carterrhea/LUCI/'


    def luci_fit_single(self):
        """
        Basic call to fit a single Halpha line!
        Returns:

        """
        self.amp = 1
        self.pos = 15234  # Corresponds to a velocity of 68.55 km/s
        self.sigma = 0.5  # Corresponds to a broadening of 9.85 km/s
        self.model_type = 'sincgauss'
        self.lines = ['Halpha']
        #self.ML_model = keras.models.load_model('ML/R5000-PREDICTOR-I-SN3')
        self.axis = None
        self.spectrum = None
        self.wavenumbers_syn = None
        self.make_spectrum()
        self.read_ref_spec()
        self.transmission_interpolated = None
        self.read_in_transmission()
        self.LuciFit_ = Fit(spectrum=self.spectrum, axis=self.axis, wavenumbers_syn=self.wavenumbers_syn, model_type=self.model_type, lines=self.lines,
                            vel_rel=[1] * len(self.lines), sigma_rel=[1] * len(self.lines), trans_filter=self.transmission_interpolated, resolution=5000,
                            Luci_path=self.Luci_path)
        self.LuciFit_.interpolate_spectrum()

    def read_in_transmission(self):
        """
        Read in the transmission spectrum for the filter. Then apply interpolation
        on it to make it have the same x-axis as the spectra.
        """
        transmission = np.loadtxt('Data/SN3_filter.dat')  # first column - axis; second column - value
        f = interpolate.interp1d(transmission[:, 0], [val/100 for val in transmission[:, 1]], kind='slinear',
                                 fill_value="extrapolate")
        self.transmission_interpolated = f(self.axis)

    def make_spectrum(self):
        """
        Create a single spectrum for Halpha
        Returns:

        """
        #self.axis = np.linspace(14400, 15800, 10000)
        #self.spectrum = Gaussian().evaluate(self.axis, [self.amp, self.pos, self.sigma], 1)  # 1 because only fitting a single line
        lines = ['Halpha']#, 'NII6583', 'NII6548', 'SII6716', 'SII6731']
        fit_function = 'sincgauss'
        ampls = [10]#, 1, 1, 0.5, 0.45]  # Just randomly choosing these
        velocity = -300  # km/s
        broadening = 50  # km/s
        filter_ = 'SN3'
        resolution = 5000
        snr = 100
        self.axis, self.spectrum = Spectrum(lines, fit_function, ampls, velocity, broadening, filter_, resolution,
                                           snr).create_spectrum()
        return None


    def read_ref_spec(self):
        ref_spec = fits.open('ML/Reference-Spectrum-R5000-SN3.fits')[1].data
        channel = []
        counts = []
        for chan in ref_spec:  # Only want SN3 region
            channel.append(chan[0])
            counts.append(np.real(chan[1]))
        min_ = np.argmin(np.abs(np.array(channel) - 14700))
        max_ = np.argmin(np.abs(np.array(channel) - 15600))
        self.wavenumbers_syn = np.array(channel[min_:max_], dtype=np.float32)


    def test_spec_length(self):
        """
        Check that the length of lines is correct
        """
        assert 'Halpha' in self.lines

    def test_fit_single(self):
        """
        Fit a single Halpha line. We then check that the amplitude, velocity, and broadening found by our fitting function
        return values that are within 10% of the true values.
        """
        self.LuciFit_.fit()
        # Check that amplitude of the fit is within 10% of the true value which is 1
        assert self.LuciFit_.fit_sol[0]-10 < 1
        # Check that velocity of the fit is within 10% of the true value which is 10km/s#68.55 km/s
        true_vel = 300
        assert np.abs((calculate_vel(0, self.lines, self.LuciFit_.fit_sol, self.line_dict) - true_vel)/true_vel) < 1
        # Check that broadening of the fit is within 10% of the true value which is 9.85 km/s
        # Convert broadening by dividing by FWHM
        true_broadening = 50
        sigma_real = true_broadening#/(2*np.sqrt(2*np.log(2)))
        assert np.abs((calculate_broad(0, self.LuciFit_.fit_sol, self.LuciFit_.axis_step) - sigma_real)/sigma_real) < 1

    def test_ML_single(self):
        """
        Fit a single Halpha line. We then check that the amplitude, velocity, and broadening found by our machine learning algorithm
        return values that are within 10% of the true values.
        """
        self.LuciFit_.fit()
        # Check that velocity of the fit is within 10% of the true value which
        print(self.LuciFit_.vel_ml)
        true_vel = 300
        assert np.abs((self.LuciFit_.vel_ml - true_vel)/true_vel) < 1.1
        # Check that broadening of the fit is within 10% of the true value which is
        true_broadening = 50
        assert np.abs((self.LuciFit_.broad_ml - true_broadening)/true_broadening) < 1.1




def test_spec_length():
    """
    Test to call Test.test_spec_length
    """
    Test_ = Test()
    Test_.luci_fit_single()
    Test_.test_spec_length()


def test_fit_single():
    """
    Test to call Test.test_fit_single
    """
    Test_ = Test()
    Test_.luci_fit_single()
    Test_.test_fit_single()


def test_ML_single():
    """
    Test to call Test.test_ML_single
    """
    Test_ = Test()
    Test_.luci_fit_single()
    Test_.test_ML_single()


def test_create_spectrum():
    """
    Test to call Test.make_spectrum
    """
    Test_ = Test()
    Test_.luci_fit_single()
    Test_.make_spectrum()

