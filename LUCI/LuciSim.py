"""
Suite of simulations tools in LUCI
"""

import numpy as np
from LUCI.LuciFunctions import Gaussian, Sinc, SincGauss


class Spectrum:

    def __init__(self, lines, fit_function, ampls, velocity, broadening, filter_, resolution, snr):
        """
        Initialize mock spectrum by creating the spectrum and adding noise

        Args:
            lines: List of lines to model (e.x. ['Halpha'])
            fit_function: Function used to model lines  (options: 'sinc', 'gaussian', 'sincgauss')
            ampls: List of amplitudes for emission lines
            velocity: List of velocities of emission lines; if not a list, then all velocities are set equal
            broadening: List of broadening of emissino lines; ditto above
            filter: SITELLE Filter (e.x. 'SN3')
            resolution: Spectral resolution
            snr: Signal to noise ratio

        """

        self.line_dict = {'Halpha': 656.280, 'NII6583': 658.341, 'NII6548': 654.803,
                      'SII6716': 671.647, 'SII6731': 673.085, 'OII3726': 372.603,
                      'OII3729': 372.882, 'OIII4959': 495.891, 'OIII5007': 500.684,
                      'Hbeta': 486.133}
        self.available_functions = ['gaussian', 'sinc', 'sincgauss']
        # Initialize (default SN3 values)
        self.delta_x = 2943  # step size in nanometers
        self.n_steps = 289  # Number of steps (default R~5000)
        self.order = 8  # Folding order
        self.theta = 11.96  # Interferometer angle in degrees
        self.lines = lines
        self.fit_function = fit_function
        self.ampls = ampls
        if not isinstance(velocity, list):
            # List not passed -- set all velocities equal
            self.velocity = [velocity for i in range(len(lines))]
        if not isinstance(broadening, list):
            # List not passed -- set all broadening equal
            self.broadening = [broadening for i in range(len(lines))]
        self.filter = filter_
        if self.filter == 'SN3':
            self.delta_x = 2943
            self.order = 8
        elif self.filter == 'SN1':
            self.delta_x = 1647
            self.order = 6
        else:
            print('We only support SN1 and SN3 at this time.')
            print('Terminating the program')
            exit()
        self.resolution = resolution
        # Calculate number of steps from resolution
        self.n_steps = 0
        self.steps_from_resolution()
        self.snr = snr  # Set signal-to-noise ratio


        # Checks
        self.check_lines()
        self.check_fitting_model()


    def steps_from_resolution(self):
        """
        Calculate the number of steps given the resolution, delta_x, and order.
        This is taken from Martin's Thesis.
        """
        temp = (self.order + 0.5) / (2 * self.delta_x)
        self.n_steps = int(np.ceil(1.20671 * self.resolution / (2 * temp * self.delta_x)))  # The constant comes from the sinc function)

    def calc_sinc_width(self):
        """
        Calculate sinc width of the sincgauss function

        """
        MPD = self.theta * self.delta_x * self.n_steps
        self.sinc_width = 1/(2*MPD)

    def gaussian_model(self, channel, amp, pos, sigma):
        """
        Function to initiate the correct number of models to fit

        Args:
            channel: Wavelength Axis in cm-1
            amp: Amplitude of the line
            pos: Position of the line
            sigma: Sigma of the line

        Return:
            Value of function given input parameters

        """
        f1 = Gaussian(channel, [amp, pos, sigma]).func
        return f1


    def sinc_model(self, channel, amp, pos, sigma):
        """
        Function to initiate the correct number of models to fit

        Args:
            channel: Wavelength Axis in cm-1
            amp: Amplitude of the line
            pos: Position of the line
            sigma: Sigma of the line

        Return:
            Value of function given input parameters

        """
        f1 = Sinc(channel, [amp, pos, sigma]).func
        return f1


    def sincgauss_model(self, channel, amp, pos, sigma):
        """
        Function to initiate the correct number of models to fit

        Args:
            channel: Wavelength Axis in cm-1
            amp: Amplitude of the line
            pos: Position of the line
            sigma: Sigma of the line

        Return:
            Value of function given input parameters

        """
        f1 = SincGauss(channel, [amp, pos, sigma], self.sinc_width).func
        return np.real(f1)

    def create_spectrum(self):
        """
        Create a mock spectrum given the lines of interest, the fitting function, the
        amplitudes of the lines, the kinematics of the lines, and the interferometer parameters.
        The interferometer parameters (step, n_steps, order & theta) will be used to construct the x-axis.
        We add noise given the snr value provided by sampling from a normal distribution with a sigma=1.


        """

        # Calculate correction factor
        corr = 1/np.cos(np.deg2rad(self.theta))
        # Construct x axis in units of cm-1
        x_min = self.order/(2*self.delta_x) * corr * 1e7
        x_max = (self.order+1)/(2*self.delta_x) * corr * 1e7
        step_ = x_max - x_min
        axis = np.array([x_min+j*step_/self.n_steps for j in range(self.n_steps)])
        # Initiate spectrum
        spectrum = np.ones_like(axis)  # Set continuum of about 2
        # Create emission lines
        for line_ct, line in enumerate(self.lines):
            print(line)
            line_lambda_cm = 1e7 / self.line_dict[line]  # Convert from nm to cm-1
            vel_ = self.velocity[line_ct]  # Get velocity
            broad_ = self.broadening[line_ct]  # Get broadening
            amp_ = self.ampls[line_ct]  # Get amplitude
            print(amp_)
            # Calculate position of line given velocity
            line_pos = (vel_/3e5)*line_lambda_cm + line_lambda_cm
            # Calculate sigma given broadening
            sigma = (broad_*line_pos)/(3e5*corr)
            # Create spectrum
            if self.fit_function == 'gaussian':
                spectrum += self.gaussian_model(axis, amp_, line_pos, sigma)
            elif self.fit_function == 'sinc':
                spectrum += self.sinc_model(axis, amp_, line_pos, sigma)
            elif self.fit_function == 'sincgauss':
                self.calc_sinc_width()
                spectrum += self.sincgauss_model(axis, amp_, line_pos, sigma)
            else:
                print('An incorrect fit function was entered. Please use either gaussian, sinc, or sincgauss.')
            print(np.max(spectrum))
        # We now add noise with our predefined SNR
        #spectrum += np.max(spectrum)*np.random.normal(0.0,1/self.snr,spectrum.shape)
        return axis, spectrum




    def check_lines(self):
        """
        This function checks to see that the lines provided are in the available options
        Return:
        Nothing if the user provides appropriate lines
        Else it will throw an error

        """
        if set(self.lines).issubset(self.line_dict):
            pass
        else:
            raise Exception('Please submit a line name in the available list: \n {}'.format(self.line_dict.keys()))

    def check_fitting_model(self):
        """
        This function checks to see that the model provided is in the available options
        Return:
        Nothing if the user provides an appropriate fitting model
        Else it will throw an error

        """
        if self.fit_function in self.available_functions:
            pass
        else:
            print(self.fit_function)
            raise Exception(
                'Please submit a fitting function name in the available list: \n {}'.format(self.available_functions))
