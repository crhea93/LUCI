import numpy as np
from scipy import special as sps
import math

# Define Constants #
SPEED_OF_LIGHT = 299792  # km/s
FWHM_COEFF = 2.*math.sqrt(2. * math.log(2.))


class Gaussian:
    """
    Class encoding all functionality of the gaussian function. This includes
    a function call, an evaluation of the function for all lines to be fit,
    and a plot call.
    """

    def __init__(self):
        pass

    def function(self, channel, params):
        A = params[0];
        x = params[1];
        sigma = params[2]
        return A * np.exp((-(channel - x) ** 2) / (2 * sigma ** 2))

    def evaluate(self, channel, theta, line_num):
        """
        Function to initiate the correct number of models to fit

        Args:
            channel: Wavelength Axis in cm-1
            theta: List of parameters for all the models in the following order
                            [amplitude, line location, sigma]
            line_num: Number of lines for fit

        Return:
            Value of function given input parameters (theta)

        """
        f1 = 0.0
        for model_num in range(line_num):
            params = theta[model_num * 3:(model_num + 1) * 3]
            f1 += self.function(channel, params)
        return f1

    def evaluate_bayes(self, channel, theta):
        """
        Function to initiate the model calculation for Bayesian Analysis

        Args:
            channel: Wavelength Axis in cm-1
            theta: List of parameters for all the models in the following order
                            [amplitude, line location, sigma]

        Return:
            Value of function given input parameters (theta)

        """
        f1 = 0.0
        params = theta
        f1 += self.function(channel, params)
        return f1

    def plot(self, channel, theta, line_num):
        """
        Function to initiate the correct number of models to fit

        Args:
            channel: Wavelength Axis in cm-1
            theta: List of parameters for all the models in the following order
                            [amplitude, line location, sigma]
            line_num: Number of lines for fit

        Return:
            Value of function given input parameters (theta)

        """
        f1 = 0.0
        for model_num in range(line_num):
            min_ind = np.argmin(np.abs(channel - theta[3*model_num+1]))
            pos_on_axis = channel[min_ind]
            params = [theta[model_num * 3], pos_on_axis, theta[model_num*3 + 2]]
            f1 += self.function(channel, params)
        #f1 += theta[-1]
        return f1


class Sinc:
    """
    Class encoding all functionality of the sinc function. This includes
    a function call, an evaluation of the function for all lines to be fit,
    and a plot call.
    """

    def __init__(self):
        pass

    def function(self, channel, params, sinc_width):
        p0 = params[0];
        p1 = params[1]
        p2 = sinc_width
        u = (channel - p1) / p2
        return [p0 * np.sinc(u_) if u_ != 0 else p0 for u_ in u]

    def evaluate(self, channel, theta, line_num, sinc_width):
        """
        Function to initiate the correct number of models to fit

        Args:
            channel: Wavelength Axis in cm-1
            theta: List of parameters for all the models in the following order
                            [amplitude, line location, sigma]
            line_num: Number of lines for fit
            sinc_width: Fixed with of the sinc function

        Return:
            Value of function given input parameters (theta)

        """
        f1 = 0.0
        for model_num in range(line_num):
            params = theta[model_num * 3:(model_num + 1) * 3]
            f1 += np.array(self.function(channel, params, sinc_width))
        return f1

    def evaluate_bayes(self, channel, theta, sinc_width):
        """
        Function to initiate the model calculation for Bayesian Analysis

        Args:
            channel: Wavelength Axis in cm-1
            theta: List of parameters for all the models in the following order
                            [amplitude, line location, sigma]
            sinc_width: Fixed with of the sinc function

        Return:
            Value of function given input parameters (theta)

        """
        f1 = 0.0
        params = theta
        f1 += np.array(self.function(channel, params, sinc_width))
        return f1


    def plot(self, channel, theta, line_num, sinc_width):
        """
        Function to initiate the correct number of models to fit

        Args:
            channel: Wavelength Axis in cm-1
            theta: List of parameters for all the models in the following order
                            [amplitude, line location, sigma]
            line_num: Number of lines for fit
            sinc_width: Fixed with of the sinc function

        Return:
            Value of function given input parameters (theta)

        """
        f1 = 0.0
        for model_num in range(line_num):
            min_ind = np.argmin(np.abs(channel - theta[3*model_num+1]))
            pos_on_axis = channel[min_ind]
            params = [theta[model_num * 3], pos_on_axis, theta[model_num*3 + 2]]
            f1 += np.array(self.function(channel, params, sinc_width))
        return f1



class SincGauss:
    """
    Class encoding all functionality of the sincgauss function. This includes
    a function call, an evaluation of the function for all lines to be fit,
    and a plot call.
    """

    def __init__(self):
        pass

    def function(self, channel, params, sinc_width):
        p0 = params[0]
        p1 = params[1]
        p2 = sinc_width/np.pi
        p3 = params[2]
        a = p3/(np.sqrt(2)*p2)
        b = (channel-p1)/(np.sqrt(2)*p3)
        dawson1 = sps.dawsn(1j * a + b) * np.exp(2.* 1j * a *b)
        dawson2 = sps.dawsn(1j * a - b) * np.exp(-2. * 1j * a *b)
        dawson3 = 2. * sps.dawsn(1j * a)
        return p0*(dawson1 + dawson2)/dawson3

    def evaluate(self, channel, theta, line_num, sinc_width):
        """
        Function to initiate the correct number of models to fit

        Args:
            channel: Wavelength Axis in cm-1
            theta: List of parameters for all the models in the following order
                            [amplitude, line location, sigma]
            line_num: Number of lines for fit
            sinc_width: Fixed width of the sinc function

        Return:
            Value of function given input parameters (theta)

        """
        f1 = 0.0
        for model_num in range(line_num):
            params = theta[model_num * 3:(model_num + 1) * 3]
            f1 += self.function(channel, params, sinc_width)
        return np.real(f1)

    def evaluate_bayes(self, channel, theta, sinc_width):
        """
        Function to initiate the model calculation for Bayesian Analysis

        Args:
            channel: Wavelength Axis in cm-1
            theta: List of parameters for all the models in the following order
                            [amplitude, line location, sigma]
            sinc_width: Fixed with of the sinc function

        Return:
            Value of function given input parameters (theta)

        """
        f1 = 0.0
        params = theta
        f1 += self.function(channel, params, sinc_width)
        return np.real(f1)

    def plot(self, channel, theta, line_num, sinc_width):
        """
        Function to initiate the correct number of models to fit

        Args:
            channel: Wavelength Axis in cm-1
            theta: List of parameters for all the models in the following order
                            [amplitude, line location, sigma]
            line_num: Number of lines for fit
            sinc_width: Fixed with of the sinc function

        Return:
            Value of function given input parameters (theta)

        """
        f1 = 0.0
        for model_num in range(line_num):
            min_ind = np.argmin(np.abs(channel - theta[3*model_num+1]))
            pos_on_axis = channel[min_ind]
            params = [theta[model_num * 3], pos_on_axis, theta[model_num*3 + 2]]
            f1 += self.function(channel, params, sinc_width)
        return np.real(f1)


class Gaussian_frozen:
    """
    Class encoding all functionality of the gaussian function. This includes
    a function call, an evaluation of the function for all lines to be fit,
    and a plot call.
    """

    def __init__(self, initial_velocity, initial_broadening, line_name):
        self.line_dict = {'Halpha': 656.280, 'NII6583': 658.341, 'NII6548': 654.803,
                          'SII6716': 671.647, 'SII6731': 673.085, 'OII3726': 372.603,
                          'OII3729': 372.882, 'OIII4959': 495.891, 'OIII5007': 500.684,
                          'Hbeta': 486.133, 'OH': 649.873}
        line_theo = self.line_dict[line_name]
        line_pos = 1e7 / ((initial_velocity/SPEED_OF_LIGHT) * line_theo + line_theo)
        line_broad = (line_pos * initial_broadening) / (SPEED_OF_LIGHT)

    def function(self, channel, params):
        A = params[0];
        x = line_pos
        sigma = line_broad
        return A * np.exp((-(channel - x) ** 2) / (2 * sigma ** 2))

    def evaluate(self, channel, theta, line_num):
        """
        Function to initiate the correct number of models to fit

        Args:
            channel: Wavelength Axis in cm-1
            theta: List of parameters for all the models in the following order
                            [amplitude, line location, sigma]
            line_num: Number of lines for fit

        Return:
            Value of function given input parameters (theta)

        """
        f1 = 0.0
        for model_num in range(line_num):
            params = theta[model_num:(model_num + 1)]
            f1 += self.function(channel, params)
        return f1

    def evaluate_bayes(self, channel, theta):
        """
        Function to initiate the model calculation for Bayesian Analysis

        Args:
            channel: Wavelength Axis in cm-1
            theta: List of parameters for all the models in the following order
                            [amplitude, line location, sigma]

        Return:
            Value of function given input parameters (theta)

        """
        f1 = 0.0
        params = theta
        f1 += self.function(channel, params)
        return f1

    def plot(self, channel, theta, line_num):
        """
        Function to initiate the correct number of models to fit

        Args:
            channel: Wavelength Axis in cm-1
            theta: List of parameters for all the models in the following order
                            [amplitude, line location, sigma]
            line_num: Number of lines for fit

        Return:
            Value of function given input parameters (theta)

        """
        f1 = 0.0
        for model_num in range(line_num):
            pos_on_axis = line_pos
            params = [theta[model_num], pos_on_axis, line_broad]
            f1 += self.function(channel, params)
        #f1 += theta[-1]
        return f1


class Sinc_frozen:
    """
    Class encoding all functionality of the sinc function. This includes
    a function call, an evaluation of the function for all lines to be fit,
    and a plot call.
    """

    def __init__(self, initial_velocity, initial_broadening, line_name):
        self.line_dict = {'Halpha': 656.280, 'NII6583': 658.341, 'NII6548': 654.803,
                          'SII6716': 671.647, 'SII6731': 673.085, 'OII3726': 372.603,
                          'OII3729': 372.882, 'OIII4959': 495.891, 'OIII5007': 500.684,
                          'Hbeta': 486.133, 'OH': 649.873}
        line_theo = self.line_dict[line_name]
        line_pos = 1e7 / ((initial_velocity/SPEED_OF_LIGHT) * line_theo + line_theo)
        line_broad = (line_pos * initial_broadening) / (SPEED_OF_LIGHT)

    def function(self, channel, params, sinc_width):
        p0 = params[0];
        p1 = line_pos
        p2 = sinc_width
        u = (channel - p1) / p2
        return [p0 * np.sinc(u_) if u_ != 0 else p0 for u_ in u]

    def evaluate(self, channel, theta, line_num, sinc_width):
        """
        Function to initiate the correct number of models to fit

        Args:
            channel: Wavelength Axis in cm-1
            theta: List of parameters for all the models in the following order
                            [amplitude, line location, sigma]
            line_num: Number of lines for fit
            sinc_width: Fixed with of the sinc function

        Return:
            Value of function given input parameters (theta)

        """
        f1 = 0.0
        for model_num in range(line_num):
            params = theta[model_num:(model_num + 1)]
            f1 += np.array(self.function(channel, params, sinc_width))
        return f1

    def evaluate_bayes(self, channel, theta, sinc_width):
        """
        Function to initiate the model calculation for Bayesian Analysis

        Args:
            channel: Wavelength Axis in cm-1
            theta: List of parameters for all the models in the following order
                            [amplitude, line location, sigma]
            sinc_width: Fixed with of the sinc function

        Return:
            Value of function given input parameters (theta)

        """
        f1 = 0.0
        params = theta
        f1 += np.array(self.function(channel, params, sinc_width))
        return f1


    def plot(self, channel, theta, line_num, sinc_width):
        """
        Function to initiate the correct number of models to fit

        Args:
            channel: Wavelength Axis in cm-1
            theta: List of parameters for all the models in the following order
                            [amplitude, line location, sigma]
            line_num: Number of lines for fit
            sinc_width: Fixed with of the sinc function

        Return:
            Value of function given input parameters (theta)

        """
        f1 = 0.0
        for model_num in range(line_num):
            pos_on_axis = line_pos
            params = [theta[model_num], pos_on_axis, line_broad]
            f1 += np.array(self.function(channel, params, sinc_width))
        return f1



class SincGauss_frozen:
    """
    Class encoding all functionality of the sincgauss function. This includes
    a function call, an evaluation of the function for all lines to be fit,
    and a plot call.
    """

    def __init__(self, initial_velocity, initial_broadening, lines):
        self.line_dict = {'Halpha': 656.280, 'NII6583': 658.341, 'NII6548': 654.803,
                          'SII6716': 671.647, 'SII6731': 673.085, 'OII3726': 372.603,
                          'OII3729': 372.882, 'OIII4959': 495.891, 'OIII5007': 500.684,
                          'Hbeta': 486.133, 'OH': 649.873}
        self.lines=lines
        self.initial_velocity=initial_velocity
        self.initial_broadening=initial_broadening

    def function(self, channel, params, line_position, line_broadening, sinc_width):
        p0 = params[0]
        p1 = line_position[0]
        p2 = sinc_width/np.pi
        p3 = line_broadening[0]
        a = p3/(np.sqrt(2)*p2)
        b = (channel-p1)/(np.sqrt(2)*p3)
        dawson1 = sps.dawsn(1j * a + b) * np.exp(2.* 1j * a *b)
        dawson2 = sps.dawsn(1j * a - b) * np.exp(-2. * 1j * a *b)
        dawson3 = 2. * sps.dawsn(1j * a)
        return p0*(dawson1 + dawson2)/dawson3

    def evaluate(self, channel, theta, line_num, sinc_width):
        """
        Function to initiate the correct number of models to fit

        Args:
            channel: Wavelength Axis in cm-1
            theta: List of parameters for all the models in the following order
                            [amplitude, line location, sigma]
            line_num: Number of lines for fit
            sinc_width: Fixed width of the sinc function

        Return:
            Value of function given input parameters (theta)

        """
        line_pos=[]
        line_broad=[]
        for line_name in self.lines:
            line_theo = self.line_dict[line_name]
            line_pos = np.append(line_pos, 1e7 / ((self.initial_velocity/SPEED_OF_LIGHT) * line_theo + line_theo))
            line_broad = np.append(line_broad, (line_pos * self.initial_broadening) / (SPEED_OF_LIGHT))
        f1 = 0.0
        for model_num in range(line_num):
            params = theta[model_num:(model_num + 1)]
            line_position = line_pos[model_num:(model_num + 1)]
            line_broadening = line_broad[model_num:(model_num + 1)]
            f1 += self.function(channel, params, line_position, line_broadening, sinc_width)
        return np.real(f1)

    def evaluate_bayes(self, channel, theta, sinc_width):
        """
        Function to initiate the model calculation for Bayesian Analysis

        Args:
            channel: Wavelength Axis in cm-1
            theta: List of parameters for all the models in the following order
                            [amplitude, line location, sigma]
            sinc_width: Fixed with of the sinc function

        Return:
            Value of function given input parameters (theta)

        """
        f1 = 0.0
        params = theta
        f1 += self.function(channel, params, sinc_width)
        return np.real(f1)

    def plot(self, channel, theta, line_num, sinc_width):
        """
        Function to initiate the correct number of models to fit

        Args:
            channel: Wavelength Axis in cm-1
            theta: List of parameters for all the models in the following order
                            [amplitude, line location, sigma]
            line_num: Number of lines for fit
            sinc_width: Fixed with of the sinc function

        Return:
            Value of function given input parameters (theta)

        """
        f1 = 0.0
        line_pos=[]
        line_broad=[]
        for line_name in self.lines:
            line_theo = self.line_dict[line_name]
            line_pos = np.append(line_pos, 1e7 / ((self.initial_velocity/SPEED_OF_LIGHT) * line_theo + line_theo))
            line_broad = np.append(line_broad, (line_pos * self.initial_broadening) / (SPEED_OF_LIGHT))
        for model_num in range(line_num):
            line_position = line_pos[model_num:(model_num + 1)]
            line_broadening = line_broad[model_num:(model_num + 1)]
            params = [theta[model_num]]
            f1 += self.function(channel, params, line_position, line_broadening, sinc_width)
        return np.real(f1)
