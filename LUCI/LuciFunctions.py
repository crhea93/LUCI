import numpy as np
from scipy import special as sps


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
            min_ind = np.argmin(np.abs(channel - theta[3*model_num+1])) - 1
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
        p1 = params[1];
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
            min_ind = np.argmin(np.abs(channel - theta[3*model_num+1])) - 1
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
        '''erf1 = sps.erf(a-1j*b)
        erf2 = sps.erf(a+1j*b)
        erf3 = 2*sps.erf(a)
        self.func = p0*np.exp(-b**2)*((erf1+erf2)/erf3)'''

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
            min_ind = np.argmin(np.abs(channel - theta[3*model_num+1]))-1
            pos_on_axis = channel[min_ind]
            params = [theta[model_num * 3], pos_on_axis, theta[model_num*3 + 2]]
            f1 += self.function(channel, params, sinc_width)
        return np.real(f1)
