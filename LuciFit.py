import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import interpolate
import keras
from scipy.optimize import Bounds
from numdifftools import Jacobian, Hessian




class Gaussian:
    def __init__(self, channel, params):
        A = params[0]; x = params[1]; sigma = params[2]
        self.func = A*np.exp((-(channel-x)**2)/(2*sigma**2))



class Fit:
    """
    Class that defines the functions necessary for the modelling aspect. This includes
    the gaussian fit functions, the prior definitions, the log likelihood, and the
    definition of the posterior (log likelihood times prior).

    The initial arguments are as follows:
    Args:

        spectrum: Spectrum of interest. This should not be the interpolated spectrum nor normalized(numpy array)

        axis: Wavelength Axis of Spectrum (numpy array)

        wavenumbers_syn: Wavelength Axis of Reference Spectrum (numpy array)

        model_type: Type of model ('gaussian')

        lines: Lines to fit (must be in line_dict)

        ML_model: Tensorflow/keras machine learning model

        Plot_bool: Boolean to determine whether or not to plot the spectrum (default = False)

    """
    def __init__(self, spectrum, axis, wavenumbers_syn, model_type, lines,
                ML_model, Plot_bool = False):
        """
        Args:
            spectrum: Spectrum of interest. This should not be the interpolated spectrum nor normalized(numpy array)
            axis: Wavelength Axis of Spectrum (numpy array)
            wavenumbers_syn: Wavelength Axis of Reference Spectrum (numpy array)
            model_type: Type of model ('gaussian')
            lines: Lines to fit (must be in line_dict)
            ML_model: Tensorflow/keras machine learning model
            Plot_bool: Boolean to determine whether or not to plot the spectrum (default = False)

        """
        self.line_dict = {'Halpha': 656.280, 'NII6583': 658.341, 'NII6548': 654.803, 'SII6716': 671.647, 'SII6731': 673.085}
        self.spectrum = spectrum
        self.axis = axis
        self.wavenumbers_syn = wavenumbers_syn
        self.model_type = model_type
        self.lines = lines
        self.line_num = len(lines)  # Number of  lines to fit
        self.spectrum_interpolated = np.zeros_like(self.spectrum)
        self.spectrum_normalized = self.spectrum/np.max(self.spectrum)  # Normalized spectrum
        self.spectrum_interp_norm = np.zeros_like(self.spectrum)
        #ADD ML_MODEL AND PLOT_BOOL
        self.ML_model = ML_model
        self.Plot_bool = Plot_bool
        self.spectrum_scale = 0.0  # Sacling factor used to normalize spectrum
        self.vel_ml = 0.0  # ML Estimate of the velocity [km/s]
        self.broad_ml = 0.0  # ML Estimate of the velocity dispersion [km/s]
        self.fit_sol = np.zeros(3*self.line_num)  # Solution to the fit
        # Set bounds
        self.A_min = 0; self.A_max = 1.1; self.x_min = 14700; self.x_max = 15600
        self.sigma_min = 0; self.sigma_max = 10


    def estimate_priors_ML(self):
        """
        TODO: Test
        Apply machine learning algorithm on spectrum in order to estimate the velocity.
        The spectrum fed into this method must be interpolated already onto the
        reference spectrum axis AND normalized as described in Rhea et al. 2020a.
        Args:
            ml_dir: Relative path to the trained ML Predictor (e.g. R5000-PREDICITOR-I)
        Return:
            Updates self.vel_ml
        """
        #print(self.spectrum_interp_norm)
        #print(self.spectrum)
        Spectrum = self.spectrum_interp_norm.reshape(1, self.spectrum_interp_norm.shape[0], 1)
        predictions = self.ML_model(Spectrum, training=False)
        #print(predictions)
        #exit()
        self.vel_ml = float(predictions[0][0])
        #print(predictions[0][1])
        self.broad_ml = float(predictions[0][1])  # Multiply value by FWHM of a gaussian
        return None

    def interpolate_spectrum(self):
        """
        Interpolate Spectrum given the wavelength axis of reference spectrum.
        Then normalize the spectrum so that the max value equals 1

        Return:
            Populates self.spectrum_interpolated, self.spectrum_scale, and self.spectrum_interp_norm.

        """
        f = interpolate.interp1d(self.axis, self.spectrum, kind='slinear')
        self.spectrum_interpolated = (f(self.wavenumbers_syn))
        self.spectrum_scale = np.max(self.spectrum_interpolated)
        self.spectrum_interp_norm = self.spectrum_interpolated/self.spectrum_scale
        #self.spectrum_interpolated = np.real(sky_corr)
        return None


    def line_vals_estimate(self, line_name):
        """
        TODO: Test

        Function to estimate the position and amplitude of a given line.

        Args:
            spec: Spectrum flux values
            line_name: Name of model. Available options are 'Halpha', 'NII6548', 'NII6543', 'SII6716', 'SII6731'

        Return:
            Estimated line amplitude in units of cm-1 (line_amp_est) and estimate line position in units of cm-1 (line_pos_est)

        """
        line_theo = self.line_dict[line_name]
        #line_cm1 = 1e7/line_theo
        #max_flux = np.argmax(self.spectrum)
        #print(axis[max_flux])
        #self.vel_ml = np.abs(3e5*((self.axis[max_flux]-line_cm1)/line_cm1))
        #print(vel_ml)
        line_pos_est = 1e7/((self.vel_ml/3e5)*line_theo + line_theo)  # Estimate of position of line in cm-1
        line_ind = np.argmin(np.abs(np.array(self.axis)-line_pos_est))
        line_amp_est = np.max([self.spectrum_normalized[line_ind-2],self.spectrum_normalized[line_ind-1], self.spectrum_normalized[line_ind], self.spectrum_normalized[line_ind+1], self.spectrum_normalized[line_ind+2]])
        line_broad_est = (line_pos_est*self.broad_ml)/3e5
        #print(line_broad_est)
        #print(line_amp_est, line_pos_est, line_broad_est)
        return line_amp_est, line_pos_est, line_broad_est

    def gaussian_model(self, channel, theta):
        """
        Function to initiate the correct number of models to fit

        Args:
            channel: Wavelength Axis in cm-1
            theta: List of parameters for all the models in the following order
                            [amplitude, line location, sigma]

        Return:
            Value of function given input parameters (theta)

        """
        f1 = 0.0
        for model_num in range(self.line_num):
            params = theta[model_num*3:(model_num+1)*3]
            f1 += Gaussian(channel, params).func
        return f1

    def log_likelihood(self, theta, yerr):
        """
        Calculate log likelihood function evaluated given parameters on spectral axis

        Args:
            theta - List of parameters for all the models in the following order
                            [amplitude, line location, sigma]
            yerr: Error on Spectrum's flux values (default 1e-2)
        Return:
            Value of log likelihood

        """
        model = self.gaussian_model(self.axis, theta)
        sigma2 = yerr ** 2
        return -0.5 * np.sum((self.spectrum_normalized - model) ** 2 / sigma2 + np.log(2*np.pi*sigma2))

    def fun_der(self, theta, yerr):
        return Jacobian(lambda theta: self.log_likelihood(theta, yerr))(theta).ravel()

    def calculate_params(self):
        """
        Calculate the amplitude, position, and sigma of the line. These values are
        calculated using the scipy.optimize.minimize function. This is called
        on the log likelood previously described. The minimization algorithm uses
        the SLSQP optimization implementation. We have applied standard bounds in order
        to speed up the fitting. We also apply the fit on the normalized spectrum.
        We then correct the flux by un-normalizing the spectrum.

        """
        nll = lambda *args: -self.log_likelihood(*args)
        initial = np.ones((3*self.line_num))
        bounds_ = []
        for mod in range(self.line_num):
            val = 3*mod + 1
            amp_est, vel_est, sigma_est = self.line_vals_estimate(self.lines[mod])
            initial[3*mod] = amp_est
            initial[3*mod + 1] = vel_est
            initial[3*mod + 2] = sigma_est
            bounds_.append((self.A_min, self.A_max))
            bounds_.append((self.x_min, self.x_max))
            bounds_.append((self.sigma_min, self.sigma_max))
        bounds_l = [val[0] for val in bounds_]
        bounds_u = [val[1] for val in bounds_]
        bounds = Bounds(bounds_l, bounds_u)
        self.inital_values = initial
        if self.line_num == 5:
            cons = ({'type': 'eq', 'fun': lambda x: 3e5*((1e7/x[4]-self.line_dict['NII6583'])/(1e7/x[4])) - 3e5*((1e7/x[1]-self.line_dict['Halpha'])/(1e7/x[1]))},
            {'type': 'eq', 'fun': lambda x: x[2] - x[5]},
            {'type': 'eq', 'fun': lambda x: x[5] - x[8]},
            {'type': 'eq', 'fun': lambda x: x[5] - x[11]},
            {'type': 'eq', 'fun': lambda x: x[5] - x[14]},
            {'type': 'eq', 'fun': lambda x: 3e5*((1e7/x[4]-self.line_dict['NII6583'])/(1e7/x[4])) - 3e5*((1e7/x[7]-self.line_dict['NII6548'])/(1e7/x[7]))},
            {'type': 'eq', 'fun': lambda x: 3e5*((1e7/x[4]-self.line_dict['NII6583'])/(1e7/x[4])) - 3e5*((1e7/x[10]-self.line_dict['SII6716'])/(1e7/x[10]))},
            {'type': 'eq', 'fun': lambda x: 3e5*((1e7/x[4]-self.line_dict['NII6583'])/(1e7/x[4])) - 3e5*((1e7/x[13]-self.line_dict['SII6731'])/(1e7/x[13]))})
        else:
            cons = ()
        soln = minimize(nll, initial, method='SLSQP',# jac=self.fun_der(),
                options={'disp': False}, bounds=bounds, tol=1e-2,
                args=(1e-2), constraints=cons)
        parameters = soln.x
        #print(parameters)
        # We now must unscale the amplitude
        #for i in range(self.line_num):
        #    parameters[i*3] *= self.spectrum_scale
        self.fit_sol = parameters
        self.fit_vector = self.gaussian_model(self.axis, self.fit_sol)
        return None


    def calculate_vel(self):
        """
        Calculate velocity given the fit of Halpha
        TODO: Test

        Return:
            Velocity of the Halpha line in units of km/s
        """
        l_calc = 1e7/self.fit_sol[1]  # Halpha
        l_shift = (l_calc - 656.28)/l_calc
        v = 3e5*l_shift
        return v


    def calculate_broad(self):
        """
        Calculate velocity dispersion given the fit of Halpha
        TODO: Test

        Return:
            Velocity Dispersion of the Halpha line in units of km/s
        """
        #print(self.fit_sol[2], self.fit_sol[1])
        broad = (3e5*self.fit_sol[2])/self.fit_sol[1]
        #print(broad)
        return broad


    def fit(self):
        """
        Primary function call for a spectrum. This will estimate the velocity using
        our machine learning algorithm described in Rhea et al. 2020a. Then we will
        fit our lines using scipy.optimize.minimize.

        Return:
            dictionary of parameters returned by the fit. The dictionary has the following form:
            {"fit_sol": Fitted spectrum, "velocity": Velocity of the line in km/s (float),
            "broadening": Velocity Dispersion of the line in km/s (float)}
        """
        # Interpolate Spectrum
        self.interpolate_spectrum()
        # Estimate the priors using machine learning algorithm
        self.estimate_priors_ML()
        # Apply Fit
        self.calculate_params()
        # Collect Amplitudes
        #ampls_dict = {}  # Create amplitude dictionary for each line
        ampls = []
        for line_ct,line_ in enumerate(self.lines):  # Step through each line
            #ampls[line_] = self.fit_sol[line_ct*3]  # Save amplitude to amplitude dictionary
            ampls.append(self.fit_sol[line_ct*3])
        # Collect parameters to return in a dictionary
        fit_dict = {'fit_sol':self.fit_vector, 'velocity': self.calculate_vel(),
                    'broadening': self.calculate_broad(), 'amplitudes': ampls}
        # Plot
        if self.Plot_bool == True:
            self.plot()
        return fit_dict

    def plot(self):
        """
        Plot initial spectrum and fitted spectrum

        """
        plt.clf()
        plt.plot(self.axis, self.sky, label='Spectrum')
        plt.plot(self.axis, self.fit_vector, label='Fit')
        plt.legend()
        plt.show()
        return None
