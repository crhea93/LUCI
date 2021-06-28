import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import interpolate
import keras
from scipy.optimize import Bounds
from numdifftools import Jacobian, Hessian
import emcee
from scipy.stats import chisquare

import warnings

warnings.filterwarnings("ignore")


class Gaussian:
    def __init__(self, channel, params):
        A = params[0];
        x = params[1];
        sigma = params[2]
        self.func = A * np.exp((-(channel - x) ** 2) / (2 * sigma ** 2))


class Sinc:
    def __init__(self, channel, params):
        p0 = params[0];
        p1 = params[1];
        p2 = params[2]
        u = (channel - p1) / p2
        self.func = p0 * (np.sin(u) / u)


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
                 ML_model, bayes_bool=False, Plot_bool=False):
        """
        Args:
            spectrum: Spectrum of interest. This should not be the interpolated spectrum nor normalized(numpy array)
            axis: Wavelength Axis of Spectrum (numpy array)
            wavenumbers_syn: Wavelength Axis of Reference Spectrum (numpy array)
            model_type: Type of model ('gaussian')
            lines: Lines to fit (must be in line_dict)
            ML_model: Tensorflow/keras machine learning model
            bayes_bool:
            Plot_bool: Boolean to determine whether or not to plot the spectrum (default = False)

        """
        self.line_dict = {'Halpha': 656.280, 'NII6583': 658.341, 'NII6548': 654.803,
                          'SII6716': 671.647, 'SII6731': 673.085, 'OII3726': 372.603,
                          'OII3729': 372.882, 'OIII4959': 495.891, 'OIII5007': 500.684,
                          'Hbeta': 486.133}
        self.available_functions = ['gaussian']
        self.spectrum = spectrum
        self.axis = axis
        self.wavenumbers_syn = wavenumbers_syn
        self.model_type = model_type
        self.lines = lines
        self.line_num = len(lines)  # Number of  lines to fit
        self.spectrum_interpolated = np.zeros_like(self.spectrum)
        self.spectrum_normalized = self.spectrum / np.max(self.spectrum)  # Normalized spectrum
        self.spectrum_interp_norm = np.zeros_like(self.spectrum)
        # ADD ML_MODEL AND PLOT_BOOL
        self.ML_model = ML_model
        self.bayes_bool = bayes_bool
        self.Plot_bool = Plot_bool
        self.spectrum_scale = 0.0  # Sacling factor used to normalize spectrum
        self.vel_ml = 0.0  # ML Estimate of the velocity [km/s]
        self.broad_ml = 0.0  # ML Estimate of the velocity dispersion [km/s]
        self.fit_sol = np.zeros(3 * self.line_num)  # Solution to the fit
        # Set bounds
        self.A_min = 0;
        self.A_max = 1.1;
        self.x_min = 14700;
        self.x_max = 15600
        self.sigma_min = 0;
        self.sigma_max = 10

        # Check that lines inputted by user are in line_dict
        self.check_lines()
        self.check_fitting_model()

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
        Spectrum = self.spectrum_interp_norm.reshape(1, self.spectrum_interp_norm.shape[0], 1)
        predictions = self.ML_model(Spectrum, training=False)
        self.vel_ml = float(predictions[0][0])
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
        self.spectrum_interpolated = f(self.wavenumbers_syn)
        self.spectrum_scale = np.max(self.spectrum_interpolated)
        self.spectrum_interp_norm = self.spectrum_interpolated / self.spectrum_scale
        # self.spectrum_interpolated = np.real(sky_corr)
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
        if self.ML_model is None or self.model_type == '':
            max_flux = np.argmax(self.spectrum)
            self.vel_ml = np.abs(3e5 * ((1e7/self.axis[max_flux] - line_theo) / line_theo))
            self.broad_ml = 10.0  # Best for now
        else:
            pass  # vel_ml and broad_ml already set using ML algorithm
        line_pos_est = 1e7 / ((self.vel_ml / 3e5) * line_theo + line_theo)  # Estimate of position of line in cm-1
        line_ind = np.argmin(np.abs(np.array(self.axis) - line_pos_est))
        line_amp_est = np.max([self.spectrum_normalized[line_ind - 2], self.spectrum_normalized[line_ind - 1],
                               self.spectrum_normalized[line_ind], self.spectrum_normalized[line_ind + 1],
                               self.spectrum_normalized[line_ind + 2]])
        line_broad_est = (line_pos_est * self.broad_ml) / 3e5
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
            params = theta[model_num * 3:(model_num + 1) * 3]
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
        return -0.5 * np.sum((self.spectrum_normalized - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))

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
        initial = np.ones((3 * self.line_num))
        bounds_ = []
        for mod in range(self.line_num):
            val = 3 * mod + 1
            amp_est, vel_est, sigma_est = self.line_vals_estimate(self.lines[mod])
            initial[3 * mod] = amp_est
            initial[3 * mod + 1] = vel_est
            initial[3 * mod + 2] = sigma_est
            bounds_.append((self.A_min, self.A_max))
            bounds_.append((self.x_min, self.x_max))
            bounds_.append((self.sigma_min, self.sigma_max))
        bounds_l = [val[0] for val in bounds_]
        bounds_u = [val[1] for val in bounds_]
        bounds = Bounds(bounds_l, bounds_u)
        self.inital_values = initial
        if self.line_num == 5:
            cons = ({'type': 'eq',
                     'fun': lambda x: 3e5 * ((1e7 / x[4] - self.line_dict['NII6583']) / (1e7 / x[4])) - 3e5 * (
                             (1e7 / x[1] - self.line_dict['Halpha']) / (1e7 / x[1]))},
                    {'type': 'eq', 'fun': lambda x: x[2] - x[5]},
                    {'type': 'eq', 'fun': lambda x: x[5] - x[8]},
                    {'type': 'eq', 'fun': lambda x: x[5] - x[11]},
                    {'type': 'eq', 'fun': lambda x: x[5] - x[14]},
                    {'type': 'eq',
                     'fun': lambda x: 3e5 * ((1e7 / x[4] - self.line_dict['NII6583']) / (1e7 / x[4])) - 3e5 * (
                             (1e7 / x[7] - self.line_dict['NII6548']) / (1e7 / x[7]))},
                    {'type': 'eq',
                     'fun': lambda x: 3e5 * ((1e7 / x[4] - self.line_dict['NII6583']) / (1e7 / x[4])) - 3e5 * (
                             (1e7 / x[10] - self.line_dict['SII6716']) / (1e7 / x[10]))},
                    {'type': 'eq',
                     'fun': lambda x: 3e5 * ((1e7 / x[4] - self.line_dict['NII6583']) / (1e7 / x[4])) - 3e5 * (
                             (1e7 / x[13] - self.line_dict['SII6731']) / (1e7 / x[13]))})
        else:
            cons = ()
        soln = minimize(nll, initial, method='SLSQP', # jac=self.fun_der(),
                        options={'disp': False, 'maxiter': 100}, bounds=bounds, tol=1e-8,
                        args=(1e-2), constraints=cons)
        parameters = soln.x
        # We now must unscale the amplitude
        for i in range(self.line_num):
            parameters[i * 3] *= self.spectrum_scale
        self.fit_sol = parameters
        self.fit_vector = self.gaussian_model(self.axis, self.fit_sol)
        return None

    def calculate_vel(self):
        """
        Calculate velocity given the fit of Halpha
        TODO: Test
        TODO: Add other lines

        Return:
            Velocity of the Halpha line in units of km/s
        """
        l_calc = 1e7 / self.fit_sol[1]  # Halpha
        l_shift = (l_calc - 656.28) / l_calc
        v = 3e5 * l_shift
        return v

    def calculate_broad(self):
        """
        Calculate velocity dispersion given the fit of Halpha
        TODO: Test
        TODO: Add other lines

        Return:
            Velocity Dispersion of the Halpha line in units of km/s
        """
        broad = (3e5 * self.fit_sol[2]) / self.fit_sol[1]
        return broad

    def calculate_flux(self, line_amp, line_sigma):
        """
        Calculate flux value given fit of line
        TODO: Test

        Args:
            line_amp: Amplitude of the line (un-normalized)
            line_sigma: Sigma of the line fit
        Return:
            Flux of the provided line in units of erg/s/cm-2
        """
        flux = 0.0  # Initialize
        if self.model_type == 'gaussian':
            flux = np.sqrt(2 * np.pi) * line_amp * line_sigma
        elif self.model_type == 'sinc':
            flux = np.sqrt(np.pi) * line_amp * line_sigma
        return flux

    def fit(self):
        """
        Primary function call for a spectrum. This will estimate the velocity using
        our machine learning algorithm described in Rhea et al. 2020a. Then we will
        fit our lines using scipy.optimize.minimize.

        Return:
            dictionary of parameters returned by the fit. The dictionary has the following form:
            {"fit_vector": Fitted spectrum, "velocity": Velocity of the line in km/s (float),
            "broadening": Velocity Dispersion of the line in km/s (float)}
        """
        # Interpolate Spectrum
        self.interpolate_spectrum()
        # Estimate the priors using machine learning algorithm
        if self.ML_model != None:
            self.estimate_priors_ML()
        # Apply Fit
        self.calculate_params()
        # Check if Bayesian approach is required
        if self.bayes_bool == True:
            self.fit_Bayes()
        # Calculate fit statistic
        chi_sqr, p_val = chisquare(self.fit_vector, self.spectrum)
        # Collect Amplitudes
        ampls = []
        fluxes = []
        for line_ct, line_ in enumerate(self.lines):  # Step through each line
            ampls.append(self.fit_sol[line_ct * 3])
            # Calculate flux
            fluxes.append(self.calculate_flux(self.fit_sol[line_ct * 3], self.fit_sol[line_ct * 3 + 2]))

        # Collect parameters to return in a dictionary
        fit_dict = {'fit_sol': self.fit_vector, 'velocity': self.calculate_vel(),
                    'broadening': self.calculate_broad(), 'amplitudes': ampls,
                    'fluxes': fluxes, 'chi2': chi_sqr}
        # Plot
        if self.Plot_bool == True:
            self.plot()
        return fit_dict

    def fit_Bayes(self):
        """
        Apply Bayesian MCMC run to constrain the parameters after solving
        """
        # Unscale the amplitude
        for i in range(self.line_num):
            self.fit_sol[i * 3] /= self.spectrum_scale
        n_dim = 3 * self.line_num
        n_walkers = n_dim * 2 + 4
        init_ = self.fit_sol + 1 * np.random.randn(n_walkers, n_dim)
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, self.log_probability,
                                        args=(self.axis, self.spectrum_normalized, 1e-2, self.lines))
        sampler.run_mcmc(init_, 100, progress=False)
        flat_samples = sampler.get_chain(discard=20, flat=True)
        parameters = []
        for i in range(n_dim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            parameters.append(mcmc[1])
        self.fit_sol = parameters
        # Now rescale the amplitude
        for i in range(self.line_num):
            self.fit_sol[i * 3] *= self.spectrum_scale

    def log_likelihood_bayes(self, theta, x, y, yerr, model__):
        """
        """
        # model = self.gaussian_model(x, theta, model)
        model = self.gaussian_model(self.axis, theta)
        sigma2 = yerr ** 2
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))

    def log_prior(self, theta, model):
        A_min = 0  # 1e-19
        A_max = 1.  # 1e-15
        x_min = 14700
        x_max = 15400
        sigma_min = 0
        sigma_max = 10
        for model_num in range(len(model)):
            params = theta[model_num * 3:(model_num + 1) * 3]
        within_bounds = True  # Boolean to determine if parameters are within bounds
        for ct, param in enumerate(params):
            if ct % 3 == 0:  # Amplitude parameter
                if param > A_min and param < A_max:
                    pass
                else:
                    within_bounds = False  # Value not in bounds
                    break
            if ct % 3 == 1:  # velocity parameter
                if param > x_min and param < x_max:
                    pass
                else:
                    within_bounds = False  # Value not in bounds
                    break
            if ct % 3 == 2:  # sigma parameter
                if param > sigma_min and param < sigma_max:
                    pass
                else:
                    within_bounds = False  # Value not in bounds
                    break
        if within_bounds:
            return 0.0
        else:
            return -np.inf
        # A_,x_,sigma_ = theta
        # if A_min < A_ < A_max and x_min < x_ < x_max and sigma_min < sigma_ < sigma_max:
        #    return 0.0#np.log(1/((t_max-t_min)*(rp_max-rp_min)*(b_max-b_min)))
        # return -np.inf

    def log_probability(self, theta, x, y, yerr, model):
        lp = self.log_prior(theta, model)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood_bayes(theta, x, y, yerr, model)

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
        if self.model_type in self.available_functions:
            pass
        else:
            print(self.model_type)
            raise Exception(
                'Please submit a fitting function name in the available list: \n {}'.format(self.available_functions))
