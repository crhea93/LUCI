import numpy as np
from scipy.optimize import minimize
from scipy import interpolate
from numdifftools import Hessian, Hessdiag
import emcee
import scipy.special as sps
import astropy.stats as astrostats
import warnings
import dynesty
from dynesty import utils as dyfunc
from LUCI.LuciFunctions import Gaussian, Sinc, SincGauss
from LUCI.LuciFitParameters import calculate_vel, calculate_vel_err, calculate_broad, calculate_broad_err, \
    calculate_flux, calculate_flux_err
from LUCI.LuciBayesian import log_probability, prior_transform, log_likelihood_bayes

warnings.filterwarnings("ignore")

# Define Constants #
SPEED_OF_LIGHT = 299792  # km/s

class Fit:
    """
    Class that defines the functions necessary for the modelling aspect. This includes
    the gaussian fit functions, the prior definitions, the log likelihood, and the
    definition of the posterior (log likelihood times prior).

    All the functions (gauss, sinc, and sincgauss) are stored in `LuciFuncitons.py`.

    All functions for calculating the velocity, broadening, and flux are in 'LuciFitParameters.py'.

    All the functions for Bayesian Inference with the exception of the fit call
    are in 'LuciBayesian.py'.

    The returned axis is the redshifted axis.

    If the initial_values argument is passed, then the fit algorithm will use these values [velocity, broadening] as
    the initial conditions for the fit **instead** of the machine learning algorithm. See example 'initial_values' for
    more details on the implementation.
    """

    def __init__(self, spectrum, axis, wavenumbers_syn, model_type, lines, vel_rel, sigma_rel,
                 ML_model, trans_filter=None,
                 theta=0, delta_x=2943, n_steps=842, zpd_index=169, filter='SN3',
                 bayes_bool=False, bayes_method='emcee',
                 uncertainty_bool=False, mdn=False,
                 nii_cons=True, sky_lines=None, sky_lines_scale=None, initial_values=False,
                 spec_min=None, spec_max=None, obj_redshift=0.0
                 ):
        """
        Args:
            spectrum: Spectrum of interest. This should not be the interpolated spectrum nor normalized(numpy array)
            axis: Wavelength Axis of Spectrum after Redshift Application (numpy array)
            axis_unshifted: Wavelength Axis of Spectrum after Redshift Application (numpy array)
            wavenumbers_syn: Wavelength Axis of Reference Spectrum (numpy array)
            model_type: Type of model ('gaussian')
            lines: Lines to fit (must be in line_dict)
            vel_rel: Constraints on Velocity/Position (must be list; e.x. [1, 2, 1])
            sigma_rel: Constraints on sigma (must be list; e.x. [1, 2, 1])
            ML_model: Tensorflow/keras machine learning model
            trans_filter: Tranmission filter interpolated on unredshifted spectral axis
            theta: Interferometric angle in degrees (defaults to 11.960 -- this is so that the correction coeff is 1)
            delta_x: Step Delta
            n_steps: Number of steps in spectra
            zpd_index: Zero Path Difference index
            filter: SITELLE filter (e.x. 'SN3')
            bayes_bool: Boolean to determine whether or not to run Bayesian analysis (default False)
            bayes_method: Bayesian Inference method. Options are '[emcee', 'dynesty'] (default 'emcee')
            uncertainty_bool: Boolean to determine whether or not to run the uncertainty analysis (default False)
            mdn: Boolean to determine which network to use (if true use MDN if false use standard CNN)
            nii_cons: Boolean to turn on or off NII doublet ratio constraint (default True)
            sky_lines: Dictionary of sky lines {OH_num: wavelength in nanometers}
            sky_lines_scale: List of relative strengths of sky lines
            initial_values: List of initial conditions for the velocity and broadening; [velocity, broadening]
            spec_min: Minimum value of the spectrum to be considered in the fit (we find the closest value)
            spec_max: Maximum value of the spectrum to be considered in the fit
            obj_redshift: Redshift of object to fit relative to cube's redshift. This is useful for fitting high redshift objects
        """
        self.line_dict = {'Halpha': 656.280, 'NII6583': 658.341, 'NII6548': 654.803,
                          'SII6716': 671.647, 'SII6731': 673.085, 'OII3726': 372.603,
                          'OII3729': 372.882, 'OIII4959': 495.891, 'OIII5007': 500.684,
                          'Hbeta': 486.133, 'OH': 649.873}
        self.available_functions = ['gaussian', 'sinc', 'sincgauss']
        self.sky_lines = sky_lines
        self.sky_lines_scale = sky_lines_scale
        self.obj_redshift_corr = 1 + obj_redshift
        for line_key in self.line_dict:
            self.line_dict[line_key] = self.line_dict[line_key] * self.obj_redshift_corr
        self.nii_cons = nii_cons
        self.spec_min = spec_min
        self.spec_max = spec_max
        self.spectrum = spectrum
        self.spectrum_clean = spectrum / np.max(spectrum)  # Clean normalized spectrum
        self.spectrum_normalized = self.spectrum / np.max(self.spectrum)  # Normalized spectrum  Yes it is duplicated
        self.axis = axis  # Redshifted axis
        self.spectrum_restricted = None
        self.spectrum_restricted_norm = None
        self.axis_restricted = None
        self.wavenumbers_syn = wavenumbers_syn
        self.model_type = model_type
        self.lines = lines
        self.line_num = len(lines)  # Number of  lines to fit
        self.trans_filter = trans_filter
        if trans_filter is not None:
            self.apply_transmission()  # Apply transmission filter if one is provided
        self.filter = filter
        self.spectrum_interpolated = np.zeros_like(self.spectrum)
        self.spectrum_interp_scale = None
        self.spectrum_interp_norm = np.zeros_like(self.spectrum)
        self.restrict_wavelength()
        self.theta = theta
        self.cos_theta = np.abs(np.cos(np.deg2rad(self.theta)))
        self.correction_factor = 1.0  # Initialize Correction factor
        self.axis_step = 0.0  # Initialize
        self.delta_x = delta_x
        self.n_steps = n_steps
        self.zpd_index = zpd_index
        self.calculate_correction()
        # Update axis with correction factor
        # self.axis = self.axis*self.correction_factor
        # Calculate noise
        self.noise = 1e-2  # Initialize
        self.calculate_noise()
        self.sigma_rel = sigma_rel
        self.vel_rel = vel_rel
        # ADD ML_MODEL
        self.ML_model = ML_model
        self.bayes_bool = bayes_bool
        self.bayes_method = bayes_method
        self.uncertainty_bool = uncertainty_bool
        self.spectrum_scale = 0.0  # Sacling factor used to normalize spectrum
        self.sinc_width = 0.0  # Width of the sinc function -- Initialize to zero
        self.calc_sinc_width()
        self.mdn = mdn
        self.vel_ml = 0.0  # ML Estimate of the velocity [km/s]
        self.broad_ml = 0.0  # ML Estimate of the velocity dispersion [km/s]
        self.vel_ml_sigma = 0.0  # ML Estimate for velocity 1-sigma error
        self.broad_ml_sigma = 0.0  # ML Estimate for velocity dispersion 1-sigma error
        self.initial_conditions = initial_values  # List for initial conditions (or default False)
        self.initial_values = initial_values  # List for initial values (or default False)
        self.fit_sol = np.zeros(3 * self.line_num + 1)  # Solution to the fit
        self.uncertainties = np.zeros(3 * self.line_num + 1)  # 1-sigma errors on fit parameters
        self.flat_samples = None
        # Check that lines inputted by user are in line_dict
        if sky_lines is None:
            self.check_lines()
        self.check_fitting_model()
        self.check_lengths()

    def apply_transmission(self):
        """
        Apply transmission curve on the spectra according to un-redshifted axis.
        This is done before we interpolate onto the wavenumbers_syn so that the axis
        align properly. Note -- the values of the x-axis are not important for this
        division since we have already interpolated the transition filter vector
        over the UNSHIFTED spectral axis.
        """
        self.spectrum = [self.spectrum[i] / self.trans_filter[i] if self.trans_filter[i] > 0.5 else self.spectrum[i] for
                         i in range(len(self.spectrum))]

    def calculate_correction(self):
        """
        Calculate correction factor based of interferometric angle. This is used to correct the broadening
        """
        self.correction_factor = 1 / self.cos_theta
        self.axis_step = self.correction_factor / (2 * self.delta_x * (self.n_steps - self.zpd_index)) * 1e7

    def calc_sinc_width(self, ):
        """
        Calculate sinc width of the sincgauss function
        """
        MPD = self.cos_theta * self.delta_x * (self.n_steps - self.zpd_index) / 1e7
        self.sinc_width = 1 / (2 * MPD)

    def restrict_wavelength(self):
        """
        Restrict the wavelength range of the fit so that the fit only occurs over the central regions of the spectra.
        We do this so that the continuum is properly calculated.
        """
        # Determine filter

        if self.spec_min is None or self.spec_max is None:  # If the user has not entered explicit bounds
            global bound_lower, bound_upper
            if self.filter == 'SN3':
                bound_lower = 14750
                bound_upper = 15400
            elif self.filter == 'SN2':
                bound_lower = 19500
                bound_upper = 20750
            elif self.filter == 'SN1':
                bound_lower = 26000
                bound_upper = 28000
            elif self.filter == 'C3' and 'OII3726' in self.lines:
                ## This is true for objects with a redshift around 0.465
                # We pretend we are looking at SN1
                bound_lower = 18000
                bound_upper = 19400
            elif self.filter == 'C4' and 'Halpha' in self.lines:
                ## This is true for objects at redshift ~0.25
                # In this case we pretend we are in SN3
                bound_lower = 14950  # LYA mod - originally the same as SN3 restrictions
                bound_upper = 15400
            else:
                print(
                    'The filter of your datacube is not supported by LUCI. We only support C3, C4, SN1, SN2, and SN3 at the moment.')
            self.spec_min = bound_lower
            self.spec_max = bound_upper
        else:
            pass
        min_ = np.argmin(np.abs(np.array(self.axis) - self.spec_min))
        max_ = np.argmin(np.abs(np.array(self.axis) - self.spec_max))
        self.spectrum_restricted = self.spectrum_normalized[min_:max_]
        self.axis_restricted = self.axis[min_:max_]
        self.spectrum_restricted_norm = self.spectrum_restricted / np.max(self.spectrum_restricted)
        return min_, max_

    def calculate_noise(self):
        """
        Calculate noise level in spectrum. We assume the noise is homogenous across the
        spectrum. We estimate it by taking a region outside of the filter and calculating
        the standard deviation in that region. We use the normalized spectrum since that
        is what is passed to the fit function.
        """
        # Determine filter
        global bound_lower, bound_upper
        if self.filter == 'SN3':
            bound_lower = 14300  # 16000
            bound_upper = 14500  # 16400
        elif self.filter == 'SN2':
            bound_lower = 18600
            bound_upper = 19000
        elif self.filter == 'SN1':
            bound_lower = 25300
            bound_upper = 25700
        elif self.filter == 'C3' and 'OII3726' in self.lines:
            ## This is true for objects at redshift ~0.465
            # In this case we pretend we are in SN1
            bound_lower = 18000
            bound_upper = 19400
        elif self.filter == 'C4' and 'Halpha' in self.lines:
            ## This is true for objects at redshift ~0.25
            # In this case we pretend we are in SN3
            bound_lower = 14600  # LYA mods, originally the same as SN3
            bound_upper = 14950
        else:
            print(
                'The filter of your datacube is not supported by LUCI. We only support C3, C4, SN1, SN2, and SN3 at the moment.')
        # Calculate standard deviation
        min_ = np.argmin(np.abs(np.array(self.axis) - bound_lower))
        max_ = np.argmin(np.abs(np.array(self.axis) - bound_upper))
        spec_noise = self.spectrum_clean[min_:max_]
        self.noise = np.nanstd(spec_noise)

    def estimate_priors_ML(self, mdn=True):
        """
        Apply machine learning algorithm on spectrum in order to estimate the velocity.
        The spectrum fed into this method must be interpolated already onto the
        reference spectrum axis AND normalized as described in Rhea et al. 2020a.
        Args:
            mdn: Boolean to use MDN or not (default True)

        Return:
            Updates self.vel_ml
        """
        Spectrum = self.spectrum_interp_norm.reshape(1, self.spectrum_interp_norm.shape[0], 1)
        if self.mdn:
            prediction_distribution = self.ML_model(Spectrum, training=False)
            prediction_mean = prediction_distribution.mean().numpy().tolist()
            prediction_stdv = prediction_distribution.stddev().numpy().tolist()
            self.vel_ml = [pred[0] for pred in prediction_mean][0]
            self.vel_ml_sigma = [pred[0] for pred in prediction_stdv][0]
            self.broad_ml = [pred[1] for pred in prediction_mean][0]
            self.broad_ml_sigma = [pred[1] for pred in prediction_stdv][0]
        elif self.mdn == False:
            predictions = self.ML_model(Spectrum, training=False)
            self.vel_ml = float(predictions[0][0])
            self.vel_ml_sigma = 0
            self.broad_ml = float(predictions[0][1])
            self.broad_ml_sigma = 0
        return None

    def interpolate_spectrum(self):
        """
        Interpolate Spectrum given the wavelength axis of reference spectrum.
        Then normalize the spectrum so that the max value equals 1

        Return:
            Populates self.spectrum_interpolated, self.spectrum_scale, and self.spectrum_interp_norm.

        """
        self.spectrum_scale = np.max(self.spectrum)
        f = interpolate.interp1d(self.axis, self.spectrum, kind='slinear', fill_value='extrapolate')
        self.spectrum_interpolated = f(self.wavenumbers_syn)
        self.spectrum_interp_scale = np.max(self.spectrum_interpolated)
        self.spectrum_interp_norm = self.spectrum_interpolated / self.spectrum_interp_scale
        return None

    def line_vals_estimate(self, line_name):
        """
        TODO: Test

        Function to estimate the position and amplitude of a given line.

        Args:
            line_name: Name of model. Available options are 'Halpha', 'NII6548', 'NII6543', 'SII6716', 'SII6731'

        Return:
            Estimated line amplitude in units of cm-1 (line_amp_est) and estimate line position in units of cm-1 (line_pos_est)

        """
        line_theo = self.line_dict[line_name]
        if self.ML_model is None or self.ML_model == '':
            if self.initial_values is not False:
                self.vel_ml = self.initial_values[0]  # Velocity component of initial conditions in km/s
                self.broad_ml = self.initial_values[1]  # Broadening component of initial conditions in km/s
            else:
                pass
        else:
            pass  # vel_ml and broad_ml already set using ML algorithm
        line_pos_est = 1e7 / ((self.vel_ml / SPEED_OF_LIGHT) * line_theo + line_theo)  # Estimate of position of line in cm-1
        line_ind = np.argmin(np.abs(np.array(self.axis) - line_pos_est))
        try:
            line_amp_est = np.max([
                #self.spectrum_normalized[line_ind - 4],
                self.spectrum_normalized[line_ind - 3],
                self.spectrum_normalized[line_ind - 2],
                self.spectrum_normalized[line_ind - 1],
                self.spectrum_normalized[line_ind],
                self.spectrum_normalized[line_ind + 1],
                self.spectrum_normalized[line_ind + 2],
                self.spectrum_normalized[line_ind + 3],
                #self.spectrum_normalized[line_ind + 4]
            ])
        except:
            line_amp_est = self.spectrum_normalized[line_ind]
        line_broad_est = (line_pos_est * self.broad_ml) / (SPEED_OF_LIGHT)
        if self.mdn:
            # Update position and sigma_gauss bounds -- looks gross but it's the usual transformation
            self.x_min = 1e7 / (((self.vel_ml + 3 * self.vel_ml_sigma) / SPEED_OF_LIGHT) * line_theo + line_theo)  # Estimate of position of line in cm-1
            self.x_max = 1e7 / (((self.vel_ml - 3 * self.vel_ml_sigma) / SPEED_OF_LIGHT) * line_theo + line_theo)  # Estimate of position of line in cm-1
            self.sigma_min = (line_pos_est * self.broad_ml) / SPEED_OF_LIGHT - 3 * (line_pos_est * self.broad_ml_sigma) / SPEED_OF_LIGHT
            self.sigma_max = (line_pos_est * self.broad_ml) / SPEED_OF_LIGHT + 3 * (line_pos_est * self.broad_ml_sigma) / SPEED_OF_LIGHT
        return line_amp_est, line_pos_est, line_broad_est

    def cont_estimate(self, sigma_level=3):
        """
        TODO: Test

        Function to estimate the continuum level. We use a sigma clipping algorithm over the(mu_vel/SPEED_OF_LIGHT)*line_dict[line[ct]] + line_dict[line[ct]]
        restricted axis/spectrum to effectively ignore emission lines. Therefore, we
        are left with the continuum. We take the minimum value of this continuum as the initial
        guess.

        Args:
            sigma_level: Sigma level to clip (Default=3)

        Return:
            Initial guess for continuum

        """
        # Define continuum regions
        min_ = 0
        max_ = 1e6
        if self.filter == 'SN3':
            min_ = 14950
            max_ = 15050
        elif self.filter == 'SN2':
            min_ = 19500
            max_ = 19550
        elif self.filter == 'SN1':
            min_ = 26000
            max_ = 26250
        elif self.filter == 'C3':
            min_ = 18000
            max_ = 19000

        # Clip values at given sigma level (defined by sigma_level)
        clipped_spec = astrostats.sigma_clip(self.spectrum_restricted[min_:max_], sigma=sigma_level,
                                             masked=False, copy=False,
                                             maxiters=3, stdfunc=astrostats.mad_std)
        if len(clipped_spec) < 1:
            clipped_spec = self.spectrum_restricted
        # Now take the minimum value to serve as the continuum value
        cont_val = np.median(clipped_spec)
        return cont_val

    def log_likelihood(self, theta):
        """
        Calculate log likelihood function evaluated given parameters on spectral axis

        Args:
            theta: List of parameters for all the models in the following order
                            [amplitude, line location, sigma, continuum constant]
                    The continuum constant is always the last argument regardless of the number of lines being modeled
        Return:
            Value of log likelihood

        """
        model = 0
        if self.model_type == 'gaussian':
            model = Gaussian().evaluate(self.axis_restricted, theta, self.line_num)
        elif self.model_type == 'sinc':
            model = Sinc().evaluate(self.axis_restricted, theta, self.line_num, self.sinc_width)
        elif self.model_type == 'sincgauss':
            model = SincGauss().evaluate(self.axis_restricted, theta, self.line_num, self.sinc_width)
        # Add constant continuum to model
        model += theta[-1]
        sigma2 = self.noise ** 2
        #return np.sum(stats.norm.logpdf(self.spectrum_restricted, loc=model, scale=sigma2))
        return -0.5 * np.sum((self.spectrum_restricted - model) ** 2 / sigma2) + np.log(2 * np.pi * sigma2)


    def sigma_constraints(self):
        """
        Set up constraints for sigma values before fitting line
        Return:
            Dictionary describing constraints
        """
        sigma_dict_list = []
        unique_rels = np.unique(self.sigma_rel)  # List of unique groups
        for unique_ in unique_rels:  # Step through each unique group
            inds_unique = [i for i, e in enumerate(self.sigma_rel) if e == unique_]  # Obtain line indices in group
            if len(inds_unique) > 1:  # If there is more than one element in the group
                ind_0_ = inds_unique[0]  # Get first element
                for ind_unique_ in inds_unique[1:]:  # Step through group elements except for the first one
                    sigma_dict_list.append({'type': 'eq', 'fun': lambda x, ind_unique=ind_unique_, ind_0=ind_0_:
                    (SPEED_OF_LIGHT * x[3 * ind_0 + 2]) / x[3 * ind_0 + 1] -
                    (SPEED_OF_LIGHT * x[3 * ind_unique + 2]) / x[3 * ind_unique + 1]}
                                           )
        return sigma_dict_list

    def vel_constraints(self):
        """
        Set up constraints for velocity values before fitting line
        Return:
            Dictionary describing constraints
        """
        vel_dict_list = []
        unique_rels = np.unique(self.vel_rel)  # List of unique groups
        for unique_ in unique_rels:  # Step through each unique group
            inds_unique = [i for i, e in enumerate(self.vel_rel) if e == unique_]  # Obtain line indices in group
            if len(inds_unique) > 1:  # If there is more than one element in the group
                ind_0 = inds_unique[0]  # Get first element
                ind_0_line = self.lines[ind_0]
                for ind_unique in inds_unique[1:]:  # Step through group elements except for the first one
                    ind_unique_line = self.lines[ind_unique]
                    expr_dict = {'type': 'eq',
                                 'fun': lambda x, ind_unique_=ind_unique, ind_0_=ind_0,
                                               ind_unique_line_=ind_unique_line, ind_0_line_=ind_0_line:
                                 SPEED_OF_LIGHT * ((1e7 / x[3 * ind_unique_ + 1] - self.line_dict[ind_unique_line_]) / (
                                 self.line_dict[ind_unique_line_]))
                                 - SPEED_OF_LIGHT * ((1e7 / x[3 * ind_0_ + 1] - self.line_dict[ind_0_line_]) / (
                                 self.line_dict[ind_0_line_]))}
                    vel_dict_list.append(expr_dict)
        return vel_dict_list

    def NII_constraints(self):
        """
        Enforce the constraint that the NII6548 lines has an amplitude that is 1/3 the amplitude of NII6583.

        Return:
            Constraint on NII doublet relative amplitudes
        """
        global func_
        nii_doublet_constraints = []
        # First we have to figure out which lines correspond to the doublet
        nii_6548_index = np.argwhere(np.array(self.lines) == 'NII6548')[0][0]
        nii_6583_index = np.argwhere(np.array(self.lines) == 'NII6583')[0][0]
        # Now tie the amplitudes together s/t that amplitude of the NII6548 line is
        # always 1/3 that of the NII6583 line
        # expr_dict = {'type': 'eq','fun': lambda x: (1/3)*x[3*nii_6548_index] - x[3*nii_6583_index]}
        if self.model_type == 'gaussian':
            func_ = lambda x: (1 / 3) * (x[3 * nii_6583_index] * x[3 * nii_6583_index + 2]) - x[3 * nii_6548_index] * x[
                3 * nii_6548_index + 2]
        elif self.model_type == 'sinc':
            func_ = lambda x: (1 / 3) * (x[3 * nii_6583_index] * x[3 * nii_6583_index + 2]) - x[3 * nii_6548_index] * x[
                3 * nii_6548_index + 2]
        elif self.model_type == 'sincgauss':
            func_ = lambda x: (1 / 3) * (x[3 * nii_6583_index] * ((np.sqrt(2 * np.pi) * x[3 * nii_6583_index + 2]) / (
                sps.erf((x[3 * nii_6583_index + 2]) / (np.sqrt(2) * self.sinc_width))))) - \
                              x[3 * nii_6548_index] * ((np.sqrt(2 * np.pi) * x[3 * nii_6548_index + 2]) / (sps.erf(
                (x[3 * nii_6548_index + 2]) / (np.sqrt(
                    2) * self.sinc_width))))  # LYA mod, this is the correct one thank you so much for putting this in Carter!!
        expr_dict = {'type': 'eq', 'fun': func_}
        nii_doublet_constraints.append(expr_dict)
        return nii_doublet_constraints

    def multiple_component_vel_constraint(self):
        """
        Constraints for the case that we have multiple components.
        If there are two components (i.e. the user passes the same line twice),
        we require that the first component has a higher velocity (wavenumber really) than the second component.
        This forces the solver to find the two components instead of simply fitting the same
        component twice.
        This should work for three or more components, but I haven't tested it.
        """
        multi_dict_list = []
        unique_lines = np.unique(self.lines)  # List of unique groups
        for unique_ in unique_lines:  # Step through each unique group
            inds_unique = [i for i, e in enumerate(self.lines) if e == unique_]  # Obtain line indices in group
            if len(inds_unique) > 1:  # If there is more than one element in the group
                ind_0 = inds_unique[0]  # Get first element
                for ind_unique in inds_unique[1:]:  # Step through group elements except for the first one
                    expr_dict_vel = {'type': 'ineq',
                                     'fun': lambda x, ind_unique=ind_unique, ind_0=ind_0: x[3 * ind_unique + 1] - x[
                                         3 * ind_0 + 1]+10}
                    multi_dict_list.append(expr_dict_vel)
        return multi_dict_list

    def calculate_params(self):
        """
        Calculate the amplitude, position, and sigma of the line. These values are
        calculated using the scipy.optimize.minimize function. This is called
        on the log likelood previously described. The minimization algorithm uses
        the SLSQP optimization implementation. We have applied standard bounds in order
        to speed up the fitting. We also apply the fit on the normalized spectrum.
        We then correct the flux by un-normalizing the spectrum.

        """
        nll = lambda *args: -self.log_likelihood(*args)  # Negative Log Likelihood function
        initial = np.ones((3 * self.line_num + 1))  # Initialize solution vector  (3*num_lines plus continuum)
        initial[-1] = self.cont_estimate(sigma_level=2)  # Add continuum constant and initialize it
        lines_fit = []  # List of lines which already have been set up for fits
        cons = None
        for mod in range(self.line_num):  # Step through each line
            lines_fit.append(self.lines[mod])  # Add to list of lines fit
            amp_est, vel_est, sigma_est = self.line_vals_estimate(self.lines[mod])  # Estimate initial values
            initial[3 * mod] = amp_est - initial[-1]  # Subtract continuum estimate from amplitude estimate
            initial[3 * mod + 1] = vel_est  # Set wavenumber
            initial[3 * mod + 2] = sigma_est  # Set sigma
        self.initial_values = initial
        sigma_cons = self.sigma_constraints()  # Call sigma constraints
        vel_cons = self.vel_constraints()  # Call velocity constraints
        vel_cons_multiple = self.multiple_component_vel_constraint()
        # CONSTRAINTS
        if 'NII6548' in self.lines and 'NII6583' in self.lines and self.nii_cons is True:  # Add additional constraint on NII doublet relative amplitudes
            nii_constraints = self.NII_constraints()
            cons = sigma_cons + vel_cons + vel_cons_multiple + nii_constraints
        else:
            cons = sigma_cons + vel_cons + vel_cons_multiple
        # Call minimize! This uses the previously defined negative log likelihood function and the restricted axis
        # We do **not** use the interpolated spectrum here!
        soln = minimize(nll, initial,
                        method='SLSQP',
                        options={'disp': False, 'maxiter': 30},
                        tol=1e-2,
                        args=(), constraints=cons
                        )
        parameters = soln.x
        # We now must unscale the amplitude
        for i in range(self.line_num):
            parameters[i * 3] *= self.spectrum_scale
            self.uncertainties[i * 3] *= self.spectrum_scale
        # Scale continuum
        parameters[-1] *= self.spectrum_scale
        self.uncertainties[-1] *= self.spectrum_scale
        if self.uncertainty_bool is True:
            # Calculate uncertainties using the negative inverse hessian  as the covariance matrix
            try:
                hessian_calc = Hessian(nll, method='backward')   # Set method to backward to speed things up
                hessian_calc = hessian_calc(parameters)
                covariance_mat = -np.linalg.inv(hessian_calc)
                self.uncertainties = np.sqrt(np.abs(np.diagonal(covariance_mat)))
            except np.linalg.LinAlgError:
                self.uncertainties = np.zeros_like(parameters)

        self.fit_sol = parameters
        # Create fit vector
        if self.model_type == 'gaussian':
            self.fit_vector = Gaussian().plot(self.axis, self.fit_sol[:-1], self.line_num) + self.fit_sol[-1]
        elif self.model_type == 'sinc':
            self.fit_vector = Sinc().plot(self.axis, self.fit_sol[:-1], self.line_num, self.sinc_width) + self.fit_sol[
                -1]
        elif self.model_type == 'sincgauss':
            self.fit_vector = SincGauss().plot(self.axis, self.fit_sol[:-1], self.line_num, self.sinc_width) + \
                              self.fit_sol[-1]
        else:
            print("Somehow all the checks missed the fact that you didn't enter a valid fit function...")

        return None

    def fit(self, sky_line=False):
        """
        Primary function call for a spectrum. This will estimate the velocity using
        our machine learning algorithm described in Rhea et al. 2020a. Then we will
        fit our lines using scipy.optimize.minimize.

        Args:
            sky_line: Boolean to fit sky lines (default False)

        Return:
            dictionary of parameters returned by the fit. The dictionary has the following form:
            {"fit_vector": Fitted spectrum, "velocity": Velocity of the line in km/s (float),
            "broadening": Velocity Dispersion of the line in km/s (float)}
        """
        if sky_line != True:
            if self.ML_model != None and self.initial_values is False:
                # Interpolate Spectrum
                self.interpolate_spectrum()
                # Estimate the priors using machine learning algorithm
                self.estimate_priors_ML()
            else:
                self.spectrum_scale = np.max(self.spectrum)
            # Apply Fit
            self.calculate_params()
            if np.isnan(self.fit_sol[0]):  # Check that there are no Nans in solution
                # If a Nan is found, then we redo the fit without the ML priors
                temp_ML = self.ML_model
                self.ML_model = ''
                self.calculate_params()
                self.ML_model = temp_ML
            # Check if Bayesian approach is required
            if self.bayes_bool:
                self.fit_Bayes()
            # Calculate fit statistic
            chi_sqr, red_chi_sqr = self.calc_chisquare(self.fit_vector, self.spectrum, self.noise,
                                                       3 * self.line_num + 1)
            # Collect Amplitudes
            ampls = []
            fluxes = []
            vels = []
            sigmas = []
            vels_errors = []
            sigmas_errors = []
            flux_errors = []
            for line_ct, line_ in enumerate(self.lines):  # Step through each line
                ampls.append(self.fit_sol[line_ct * 3])
                # Calculate flux
                fluxes.append(calculate_flux(self.fit_sol[line_ct * 3], self.fit_sol[line_ct * 3 + 2], self.model_type,
                                             self.sinc_width))
                vels.append(calculate_vel(line_ct, self.lines, self.fit_sol, self.line_dict))
                sigmas.append(calculate_broad(line_ct, self.fit_sol, self.axis_step))
                vels_errors.append(
                    calculate_vel_err(line_ct, self.lines, self.fit_sol, self.line_dict, self.uncertainties))
                sigmas_errors.append(calculate_broad_err(line_ct, self.fit_sol, self.axis_step, self.uncertainties))
                flux_errors.append(
                    calculate_flux_err(line_ct, self.fit_sol, self.uncertainties, self.model_type, self.sinc_width))
            # Collect parameters to return in a dictionary
            fit_dict = {'fit_sol': self.fit_sol, 'fit_uncertainties': self.uncertainties,
                        'amplitudes': ampls, 'fluxes': fluxes, 'flux_errors': flux_errors, 'chi2': red_chi_sqr,
                        'velocities': vels, 'sigmas': sigmas,
                        'vels_errors': vels_errors, 'sigmas_errors': sigmas_errors,
                        'axis_step': self.axis_step, 'corr': self.correction_factor,
                        'continuum': self.fit_sol[-1], 'scale': self.spectrum_scale,
                        'vel_ml': self.vel_ml, 'vel_ml_sigma': self.vel_ml_sigma,
                        'broad_ml': self.broad_ml, 'broad_ml_sigma': self.broad_ml_sigma,
                        'fit_vector': self.fit_vector, 'fit_axis': self.axis,
                        }
            return fit_dict
        else:  # Fit sky line
            self.spectrum_scale = np.max(self.spectrum)
            # Apply Fit
            nll = lambda *args: self.log_likelihood(*args)  # Negative Log Likelihood function
            initial = np.ones(3*self.line_num + 1)
            skylines_vals = list(self.sky_lines.values())
            for mod in range(self.line_num):
                initial[3 * mod] = self.cont_estimate(sigma_level=5) * self.sky_lines_scale[mod]
                initial[3 * mod + 1] = 1e7 / ((80 * (skylines_vals[mod]) / SPEED_OF_LIGHT) + skylines_vals[mod])
                initial[3 * mod + 2] = 1
            initial[-1] = self.cont_estimate(sigma_level=5)
            self.initial_values = initial
            sigma_cons = self.sigma_constraints()  # Call sigma constaints
            vel_cons = self.vel_constraints()  # Call velocity constraints
            cons = sigma_cons+vel_cons
            soln = minimize(nll, initial,method='trust-constr',
                            options={'disp': False, 'maxiter': 30}, tol=1e-3,
                            args=())#, constraints=cons)
            parameters = soln.x

            # We now must unscale the amplitude
            for i in range(self.line_num):
                parameters[i * 3] *= self.spectrum_scale
                self.uncertainties[i * 3] *= self.spectrum_scale
            # Scale continuum
            parameters[-1] *= self.spectrum_scale
            self.fit_sol = parameters
            velocity = SPEED_OF_LIGHT * ((1e7 / self.fit_sol[1] - skylines_vals[0]) / skylines_vals[0])
            fit_vector = Sinc().plot(self.axis, self.fit_sol[:-1], self.line_num, self.sinc_width) + parameters[-1]
            return velocity, fit_vector

    def fit_Bayes(self):
        """
        Apply Bayesian MCMC run to constrain the parameters after solving
        """
        # Unscale the amplitude
        for i in range(self.line_num):
            self.fit_sol[i * 3] /= self.spectrum_scale
        self.fit_sol[-1] /= self.spectrum_scale
        # Set the number of dimensions -- this is somewhat arbitrary
        n_dim = 3 * self.line_num + 1
        # Set number of MCMC walkers. Again, this is somewhat arbitrary
        n_walkers = n_dim * 5
        # Initialize walkers
        random_ = 1e-1 * np.random.randn(n_walkers, n_dim)
        for i in range(self.line_num):
            random_[:, 3 * i + 1] *= 1e2
            random_[:, 3 * i + 2] *= 1e1
        init_ = self.fit_sol + random_  # + self.fit_sol[-1] + random_
        # Ensure continuum values for walkers are positive
        init_[:, -1] = np.abs(init_[:, -1])
        if self.bayes_method == 'dynesty':
            # Run nested sampling
            dsampler = dynesty.NestedSampler(log_likelihood_bayes, prior_transform, ndim=n_dim,
                                             logl_args=(self.axis_restricted, self.spectrum_restricted,
                                                        self.noise, self.model_type, self.line_num, self.sinc_width,
                                                        self.vel_rel, self.sigma_rel),
                                                        sample='rwalk', maxiter=1000, bound='balls'
                                             )
            dsampler.run_nested()
            dres = dsampler.results
            samples, weights = dres.samples, np.exp(dres.logwt - dres.logz[-1])
            mean, cov = dyfunc.mean_and_cov(samples, weights)
            std = np.sqrt(np.diag(cov))
            parameters_med = mean
            parameters_std = std
        elif self.bayes_method == 'emcee':
            # Set Ensemble Sampler
            sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability,
                                            args=(self.axis_restricted, self.spectrum_restricted,
                                                  self.noise, self.model_type, self.line_num, self.lines,
                                                  self.line_dict, self.sinc_width,
                                                  [self.vel_ml, self.broad_ml, self.vel_ml_sigma, self.broad_ml_sigma],
                                                  self.vel_rel, self.sigma_rel, self.mdn
                                                  )  # End additional args
                                            )  # End EnsembleSampler
            # Call Ensemble Sampler setting 2000 walks
            sampler.run_mcmc(init_, 10000, progress=False)
            # Obtain Ensemble Sampler results and discard first 200 walks (10%)
            flat_samples = sampler.get_chain(discard=1000, flat=True)
            parameters_med = []
            parameters_std = []
            self.flat_samples = flat_samples
            for i in range(n_dim):  # Calculate and store these results
                median = np.median(flat_samples[:, i])
                std = np.std(flat_samples[:, i])
                parameters_med.append(median)
                parameters_std.append(std)
            # self.fit_sol = parameters_med
            # self.uncertainties = parameters_std

        else:
            print("The bayes_method parameter has been incorrectly set to '%s'" % self.bayes_method)
            print("Please enter either 'emcee' or 'dynesty' instead.")
        # We now must scale the amplitude
        for i in range(self.line_num):
            parameters_med[i * 3] *= self.spectrum_scale
            parameters_std[i * 3] *= self.spectrum_scale
        # Scale continuum
        parameters_med[-1] *= self.spectrum_scale
        parameters_std[-1] *= self.spectrum_scale

        self.fit_sol = parameters_med
        self.uncertainties = parameters_std
        # Calculate fit vector using updated values
        if self.model_type == 'gaussian':
            self.fit_vector = Gaussian().plot(self.axis, self.fit_sol[:-1], self.line_num) + self.fit_sol[-1]
        elif self.model_type == 'sinc':
            self.fit_vector = Sinc().plot(self.axis, self.fit_sol[:-1], self.line_num, self.sinc_width) + self.fit_sol[
                -1]
        elif self.model_type == 'sincgauss':
            self.fit_vector = SincGauss().plot(self.axis, self.fit_sol[:-1], self.line_num, self.sinc_width) + \
                              self.fit_sol[-1]

    def calc_chisquare(self, fit_vector, init_spectrum, init_errors, n_dof):
        """
        Calculate reduced chi 2

        Args:
            fit_vector: Spectrum obtained from fit
            init_spectrum: Observed spectrum
            init_errors: Errors on observed spectrum
            n_dof: Number of degrees of freedom

        Return:
            chi2: Chi squared value
            chi2dof: Reduced chi squared value
        """
        # compute the mean and the chi^2/dof
        min_restricted, max_restricted = self.restrict_wavelength()
        z = (init_spectrum[min_restricted: max_restricted]- fit_vector[min_restricted: max_restricted])
        chi2 = np.sum((z ** 2)/(init_errors*self.spectrum_scale))
        chi2dof = chi2 / (len(fit_vector[min_restricted: max_restricted]) - n_dof )
        return chi2, chi2dof

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
            raise Exception(
                'Please submit a fitting function name in the available list: \n {}'.format(self.available_functions))

    def check_lengths(self):
        """
        This function checks to see that the length of the sigma_rel and vel_rel arguments are correct
        Return:
        Nothing if the user provides appropriate length
        Else it will throw an error

        """
        if len(self.vel_rel) != len(self.lines):
            raise Exception("The argument vel_rel has %i arguments, but it should have %i arguments" % (
                len(self.vel_rel), len(self.lines)))
        elif len(self.sigma_rel) != len(self.lines):
            raise Exception("The argument sigma_rel has %i arguments, but it should have %i arguments" % (
                len(self.sigma_rel), len(self.lines)))
        else:
            pass
