from LUCI.LuciFunctions import Gaussian, Sinc, SincGauss
import numpy as np


def log_likelihood_bayes(theta, axis_restricted, spectrum_restricted, yerr, model_type, line_num, sinc_width):
    """
    Calculate a Gaussian likelihood function given a certain fitting function: gaussian, sinc, or sincgauss

    Args:
        theta: Fit parameters
        axis_restricted: Wavelength axis restricted to fitting region
        spectrum_restricted: Flux values corresponded to restricted wavelength axis
        yerr: Noise in data
        model_type: Fitting function (i.e. 'gaussian', 'sinc', or 'sincgauss')
        line_num: Number of lines for fit
        sinc_width: Fixed with of the sinc function

    Return:
        Gaussian log likelihood function evaluation
    """
    if model_type == 'gaussian':
        model = Gaussian().evaluate(axis_restricted, theta, line_num)
    elif model_type == 'sinc':
        model = Sinc().evaluate(axis_restricted, theta, line_num , sinc_width)
    elif model_type == 'sincgauss':
        model = SincGauss().evaluate(axis_restricted, theta, line_num, sinc_width)
    # Add constant contimuum to model
    model += theta[-1]
    sigma2 = yerr ** 2
    return -0.5 * np.sum((spectrum_restricted - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))

def log_prior(theta, line_num):
    """
    Calculate log prior assuming uniform priors

    Args:
        theta: Fit parameters
        line_nun: Number of lines for fit

    Return:
        If theta parameters are within the limits, we return the negative log prior value.
        Else we return -np.inf

    """
    A_min = 0  # 1e-19
    A_max = 1.1  # 1e-15
    x_min = 0#14700
    x_max = 1e8#15400
    sigma_min = 0
    sigma_max = 30
    continuum_min = 0
    continuum_max = 1
    val_prior = 1
    for model_num in range(line_num):
        params = theta[model_num * 3:(model_num + 1) * 3]
    within_bounds = True  # Boolean to determine if parameters are within bounds
    for ct, param in enumerate(params):
        if ct % 3 == 0:  # Amplitude parameter
            if param > A_min and param < A_max:
                val_prior *= (A_max-A_min)
            else:
                within_bounds = False  # Value not in bounds
                break
        if ct % 3 == 1:  # velocity parameter
            if param > x_min and param < x_max:
                val_prior *= (x_max-x_min)
            else:
                within_bounds = False  # Value not in bounds
                break
        if ct % 3 == 2:  # sigma parameter
            if param > sigma_min and param < sigma_max:
                val_prior *= (sigma_max-sigma_min)
            else:
                within_bounds = False  # Value not in bounds
                break
    # Check continuum
    if theta[-1] > continuum_min and theta[-1] < continuum_max:
        val_prior *= (continuum_max-continuum_min)
    else:
        within_bounds = False  # Value not in bounds
    if within_bounds:
        return -np.log(val_prior)
    else:
        return -np.inf


def log_probability(theta, axis_restricted, spectrum_restricted, yerr, model_type, line_num, sinc_width):
    """
    Calculate a Gaussian likelihood function given a certain fitting function: gaussian, sinc, or sincgauss

    Args:
        theta: Fit parameters
        axis_restricted: Wavelength axis restricted to fitting region
        spectrum_restricted: Flux values corresponded to restricted wavelength axis
        yerr: Noise in data
        model_type: Fitting function (i.e. 'gaussian', 'sinc', or 'sincgauss')
        line_num: Number of lines for fit
        sinc_width: Fixed with of the sinc function

    Return:
        If not finite or if an nan we return -np.inf. Otherwise, we return the log likelihood + log prior
    """
    lp = log_prior(theta, line_num)
    if not np.isfinite(lp):
        return -np.inf
    if np.isnan(lp):
        return -np.inf
    if np.isnan(lp + log_likelihood_bayes(theta, axis_restricted, spectrum_restricted, yerr, model_type, line_num, sinc_width)):
        return -np.inf
    else:
        return lp + log_likelihood_bayes(theta, axis_restricted, spectrum_restricted, yerr, model_type, line_num, sinc_width)
