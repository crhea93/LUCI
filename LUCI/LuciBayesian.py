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
    return -0.5 * np.sum((spectrum_restricted - model) ** 2 / sigma2) + np.log(2 * np.pi * sigma2)

def log_prior(theta, axis_restricted, line_num, mu_vel, mu_broad, sigma_vel, sigma_broad):
    """
    Calculate log prior assuming uniform priors on the amplitude and continuum values.

    The uniform prior on the amplitude is 1/(A_max-A_min)

    The uniform prior on the continuum is 1/(C_min-C_max)

    The priors on the line position and Gaussian broadening are defined by Gaussian Priors
    The mu and std of these priors are calculated using a MDN (see LuciNetwork).



    Args:
        theta: Fit parameters
        axis_restricted:
        line_nun: Number of lines for fit

    Return:
        If theta parameters are within the limits, we return the negative log prior value.
        Else we return -np.inf

    """
    A_min = -0.1
    A_max = 1.0
    continuum_min = 0
    continuum_max = .75
    val_prior = 0
    for model_num in range(line_num):
        params = theta[model_num * 3:(model_num + 1) * 3]
    within_bounds = True  # Boolean to determine if parameters are within bounds
    for ct, param in enumerate(params):
        if ct % 3 == 0:  # Amplitude parameter
            if param > A_min and param < A_max:
                val_prior -= np.log(A_max-A_min)
            else:
                within_bounds = False  # Value not in bounds
                break
        if ct % 3 == 1:  # velocity parameter
            #val_prior += np.log(1.0/(np.sqrt(2*np.pi)*sigma_vel))-0.5*(param-mu_vel)**2/sigma_vel**2
            if param > mu_vel - 3*sigma_vel and param < mu_vel + 3*sigma_vel:
                val_prior -= np.log(6*sigma_vel)
            else:
                within_bounds = False  # Value not in bounds
                break
        if ct % 3 == 2:  # sigma parameter
            #val_prior += np.log(1.0/(np.sqrt(2*np.pi)*sigma_broad))-0.5*(param-mu_broad)**2/sigma_broad**2
            if param > mu_broad - 3*sigma_broad and param < mu_broad + 3*sigma_broad:
                val_prior -= np.log(6*sigma_broad)
            else:
                within_bounds = False  # Value not in bounds
                break
    # Check continuum
    if theta[-1] > continuum_min and theta[-1] < continuum_max:
        val_prior -= np.log(continuum_max-continuum_min)
    else:
        within_bounds = False  # Value not in bounds
    if within_bounds:
        return -val_prior
    else:
        return -np.inf

def log_prior_uniform(theta, line_num):
    """
    Calculate log prior assuming uniform priors on the amplitude, line_position,
    broadening, and continuum values.

    The uniform prior on the amplitude is 1/(A_max-A_min)

    The uniform prior on the line position is 1/(x_max-x_min)

    The uniform prior on the gaussian broadening is 1/(sigma_min-sigma_max)

    The uniform prior on the continuum is 1/(C_min-C_max)


    Args:
        theta: Fit parameters
        line_nun: Number of lines for fit

    Return:
        If theta parameters are within the limits, we return the negative log prior value.
        Else we return -np.inf

    """
    A_min = -0.5
    A_max = 1.1
    x_min = 0
    x_max = 1e6
    sigma_min = 0
    sigma_max = 30
    continuum_min = 0
    continuum_max = 1
    val_prior = 0
    for model_num in range(line_num):
        params = theta[model_num * 3:(model_num + 1) * 3]
    within_bounds = True  # Boolean to determine if parameters are within bounds
    for ct, param in enumerate(params):
        if ct % 3 == 0:  # Amplitude parameter
            if param > A_min and param < A_max:
                val_prior -= np.log(A_max-A_min)
            else:
                within_bounds = False  # Value not in bounds
                break
        if ct % 3 == 1:  # velocity parameter
            if param > x_min and param < x_max:
                val_prior -= np.log(x_max-x_min)
            else:
                within_bounds = False  # Value not in bounds
                break
        if ct % 3 == 2:  # sigma parameter
            if param > sigma_min and param < sigma_max:
                val_prior -= np.log(sigma_max-sigma_min)
            else:
                within_bounds = False  # Value not in bounds
                break
    # Check continuum
    if theta[-1] > continuum_min and theta[-1] < continuum_max:
        val_prior -= np.log(continuum_max-continuum_min)
    else:
        within_bounds = False  # Value not in bounds
    if within_bounds:
        return -val_prior
    else:
        return -np.inf

def prior_transform_nestle(theta):
    A_min = -0.5
    A_max = 1.1
    x_min = 0
    x_max = 1e6
    sigma_min = 0
    sigma_max = 30
    continuum_min = 0
    continuum_max = 1
    prior_list = []
    for ct, param in enumerate(theta[:-1]):  # Don't include continuum
        if ct % 3 == 0:  # Amplitude parameter
            prior_list.append(A_max-A_min)
        if ct % 3 == 1:  # velocity parameter
            prior_list.append(x_max-x_min)
        if ct % 3 == 2:  # sigma parameter
            prior_list.append(sigma_max-sigma_min)
    prior_list.append(continuum_max-continuum_min)  # Include continuum
    return tuple(np.array(prior_list) * theta)


def prior_transform(u):
    A_min = 0.001
    A_max = 1.1
    x_min = 1e4
    x_max = 1e5
    sigma_min = 0.01
    sigma_max = 100
    continuum_min = 0.001
    continuum_max = 0.9
    prior_list = []
    for ct, param in enumerate(u[:-1]):  # Don't include continuum
        if ct % 3 == 0:  # Amplitude parameter
            prior_list.append(u[ct])
        if ct % 3 == 1:  # velocity parameter
            prior_list.append((x_max)*u[ct])
        if ct % 3 == 2:  # sigma parameter
            prior_list.append((sigma_max)*u[ct]+0.01)
    prior_list.append((continuum_max)*u[ct])  # Include continuum
    #print(prior_list)
    return prior_list


def log_probability(theta, axis_restricted, spectrum_restricted, yerr, model_type, line_num, sinc_width, prior_gauss):
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
        prior_gauss: Parameters for Gaussian priors [mu_vel, mu_broad, sigma_vel, sigma_broad]

    Return:
        If not finite or if an nan we return -np.inf. Otherwise, we return the log likelihood + log prior
    """
    mu_vel, mu_broad, sigma_vel, sigma_broad = prior_gauss
    lp = log_prior(theta, axis_restricted, line_num, mu_vel, mu_broad, sigma_vel, sigma_broad)
    #lp = log_prior_uniform(theta, line_num)
    if not np.isfinite(lp):
        return -np.inf
    if np.isnan(lp):
        return -np.inf
    if np.isnan(lp + log_likelihood_bayes(theta, axis_restricted, spectrum_restricted, yerr, model_type, line_num, sinc_width)):
        return -np.inf
    else:
        return lp + log_likelihood_bayes(theta, axis_restricted, spectrum_restricted, yerr, model_type, line_num, sinc_width)
