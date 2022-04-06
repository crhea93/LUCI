from LUCI.LuciFunctions import Gaussian, Sinc, SincGauss
import numpy as np




def log_likelihood_bayes(theta, axis_restricted, spectrum_restricted, yerr, model_type, line_num, sinc_width, vel_rel, sigma_rel):
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
        vel_rel:
        sigma_rel

    Return:
        Gaussian log likelihood function evaluation
    """
    model = 0  # Initialize
    '''unique_vel_ct = 0  # Ct of how many uniques used for velocity
    unique_broad_ct = 0  # Ct of how many uniques used for broadening
    for line_ct in range(line_num):  # Each line
        amp = theta[3*line_ct]  # Amplitude
        vel = theta[3*line_ct+1]  # Velocity
        broad = theta[3*line_ct+2]  # Broadening
        if line_ct > 0:  # Only check for lines that aren't the first line being fit.
            if vel_rel[line_ct] == vel_rel[line_ct - 1]:  # if  the vel is tied to the previous one
                unique_vel_ct += 1
                vel = theta[3*(line_ct-unique_vel_ct)]  # Velocity from the first of the uniques
            else:
                unique_vel_ct = 0  # Reset unique velocities
            if sigma_rel[line_ct] == sigma_rel[line_ct - 1]:  # if  the vel is tied to the previous one
                unique_broad_ct += 1
                broad = theta[3*(line_ct-unique_broad_ct)]
            else:
                unique_broad_ct = 0  # Reset unique broadening
        else:  # First line in list
            pass
        # Now get actual values
        theta_line = [amp, vel, broad]
        if model_type == 'gaussian':
            model = Gaussian().evaluate_bayes(axis_restricted, theta_line)
        elif model_type == 'sinc':
            model = Sinc().evaluate_bayes(axis_restricted, theta_line, sinc_width)
        elif model_type == 'sincgauss':
            model = SincGauss().evaluate_bayes(axis_restricted, theta_line, sinc_width)'''
    # Add constant contimuum to model
    if model_type == 'gaussian':
        model = Gaussian().evaluate(axis_restricted, theta, line_num)
    elif model_type == 'sinc':
        model = Sinc().evaluate(axis_restricted, theta, line_num , sinc_width)
    elif model_type == 'sincgauss':
        model = SincGauss().evaluate(axis_restricted, theta, line_num, sinc_width)
    #model += theta[-1]
    sigma2 = yerr ** 2
    #print(theta)
    #print(axis_restricted)
    #print(model)
    #print(sigma2)
    #exit()
    val = 0.5 * np.sum((spectrum_restricted - model) ** 2 / sigma2) + np.log(2 * np.pi * sigma2)
    if np.isnan(val):
        return -np.inf
    else:
        return val

def log_prior(theta, axis_restricted, line_num, line, line_dict, mu_vel, mu_broad, sigma_vel, sigma_broad):
    """
    Calculate log prior assuming uniform priors on the amplitude and continuum values.

    The uniform prior on the amplitude is 1/(A_max-A_min)

    The uniform prior on the continuum is 1/(C_min-C_max)

    The priors on the line position and Gaussian broadening are defined by Gaussian Priors
    The mu and std of these priors are calculated using a MDN (see LuciNetwork).


    TODO: The velocity and broadening need to be updated to be in cm^-1 using the formulas in LuciDocs

    Args:
        theta: Fit parameters
        axis_restricted:
        line_nun: Number of lines for fit

    Return:
        If theta parameters are within the limits, we return the negative log prior value.
        Else we return -np.inf

    """
    A_min = -0.1
    A_max = 1.25
    continuum_min = 0
    continuum_max = .75
    val_prior = 0
    within_bounds = True  # Boolean to determine if parameters are within bounds
    line_num = -1  # Current line  -- initialization
    for ct, param in enumerate(theta):
        if ct % 3 == 0:  # Amplitude parameter & continuum parameter
            line_num += 1
            if param > A_min and param < A_max:
                val_prior -= np.log(A_max-A_min)
            else:
                #print('amp: %.2E'%param)
                within_bounds = False  # Value not in bounds
                break
        if ct % 3 == 1:  # velocity parameter
            mu_vel_min = (mu_vel/3e5+10*sigma_vel/3e5)*line_dict[line[line_num]] + line_dict[line[line_num]]  # Convert to position in nm
            mu_vel_min = 1e7/mu_vel_min  # Convert to cm-1
            mu_vel_max = (mu_vel/3e5-10*sigma_vel/3e5)*line_dict[line[line_num]] + line_dict[line[line_num]]  # Convert to position in nm
            mu_vel_max = 1e7/mu_vel_max  # Convert to cm-1
            if param > mu_vel_min and param < mu_vel_max:
                val_prior -= np.log(6*sigma_vel)
            else:
                within_bounds = False  # Value not in bounds
                break
        if ct % 3 == 2:  # sigma parameter
            if param > mu_broad - 3*sigma_broad and param < mu_broad + 3*sigma_broad:
                val_prior -= np.log(6*sigma_broad)
            else:
                #print('broad: %i'%param)
                within_bounds = False  # Value not in bounds
                break
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
    x_max = 1e7
    sigma_min = 0
    sigma_max = 30
    continuum_min = -1e-8
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


def prior_transform(u):
    A_min = 0.001
    A_max = 1.1
    x_min = 1e4
    x_max = 1e7
    sigma_min = 0.01
    sigma_max = 100
    continuum_min = 0.001
    continuum_max = 0.9
    prior_list = []
    '''for ct, param in enumerate(u[:-1]):  # Don't include continuum
        if ct % 3 == 0:  # Amplitude parameter
            prior_list.append(u[ct])
        if ct % 3 == 1:  # velocity parameter
            prior_list.append((x_max)*u[ct])
        if ct % 3 == 2:  # sigma parameter
            prior_list.append((sigma_max)*u[ct]+0.01)'''
    uA, usigma, ux0, ucont = u
    A = 3*uA
    sigma = 50*usigma
    x0 = 500*(ux0+15000)
    cont = ucont
    return A, x0, sigma, cont
    #prior_list.append((continuum_max)*u[ct])  # Include continuum
    #return prior_list


def log_probability(theta, axis_restricted, spectrum_restricted, yerr, model_type, line_num, lines, line_dict, sinc_width, prior_gauss, vel_rel, sigma_rel, mdn):
    """
    Calculate a Gaussian likelihood function given a certain fitting function: gaussian, sinc, or sincgauss

    Args:
        theta: Fit parameters
        axis_restricted: Wavelength axis restricted to fitting region
        spectrum_restricted: Flux values corresponded to restricted wavelength axis
        yerr: Noise in data
        model_type: Fitting function (i.e. 'gaussian', 'sinc', or 'sincgauss')
        line_num: Number of lines for fit
        lines: List of lines to fit
        line_dict: Dictionary of available lines with wavelength in nm
        sinc_width: Fixed with of the sinc function
        prior_gauss: Parameters for Gaussian priors [mu_vel, mu_broad, sigma_vel, sigma_broad]
        vel_rel:
        sigma_rel:
        mdn: Boolean for MDN

    Return:
        If not finite or if an nan we return -np.inf. Otherwise, we return the log likelihood + log prior
    """
    mu_vel, mu_broad, sigma_vel, sigma_broad = prior_gauss
    lp = 0
    if mdn:
        lp = log_prior(theta, axis_restricted, line_num, lines, line_dict, mu_vel, mu_broad, sigma_vel, sigma_broad)
    else:
        lp = log_prior_uniform(theta, line_num)
    if not np.isfinite(lp):
        return -np.inf
    if np.isnan(lp):
        return -np.inf
    if np.isnan(lp + log_likelihood_bayes(theta, axis_restricted, spectrum_restricted, yerr, model_type, line_num, sinc_width, vel_rel, sigma_rel)):
        return -np.inf
    else:
        return lp + log_likelihood_bayes(theta, axis_restricted, spectrum_restricted, yerr, model_type, line_num, sinc_width, vel_rel, sigma_rel)
