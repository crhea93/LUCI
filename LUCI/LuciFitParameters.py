"""
In this file we have the functions used to calculate the velocity, the velocity dispersion (broadening),
and the flux as well as their uncertainties.
"""
import math
import numpy as np
from scipy import special as sps

# Define Constants #
SPEED_OF_LIGHT = 299792  # km/s
FWHM_COEFF = 2.*math.sqrt(2. * math.log(2.))


def calculate_vel(ind, lines, fit_sol, line_dict):
    """
    Calculate velocity.

    .. math::
        v = SPEED_OF_LIGHT*(l\_calc - l\_rest)/l\_calc

    Where :math:`l\_calc = 1e7/fit\_vel` and :math:`l\_rest` is the rest wavelength of the line.
    :math:`fit\_vel` is the shifted position of the line in units of cm-1.

    Args:
        ind: Index of line in lines
        lines:Lines to be fit (e.x. ['Halpha'])
        fit_sol: Solution from fitting algorithm
        line_dict: Dictionary of Line Names and their wavelengths in nm
    Return:
        Velocity of the Halpha line in units of km/s
    """
    line_name = lines[ind]
    l_calc = 1e7 / fit_sol[3*ind+1]
    l_shift = (l_calc - line_dict[line_name]) /  line_dict[line_name]
    v = SPEED_OF_LIGHT * l_shift
    return v


def calculate_vel_err(ind, lines, fit_sol, line_dict, uncertainties):
    """
    Calculate velocity error

    We simply take the difference between the velocities with and without the uncertainty.

    Args:
        ind: Index of line in lines
        lines:Lines to be fit (e.x. ['Halpha'])
        fit_sol: Solution from fitting algorithm
        line_dict: Dictionary of Line Names and their wavelengths in nm
        uncertaintes: Uncertainties from fitting algoritm
    Return:
        Velocity of the Halpha line in units of km/s
    """
    line_name = lines[ind]
    l_calc1 = 1e7 / (fit_sol[3*ind+1])
    l_calc2 = 1e7 / (fit_sol[3*ind+1] + uncertainties[3*ind+1])
    #print()
    #print(l_calc1, l_calc2)
    #print(l_calc1, l_calc2)
    l_shift1 = (l_calc1 - line_dict[line_name]) / line_dict[line_name]
    l_shift2 = (l_calc2 - line_dict[line_name]) / line_dict[line_name]
    #print(l_shift1, l_shift2)
    v1 = SPEED_OF_LIGHT * l_shift1
    v2 = SPEED_OF_LIGHT * l_shift2
    #print(v1, v2)
    return np.abs(v1 - v2)


def calculate_broad(ind, fit_sol, axis_step):
    """
    Calculate velocity dispersion

    .. math::
        \sigma = (SPEED_OF_LIGHT*fit\_\sigma * axis\_step)/(fit\_vel)

    where :math:`fit\_sigma` is the gaussian broadening parameter found in the fit, :math:`axis\_step` is defined in the HowLuciWorks section,
    and :math:`fit\_vel` is the shifted position of the line in units of cm-1.

    Args:
        ind: Index of line in lines
        fit_sol: Solution from fitting algorithm
        axis_step: Step due to correction factor (see LuciFit.calculate_correction)
    Return:
        Velocity Dispersion of the Halpha line in units of km/s
    """
    #broad = (SPEED_OF_LIGHT * fit_sol[3*ind+2] * axis_step) / fit_sol[3*ind+1]
    #return np.abs(broad)/abs(2.*np.sqrt(2. * np.log(2.)))  # Add FWHM correction
    broad = (SPEED_OF_LIGHT * fit_sol[3*ind+2]) / fit_sol[3*ind+1]
    return np.abs(broad)/FWHM_COEFF   # Add FWHM correction


def calculate_broad_err(ind, fit_sol, axis_step, uncertainties):
    """
    Calculate velocity dispersion error
    We simply take the difference between the velocity dispersions with and without the uncertainty.

    Args:
        ind: Index of line in lines
        fit_sol: Solution from fitting algorithm
        axis_step: Step due to correction factor (see LuciFit.calculate_correction)
        uncertaintes: Uncertainties from fitting algorithm
    Return:
        Velocity Dispersion of the Halpha line in units of km/s
    """
    broad1 = (SPEED_OF_LIGHT * fit_sol[3*ind+2]* axis_step) / fit_sol[3*ind+1]
    broad1 /= FWHM_COEFF  # Add FWHM correction
    broad2 = (SPEED_OF_LIGHT * (fit_sol[3*ind+2]+uncertainties[3*ind+2])* axis_step) / (fit_sol[3*ind+1]+uncertainties[3*ind+1])
    broad2 /= FWHM_COEFF  # Add FWHM correction
    return np.abs(broad1-broad2)


def calculate_flux(line_amp, line_sigma, model_type, sinc_width):
    """
    Calculate flux value given fit of line
    See HowLuciWorks for calculations


    Args:
        line_amp: Amplitude of the line (un-normalized)
        line_sigma: Sigma of the line fit
        model_type: Fitting function (i.e. 'gaussian', 'sinc', or 'sincgauss')
        sinc_width: Fixed with of the sinc function
    Return:
        Flux of the provided line in units of erg/s/cm-2
    """
    flux = 0.0  # Initialize
    if model_type == 'gaussian':
        flux =  (1.20671/FWHM_COEFF) * np.sqrt(2 * np.pi) * line_amp * line_sigma
    elif model_type == 'sinc':
        flux =  np.pi *np.sqrt(np.pi) * line_amp * line_sigma
    elif model_type == 'sincgauss':
        flux = (1.20671/(np.pi*FWHM_COEFF)) * line_amp * ((np.sqrt(2*np.pi)*line_sigma) / (sps.erf(line_sigma / (np.sqrt(2) * sinc_width))))
    else:
        print("ERROR: INCORRECT FIT FUNCTION")
    return flux


def calculate_flux_err(ind, fit_sol, uncertainties, model_type, sinc_width):
    """
    Calculate flux error

    We simply take the difference between the fluxes with and without the uncertainty.


    Args:
        ind: Index of line in lines
        fit_sol: Solution from fitting algorithm
        uncertaintes: Uncertainties from fitting algoritm
        model_type: Fitting function (i.e. 'gaussian', 'sinc', or 'sincgauss')
        sinc_width: Fixed with of the sinc function

    Return:
        Error of the provided line in units of ergs/s/cm-2
    """

    p0 = fit_sol[3*ind]
    p2 = fit_sol[3*ind + 2]
    p0_err = uncertainties[3*ind]
    p2_err = uncertainties[3*ind + 2]


    if model_type == 'gaussian':
        flux_err = np.sqrt(2*np.pi) * calculate_flux(p0 , p2, model_type, sinc_width) * \
                   np.sqrt( (p0_err / p0 )**2 + (p2_err / p2)**2  )

    elif model_type == 'sinc':
        flux_err = np.sqrt(np.pi) * calculate_flux(p0 , p2, model_type, sinc_width) * \
                   np.sqrt( (p0_err / p0 )**2 + (p2_err / p2)**2  )

    elif model_type == 'sincgauss':
        erf_func = sps.erf(p2 / (np.sqrt(2)*sinc_width)) #Shortcut for the error function
        flux_err = np.sqrt(2*np.pi) * np.sqrt(   (p2*p0_err / erf_func)**2 + \
                   (  p0*p2_err *  (erf_func - (np.sqrt(2)*p2*np.exp(-(p2/(np.sqrt(2)*sinc_width))**2)/np.sqrt(np.pi))) / erf_func**2  )**2  )

    else:
        print('The fit function you have entered, %s, does not exist!'%model_type)
        print('The program is terminating!')
        exit()

    return flux_err
