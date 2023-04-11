"""
A hodge-podge of convenience functions for luci

"""
import numpy as np
from LUCI.LuciFunctions import Gaussian, Sinc, SincGauss
import pyregion


def get_individual_components(fit_dictionary):
    """
    This function will take a fit solution and calculate the fit vectors for each component in the fit. This is useful
    if you have a multiple component fit and wish to obtain (i.e. visualize) the various components. The continuum is added as
    a separate fit_vector and can be queried with fit_vectors['continuum']

    Args:
        fit_dictionary: Fit dictionary returned by a fitting function. This contains the amplitude, velocity, and broadening
            of the lines of interest.
    Return:
        Dictionary of components with fit vector {'Halpha_1': fit_vector}
    """
    # Pull necessary information from the fit dictionary
    fit_solution = fit_dictionary['fit_sol']  # Solutions of the following form [line_1_amp, line_1_vel, line_1_broad, line_2_amp, ..., continuum]
    fit_axis = fit_dictionary['fit_axis']
    fit_function = fit_dictionary['fit_function']  # Function used in fitting the lines
    fit_lines = fit_dictionary['fit_lines']  # Dictionary of lines fit in order of solutions
    sinc_width = fit_dictionary['sinc_width']
    # Initialize output
    fit_vectors = {}  # {line_name: fit_vector}
    # Step through each line in fit lines and create fit vector
    for ct, fit_line in enumerate(fit_lines):
        if fit_line in fit_vectors.keys():  # Line has already been in fit_vectors so we need to append a number to its name -- # TODO: Only works for 2 components -- must generalize
            name = fit_line+'_2'
        else:
            name = fit_line
        fit_solution_ = fit_solution[3*ct : 3*(ct+1)]  # Get correct solutions
        # Based on fit function create appropriate fit vector
        if fit_function == 'gaussian':
            fit_vector = Gaussian().plot(fit_axis, fit_solution_[:-1])
        elif fit_function == 'sinc':
            fit_vector = Sinc().plot(fit_axis, fit_solution_[:-1], sinc_width)
        elif fit_function == 'sincgauss':
            fit_vector = SincGauss().plot(fit_axis, fit_solution_[:-1], sinc_width)
        fit_vectors[name] = fit_vector
    # Add continuum
    fit_vectors['continuum'] = fit_solution[:-1]*np.ones(len(fit_axis))
    return fit_vectors

def reg_to_mask(region, header):
    """
    Utility function to convert a .reg file to a mask the fitting algorithm can use
    """
    shape = (2064, 2048)  # (self.header["NAXIS1"], self.header["NAXIS2"])  # Get the shape
    r = pyregion.open(region).as_imagecoord(header)  # Obtain pyregion region
    mask = r.get_mask(shape=shape).T  # Calculate mask from pyregion region
    return mask