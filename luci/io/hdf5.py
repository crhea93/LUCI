"""Layout of the SITELLE HDF5 cube files."""

import numpy as np


def get_quadrant_dims(quad_number, quad_nb, dimx, dimy):
    """
    Calculate the x and y limits of a given quadrant in the HDF5 file. The
    data cube is saved in 9 individual arrays in the original HDF5 cube. This
    function gets the bouunds for each quadrant.

    Args:
        quad_number: Current Quadrant Number
        quand_nb: Total number of quadrants
        dimx: Number of x dimensions
        dimy: Number of y dimensions
    Return:
        x_min, x_max, y_min, y_max: Spatial bounds of quadrant
    """
    div_nb = int(np.sqrt(quad_nb))
    if (quad_number < 0) or (quad_number > quad_nb - 1):
        raise Exception("quad_number out of bounds [0," + str(quad_nb - 1) + "]")
    index_x = quad_number % div_nb
    index_y = (quad_number - index_x) / div_nb
    x_min = index_x * np.floor(dimx / div_nb)
    if index_x != div_nb - 1:
        x_max = (index_x + 1) * np.floor(dimx / div_nb)
    else:
        x_max = dimx
    y_min = index_y * np.floor(dimy / div_nb)
    if index_y != div_nb - 1:
        y_max = (index_y + 1) * np.floor(dimy / div_nb)
    else:
        y_max = dimy
    return int(x_min), int(x_max), int(y_min), int(y_max)
