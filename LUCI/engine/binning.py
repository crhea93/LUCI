"""Spatial binning of cubes and masks."""

import numpy as np


def bin_cube_function(cube_final, header, binning, x_min, x_max, y_min, y_max):
    """
    Function to bin cube into bin x bin sub cubes

    Args:
        binning: Size of binning (equal in x and y direction)
        x_min: Lower bound in x
        x_max: Upper bound in x
        y_min: Lower bound in y
        y_max: Upper bound in y
    Return:
        Binned cubed called self.cube_binned and new spatial limits
    """
    x_shape_new = int((x_max - x_min) / binning)
    y_shape_new = int((y_max - y_min) / binning)
    binned_cube = np.zeros((x_shape_new, y_shape_new, cube_final.shape[2]))
    for i in range(x_shape_new):
        for j in range(y_shape_new):
            summed_spec = cube_final[
                x_min + int(i * binning) : x_min + int((i + 1) * binning),
                y_min + int(j * binning) : y_min + int((j + 1) * binning),
                :,
            ]
            summed_spec = np.nansum(summed_spec, axis=0)
            summed_spec = np.nansum(summed_spec, axis=0)
            binned_cube[i, j] = summed_spec[:]
    header_binned = header
    header_binned["CRPIX1"] = (header_binned["CRPIX1"] - x_min - 0.5) / binning + 0.5
    header_binned["CRPIX2"] = (header_binned["CRPIX2"] - y_min - 0.5) / binning + 0.5
    # header_binned['CDELT1'] = header_binned['CDELT1'] * binning
    # header_binned['CDELT2'] = header_binned['CDELT2'] * binning
    try:
        header_binned["PC1_1"] = header_binned["PC1_1"] * binning
        header_binned["PC1_2"] = header_binned["PC1_2"] * binning
        header_binned["PC2_1"] = header_binned["PC2_1"] * binning
        header_binned["PC2_2"] = header_binned["PC2_2"] * binning
    except KeyError:
        pass  # Header doesn't contain PC info
    cube_binned = binned_cube  # / (binning ** 2)
    return header_binned, cube_binned


def bin_mask(mask, binning, x_min, x_max, y_min, y_max):
    """
    Function to bin mask. This is effectively the same as `self.bin_cube` with
    the exception that it is for the mask only. For now, this function is only
    triggered when the mask is in the form of a '.npy' file. Region files
    passed as '.reg' files do not need to be additionally masked since they use
    the binned header information.

    Args:
        mask: Mask to be binned
        binning: Size of binning (equal in x and y direction)
        x_min: Lower bound in x
        x_max: Upper bound in x
        y_min: Lower bound in y
        y_max: Upper bound in y
    Return:
        Binned cubed called self.cube_binned and new spatial limits
    """
    x_shape_new = int((x_max - x_min) / binning)
    y_shape_new = int((y_max - y_min) / binning)
    # Boolean: a bin is selected if any pixel in it is. The old code also divided
    # by binning**2, turning True into 0.25 and the result into floats (B8).
    binned_mask = np.zeros((x_shape_new, y_shape_new), dtype=bool)
    for i in range(x_shape_new):
        for j in range(y_shape_new):
            block = mask[
                x_min + int(i * binning) : x_min + int((i + 1) * binning),
                y_min + int(j * binning) : y_min + int((j + 1) * binning),
            ]
            binned_mask[i, j] = bool(block.any())
    return binned_mask
