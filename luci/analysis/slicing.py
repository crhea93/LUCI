"""Extract and write narrow-band slices around fitted lines."""

import os

import numpy as np
from astropy.io import fits
from tqdm import tqdm

from luci.log import get_logger

logger = get_logger(__name__)


def slicing(cube, lines, bkg=None):
    """
    Slices the cube along the spectral axis around the specified emission line wavelengths. The input velocity dispersion
    serves as a rough estimate of the width of the spectral lines to make sure the slices cover a wide enough range on each side
    of the emission lines. The slices are saved in a new folder for each input emission line.

    We currently have the following lines available:

    'Halpha': 656.280,
    'NII6583': 658.341,
    'NII6548': 654.803,
    'SII6716': 671.647,
    'SII6731': 673.085,
    'OII3726': 372.603,
    'OII3729': 372.882,
    'OIII4959': 495.891,
    'OIII5007': 500.684,
    'Hbeta': 486.133,
    'OH': 649.873,
    'HalphaC4': 807.88068,
    'NII6583C4': 810.417771,
    'NII6548C4': 806.062493,
    'OIII5007C2': 616.342,
    'OIII4959C2': 610.441821,
    'HbetaC2': 598.429723,
    'OII3729C1': 459.017742,
    'OII3726C1': 458.674293,
    'FeXIV5303': 530.286,
    'NI5200': 520.026,
    'FeVII5158': 515.89,
    'HeII5411': 541.152

    Args:
        lines: Lines to extract (e.x. ['Halpha', 'NII6583'])
        bkg: Background Spectrum (default None)
    """
    line_dict = {
        "Halpha": 656.280,
        "NII6583": 658.341,
        "NII6548": 654.803,
        "SII6716": 671.647,
        "SII6731": 673.085,
        "OII3726": 372.603,
        "OII3729": 372.882,
        "OIII4959": 495.891,
        "OIII5007": 500.684,
        "Hbeta": 486.133,
        "OH": 649.873,
        "HalphaC4": 807.88068,
        "NII6583C4": 810.417771,
        "NII6548C4": 806.062493,
        "OIII5007C2": 616.342,
        "OIII4959C2": 610.441821,
        "HbetaC2": 598.429723,
        "OII3729C1": 459.017742,
        "OII3726C1": 458.674293,
        "FeXIV5303": 530.286,
        "NI5200": 520.026,
        "FeVII5158": 515.89,
        "HeII5411": 541.152,
        "FeXI6624": 662.43,
        "NiXV6703": 670.332,
    }

    filter_line = {
        "SN1": ["OII3726", "OII3729"],
        "SN2": ["Hbeta", "OIII4959", "OIII5007"],
        "SN3": ["Halpha", "NII6583", "NII6548", "SII6716", "SII6731"],
        "SN4": ["Halpha", "NII6583", "NII6548"],
        "C3": ["FeXIV5303", "NI5200", "FeVII5158", "HeII5411", "FeXI6624", "NiXV6703"],
    }

    spectral_axis = 1e7 / cube.spectrum_axis  # Convert wavenumber in cm-1 to nm
    # If a background is provided then we need to update the cube to be background subtracted
    if bkg != None:
        # Reshape the background to (1, 1, lambda) so it can be broadcasted
        bkg_reshaped = bkg.reshape(1, 1, -1)
        # Subtract the vector from the matrix using broadcasting
        cube_to_use = cube.cube_final - bkg_reshaped
    else:
        cube_to_use = cube.cube_final  # Use non-bkg subtracted cube
    # Make sure the specified lines are part of the filter of the cube
    if all(line in (filter_line[cube.filter]) for line in lines):
        wavelengths = np.array(list(map(line_dict.get, lines)))  # Get the rest wavelength values of the specified lines
        wavelength_redshift = (1 + cube.redshift) * wavelengths  # Calculate the wavelength in the redshifted frame

        # Match the minimum and maximum wavelengths of each group of slices around the lines with the spectral axis of the cube
        def find_nearest(spectral_axis, line_lambda):
            indices = np.abs(np.subtract.outer(spectral_axis, line_lambda)).argmin(0)
            return indices

        slice_sum = np.zeros_like(cube.cube_final[:, :, 0])
        # Loop for every emission line
        for i in range(len(wavelength_redshift)):
            delta_lambda = (
                200 * wavelength_redshift[i] / 299792
            )  # In units of nm. The 200km/s is a reasonable maximum value for the velocity dispersion of galaxies
            lambda_range = np.array(
                [wavelength_redshift[i] - 10 * delta_lambda, wavelength_redshift[i] + 10 * delta_lambda]
            )
            idx_axis = find_nearest(spectral_axis, lambda_range)[::-1]

            directory = cube.output_dir + "/Slice_{}".format(lines[i])

            if not os.path.exists(directory):
                os.makedirs(directory)

            # Loop for every slice
            for j in tqdm(range(idx_axis[0], idx_axis[1] + 1)):
                cube_slice = cube_to_use[:, :, j]
                slice_sum += cube_slice
                hdu = fits.PrimaryHDU(cube_slice, header=cube.header)
                hdu.writeto(directory + "/slice_{}.fits".format(j - (idx_axis[0] - 1)), overwrite=True)
            hdu = fits.PrimaryHDU(slice_sum, header=cube.header)
            hdu.writeto(directory + "/slice_sum.fits", overwrite=True)
            logger.info("")
            logger.info("#######################################################################")
            logger.info(
                "Wavelength of the {} line in the redshifted frame: {} nm".format(
                    lines[i], np.round(wavelength_redshift[i], 2)
                )
            )
            logger.info(
                "Wavelength of the last slice: " + str(np.round(spectral_axis[idx_axis[1]], 2)) + " nm"
            )  # .format() was not giving the right number of decimals..
            logger.info(
                "Wavelength of the first slice: " + str(np.round(spectral_axis[idx_axis[0]], 2)) + " nm"
            )  # so not the cleanest way to do it but still it works
            logger.info("")

    else:
        logger.info("The specified lines are not in the wavelength range covered by the filter of this cube")
