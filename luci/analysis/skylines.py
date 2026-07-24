"""Sky-line based velocity calibration."""

import numpy as np
import pandas
from astropy.io import fits
from tqdm import tqdm

from luci.fitting.spectrum_fitter import SpectrumFitter as Fit


def skyline_calibration(cube, Luci_path, n_grid, bin_size=30):
    """
    Compute skyline calibration by fitting the 6498.729 Angstrom line. Flexures
    of the telescope lead to minor offset that can be measured by high resolution
    spectra (R~5000). This function divides the FOV into a grid of NxN spaxel regions
    of 10x10 pixels to increase the signal. The function will output a map of the
    velocity offset. The initial velocity guess is set to 80 km/s. Additionally,
    we fit with a simple sinc function.

    Args:
        n_grid: NxN grid (int)
        Luci_path: Full path to LUCI (str)
        bin_size: Size of grouping used for each region (optional int; default=30)

    Return:
        Velocity offset map
    """
    velocity = None
    fit_vector = None
    sky = None
    # Read in sky lines
    sky_lines_df = pandas.read_csv(Luci_path + "/Data/sky_lines.dat", skiprows=2)
    sky_lines = sky_lines_df["Wavelength"]  # Get wavelengths
    sky_lines = [sky_line / 10 for sky_line in sky_lines]  # Convert from angstroms to nanometers
    sky_lines_scale = [sky_line for sky_line in sky_lines_df["Strength"]]  # Get the relative strengths
    # Create skyline dictionary
    sky_line_dict = {}  # {OH_num: wavelength in nm}
    for line_ct, line_wvl in enumerate(sky_lines):
        sky_line_dict["OH_%i" % line_ct] = line_wvl
    # Calculate grid
    x_min = 200
    x_max = cube.cube_final.shape[0] - x_min
    x_step = int(
        (x_max - x_min) / n_grid
    )  # Calculate step size based on min and max values and the number of grid points
    y_min = 200
    y_max = cube.cube_final.shape[1] - y_min
    y_step = int(
        (y_max - y_min) / n_grid
    )  # Calculate step size based on min and max values and the number of grid points
    vel_grid = np.zeros((n_grid, n_grid))  # Initialize velocity grid
    vel_uncertainty_grid = np.zeros((n_grid, n_grid))

    for x_grid in tqdm(range(n_grid)):  # Step through x steps
        for y_grid in range(n_grid):  # Step through y steps
            # Collect spectrum in 10x10 region
            x_center = x_min + int(0.5 * (x_step) * (x_grid + 1))
            y_center = y_min + int(0.5 * (y_step) * (y_grid + 1))
            integrated_spectrum = np.zeros_like(cube.cube_final[x_center, y_center, :])  # Initialize as zeros
            for i in range(bin_size):  # Take bin_size x bin_size bins
                for j in range(bin_size):
                    integrated_spectrum += cube.cube_final[x_center + i, y_center + i, :]
            # Collapse to single spectrum
            good_sky_inds = ~np.isnan(integrated_spectrum)  # Clean up spectrum
            sky = integrated_spectrum[good_sky_inds]
            axis = cube.spectrum_axis[good_sky_inds]
            # Call fit!
            fit = Fit(
                sky,
                axis,
                cube.wavenumbers_syn,
                "sinc",
                ["OH_%i" % num for num in sky_lines],
                len(sky_lines) * [1],
                len(sky_lines) * [1],
                trans_filter=cube.transmission_interpolated,
                theta=cube.interferometer_theta[x_center, y_center],
                delta_x=cube.hdr_dict["STEP"],
                n_steps=cube.step_nb,
                zpd_index=cube.zpd_index,
                uncertainty_bool=True,
                filter=cube.hdr_dict["FILTER"],
                ML_bool=cube.ML_bool,
                bayes_bool=False,
                bayes_method="emcee",
                sky_lines=sky_line_dict,
                sky_lines_scale=sky_lines_scale,
                resolution=cube.resolution,
                Luci_path=cube.Luci_path,
            )

            velocity, velocity_error, fit_vector = fit.fit(sky_line=True)
            vel_grid[x_grid, y_grid] = float(velocity)
            vel_uncertainty_grid[x_grid, y_grid] = float(velocity_error)
    # Now that we have the grid, we need to reproject it onto the original pixel grid
    vel_grid_final = np.zeros((x_max, y_max))
    for x_grid in range(n_grid):  # Step through x steps
        for y_grid in range(n_grid):  # Step through y steps
            # Collect spectrum in 10x10 region
            x_center = x_min + int(0.5 * (x_step) * (x_grid + 1))
            y_center = y_min + int(0.5 * (y_step) * (y_grid + 1))
            vel_grid_final[x_center - x_step : x_center + x_step, y_center - y_step : y_center + y_step] = vel_grid[
                x_grid, y_grid
            ]
    fits.writeto(cube.output_dir + "/velocity_correction.fits", vel_grid, cube.header, overwrite=True)
    return velocity, fit_vector, sky, vel_grid, vel_uncertainty_grid, cube.spectrum_axis
