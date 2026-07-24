import glob
import logging
import multiprocessing as mp
import os
import pickle
import time
import warnings

import astropy.stats as astrostats
import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.nddata import Cutout2D
from astropy.time import Time
from astropy.wcs import WCS
from joblib import Parallel, delayed
from sklearn import decomposition
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from luci.analysis.components import calculate_components_in_region_function, create_component_map_function
from luci.analysis.skylines import skyline_calibration as _skyline_calibration
from luci.analysis.slicing import slicing as _slicing
from luci.analysis.snr import create_snr_map as _create_snr_map
from luci.analysis.snr import detection_map as _detection_map
from luci.analysis.wvt import *
from luci.analysis.wvt import create_wvt as _create_wvt
from luci.analysis.wvt import fit_wvt as _fit_wvt
from luci.analysis.wvt import wvt_fit_region as _wvt_fit_region
from luci.background.detection import find_background_pixels
from luci.background.pca import create_background_subspace as _create_background_subspace
from luci.background.subtraction import pca_background, subtract_pca, subtract_standard
from luci.engine import FitMaps, deep_image_cutout, resolve_initial_values, run_fit
from luci.engine.selection import reg_to_mask, resolve_mask
from luci.fitting.spectrum_fitter import SpectrumFitter as Fit
from luci.instrument.flux import flux_calibration_vector, is_flux_calibrated
from luci.io.assets import resolve_luci_path
from luci.log import get_logger
from luci.LuciUtility import (
    bin_cube_function,
    bin_mask,
    get_interferometer_angles,
    get_quadrant_dims,
    read_in_reference_spectrum,
    read_in_transmission,
    save_fits,
    spectrum_axis_func,
    update_header,
)
from luci.viz.visualize import visualize as LUCIvisualize

logger = get_logger(__name__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)


class SitelleCube:
    """
    A SITELLE data cube: the data, its geometry, and the fits run over it.

    This is the primary class for the general purpose line fitting code LUCI. This contains
    all io/administrative functionality. The fitting functionality can be found in the
    Fit class (Lucifit.py).
    """

    def __init__(
        self,
        Luci_path=None,
        cube_path=None,
        output_dir=None,
        object_name=None,
        redshift=0.0,
        resolution=5000,
        ML_bool=True,
        mdn=False,
        flux_calibration=True,
    ):
        """
        Initialize our Luci class -- this acts similar to the SpectralCube class
        of astropy or spectral-cube.

        Args:
            Luci_path: Path to the LUCI checkout holding ML/ and Data/. Optional -- if omitted it
                is taken from $LUCI_DATA_DIR, or from the installed package's location.
            cube_path: Full path to hdf5 cube with the hdf5 extension (e.x. '/user/home/M87.hdf5'; No trailing "/")
            output_dir: Full path to output directory
            object_name: Name of the object to fit. This is used for naming purposes. (e.x. 'M87')
            redshift: Redshift to the object. (e.x. 0.00428)
            resolution: Resolution requested of machine learning algorithm reference spectrum
            ML_bool: Boolean for applying machine learning; default=True
            mdn: Boolean for using the Mixed Density Network models; If true, then we use the posterior distributions calculated by our network as our priors for bayesian fits
            flux_calibration: Convert the cube from counts to erg/cm2/s/A when its header says
                it is not already calibrated (default True). ORB's level-3 cubes are stored in
                counts with the calibration held in the `flambda` header vector; without this
                every flux, amplitude and continuum map is in instrumental units despite being
                labelled erg/cm2/s/A. Set False to fit the raw counts.
        """
        self.header_binned = None
        self.Luci_path = resolve_luci_path(Luci_path)
        self.cube_path = cube_path
        self.output_dir = output_dir + "/Luci_outputs"
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.object_name = object_name
        self.redshift = redshift
        self.resolution = resolution
        self.mdn = mdn
        self.quad_nb = 0  # Number of quadrants in Hdf5
        self.dimx = 0  # X dimension of cube
        self.dimy = 0  # Y dimension of cube
        self.dimz = 0  # Z dimension of cube
        self.cube_final = None  # Complete data cube
        self.cube_binned = None  # Binned data cube
        self.header = None
        self.deep_image = None
        self.spectrum_axis = None
        self.spectrum_axis_unshifted = None  # Spectrum axis without redshift change
        self.wavenumbers_syn = None
        self.wavenumbers_syn_full = None  # Unclipped reference spectrum
        self.hdr_dict = None
        self.interferometer_theta = None
        self.transmission_interpolated = None
        self.read_in_cube()
        self.flux_calibrated = False  # Whether cube_final is in erg/cm2/s/A
        if flux_calibration:
            self.apply_flux_calibration()
        elif not is_flux_calibrated(self.hdr_dict):
            logger.warning(
                "Cube is not flux calibrated and flux_calibration=False, so every flux, "
                "amplitude and continuum map will be in counts, not erg/cm2/s/A."
            )
        self.step_nb = self.hdr_dict["STEPNB"]
        self.zpd_index = self.hdr_dict["ZPDINDEX"]
        self.filter = self.hdr_dict["FILTER"]
        self.spectrum_axis, self.spectrum_axis_unshifted = spectrum_axis_func(self.hdr_dict, self.redshift)
        if self.filter == "C4" or self.filter == "C2" or self.filter == "C1":
            self.spectrum_axis = self.spectrum_axis_unshifted  # LYA mod
        self.ref_spec = self.Luci_path + "ML/Reference-Spectrum-R%i-%s.fits" % (resolution, self.filter)
        self.wavenumbers_syn, self.wavenumbers_syn_full = read_in_reference_spectrum(self.ref_spec, self.hdr_dict)
        self.ML_bool = ML_bool

        self.transmission_interpolated = read_in_transmission(
            self.Luci_path, self.hdr_dict, self.spectrum_axis_unshifted
        )

    def read_in_cube(self):
        """
        Function to read the hdf5 data into a 3d numpy array (data cube). We also
        translate the header to standard wcs format by calling the update_header function.
        Note that the data are saved in several quadrants which is why we have to loop
        through them and place all the spectra in a single cube.
        """
        logger.info("Reading in data...")
        file = h5py.File(self.cube_path + ".hdf5", "r")  # Read in file
        # file = ht.load(self.cube_path + '.hdf5')
        # print(file.keys())
        try:
            self.quad_nb = file.attrs["quad_nb"]  # Get the number of quadrants
            self.dimx = file.attrs["dimx"]  # Get the dimensions in x
            self.dimy = file.attrs["dimy"]  # Get the dimensions in y
            self.dimz = file.attrs["dimz"]  # Get the dimensions in z (spectral axis)
            self.cube_final = np.zeros((self.dimx, self.dimy, self.dimz))  # Complete data cube
            for iquad in tqdm(range(self.quad_nb)):
                xmin, xmax, ymin, ymax = get_quadrant_dims(iquad, self.quad_nb, self.dimx, self.dimy)
                iquad_data = file["quad00%i" % iquad]["data"][:]  # Save data to intermediate array
                iquad_data[(np.isfinite(iquad_data) == False)] = 1e-22  # Set infinite values to 1-e22
                iquad_data[(iquad_data < -1e-16)] = 1e-22  # Set high negative flux values to 1e-22
                iquad_data[(iquad_data > 1e-9)] = 1e-22  # Set unrealistically high positive flux values to 1e-22
                self.cube_final[xmin:xmax, ymin:ymax, :] = iquad_data  # Save to correct location in main cube
                # iquad_data = None
        except KeyError:
            self.cube_final = np.real(file["data"])
        self.cube_final = self.cube_final  # .transpose(1, 0, 2)
        """folder = './joblib_memmap'
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass"""

        # data_filename_memmap = os.path.join(folder, 'data_memmap')
        # dump(self.cube_final, data_filename_memmap)
        # self.cube_final = load(data_filename_memmap, mmap_mode='readwrite')
        self.header, self.hdr_dict = update_header(file)
        self.interferometer_theta = get_interferometer_angles(file, self.hdr_dict)
        # file.close()

    def apply_flux_calibration(self):
        """
        Scale the cube from counts to erg/cm2/s/A if its header says it is not already calibrated.

        ORB level-3 cubes hold counts and carry the conversion in the `flambda` header vector
        (see :mod:`luci.instrument.flux`). LUCI's flux formulas are amplitude x width, so they
        inherit whatever units the cube is in -- calibrating once, here, is what makes the
        erg/cm2/s/A on every downstream axis label true. Cubes ORBS already calibrated, and
        cubes with no usable `flambda`, are left alone.

        Sets `self.flux_calibrated` and returns it.
        """
        calibration = flux_calibration_vector(self.hdr_dict, self.cube_final.shape[-1])
        if calibration is None:
            self.flux_calibrated = is_flux_calibrated(self.hdr_dict)
            return self.flux_calibrated
        logger.info(
            "Applying flux calibration: counts -> erg/cm2/s/A (median factor %.4g per count)",
            np.median(calibration),
        )
        # A contiguous float32 copy: the cube may be a strided `np.real(...)` view of a complex
        # dataset, which both doubles the memory and makes this multiply crawl.
        self.cube_final = np.ascontiguousarray(self.cube_final, dtype=np.float32)
        self.cube_final *= calibration.astype(np.float32)
        self.flux_calibrated = True
        return True

    def create_deep_image(self, output_name=None, binning=None):
        """
        Create deep image fits file of the cube. This takes the cube and sums
        the spectral axis. Then the deep image is saved as a fits file with the following
        naming convention: output_dir+'/'+object_name+'_deep.fits'. We also allow for
        the binning of the deep image -- this is used primarily for astrometry purposes.

        Args:
            output_name: Full path to output (optional)
            binning: Binning number (optional integer; default=None)
        """
        # hdu = fits.PrimaryHDU()
        # We are going to break up this calculation into chunks so that  we can have a progress bar
        # self.deep_image = np.sum(self.cube_final, axis=2).T

        hdf5_file = h5py.File(self.cube_path + ".hdf5", "r")  # Open and read hdf5 file

        if "deep_frame" in hdf5_file:  # A deep image already exists
            logger.info("Existing deep frame extracted from hdf5 file.")
            self.deep_image = hdf5_file["deep_frame"][:]
            if self.dimz != 0:  # had to put this because of new version of cubes
                self.deep_image *= self.dimz
        else:  # Create new deep image
            logger.info("New deep frame created from data.")
            self.deep_image = np.zeros((self.cube_final.shape[0], self.cube_final.shape[1]))
            # Slabs drive the progress bar. Deriving the count from the step size
            # covers the whole cube; ten fixed slabs dropped the remainder (B7).
            n_rows = self.cube_final.shape[0]
            step_size = max(1, int(n_rows / 10))
            for start in tqdm(range(0, n_rows, step_size)):
                stop = min(start + step_size, n_rows)
                self.deep_image[start:stop] = np.nansum(self.cube_final[start:stop], axis=2)
        self.deep_image = self.deep_image.T
        header_to_use = self.header  # Set header to be used
        # Bin data
        if binning != None and binning != 1:
            # Get cube size
            x_min = 0
            x_max = self.cube_final.shape[0]
            y_min = 0
            y_max = self.cube_final.shape[1]
            # Get new bin shape
            x_shape_new = int((x_max - x_min) / binning)
            y_shape_new = int((y_max - y_min) / binning)
            # Set to zero
            binned_deep = np.zeros((x_shape_new, y_shape_new))
            for i in range(x_shape_new):
                for j in range(y_shape_new):
                    # Bin
                    summed_deep = self.deep_image[
                        x_min + int(i * binning) : x_min + int((i + 1) * binning),
                        y_min + int(j * binning) : y_min + int((j + 1) * binning),
                    ]
                    summed_deep = np.nansum(summed_deep, axis=0)  # Sum along x
                    summed_deep = np.nansum(summed_deep, axis=0)  # Sum along y
                    binned_deep[i, j] = summed_deep  # Set to global
            # Update header information
            header_binned = self.header
            header_binned["CRPIX1"] = header_binned["CRPIX1"] / binning
            header_binned["CRPIX2"] = header_binned["CRPIX2"] / binning
            # header_binned['CDELT1'] = header_binned['CDELT1'] * binning
            # header_binned['CDELT2'] = header_binned['CDELT2'] * binning
            header_binned["PC1_1"] = header_binned["PC1_1"] * binning
            header_binned["PC1_2"] = header_binned["PC1_2"] * binning
            header_binned["PC2_1"] = header_binned["PC2_1"] * binning
            header_binned["PC2_2"] = header_binned["PC2_2"] * binning
            header_to_use = header_binned
            self.deep_image = binned_deep / (binning**2)
        if output_name == None:
            output_name = self.output_dir + "/" + self.object_name + "_deep.fits"
        fits.writeto(output_name, self.deep_image, header_to_use, overwrite=True)
        hdf5_file.close()

    def visualize(self):
        """
        Wrapper function for LUCI.LuciVisualize()
        """
        if self.deep_image is None:
            try:
                deep_image = fits.open("Luci_outputs/%s_deep.fits" % self.object_name)[0].data
            except (OSError, IndexError):
                self.create_deep_image()
                deep_image = fits.open("Luci_outputs/%s_deep.fits" % self.object_name)[0].data
        else:
            deep_image = self.deep_image
        LUCIvisualize(deep_image, self.spectrum_axis, self.cube_final, self.hdr_dict)

    def fit_entire_cube(
        self,
        lines,
        fit_function,
        vel_rel,
        sigma_rel,
        bkg=None,
        bkgType=None,
        binning=None,
        bayes_bool=False,
        output_name=None,
        uncertainty_bool=False,
        n_threads=1,
    ):
        """
        Fit the entire cube (all spatial dimensions)

        Args:
            lines: Lines to fit (e.x. ['Halpha', 'NII6583'])
            fit_function: Fitting function to use (e.x. 'gaussian')
            vel_rel: Constraints on Velocity/Position (must be list; e.x. [1, 2, 1])
            sigma_rel: Constraints on sigma (must be list; e.x. [1, 2, 1])
            bkg: Background Spectrum (1D numpy array; default None)
            binning:  Value by which to bin (default None)
            bayes_bool: Boolean to determine whether or not to run Bayesian analysis (default False)
            output_name: User defined output path/name (default None)
            uncertainty_bool: Boolean to determine whether or not to run the uncertainty analysis (default False)
            n_threads: Number of threads to be passed to joblib for parallelization (default = 1)

        Return:
            Velocity and Broadening arrays (2d). Also return amplitudes array (3D).
        """
        x_min = 0
        x_max = self.cube_final.shape[0]
        y_min = 0
        y_max = self.cube_final.shape[1]
        return self.fit_cube(
            lines,
            fit_function,
            vel_rel,
            sigma_rel,
            x_min,
            x_max,
            y_min,
            y_max,
            bkg=bkg,
            bkgType=bkgType,
            binning=binning,
            bayes_bool=bayes_bool,
            output_name=output_name,
            uncertainty_bool=uncertainty_bool,
            n_threads=n_threads,
        )

    # @jit(nopython=False, parallel=True, nogil=True)
    @staticmethod
    def fit_calc(
        i,
        x_min,
        x_max,
        y_min,
        fit_function,
        lines,
        vel_rel,
        sigma_rel,
        cube_slice,
        spectrum_axis,
        wavenumbers_syn,
        transmission_interpolated,
        interferometer_theta,
        hdr_dict,
        step_nb,
        zpd_index,
        mdn,
        mask=None,
        ML_bool=True,
        bayes_bool=False,
        bayes_method="emcee",
        absorp=None,
        uncertainty_bool=False,
        nii_cons=False,
        bkg=None,
        bkgType=None,
        binning=None,
        spec_min=None,
        spec_max=None,
        initial_values=[False],
        obj_redshift=0.0,
        n_stoch=1,
        resolution=1000,
        Luci_path=None,
        pca_coefficient_array=None,
        pca_vectors=None,
        pca_mean=None,
    ):
        """
        Function for calling fit for a given y coordinate.

        Args:
            i: Y coordinate step
            lines: Lines to fit (e.x. ['Halpha', 'NII6583'])
            fit_function: Fitting function to use (e.x. 'gaussian')
            vel_rel: Constraints on Velocity/Position (must be list; e.x. [1, 2, 1])
            sigma_rel: Constraints on sigma (must be list; e.x. [1, 2, 1])
            x_min: Lower bound in x
            x_max: Upper bound in x
            y_min: Lower bound in y
            bkg: Background Spectrum (1D numpy array; default None)
            bkgType: default None
            binning:  Value by which to bin (default None)
            ML_bool: Boolean to determione whether or not we use ML priors
            bayes_bool: Boolean to determine whether or not to run Bayesian analysis (default False)
            bayes_method: Bayesian Inference method. Options are '[emcee', 'dynesty'] (default 'emcee')
            uncertainty_bool: Boolean to determine whether or not to run the uncertainty analysis (default False)
            nii_cons: Boolean to turn on or off NII doublet ratio constraint (default True)
            initial_values: List of files containing initial conditions (default False)
            spec_min: Minimum value of the spectrum to be considered in the fit (we find the closest value)
            spec_max: Maximum value of the spectrum to be considered in the fit
            obj_redshift: Redshift of object to fit relative to cube's redshift. This is useful for fitting high redshift objects
            n_stoch: The number of stochastic runs -- set to 50 for fitting double components (default 1)
            pca_coefficient_array: Array of PCA Coefficients (default None)
            pca_vectors: Vectors corresponding to principal components (default None)
            pca_mean: Mean vector from PCA analysis (default None)

        Return:
            all fit parameters for y-slice
        """
        # B26: a caller that passes bkg= but not bkgType= plainly means "subtract
        # this background". The old code required both and silently ignored bkg
        # otherwise -- which is what Examples/BasicExample.ipynb does.
        if bkg is not None and bkgType is None:
            bkgType = "standard"
        y_pix = y_min + i  # Step y coordinate
        # Set up all the local lists for the current y_pixel step
        ampls_local = []
        flux_local = []
        flux_errs_local = []
        vels_local = []
        broads_local = []
        vels_errs_local = []
        broads_errs_local = []
        chi2_local = []
        corr_local = []
        step_local = []
        continuum_local = []
        continuum_errs_local = []
        bool_fit = True  # Boolean to fit
        # Step through x coordinates
        for j in range(x_max - x_min):
            x_pix = x_min + j  # Set current x pixel
            if mask is not None:  # Check if there is a mask
                if mask[x_pix, y_pix]:  # Check that the mask is true
                    bool_fit = True
                else:
                    bool_fit = False
            sky = np.copy(cube_slice[x_pix, :])  # cube_binned[x_pix, y_pix, :]
            if bkgType == "standard":
                sky = subtract_standard(sky, bkg, binning)
            elif bkgType == "pca":
                if binning:  # Group the coefficients over the bin
                    coefficients = pca_coefficient_array[
                        x_min + int(j * binning) : x_min + int((j + 1) * binning),
                        y_min + int(i * binning) : y_min + int((i + 1) * binning),
                        :,
                    ]
                    coefficients = np.nansum(np.nansum(coefficients, axis=0), axis=0)
                else:
                    coefficients = pca_coefficient_array[x_pix, y_pix]
                background = pca_background(coefficients, pca_vectors, pca_mean)
                sky = subtract_pca(sky, background, spectrum_axis, hdr_dict["FILTER"])

            if absorp is not None:
                spectralcut = int(len(sky) * 0.25)
                sky = (
                    sky
                    - absorp / np.nanmedian(absorp) * np.nanmedian(sky[spectralcut:-spectralcut])
                    + np.nanmedian(sky[spectralcut:-spectralcut])
                )

            good_sky_inds = ~np.isnan(sky)  # Find all NaNs in sky spectru
            sky = sky[good_sky_inds]  # Clean up spectrum by dropping any Nan values
            axis = spectrum_axis[good_sky_inds]  # Clean up axis  accordingly
            # ...and the transmission curve, which `Fit.apply_transmission` walks by position.
            # Leaving it at full length divides each surviving channel by the transmission of a
            # different wavelength -- for an SN4 cube that is a ~140 channel shift, which
            # deforms the continuum and takes the fit with it.
            trans_filter = transmission_interpolated[good_sky_inds] if transmission_interpolated is not None else None
            if initial_values[0] is not False:  # Frozen parameter
                initial_values_to_pass = [initial_values[0][i][j], initial_values[1][i][j]]
            else:
                initial_values_to_pass = initial_values
            # Call fit!
            if len(sky) > 0 and bool_fit == True:  # Ensure that there are values in sky
                fit = Fit(
                    sky,
                    axis,
                    wavenumbers_syn,
                    fit_function,
                    lines,
                    vel_rel,
                    sigma_rel,
                    trans_filter=trans_filter,
                    theta=interferometer_theta[x_pix, y_pix],
                    delta_x=hdr_dict["STEP"],
                    n_steps=step_nb,
                    zpd_index=zpd_index,
                    filter=hdr_dict["FILTER"],
                    ML_bool=ML_bool,
                    bayes_bool=bayes_bool,
                    bayes_method=bayes_method,
                    uncertainty_bool=uncertainty_bool,
                    mdn=mdn,
                    nii_cons=nii_cons,
                    initial_values=initial_values_to_pass,
                    spec_min=spec_min,
                    spec_max=spec_max,
                    obj_redshift=obj_redshift,
                    n_stoch=n_stoch,
                    resolution=resolution,
                    Luci_path=Luci_path,
                )
                fit_dict = fit.fit()  # Collect fit dictionary
                # Save local list of fit values
                ampls_local.append(fit_dict["amplitudes"])
                flux_local.append(fit_dict["fluxes"])
                flux_errs_local.append(fit_dict["flux_errors"])
                vels_local.append(fit_dict["velocities"])
                broads_local.append(fit_dict["sigmas"])
                vels_errs_local.append(fit_dict["vels_errors"])
                broads_errs_local.append(fit_dict["sigmas_errors"])
                chi2_local.append(fit_dict["chi2"])
                corr_local.append(fit_dict["corr"])
                step_local.append(fit_dict["axis_step"])
                continuum_local.append(fit_dict["continuum"])
                continuum_errs_local.append(fit_dict["continuum_error"])
            else:  # If the sky is empty (this rarely rarely rarely happens), then return zeros for everything
                ampls_local.append([0] * len(lines))
                flux_local.append([0] * len(lines))
                flux_errs_local.append([0] * len(lines))
                vels_local.append([0] * len(lines))
                broads_local.append([0] * len(lines))
                vels_errs_local.append([0] * len(lines))
                broads_errs_local.append([0] * len(lines))
                chi2_local.append(0)
                corr_local.append(0)
                step_local.append(0)
                continuum_local.append(0)
                continuum_errs_local.append(0)
        return (
            i,
            ampls_local,
            flux_local,
            flux_errs_local,
            vels_local,
            vels_errs_local,
            broads_local,
            broads_errs_local,
            chi2_local,
            corr_local,
            step_local,
            continuum_local,
            continuum_errs_local,
        )

    # @jit(nopython=False, parallel=True, nogil=True)
    def fit_cube(
        self,
        lines,
        fit_function,
        vel_rel,
        sigma_rel,
        x_min,
        x_max,
        y_min,
        y_max,
        bkg=None,
        bkgType=None,
        binning=None,
        bayes_bool=False,
        bayes_method="emcee",
        absorp=None,
        uncertainty_bool=False,
        n_threads=2,
        nii_cons=True,
        initial_values=[False],
        spec_min=None,
        spec_max=None,
        obj_redshift=0.0,
        n_stoch=1,
        pca_coefficient_array=None,
        pca_vectors=None,
        pca_mean=None,
    ):
        """
        Primary fit call to fit rectangular regions in the data cube. This wraps the
        LuciFits.FIT().fit() call which applies all the fitting steps. This also
        saves the velocity and broadening fits files. All the files will be saved
        in the folder Luci. The files are the fluxes, velocities, broadening, amplitudes,
        and continuum (and their associated errors) for each linespectrum_axis.

        Args:
            lines: Lines to fit (e.x. ['Halpha', 'NII6583'])
            fit_function: Fitting function to use (e.x. 'gaussian')
            vel_rel: Constraints on Velocity/Position (must be list; e.x. [1, 2, 1])
            sigma_rel: Constraints on sigma (must be list; e.x. [1, 2, 1])
            x_min: Lower bound in x
            x_max: Upper bound in x
            y_min: Lower bound in y
            y_max: Upper bound in y
            bkg: Background Spectrum (1D numpy array; default None)
            bkgType: Type of background (default 'standard'; options ['standard', 'pca'])
            binning:  Value by which to bin (default None)
            bayes_bool: Boolean to determine whether or not to run Bayesian analysis (default False)
            bayes_method = Bayesian Inference method. Options are '[emcee', 'dynesty'] (default 'emcee')
            uncertainty_bool: Boolean to determine whether or not to run the uncertainty analysis (default False)
            n_threads: Number of threads to be passed to joblib for parallelization (default = 1)
            nii_cons: Boolean to turn on or off NII doublet ratio constraint (default True)
            initial_values: List of files containing initial conditions (default [False])
            spec_min: Minimum value of the spectrum to be considered in the fit (we find the closest value)
            spec_max: Maximum value of the spectrum to be considered in the fit
            obj_redshift: Redshift of object to fit relative to cube's redshift. This is useful for fitting high redshift objects
            n_stoch: The number of stochastic runs -- set to 50 for fitting double components (default 1)
            pca_coefficient_array: Array of PCA Coefficients (default None)
            pca_vectors: Vectors corresponding to principal components (default None)
            pca_mean: Mean vector from PCA analysis (default None)


        Return:
            Velocity and Broadening arrays (2d). Also return amplitudes array (3D).

        Examples:
            As always, we must first have the cube initialized (see basic example).

            If we want to fit all five lines in SN3 with a sincgauss function and binning of 2
            over a rectangular region defined in image coordinates as 800<x<1500; 250<y<1250,
            we would run the following:

            >>> vel_map, broad_map, flux_map, chi2_fits = cube.fit_cube(['Halpha', 'NII6548', 'NII6583', 'SII6716', 'SII6731'], 'sincgauss', [1,1,1,1,1], [1,1,1,1,1], 800, 1500, 250, 750, binning=2)

        """
        # Initialize fit solution arrays
        if binning != None and binning != 1:
            self.bin_cube(self.cube_final, self.header, binning, x_min, x_max, y_min, y_max)
            x_max = int((x_max - x_min) / binning)
            y_max = int((y_max - y_min) / binning)
            x_min = 0
            y_min = 0
        elif binning == 1:
            pass  # Don't do anything if binning is set to 1
        cube_to_slice = self.cube_binned if (binning is not None and binning != 1) else self.cube_final
        # TODO: ALLOW BINNING OF INITIAL CONDITIONS
        vel_init, broad_init = resolve_initial_values(initial_values)
        cutout = deep_image_cutout(self, x_min, x_max, y_min, y_max, binning)
        maps = run_fit(
            self,
            cube_to_slice,
            lines,
            fit_function,
            vel_rel,
            sigma_rel,
            x_min,
            x_max,
            y_min,
            y_max,
            n_threads=n_threads,
            bayes_bool=bayes_bool,
            absorp=absorp,
            bayes_method=bayes_method,
            spec_min=spec_min,
            spec_max=spec_max,
            uncertainty_bool=uncertainty_bool,
            bkg=bkg,
            bkgType=bkgType,
            nii_cons=nii_cons,
            initial_values=[vel_init, broad_init],
            obj_redshift=obj_redshift,
            n_stoch=n_stoch,
            pca_coefficient_array=pca_coefficient_array,
            pca_vectors=pca_vectors,
            pca_mean=pca_mean,
        )
        maps.save(self.output_dir, self.object_name, lines, cutout.wcs.to_header(), binning, fit_function=fit_function)
        return maps.velocities, maps.broadenings, maps.fluxes, maps.amplitudes

    def fit_region(
        self,
        lines,
        fit_function,
        vel_rel,
        sigma_rel,
        region,
        bkg=None,
        bkgType="standard",
        binning=None,
        bayes_bool=False,
        bayes_method="emcee",
        output_name=None,
        uncertainty_bool=False,
        n_threads=1,
        nii_cons=True,
        spec_min=None,
        spec_max=None,
        obj_redshift=0.0,
        initial_values=[False],
        n_stoch=1,
        pixel_list=False,
    ):
        """
        Fit the spectrum in a region. This is an extremely similar command to fit_cube except
        it works for ds9 regions. We first create a mask from the ds9 region file. Then
        we step through the cube and only fit the unmasked pixels. Although this may not
        be the most efficient method, it does ensure the fidelity of the wcs system.
        All the files will be saved
        in the folder Luci. The files are the fluxes, velocities, broadening, amplitudes,
        and continuum (and their associated errors) for each line.

        Args:
            lines: Lines to fit (e.x. ['Halpha', 'NII6583'])
            fit_function: Fitting function to use (e.x. 'gaussian')
            vel_rel: Constraints on Velocity/Position (must be list; e.x. [1, 2])
            sigma_rel: Constraints on sigma (must be list; e.x. [1, 2])
            region: Name of ds9 region file (e.x. 'region.reg'). You can also pass a boolean mask array.
            bkg: Background Spectrum (1D numpy array; default None)
            binning:  Value by which to bin (default None)
            bayes_bool: Boolean to determine whether or not to run Bayesian analysis (default False)
            bayes_method: Bayesian Inference method. Options are '[emcee', 'dynesty'] (default 'emcee')
            output_name: User defined output path/name
            uncertainty_bool: Boolean to determine whether or not to run the uncertainty analysis (default False)
            n_threads: Number of threads to be passed to joblib for parallelization (default = 1)
            nii_cons: Boolean to turn on or off NII doublet ratio constraint (default True)
            spec_min: Minimum value of the spectrum to be considered in the fit (we find the closest value)
            spec_max: Maximum value of the spectrum to be considered in the fit
            obj_redshift: Redshift of object to fit relative to cube's redshift. This is useful for fitting high redshift objects
            initial_values: List of files containing initial conditions (default [False])
            n_stoch: The number of stochastic runs -- set to 50 for fitting double components (default 1)
            pixel_list: Boolean indicating if the user passes a 2D list containing pixel IDs for the region (default False)

        Return:
            Velocity and Broadening arrays (2d). Also return amplitudes array (3D).

        Examples:
            As always, we must first have the cube initialized (see basic example).

            If we want to fit all five lines in SN3 with a gaussian function and no binning
            over a ds9 region called main.reg, we would run the following:

            >>> vel_map, broad_map, flux_map, chi2_fits = cube.fit_region(['Halpha', 'NII6548', 'NII6583', 'SII6716', 'SII6731'], 'gaussian', [1,1,1,1,1], [1,1,1,1,1],region='main.reg')

            We could also enable uncertainty calculations and parallel fitting:

            >>> vel_map, broad_map, flux_map, chi2_fits = cube.fit_region(['Halpha', 'NII6548', 'NII6583', 'SII6716', 'SII6731'], 'gaussian', [1,1,1,1,1], [1,1,1,1,1], region='main.reg', uncertatinty_bool=True, n_threads=4)

        """
        # Set spatial bounds for entire cube
        x_min = 0
        x_max = self.cube_final.shape[0]
        y_min = 0
        y_max = self.cube_final.shape[1]
        cube_to_slice = self.cube_final  # Set cube for slicing
        mask = None  # Initialize
        # Initialize fit solution arrays
        if binning != None and binning > 1:  # Bin if we need to
            self.bin_cube(self.cube_final, self.header, binning, x_min, x_max, y_min, y_max)
            x_max = int((x_max - x_min) / binning)
            y_max = int((y_max - y_min) / binning)
            x_min = 0
            y_min = 0
            cube_to_slice = self.cube_binned
        # Resolve whatever the caller passed (.reg / .npy / array / pixel list)
        # to a boolean mask.
        header = self.header_binned if (binning is not None and binning > 1) else self.header
        mask = resolve_mask(region, header, self.cube_final.shape[:2], pixel_list=pixel_list)

        if binning != None and binning > 1:
            mask = bin_mask(mask, binning, x_min, self.cube_final.shape[0], y_min, self.cube_final.shape[1])  # Bin Mask
        # Clean up output name
        if isinstance(region, str):
            if len(region.split("/")) > 1:  # If region file is a path, just keep the name for output purposes
                region = region.split("/")[-1]
            if output_name is None:
                output_name = self.output_dir + "/" + self.object_name + "_" + region.split(".")[0]
        else:  # Passed mask not region file
            if output_name is None:
                output_name = self.output_dir + "/" + self.object_name + "_mask"

        # TODO: ALLOW BINNING OF INITIAL CONDITIONS
        vel_init, broad_init = resolve_initial_values(initial_values)
        cutout = deep_image_cutout(self, x_min, x_max, y_min, y_max, binning)
        maps = run_fit(
            self,
            cube_to_slice,
            lines,
            fit_function,
            vel_rel,
            sigma_rel,
            x_min,
            x_max,
            y_min,
            y_max,
            n_threads=n_threads,
            mask=mask,
            bayes_bool=bayes_bool,
            bayes_method=bayes_method,
            spec_min=spec_min,
            spec_max=spec_max,
            uncertainty_bool=uncertainty_bool,
            bkg=bkg,
            bkgType=bkgType,
            nii_cons=nii_cons,
            initial_values=[vel_init, broad_init],
            obj_redshift=obj_redshift,
            n_stoch=n_stoch,
        )
        maps.save(self.output_dir, self.object_name, lines, cutout.wcs.to_header(), binning, fit_function=fit_function)
        return maps.velocities, maps.broadenings, maps.fluxes, maps.chi2, mask

    def fit_pixel(
        self,
        lines,
        fit_function,
        vel_rel,
        sigma_rel,
        pixel_x,
        pixel_y,
        binning=None,
        bkg=None,
        absorp=None,
        bayes_bool=False,
        bayes_method="emcee",
        uncertainty_bool=False,
        nii_cons=True,
        spec_min=None,
        spec_max=None,
        obj_redshift=0.0,
        n_stoch=1,
        bkgType="standard",
        pca_coefficient_array=None,
        pca_vectors=None,
        pca_mean=None,
    ):
        """
        Primary fit call to fit a single pixel in the data cube. This wraps the
        LuciFits.FIT().fit() call which applies all the fitting steps.

        Args:
            lines: Lines to fit (e.x. ['Halpha', 'NII6583'])
            fit_function: Fitting function to use (e.x. 'gaussian')
            vel_rel: Constraints on Velocity/Position (must be list; e.x. [1, 2, 1])
            sigma_rel: Constraints on sigma (must be list; e.x. [1, 2, 1])
            pixel_x: X coordinate (physical)
            pixel_y: Y coordinate (physical)
            binning: Number of pixels to take around coordinate (i.e. bin=1 will take all pixels touching the X and Y coordinates.
            bkg: Background Spectrum (1D numpy array; default None)
            bkgType: Type of background (default 'standard'; options ['standard', 'pca'])
            bayes_bool: Boolean to determine whether or not to run Bayesian analysis (default False)
            bayes_method: Bayesian Inference method. Options are '[emcee', 'dynesty'] (default 'emcee')
            uncertainty_bool: Boolean to determine whether or not to run the uncertainty analysis (default False)
            nii_cons: Boolean to turn on or off NII doublet ratio constraint (default True)
            spec_min: Minimum value of the spectrum to be considered in the fit (we find the closest value)
            spec_max: Maximum value of the spectrum to be considered in the fit
            obj_redshift: Redshift of object to fit relative to cube's redshift. This is useful for fitting high redshift objects
            n_stoch: The number of stochastic runs -- set to 50 for fitting double components (default 1)
            bkgType: Type of background (default 'standard'; options ['standard', 'pca'])
            pca_coefficient_array: Array of PCA Coefficients (default None)
            pca_vectors: Vectors corresponding to principal components (default None)
            pca_mean: Mean vector from PCA analysis (default None)


        Return:
            Returns the x-axis (redshifted), sky, and fit dictionary


        """
        sky = None
        if binning is not None and binning != 1:  # If data is binned
            sky = self.cube_final[pixel_x - binning : pixel_x + binning, pixel_y - binning : pixel_y + binning, :]
            sky = np.nansum(sky, axis=0)
            sky = np.nansum(sky, axis=0)

        else:
            sky = self.cube_final[pixel_x, pixel_y, :]
        if bkgType == "standard":
            # A binned pixel spans 2*binning per side; unbinned is a single spaxel.
            sky = subtract_standard(sky, bkg, 2 * binning if binning else None)
        elif bkgType == "pca":
            if binning:
                coefficients = pca_coefficient_array[
                    pixel_x - binning : pixel_x + binning, pixel_y - binning : pixel_y + binning, :
                ]
                coefficients = np.nansum(np.nansum(coefficients, axis=0), axis=0)
            else:
                coefficients = pca_coefficient_array[pixel_x, pixel_y]
            background = pca_background(coefficients, pca_vectors, pca_mean)
            sky = subtract_pca(sky, background, self.spectrum_axis, self.hdr_dict["FILTER"])
        elif bkgType is not None:
            raise ValueError("bkgType must be 'standard', 'pca', or None; got %r" % (bkgType,))

        if absorp is not None:
            spectralcut = int(len(sky) * 0.25)
            sky = (
                sky
                - absorp / np.nanmedian(absorp) * np.nanmedian(sky[spectralcut:-spectralcut])
                + np.nanmedian(sky[spectralcut:-spectralcut])
            )

        good_sky_inds = ~np.isnan(sky)  # Clean up spectrum
        sky = sky[good_sky_inds]  # Apply clean to sky
        axis = self.spectrum_axis[good_sky_inds]  # Apply clean to axis
        # The transmission curve is indexed by position in `Fit.apply_transmission`, so it has
        # to be masked in step with the spectrum (see fit_calc).
        trans_filter = self.transmission_interpolated[good_sky_inds]
        # Call fit!
        fit = Fit(
            sky,
            axis,
            self.wavenumbers_syn,
            fit_function,
            lines,
            vel_rel,
            sigma_rel,
            trans_filter=trans_filter,
            theta=self.interferometer_theta[pixel_x, pixel_y],
            delta_x=self.hdr_dict["STEP"],
            n_steps=self.step_nb,
            zpd_index=self.zpd_index,
            filter=self.hdr_dict["FILTER"],
            ML_bool=self.ML_bool,
            bayes_bool=bayes_bool,
            bayes_method=bayes_method,
            uncertainty_bool=uncertainty_bool,
            mdn=self.mdn,
            nii_cons=nii_cons,
            spec_min=spec_min,
            spec_max=spec_max,
            obj_redshift=obj_redshift,
            n_stoch=n_stoch,
            resolution=self.resolution,
            Luci_path=self.Luci_path,
        )
        """fit = self.fit_calc(0, x_min, x_max, y_min, fit_function, lines, vel_rel, sigma_rel,
                            cube_slice=cube_to_slice[:, y_min + 0, :],
                            spectrum_axis=self.spectrum_axis, wavenumbers_syn=self.wavenumbers_syn,
                            transmission_interpolated=self.transmission_interpolated,
                            interferometer_theta=self.interferometer_theta, hdr_dict=self.hdr_dict,
                            step_nb=self.step_nb, zpd_index=self.zpd_index, mdn=self.mdn,
                            ML_bool=self.ML_bool, bayes_bool=bayes_bool,
                            bayes_method=bayes_method, spec_min=spec_min, spec_max=spec_max,
                            uncertainty_bool=uncertainty_bool, bkg=bkg,
                            bkgType=bkgType, nii_cons=nii_cons,
                            initial_values=[vel_init, broad_init],
                            obj_redshift=obj_redshift, n_stoch=n_stoch, resolution=self.resolution,
                            Luci_path=self.Luci_path,
                            pca_coefficient_array=pca_coefficient_array, pca_vectors=pca_vectors, pca_mean=pca_mean) """
        fit_dict = fit.fit()
        return axis, sky, fit_dict

    def bin_cube(self, cube_final, header, binning, x_min, x_max, y_min, y_max):
        self.header_binned, self.cube_binned = bin_cube_function(
            cube_final, header, binning, x_min, x_max, y_min, y_max
        )

    def extract_spectrum(self, x_min, x_max, y_min, y_max, bkg=None, binning=None, mean=False):
        """
        Extract spectrum in region. This is primarily used to extract background regions.
        The spectra in the region are summed and then averaged (if mean is selected).
        Using the 'mean' argument, we can either calculate the total summed spectrum (False)
        or the averaged spectrum for background spectra (True).

        Args:
            x_min: Lower bound in x
            x_max: Upper bound in x
            y_min: Lower bound in y
            y_max: Upper bound in y
            bkg: Background Spectrum (1D numpy array; default None)
            binning:  Value by which to bin (default None)
            mean: Boolean to determine whether or not the mean spectrum is taken. This is used for calculating background spectra.
        Return:
            X-axis (redshifted) and spectral axis of region.

        """
        integrated_spectrum = np.zeros(self.cube_final.shape[2])
        spec_ct = 0
        axis = None  # Initialize
        # Initialize fit solution arrays
        if binning != None and binning != 1:
            self.bin_cube(self.cube_final, self.header, binning, x_min, x_max, y_min, y_max)
            x_max = int((x_max - x_min) / binning)
            y_max = int((y_max - y_min) / binning)
            x_min = 0
            y_min = 0
        for i in tqdm(range(y_max - y_min)):
            y_pix = y_min + i
            for j in range(x_max - x_min):
                x_pix = x_min + j
                if binning is not None and binning != 1:
                    sky = self.cube_binned[x_pix, y_pix, :]
                else:
                    sky = self.cube_final[x_pix, y_pix, :]
                if bkg is not None:
                    if binning:
                        sky -= bkg * binning**2  # Subtract background spectrum
                    else:
                        sky -= bkg  # Subtract background spectrum
                integrated_spectrum += sky[~np.isnan(sky)]
                if axis is None:
                    axis = self.spectrum_axis[~np.isnan(sky)]
                # Counted per spaxel; previously only incremented inside the
                # axis-init guard, making mean=True a no-op (B2).
                spec_ct += 1
        if mean and spec_ct > 0:
            integrated_spectrum /= spec_ct
        return axis, integrated_spectrum

    def extract_spectrum_region(self, region, mean=False):
        """
        Extract spectrum in region. This is primarily used to extract background regions.
        The spectra in the region are summed and then averaged (if mean is selected).
        Using the 'mean' argument, we can either calculate the total summed spectrum (False)
        or the averaged spectrum for background spectra (True).

        Args:
            region: Name of ds9 region file (e.x. 'region.reg'). You can also pass a boolean mask array.
            mean: Boolean to determine whether or not the mean spectrum is taken. This is used for calculating background spectra.
        Return:
            X-axis and spectral axis of region.

        """
        # Create mask
        # if '.reg' in region:
        #    mask = reg_to_mask(region, self.header)
        # elif '.npy' in region:
        #    mask = np.load(region)
        # else:
        #    print("At the moment, we only support '.reg' and '.npy' files for masks.")
        #    print("Terminating Program!")
        if ".reg" in region:  # If passed a .reg file
            header = self.header
            # From the cube, not the standard detector size (bug B10).
            header.set("NAXIS1", self.cube_final.shape[1])  # Need this for astropy
            header.set("NAXIS2", self.cube_final.shape[0])
            mask = reg_to_mask(region, header)
        elif ".npy" in region:  # If passed numpy file
            mask = np.load(region)
        elif region is not None:  # If passed numpy array
            mask = region
        else:  # Not passed a mask in any of the correct formats
            logger.info("Mask was incorrectly passed. Please use either a .reg file or a .npy file or a numpy ndarray")
        # Set spatial bounds for entire cube
        x_min = 0
        x_max = self.cube_final.shape[0]
        y_min = 0
        y_max = self.cube_final.shape[1]
        integrated_spectrum = np.zeros(self.cube_final.shape[2])
        # Index the masked pixels rather than walking the whole cube in Python -- see the note in
        # `fit_spectrum_region`. The mask is transposed so the pixels are visited in the same
        # y-then-x order, and the float64 accumulator is kept.
        mask = np.asarray(mask)
        ys, xs = np.where(mask.T)
        spec_ct = int(xs.size)
        if spec_ct:
            integrated_spectrum = self.cube_final[xs, ys, :].sum(axis=0, dtype=np.float64)
        if mean:
            integrated_spectrum /= spec_ct
        return self.spectrum_axis, integrated_spectrum

    def fit_spectrum_region(
        self,
        lines,
        fit_function,
        vel_rel,
        sigma_rel,
        region,
        initial_values=[False],
        bkg=None,
        bayes_bool=False,
        bayes_method="emcee",
        uncertainty_bool=False,
        mean=False,
        nii_cons=True,
        spec_min=None,
        spec_max=None,
        obj_redshift=0.0,
        n_stoch=1,
    ):
        """
        Fit spectrum in region.
        The spectra in the region are summed and then averaged (if mean is selected).
        Using the 'mean' argument, we can either calculate the total summed spectrum (False)
        or the averaged spectrum for background spectra (True).

        Args:
            lines: Lines to fit (e.x. ['Halpha', 'NII6583'])
            fit_function: Fitting function to use (e.x. 'gaussian')
            vel_rel: Constraints on Velocity/Position (must be list; e.x. [1, 2, 1])
            sigma_rel: Constraints on sigma (must be list; e.x. [1, 2, 1])
            region: Name of ds9 region file (e.x. 'region.reg'). You can also pass a boolean mask array.
            initial_values:
            bkg: Background Spectrum (1D numpy array; default None)
            bayes_bool: Boolean to determine whether or not to run Bayesian analysis
            bayes_method: Bayesian Inference method. Options are '[emcee', 'dynesty'] (default 'emcee')
            uncertainty_bool: Boolean to determine whether or not to run the uncertainty analysis (default False)
            mean: Boolean to determine whether or not the mean spectrum is taken. This is used for calculating background spectra.
            nii_cons: Boolean to turn on or off NII doublet ratio constraint (default True)
            spec_min: Minimum value of the spectrum to be considered in the fit (we find the closest value)
            spec_max: Maximum value of the spectrum to be considered in the fit
            obj_redshift: Redshift of object to fit relative to cube's redshift. This is useful for fitting high redshift objects
            n_stoch: The number of stochastic runs -- set to 50 for fitting double components (default 1)

        Return:
            X-axis and spectral axis of region.

        """
        # Create mask
        integrated_spectrum = np.zeros(self.cube_final.shape[2])
        if isinstance(region, tuple) and len(region) == 2:
            # An (xs, ys) index pair. A WVT bin is a dozen pixels out of four million, so handing the
            # indices over directly avoids materialising -- and, in `fit_wvt`, storing -- a
            # full-field mask per bin. See `luci.analysis.wvt.fit_wvt`.
            xs = np.asarray(region[0], dtype=np.intp)
            ys = np.asarray(region[1], dtype=np.intp)
        else:
            mask = None  # Initialize
            if isinstance(region, str) and region.endswith(".reg"):  # If passed a .reg file
                header = self.header
                # From the cube, not the standard detector size (bug B10).
                header.set("NAXIS1", self.cube_final.shape[1])  # Need this for astropy
                header.set("NAXIS2", self.cube_final.shape[0])
                mask = reg_to_mask(region, header)
            elif isinstance(region, str) and region.endswith(".npy"):  # If passed numpy file
                mask = np.load(region)
            elif region is not None:  # If passed numpy array
                mask = region
            else:  # Not passed a mask in any of the correct formats
                raise ValueError(
                    "No region given. Pass a .reg path, a .npy path, a boolean array, or an (xs, ys) "
                    "index pair. This used to log a message and carry on with mask unset, which then "
                    "failed further down with an unrelated TypeError."
                )
            # Transposing makes `np.where` walk the pixels in the same y-then-x order the original
            # double loop used, so the sum is accumulated in the same order.
            ys, xs = np.where(np.asarray(mask).T)
        spec_ct = int(xs.size)
        # Initialize initial conditions for velocity and broadening as False --> Assuming we don't have them
        vel_init = False
        broad_init = False
        # TODO: ALLOW BINNING OF INITIAL CONDITIONS
        if len(initial_values) == 2:
            try:  # Obtain initial condition maps from files
                vel_init = fits.open(initial_values[0])[0].data
                broad_init = fits.open(initial_values[1])[0].data
            except (OSError, TypeError, ValueError):  # arrays from a previous fit, not FITS paths
                vel_init = initial_values[0]
                broad_init = initial_values[1]
        # Sum the selected pixels by indexing them directly. This used to be a Python double loop over
        # every pixel of the cube, which costs the same 4.2 million iterations whether the selection
        # holds one pixel or a million: ~0.2 s per call, and `fit_wvt` makes one call per bin, so a
        # full-field WVT run spent ~18 hours here. The float64 accumulator is kept, so the only
        # difference from the old sum is that numpy adds pairwise rather than sequentially -- a
        # relative change of order 1e-16 -- and NaN propagation is unchanged (a NaN channel in any
        # selected pixel still poisons that channel).
        if spec_ct:
            integrated_spectrum = self.cube_final[xs, ys, :].sum(axis=0, dtype=np.float64)
        if mean:
            integrated_spectrum /= spec_ct  # Take mean spectrum
        if bkg is not None:
            integrated_spectrum -= bkg * spec_ct  # Subtract background spectrum
        good_sky_inds = ~np.isnan(integrated_spectrum)  # Clean up spectrum

        sky = integrated_spectrum[good_sky_inds]
        axis = self.spectrum_axis[good_sky_inds]
        # Masked in step with the spectrum -- see fit_calc.
        trans_filter = self.transmission_interpolated[good_sky_inds]
        # Incidence angle of the region, which sets the wavelength correction factor. It has to come
        # from the region itself: this read used to be `interferometer_theta[x_pix, y_pix]` after the
        # summation loop, and since that loop always ran to completion over the whole cube those
        # indices were always the far corner (dimx-1, dimy-1) -- the same irrelevant pixel for every
        # region ever fit. `luci.analysis.skylines` already uses the region's centre, so a
        # representative pixel is what was meant; the mean over the region is the right one for a
        # spectrum that is itself a sum over those pixels.
        theta = float(np.mean(self.interferometer_theta[xs, ys])) if spec_ct else 0.0
        # Call fit!
        fit = Fit(
            sky,
            axis,
            self.wavenumbers_syn,
            fit_function,
            lines,
            vel_rel,
            sigma_rel,
            trans_filter=trans_filter,
            theta=theta,
            delta_x=self.hdr_dict["STEP"],
            n_steps=self.step_nb,
            zpd_index=self.zpd_index,
            filter=self.hdr_dict["FILTER"],
            ML_bool=self.ML_bool,
            bayes_bool=bayes_bool,
            bayes_method=bayes_method,
            uncertainty_bool=uncertainty_bool,
            nii_cons=nii_cons,
            mdn=self.mdn,
            initial_values=initial_values,
            spec_min=spec_min,
            spec_max=spec_max,
            obj_redshift=obj_redshift,
            n_stoch=n_stoch,
            resolution=self.resolution,
            Luci_path=self.Luci_path,
        )
        fit_dict = fit.fit()
        return axis, sky, fit_dict

    def create_snr_map(self, *args, **kwargs):
        """See LUCI.analysis.snr.create_snr_map."""
        return _create_snr_map(self, *args, **kwargs)

    def skyline_calibration(self, *args, **kwargs):
        """See LUCI.analysis.skylines.skyline_calibration."""
        return _skyline_calibration(self, *args, **kwargs)

    def heliocentric_correction(self):
        """
        Calculate heliocentric correction for observation given the location of SITELLE/CFHT
        and the time of the observation
        """
        CFHT = EarthLocation.of_site("CFHT")
        sc = SkyCoord(ra=self.hdr_dict["CRVAL1"] * u.deg, dec=self.hdr_dict["CRVAL2"] * u.deg)
        heliocorr = sc.radial_velocity_correction(
            "heliocentric", obstime=Time(self.hdr_dict["DATE-OBS"]), location=CFHT
        )
        helio_kms = heliocorr.to(u.km / u.s)
        return helio_kms

    def calculate_component_map(self, x_min=0, x_max=None, y_min=0, y_max=None, bkg=None, n_threads=2, region=None):
        # TODO: ADD Documentation and example
        # Bounds default to this cube's extent rather than the standard detector
        # size (bug B10).
        if x_max is None:
            x_max = self.cube_final.shape[0]
        if y_max is None:
            y_max = self.cube_final.shape[1]
        return create_component_map_function(
            self.header,
            self.hdr_dict,
            self.Luci_path,
            self.resolution,
            self.filter,
            self.cube_final,
            self.spectrum_axis,
            self.wavenumbers_syn_full,
            self.output_dir,
            self.object_name,
            x_min,
            x_max,
            y_min,
            y_max,
            bkg,
            n_threads,
            region,
        )

    def calculate_components_in_region(self, region, bkg):
        # TODO: ADD Documentation and example
        return calculate_components_in_region_function(
            self.header,
            self.hdr_dict,
            self.Luci_path,
            self.resolution,
            self.filter,
            self.cube_final,
            self.spectrum_axis,
            self.wavenumbers_syn_full,
            region,
            bkg,
        )

    def close(self):
        """
        Functionality to delete Luci object (and thus the cube) from memory
        """
        del self.cube_final
        del self.header
        if self.cube_binned:
            del self.cube_binned

    def create_wvt(self, *args, **kwargs):
        """See LUCI.analysis.wvt.create_wvt."""
        return _create_wvt(self, *args, **kwargs)

    def fit_wvt(self, *args, **kwargs):
        """See LUCI.analysis.wvt.fit_wvt."""
        return _fit_wvt(self, *args, **kwargs)

    def wvt_fit_region(self, *args, **kwargs):
        """See LUCI.analysis.wvt.wvt_fit_region."""
        return _wvt_fit_region(self, *args, **kwargs)

    def export_fits(self):
        """
        Export HDF file as fits file. The data will be saved in the same location as the cube (`cube_dir`) with the same
        name (`object_name`).
        """
        header3D = self.header.copy()
        header3D["NAXIS"] = 4
        # header3D['CUNIT3'] = 'WAVN'
        header3D["CTYPE3"] = "FREQ"
        header3D["CRPIX3"] = 1
        header3D["CRVAL3"] = self.hdr_dict["axis_min"]
        header3D["CDELT3"] = self.hdr_dict["axis_step"]
        # header3D['CUNIT3'] = '/cm'
        header3D["NAXIS4"] = 1
        header3D["CRPIX4"] = 1
        header3D["CRVAL4"] = 1
        header3D["CDELT4"] = 1
        header3D.remove("WCSAXES")
        fits_header = fits.PrimaryHDU(header=header3D, data=self.cube_final.transpose(2, 1, 0))
        hdu = fits.HDUList([fits_header])
        hdu.writeto(os.path.join(self.output_dir, self.object_name + ".fits"), overwrite=True)
        return None

    def detection_map(self, *args, **kwargs):
        """See LUCI.analysis.snr.detection_map."""
        return _detection_map(self, *args, **kwargs)

    def slicing(self, *args, **kwargs):
        """See LUCI.analysis.slicing.slicing."""
        return _slicing(self, *args, **kwargs)

    def create_background_subspace(self, *args, **kwargs):
        """See LUCI.background.pca.create_background_subspace."""
        return _create_background_subspace(self, *args, **kwargs)


# ``Luci`` was this class's name for its whole published life; keep it working.
Luci = SitelleCube
