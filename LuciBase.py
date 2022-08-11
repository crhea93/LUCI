import h5py
import glob
import pandas
from astropy.wcs import WCS
import astropy.units as u
from tqdm import tqdm
import keras
from joblib import Parallel, delayed, dump, load
from LUCI.LuciComponentCalculations import calculate_components_in_region_function, create_component_map_function
from LUCI.LuciConvenience import reg_to_mask
from LUCI.LuciFit import Fit
from astropy.nddata import Cutout2D
import astropy.stats as astrostats
# from astroquery.astrometry_net import AstrometryNet
from astropy.time import Time
import numpy.ma as ma
from astropy.coordinates import SkyCoord, EarthLocation
from numba import jit, set_num_threads, prange
from LUCI.LuciNetwork import create_MDN_model, negative_loglikelihood
from LUCI.LuciUtility import save_fits, get_quadrant_dims, get_interferometer_angles, update_header, \
    read_in_reference_spectrum, read_in_transmission, check_luci_path, spectrum_axis_func, bin_cube_function
from LUCI.LuciWVT import *
from LUCI.LuciVisualize import visualize as LUCIvisualize
import multiprocessing as mp
import time


class Luci():
    """
    This is the primary class for the general purpose line fitting code LUCI. This contains
    all io/administrative functionality. The fitting functionality can be found in the
    Fit class (Lucifit.py).
    """

    def __init__(self, Luci_path, cube_path, output_dir, object_name, redshift, resolution, ML_bool=True, mdn=False):
        """
        Initialize our Luci class -- this acts similar to the SpectralCube class
        of astropy or spectral-cube.

        Args:
            Luci_path: Path to Luci (must include trailing "/")
            cube_path: Full path to hdf5 cube with the hdf5 extension (e.x. '/user/home/M87.hdf5'; No trailing "/")
            output_dir: Full path to output directory
            object_name: Name of the object to fit. This is used for naming purposes. (e.x. 'M87')
            redshift: Redshift to the object. (e.x. 0.00428)
            resolution: Resolution requested of machine learning algorithm reference spectrum
            ML_bool: Boolean for applying machine learning; default=True
            mdn: Boolean for using the Mixed Density Network models; If true, then we use the posterior distributions calculated by our network as our priors for bayesian fits
        """
        self.header_binned = None
        self.Luci_path = Luci_path
        check_luci_path(Luci_path)  # Make sure the path is correctly written
        self.cube_path = cube_path
        self.output_dir = output_dir + '/Luci_outputs'
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
        self.step_nb = self.hdr_dict['STEPNB']
        self.zpd_index = self.hdr_dict['ZPDINDEX']
        self.filter = self.hdr_dict['FILTER']
        self.spectrum_axis, self.spectrum_axis_unshifted = spectrum_axis_func(self.hdr_dict, self.redshift)
        print('LYA check: Using dev branch')
        if ML_bool is True:
            if not self.mdn:
                if self.filter in ['SN1', 'SN2', 'SN3', 'C4', 'C2', 'C3', 'C1']:
                    self.ref_spec = self.Luci_path + 'ML/Reference-Spectrum-R%i-%s.fits' % (resolution, self.filter)
                    self.wavenumbers_syn, self.wavenumbers_syn_full = read_in_reference_spectrum(self.ref_spec,
                                                                                                 self.hdr_dict)
                    self.model_ML = keras.models.load_model(
                        self.Luci_path + 'ML/R%i-PREDICTOR-I-%s' % (resolution, self.filter))
                else:
                    print(
                        'LUCI does not support machine learning parameter estimates for the filter you entered. Please set ML_bool=False.')
            else:  # mdn == True
                if self.filter in ['SN3']:
                    self.ref_spec = self.Luci_path + 'ML/Reference-Spectrum-R%i-%s.fits' % (resolution, self.filter)
                    self.wavenumbers_syn, self.wavenumbers_syn_full = read_in_reference_spectrum(self.ref_spec,
                                                                                                 self.hdr_dict)
                    self.model_ML = create_MDN_model(len(self.wavenumbers_syn), negative_loglikelihood)
                    self.model_ML.load_weights(self.Luci_path + 'ML/R%i-PREDICTOR-I-MDN-%s/R%i-PREDICTOR-I-MDN-%s' % (
                        resolution, self.filter, resolution, self.filter))
                else:
                    print(
                        'LUCI does not support machine learning parameter estimates using a MDN for the filter you entered. Please set ML_bool=False or mdn=False.')
        else:
            self.model_ML = None
        self.transmission_interpolated = read_in_transmission(self.Luci_path, self.hdr_dict,
                                                              self.spectrum_axis_unshifted)

    def read_in_cube(self):
        """
        Function to read the hdf5 data into a 3d numpy array (data cube). We also
        translate the header to standard wcs format by calling the update_header function.
        Note that the data are saved in several quadrants which is why we have to loop
        through them and place all the spectra in a single cube.
        """
        print('Reading in data...')
        file = h5py.File(self.cube_path + '.hdf5', 'r')  # Read in file
        # file = ht.load(self.cube_path + '.hdf5')
        self.quad_nb = file.attrs['quad_nb']  # Get the number of quadrants
        self.dimx = file.attrs['dimx']  # Get the dimensions in x
        self.dimy = file.attrs['dimy']  # Get the dimensions in y
        self.dimz = file.attrs['dimz']  # Get the dimensions in z (spectral axis)
        self.cube_final = np.zeros((self.dimx, self.dimy, self.dimz))  # Complete data cube
        for iquad in tqdm(range(self.quad_nb)):
            xmin, xmax, ymin, ymax = get_quadrant_dims(iquad, self.quad_nb, self.dimx, self.dimy)
            iquad_data = file['quad00%i' % iquad]['data'][:]  # Save data to intermediate array
            iquad_data[(np.isfinite(iquad_data) == False)] = 1e-22  # Set infinite values to 1-e22
            iquad_data[(iquad_data < -1e-16)] = 1e-22  # Set high negative flux values to 1e-22
            iquad_data[(iquad_data > 1e-9)] = 1e-22  # Set unrealistically high positive flux values to 1e-22
            self.cube_final[xmin:xmax, ymin:ymax, :] = iquad_data  # Save to correct location in main cube
            iquad_data = None
        self.cube_final = self.cube_final  # .transpose(1, 0, 2)
        folder = './joblib_memmap'
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass

        data_filename_memmap = os.path.join(folder, 'data_memmap')
        dump(self.cube_final, data_filename_memmap)
        self.cube_final = load(data_filename_memmap, mmap_mode='readwrite')
        self.header, self.hdr_dict = update_header(file)
        self.interferometer_theta = get_interferometer_angles(file, self.hdr_dict)
        #file.close()

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

        hdf5_file = h5py.File(self.cube_path + '.hdf5', 'r')  # Open and read hdf5 file

        if 'deep_frame' in hdf5_file:  # A deep image already exists
            print('Existing deep frame extracted from hdf5 file.')
            self.deep_image = hdf5_file['deep_frame'][:]
            self.deep_image *= self.dimz
        else:  # Create new deep image
            print('New deep frame created from data.')
            self.deep_image = np.zeros(
                (self.cube_final.shape[0], self.cube_final.shape[1]))  # np.sum(self.cube_final, axis=2).T
            iterations_ = 10
            step_size = int(self.cube_final.shape[0] / iterations_)
            for i in tqdm(range(10)):
                self.deep_image[step_size * i:step_size * (i + 1)] = np.nansum(
                    self.cube_final[step_size * i:step_size * (i + 1)], axis=2)
        self.deep_image = self.deep_image.T
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
                    summed_deep = self.deep_image[x_min + int(i * binning):x_min + int((i + 1) * binning),
                                  y_min + int(j * binning):y_min + int((j + 1) * binning)]
                    summed_deep = np.nansum(summed_deep, axis=0)  # Sum along x
                    summed_deep = np.nansum(summed_deep, axis=0)  # Sum along y
                    binned_deep[i, j] = summed_deep  # Set to global
            # Update header information
            header_binned = self.header
            header_binned['CRPIX1'] = header_binned['CRPIX1'] / binning
            header_binned['CRPIX2'] = header_binned['CRPIX2'] / binning
            header_binned['CDELT1'] = header_binned['CDELT1'] * binning
            header_binned['CDELT2'] = header_binned['CDELT2'] * binning
            self.deep_image = binned_deep / (binning ** 2)
        if output_name == None:
            output_name = self.output_dir + '/' + self.object_name + '_deep.fits'
        fits.writeto(output_name, self.deep_image, self.header, overwrite=True)
        hdf5_file.close()

    def visualize(self):
        """
        Wrapper function for LUCI.LuciVisualize()
        """
        if self.deep_image is None:
            try:
                deep_image = fits.open('Luci_outputs/%s_deep.fits' % self.object_name)[0].data
            except:
                self.create_deep_image()
                deep_image = fits.open('Luci_outputs/%s_deep.fits' % self.object_name)[0].data
        else:
            deep_image = self.deep_image
        LUCIvisualize(deep_image, self.spectrum_axis, self.cube_final)

    def fit_entire_cube(self, lines, fit_function, vel_rel, sigma_rel, bkg=None, binning=None, bayes_bool=False,
                        output_name=None, uncertainty_bool=False, n_threads=1):
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
        self.fit_cube(lines, fit_function, vel_rel, sigma_rel, x_min, x_max, y_min, y_max)

    def fit_calc(self, i, x_min, x_max, y_min, fit_function, lines, vel_rel, sigma_rel, mask=None, bayes_bool=False,
                 bayes_method='emcee',
                 uncertainty_bool=False, nii_cons=False,
                 bkg=None, binning=None, spec_min=None, spec_max=None, initial_values=False,
                 obj_redshift=0.0):
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
            binning:  Value by which to bin (default None)
            bayes_bool: Boolean to determine whether or not to run Bayesian analysis (default False)
            bayes_method = Bayesian Inference method. Options are '[emcee', 'dynesty'] (default 'emcee')
            uncertainty_bool: Boolean to determine whether or not to run the uncertainty analysis (default False)
            nii_cons: Boolean to turn on or off NII doublet ratio constraint (default True)
            initial_values: List of files containing initial conditions (default False)
            spec_min: Minimum value of the spectrum to be considered in the fit (we find the closest value)
            spec_max: Maximum value of the spectrum to be considered in the fit
            obj_redshift: Redshift of object to fit relative to cube's redshift. This is useful for fitting high redshift objects
        """
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
        # Step through x coordinates
        for j in range(x_max - x_min):
            x_pix = x_min + j  # Set current x pixel
            if mask is not None:  # Check if there is a mask
                if not mask[x_pix, y_pix]:  # Check that the mask is true
                    pass
            if binning is not None and binning != 1:  # If binning, then take spectrum from binned cube
                sky = self.cube_binned[x_pix, y_pix, :]
            else:  # If not, then take from the unbinned cube
                sky = self.cube_final[x_pix, y_pix, :]
            if bkg is not None:  # If there is a background variable subtract the bkg spectrum
                if binning:  # If binning, then we have to take into account how many pixels are in each bin
                    sky -= bkg * binning ** 2  # Subtract background spectrum
                else:  # No binning so just subtract the background directly
                    sky -= bkg  # Subtract background spectrum
            good_sky_inds = [~np.isnan(sky)]  # Find all NaNs in sky spectrum
            sky = sky[good_sky_inds]  # Clean up spectrum by dropping any Nan values
            axis = self.spectrum_axis#[good_sky_inds]  # Clean up axis  accordingly
            if initial_values[0] is not False:   #Frozen parameter
                initial_values_to_pass = [initial_values[0][i][j], initial_values[1][i][j]]
            else:
                initial_values_to_pass = initial_values
            # Call fit!
            if len(sky) > 0:  # Ensure that there are values in sky
                fit = Fit(sky, axis, self.wavenumbers_syn, fit_function, lines, vel_rel, sigma_rel,
                          self.model_ML, trans_filter=self.transmission_interpolated,
                          theta=self.interferometer_theta[x_pix, y_pix],
                          delta_x=self.hdr_dict['STEP'], n_steps=self.step_nb,
                          zpd_index=self.zpd_index,
                          filter=self.hdr_dict['FILTER'],
                          bayes_bool=bayes_bool, bayes_method=bayes_method,
                          uncertainty_bool=uncertainty_bool,
                          mdn=self.mdn, nii_cons=nii_cons, initial_values=initial_values_to_pass,
                          spec_min=spec_min, spec_max=spec_max, obj_redshift=obj_redshift
                          )
                fit_dict = fit.fit()  # Collect fit dictionary
                # Save local list of fit values
                ampls_local.append(fit_dict['amplitudes'])
                flux_local.append(fit_dict['fluxes'])
                flux_errs_local.append(fit_dict['flux_errors'])
                vels_local.append(fit_dict['velocities'])
                broads_local.append(fit_dict['sigmas'])
                vels_errs_local.append(fit_dict['vels_errors'])
                broads_errs_local.append(fit_dict['sigmas_errors'])
                chi2_local.append(fit_dict['chi2'])
                corr_local.append(fit_dict['corr'])
                step_local.append(fit_dict['axis_step'])
                continuum_local.append(fit_dict['continuum'])
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
        return i, ampls_local, flux_local, flux_errs_local, vels_local, vels_errs_local, broads_local, broads_errs_local, chi2_local, corr_local, step_local, continuum_local

    def fit_cube(self, lines, fit_function, vel_rel, sigma_rel,
                 x_min, x_max, y_min, y_max, bkg=None, binning=None,
                 bayes_bool=False, bayes_method='emcee',
                 uncertainty_bool=False, n_threads=2, nii_cons=True, initial_values=[False],
                 spec_min=None, spec_max=None):

        """
        Primary fit call to fit rectangular regions in the data cube. This wraps the
        LuciFits.FIT().fit() call which applies all the fitting steps. This also
        saves the velocity and broadening fits files. All the files will be saved
        in the folder Luci. The files are the fluxes, velocities, broadening, amplitudes,
        and continuum (and their associated errors) for each line.

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
            self.bin_cube(self.cube_final, self.header, binning, x_min, x_max, y_min,
                          y_max)
            x_max = int((x_max - x_min) / binning);
            y_max = int((y_max - y_min) / binning)
            x_min = 0;
            y_min = 0
        elif binning == 1:
            pass  # Don't do anything if binning is set to 1
        chi2_fits = np.zeros((x_max - x_min, y_max - y_min), dtype=np.float32).T
        corr_fits = np.zeros((x_max - x_min, y_max - y_min), dtype=np.float32).T
        step_fits = np.zeros((x_max - x_min, y_max - y_min), dtype=np.float32).T
        # First two dimensions are the X and Y dimensions.
        # The third dimension corresponds to the line in the order of the lines input parameter.
        ampls_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0, 2)
        flux_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0, 2)
        flux_errors_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0, 2)
        velocities_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0, 2)
        broadenings_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0, 2)
        velocities_errors_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0,
                                                                                                                  2)
        broadenings_errors_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0,
                                                                                                                   2)
        continuum_fits = np.zeros((x_max - x_min, y_max - y_min), dtype=np.float32).T
        # Initialize initial conditions for velocity and broadening as False --> Assuming we don't have them
        vel_init = False
        broad_init = False
        # TODO: ALLOW BINNING OF INITIAL CONDITIONS
        if len(initial_values) == 2:
            try:  # Obtain initial condition maps from files
                vel_init = fits.open(initial_values[0])[0].data
                broad_init = fits.open(initial_values[1])[0].data
            except:  # Initial conditions passed are arrays from a previous fit and not  fits files
                vel_init = initial_values[0]
                broad_init = initial_values[1]
        if binning is not None and binning != 1:
            # Check if deep image exists: if not, create it
            if not os.path.exists(self.output_dir + '/' + self.object_name + '_deep.fits'):
                self.create_deep_image()
            wcs = WCS(self.header_binned)
        else:
            # Check if deep image exists: if not, create it
            if not os.path.exists(self.output_dir + '/' + self.object_name + '_deep.fits'):
                self.create_deep_image()
            wcs = WCS(self.header, naxis=2)
        cutout = Cutout2D(fits.open(self.output_dir + '/' + self.object_name + '_deep.fits')[0].data,
                          position=((x_max + x_min) / 2, (y_max + y_min) / 2), size=(x_max - x_min, y_max - y_min),
                          wcs=wcs)
        results = Parallel(n_jobs=n_threads, require='sharedmem') \
            (delayed(self.fit_calc)(sl, x_min, x_max, y_min, fit_function, lines, vel_rel, sigma_rel, bayes_bool=bayes_bool,
                                    bayes_method=bayes_method,
                                    uncertainty_bool=uncertainty_bool, bkg=bkg, nii_cons=nii_cons, initial_values=[vel_init, broad_init]) for sl in tqdm(range(y_max - y_min)))
        for result in results:
            i, ampls_local, flux_local, flux_errs_local, vels_local, vels_errs_local, broads_local, broads_errs_local, chi2_local, corr_local, step_local, continuum_local = result
            ampls_fits[i] = ampls_local
            flux_fits[i] = flux_local
            flux_errors_fits[i] = flux_errs_local
            velocities_fits[i] = vels_local
            broadenings_fits[i] = broads_local
            velocities_errors_fits[i] = vels_errs_local
            broadenings_errors_fits[i] = broads_errs_local
            chi2_fits[i] = chi2_local
            corr_fits[i] = corr_local
            step_fits[i] = step_local
            continuum_fits[i] = continuum_local
        save_fits(self.output_dir, self.object_name, lines, ampls_fits, flux_fits, flux_errors_fits, velocities_fits,
                  broadenings_fits,
                  velocities_errors_fits, broadenings_errors_fits, chi2_fits, continuum_fits,
                  cutout.wcs.to_header(), binning)

        return velocities_fits, broadenings_fits, flux_fits, chi2_fits

    def fit_region(self, lines, fit_function, vel_rel, sigma_rel, region,
                   bkg=None, binning=None, bayes_bool=False, bayes_method='emcee',
                   output_name=None, uncertainty_bool=False, n_threads=1, nii_cons=True,
                   spec_min=None, spec_max=None, obj_redshift=0.0, initial_values=[False]):
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
        # Initialize fit solution arrays
        if binning != None and binning != 1:
            self.bin_cube(self.cube_final, self.header, binning, x_min, x_max, y_min,
                          y_max)
            x_max = int((x_max - x_min) / binning);
            y_max = int((y_max - y_min) / binning)
            x_min = 0;
            y_min = 0
        # Create mask
        if '.reg' in region:
            shape = (2064, 2048)  # (self.header["NAXIS1"], self.header["NAXIS2"])  # Get the shape
            if binning != None and binning > 1:
                header = self.header_binned
            else:
                header = self.header
            header.set('NAXIS1', 2064)
            header.set('NAXIS2', 2048)
            mask = reg_to_mask(region, header)
        elif '.npy' in region:
            mask = np.load(region).T
        else:
            print('Mask was incorrectly passed. Please use either a .reg file or a .npy file')
        # Clean up output name
        if isinstance(region, str):
            if len(region.split('/')) > 1:  # If region file is a path, just keep the name for output purposes
                region = region.split('/')[-1]
            if output_name is None:
                output_name = self.output_dir + '/' + self.object_name + '_' + region.split('.')[0]
        else:  # Passed mask not region file
            if output_name is None:
                output_name = self.output_dir + '/' + self.object_name + '_mask'

        chi2_fits = np.zeros((x_max - x_min, y_max - y_min), dtype=np.float32).T
        # First two dimensions are the X and Y dimensions.
        # The third dimension corresponds to the line in the order of the lines input parameter.
        ampls_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0, 2)
        flux_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0, 2)
        flux_errors_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0, 2)
        velocities_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0, 2)
        broadenings_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0, 2)
        velocities_errors_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0, 2)
        broadenings_errors_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0,2)
        continuum_fits = np.zeros((x_max - x_min, y_max - y_min), dtype=np.float32).T

        # Initialize initial conditions for velocity and broadening as False --> Assuming we don't have them
        vel_init = False
        broad_init = False
        # TODO: ALLOW BINNING OF INITIAL CONDITIONS
        if len(initial_values) == 2:
            try:  # Obtain initial condition maps from files
                vel_init = fits.open(initial_values[0])[0].data
                broad_init = fits.open(initial_values[1])[0].data
            except:  # Initial conditions passed are arrays from a previous fit and not  fits files
                vel_init = initial_values[0]
                broad_init = initial_values[1]
        if binning is not None and binning > 1:
            # Check if deep image exists: if not, create it
            if not os.path.exists(self.output_dir + '/' + self.object_name + '_deep.fits'):
                self.create_deep_image()
            wcs = WCS(self.header_binned)
        else:
            # Check if deep image exists: if not, create it
            if not os.path.exists(self.output_dir + '/' + self.object_name + '_deep.fits'):
                self.create_deep_image()
            wcs = WCS(self.header, naxis=2)
        cutout = Cutout2D(fits.open(self.output_dir + '/' + self.object_name + '_deep.fits')[0].data,
                          position=((x_max + x_min) / 2, (y_max + y_min) / 2), size=(x_max - x_min, y_max - y_min),
                          wcs=wcs)
        results = Parallel(n_jobs=n_threads, require='sharedmem') \
            (delayed(self.fit_calc)(sl, x_min, x_max, y_min, fit_function, lines, vel_rel, sigma_rel, mask,
                                    bayes_bool=bayes_bool,
                                    bayes_method=bayes_method,
                                    uncertainty_bool=uncertainty_bool, bkg=bkg, nii_cons=nii_cons,
                                    initial_values=[vel_init, broad_init]) for sl in tqdm(range(y_max - y_min)))
        for result in results:
            i, ampls_local, flux_local, flux_errs_local, vels_local, vels_errs_local, broads_local, broads_errs_local, chi2_local, corr_local, step_local, continuum_local = result.get()
            ampls_fits[i] = ampls_local
            flux_fits[i] = flux_local
            flux_errors_fits[i] = flux_errs_local
            velocities_fits[i] = vels_local
            broadenings_fits[i] = broads_local
            velocities_errors_fits[i] = vels_errs_local
            broadenings_errors_fits[i] = broads_errs_local
            chi2_fits[i] = chi2_local
            continuum_fits[i] = continuum_local
        save_fits(self.output_dir, self.object_name, lines, ampls_fits, flux_fits, flux_errors_fits, velocities_fits,
                  broadenings_fits,
                  velocities_errors_fits, broadenings_errors_fits, chi2_fits, continuum_fits,
                  cutout.wcs.to_header(), binning)
        return velocities_fits, broadenings_fits, flux_fits, chi2_fits, mask

    def fit_pixel(self, lines, fit_function, vel_rel, sigma_rel,
                  pixel_x, pixel_y, bin=None, bkg=None,
                  bayes_bool=False, bayes_method='emcee',
                  uncertainty_bool=False,
                  nii_cons=True, spec_min=None, spec_max=None, obj_redshift=0.0):
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
            bin: Number of pixels to take around coordinate (i.e. bin=1 will take all pixels touching the X and Y coordinates.
            bkg: Background Spectrum (1D numpy array; default None)
            bayes_bool: Boolean to determine whether or not to run Bayesian analysis (default False)
            bayes_method: Bayesian Inference method. Options are '[emcee', 'dynesty'] (default 'emcee')
            uncertainty_bool: Boolean to determine whether or not to run the uncertainty analysis (default False)
            nii_cons: Boolean to turn on or off NII doublet ratio constraint (default True)
            spec_min: Minimum value of the spectrum to be considered in the fit (we find the closest value)
            spec_max: Maximum value of the spectrum to be considered in the fit
            obj_redshift: Redshift of object to fit relative to cube's redshift. This is useful for fitting high redshift objects
        Return:
            Returns the x-axis (redshifted), sky, and fit dictionary


        """
        sky = None
        if bin is not None and bin != 1:  # If data is binned
            sky = self.cube_final[pixel_x - bin:pixel_x + bin, pixel_y - bin:pixel_y + bin, :]
            sky = np.nansum(sky, axis=0)
            sky = np.nansum(sky, axis=0)
            if bkg is not None:
                sky -= bkg * (2 + bin) ** 2  # Subtract background times number of pixels
        else:
            sky = self.cube_final[pixel_x, pixel_y, :]
            if bkg is not None:
                sky -= bkg  # Subtract background spectrum
        good_sky_inds = [~np.isnan(sky)]  # Clean up spectrum
        sky = sky[good_sky_inds]  # Apply clean to sky
        axis = self.spectrum_axis[good_sky_inds]  # Apply clean to axis
        # Call fit!
        fit = Fit(sky, axis, self.wavenumbers_syn, fit_function, lines, vel_rel, sigma_rel,
                  self.model_ML, trans_filter=self.transmission_interpolated,
                  theta=self.interferometer_theta[pixel_x, pixel_y],
                  delta_x=self.hdr_dict['STEP'], n_steps=self.step_nb,
                  zpd_index=self.zpd_index,
                  filter=self.hdr_dict['FILTER'],
                  bayes_bool=bayes_bool, bayes_method=bayes_method,
                  uncertainty_bool=uncertainty_bool,
                  mdn=self.mdn, nii_cons=nii_cons,
                  spec_min=spec_min, spec_max=spec_max, obj_redshift=obj_redshift)
        fit_dict = fit.fit()
        return axis, sky, fit_dict

    def bin_cube(self, cube_final, header, binning, x_min, x_max, y_min, y_max):
        self.header_binned, self.cube_binned = bin_cube_function(cube_final, header, binning, x_min, x_max, y_min,
                                                                 y_max)

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
            self.bin_cube(self.cube_final, self.header, binning, x_min, x_max, y_min,
                          y_max)
            # x_min = int(x_min/binning) ; y_min = int(y_min/binning) ; x_max = int(x_max/binning) ;  y_max = int(y_max/binning)
            x_max = int((x_max - x_min) / binning);
            y_max = int((y_max - y_min) / binning)
            x_min = 0;
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
                        sky -= bkg * binning ** 2  # Subtract background spectrum
                    else:
                        sky -= bkg  # Subtract background spectrum
                good_sky_inds = [~np.isnan(sky)]  # Clean up spectrum
                integrated_spectrum += sky[good_sky_inds]
                if spec_ct == 0:
                    axis = self.spectrum_axis[good_sky_inds]
                    spec_ct += 1
        if mean:
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
        if '.reg' in region:
            mask = reg_to_mask(region, self.header)
        elif '.npy' in region:
            mask = np.load(region)
        else:
            print("At the moment, we only support '.reg' and '.npy' files for masks.")
            print("Terminating Program!")

        # Set spatial bounds for entire cube
        x_min = 0
        x_max = self.cube_final.shape[0]
        y_min = 0
        y_max = self.cube_final.shape[1]
        integrated_spectrum = np.zeros(self.cube_final.shape[2])
        spec_ct = 0
        for i in tqdm(range(y_max - y_min)):
            y_pix = y_min + i
            for j in range(x_max - x_min):
                x_pix = x_min + j
                # Check if pixel is in the mask or not
                if mask[x_pix, y_pix]:
                    integrated_spectrum += self.cube_final[x_pix, y_pix, :]
                    spec_ct += 1
                else:
                    pass
        if mean:
            integrated_spectrum /= spec_ct
        return self.spectrum_axis, integrated_spectrum

    def fit_spectrum_region(self, lines, fit_function, vel_rel, sigma_rel,
                            region, initial_values=[False], bkg=None,
                            bayes_bool=False, bayes_method='emcee',
                            uncertainty_bool=False, mean=False, nii_cons=True,
                            spec_min=None, spec_max=None, obj_redshift=0.0
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
        Return:
            X-axis and spectral axis of region.

        """
        # Create mask
        mask = None  # Initialize
        if '.reg' in region:
            mask = reg_to_mask(region, self.header)
        elif '.npy' in region:
            mask = np.load(region)
        else:
            print("At the moment, we only support '.reg' and '.npy' files for masks.")
            print("Terminating Program!")
        # Set spatial bounds for entire cube
        x_min = 0
        x_max = self.cube_final.shape[0]
        y_min = 0
        y_max = self.cube_final.shape[1]
        integrated_spectrum = np.zeros(self.cube_final.shape[2])
        spec_ct = 0
        # Initialize initial conditions for velocity and broadening as False --> Assuming we don't have them
        vel_init = False
        broad_init = False
        # TODO: ALLOW BINNING OF INITIAL CONDITIONS
        if len(initial_values) == 2:
            try:  # Obtain initial condition maps from files
                vel_init = fits.open(initial_values[0])[0].data
                broad_init = fits.open(initial_values[1])[0].data
            except:  # Initial conditions passed are arrays from a previous fit and not  fits files
                vel_init = initial_values[0]
                broad_init = initial_values[1]
        for i in range(y_max - y_min):
            y_pix = y_min + i
            for j in range(x_max - x_min):
                x_pix = x_min + j
                # Check if pixel is in the mask or not
                if mask[x_pix, y_pix]:
                    integrated_spectrum += self.cube_final[x_pix, y_pix, :]
                    spec_ct += 1
                else:
                    pass
        if mean:
            integrated_spectrum /= spec_ct  # Take mean spectrum
        if bkg is not None:
            integrated_spectrum -= bkg * spec_ct  # Subtract background spectrum
        good_sky_inds = [~np.isnan(integrated_spectrum)]  # Clean up spectrum
        sky = integrated_spectrum[good_sky_inds]
        axis = self.spectrum_axis[good_sky_inds]
        # Call fit!
        fit = Fit(sky, axis, self.wavenumbers_syn, fit_function, lines, vel_rel, sigma_rel,
                  self.model_ML, trans_filter=self.transmission_interpolated,
                  theta=self.interferometer_theta[x_pix, y_pix],
                  delta_x=self.hdr_dict['STEP'], n_steps=self.step_nb,
                  zpd_index=self.zpd_index,
                  filter=self.hdr_dict['FILTER'],
                  bayes_bool=bayes_bool, bayes_method=bayes_method,
                  uncertainty_bool=uncertainty_bool, nii_cons=nii_cons,
                  mdn=self.mdn, initial_values=initial_values,
                  spec_min=spec_min, spec_max=spec_max,
                  obj_redshift=obj_redshift)
        fit_dict = fit.fit()
        return axis, sky, fit_dict

    def create_snr_map(self, x_min=0, x_max=2048, y_min=0, y_max=2064, method=1, n_threads=2):
        """
        Create signal-to-noise ratio (SNR) map of a given region. If no bounds are given,
        a map of the entire cube is calculated.

        Args:
            x_min: Minimal X value (default 0)
            x_max: Maximal X value (default 2048)
            y_min: Minimal Y value (default 0)
            y_max: Maximal Y value (default 2064)
            method: Method used to calculate SNR (default 1; options 1 or 2)
            n_threads: Number of threads to use
        Return:
            snr_map: Signal-to-Noise ratio map

        """
        SNR = np.zeros((2048, 2064), dtype=np.float32).T
        flux_min = 0;
        flux_max = 0;
        noise_min = 0;
        noise_max = 0  # Initializing bounds for flux and noise calculation regions
        if self.hdr_dict['FILTER'] == 'SN3':
            flux_min = 15150
            flux_max = 15300
            noise_min = 14500
            noise_max = 14600
        elif self.hdr_dict['FILTER'] == 'SN2':
            flux_min = 19800
            flux_max = 20750
            noise_min = 19000
            noise_max = 19500
        elif self.hdr_dict['FILTER'] == 'SN1':
            flux_min = 26550
            flux_max = 27550
            noise_min = 25700
            noise_max = 26300
        elif self.hdr_dict['FILTER'] == 'C3':  # Only for MACSJ1621
            flux_min = 18100
            flux_max = 19500
            noise_min = 17450
            noise_max = 17550
        else:
            print('SNR Calculation for this filter has not been implemented')

        def SNR_calc(i):
            y_pix = y_min + i
            snr_local = np.zeros(2048)
            for j in range(x_max - x_min):
                x_pix = x_min + j
                # Calculate SNR
                min_ = np.argmin(np.abs(np.array(self.spectrum_axis) - flux_min))
                max_ = np.argmin(np.abs(np.array(self.spectrum_axis) - flux_max))
                in_region = self.cube_final[x_pix, y_pix, min_:max_]
                flux_in_region = np.nansum(self.cube_final[x_pix, y_pix, min_:max_])
                # Subtract off continuum estimate
                clipped_spec = astrostats.sigma_clip(in_region, sigma=3, masked=False,
                                                     copy=False, maxiters=10)
                # Now take the mean value to serve as the continuum value
                cont_val = np.min(clipped_spec)
                flux_in_region -= cont_val * (max_ - min_)  # Need to scale by the number of steps along wavelength axis
                min_ = np.argmin(np.abs(np.array(self.spectrum_axis) - noise_min))
                max_ = np.argmin(np.abs(np.array(self.spectrum_axis) - noise_max))
                out_region = self.cube_final[x_pix, y_pix, min_:max_]
                std_out_region = np.nanstd(out_region)
                if method == 1:
                    signal = np.nanmax(in_region) - np.nanmedian(in_region)
                    noise = np.abs(np.nanstd(out_region))
                    snr = float(signal / np.sqrt(noise))
                    if snr < 0:
                        snr = 0
                    else:
                        snr = snr / (np.sqrt(np.nanmean(np.abs(in_region))))
                else:
                    snr = float(flux_in_region / std_out_region)
                    if snr < 0:
                        snr = 0
                    else:
                        pass
                snr_local[x_pix] = snr
            return snr_local, i

        res = Parallel(n_jobs=n_threads, backend="threading")(delayed(SNR_calc)(i) for i in tqdm(range(y_max - y_min)));
        # Save
        for snr_ind in res:
            snr_vals, step_i = snr_ind
            SNR[y_min + step_i] = snr_vals
        SNR[SNR == np.inf] = 0
        SNR[SNR == -np.inf] = 0
        fits.writeto(self.output_dir + '/' + self.object_name + '_SNR.fits', SNR, self.header, overwrite=True)

        # Save masks for SNr 3, 5, and 10
        masks = []
        for snr_val in [1, 3, 5, 10]:
            mask = ma.masked_where(SNR >= snr_val, SNR)
            masks.append(mask)
            np.save("%s/SNR_%i_mask.npy" % (self.output_dir, snr_val), mask.mask)
        return masks

    def heliocentric_correction(self):
        """
        Calculate heliocentric correction for observation given the location of SITELLE/CFHT
        and the time of the observation
        """
        CFHT = EarthLocation.of_site('CFHT')
        sc = SkyCoord(ra=self.hdr_dict['CRVAL1'] * u.deg, dec=self.hdr_dict['CRVAL2'] * u.deg)
        heliocorr = sc.radial_velocity_correction('heliocentric', obstime=Time(self.hdr_dict['DATE-OBS']),
                                                  location=CFHT)
        helio_kms = heliocorr.to(u.km / u.s)
        return helio_kms

    def skyline_calibration(self, n_grid, bin_size=30):
        """
        Compute skyline calibration by fitting the 6498.729 Angstrom line. Flexures
        of the telescope lead to minor offset that can be measured by high resolution
        spectra (R~5000). This function divides the FOV into a grid of NxN spaxel regions
        of 10x10 pixels to increase the signal. The function will output a map of the
        velocity offset. The initial velocity guess is set to 80 km/s. Additionally,
        we fit with a simple sinc function.

        Args:
            n_grid: NxN grid (int)
            bin_size: Size of grouping used for each region (optional int; default=30)

        Return:
            Velocity offset map
        """
        velocity = None;
        fit_vector = None;
        sky = None;
        # Read in sky lines
        sky_lines_df = pandas.read_csv('Data/sky_lines.csv', skiprows=2)
        sky_lines = sky_lines_df['Wavelength']  # Get wavelengths
        sky_lines = [sky_line / 10 for sky_line in sky_lines]  # Convert from angstroms to nanometers
        sky_lines_scale = [sky_line for sky_line in sky_lines_df['Strength']]  # Get the relative strengths
        # Create skyline dictionary
        sky_line_dict = {}  # {OH_num: wavelength in nm}
        for line_ct, line_wvl in enumerate(sky_lines):
            sky_line_dict['OH_%i' % line_ct] = line_wvl
        # Calculate grid
        x_min = 100
        x_max = self.cube_final.shape[0] - 100
        x_step = int(
            (x_max - x_min) / n_grid)  # Calculate step size based on min and max values and the number of grid points
        y_min = 1000
        y_max = self.cube_final.shape[1] - 100
        y_step = int(
            (y_max - y_min) / n_grid)  # Calculate step size based on min and max values and the number of grid points
        vel_grid = np.zeros((n_grid, n_grid))  # Initialize velocity grid
        for x_grid in range(n_grid):  # Step through x steps
            for y_grid in range(n_grid):  # Step through y steps
                # Collect spectrum in 10x10 region
                x_center = x_min + int(0.5 * (x_step) * (x_grid + 1))
                y_center = y_min + int(0.5 * (y_step) * (y_grid + 1))
                integrated_spectrum = np.zeros_like(self.cube_final[x_center, y_center, :])  # Initialize as zeros
                for i in range(bin_size):  # Take bin_size x bin_size bins
                    for j in range(bin_size):
                        integrated_spectrum += self.cube_final[x_center + i, y_center + i, :]
                # Collapse to single spectrum
                good_sky_inds = [~np.isnan(integrated_spectrum)]  # Clean up spectrum
                sky = integrated_spectrum[good_sky_inds]
                axis = self.spectrum_axis[good_sky_inds]
                # Call fit!
                fit = Fit(sky, axis, self.wavenumbers_syn, 'sinc', ['OH_%i' % num for num in sky_lines],
                          len(sky_lines) * [1], len(sky_lines) * [1],
                          self.model_ML, trans_filter=self.transmission_interpolated,
                          theta=self.interferometer_theta[x_center, y_center],
                          delta_x=self.hdr_dict['STEP'], n_steps=self.step_nb,
                          zpd_index=self.zpd_index,
                          filter=self.hdr_dict['FILTER'], bayes_bool=False, bayes_method='emcee',
                          sky_lines=sky_line_dict, sky_lines_scale=sky_lines_scale
                          )

                velocity, fit_vector = fit.fit(sky_line=True)
                vel_grid[x_grid, y_grid] = float(velocity)
        # Now that we have the grid, we need to reproject it onto the original pixel grid
        vel_grid_final = np.zeros((x_max, y_max))
        for x_grid in range(n_grid):  # Step through x steps
            for y_grid in range(n_grid):  # Step through y steps
                # Collect spectrum in 10x10 region
                x_center = x_min + int(0.5 * (x_step) * (x_grid + 1))
                y_center = y_min + int(0.5 * (y_step) * (y_grid + 1))
                vel_grid_final[x_center - x_step:x_center + x_step, y_center - y_step:y_center + y_step] = vel_grid[
                    x_grid, y_grid]
        fits.writeto(self.output_dir + '/velocity_correction.fits', vel_grid, self.header, overwrite=True)
        return velocity, fit_vector, sky

    def calculate_component_map(self, x_min=0, x_max=2048, y_min=0, y_max=2064, bkg=None, n_threads=2, region=None):
        # TODO: ADD Documentation and example
        return create_component_map_function(self.header, self.hdr_dict, self.Luci_path, self.resolution, self.filter,
                                             self.cube_final, self.spectrum_axis,
                                             self.wavenumbers_syn_full, self.output_dir, self.object_name, x_min, x_max,
                                             y_min,
                                             y_max, bkg,
                                             n_threads, region)

    def calculate_components_in_region(self, region, bkg):
        # TODO: ADD Documentation and example
        return calculate_components_in_region_function(self.header, self.hdr_dict, self.Luci_path, self.resolution,
                                                       self.filter,
                                                       self.cube_final, self.spectrum_axis,
                                                       self.wavenumbers_syn_full, region, bkg)

    def close(self):
        """
        Functionality to delete Luci object (and thus the cube) from memory
        """
        del self.cube_final
        del self.header
        if self.cube_binned:
            del self.cube_binned

    def create_wvt(self, x_min_init, x_max_init, y_min_init, y_max_init, pixel_size, stn_target, roundness_crit, ToL):
        """
        Written by Benjamin Vigneron.

        Functionality to create a weighted Voronoi tesselation map from a region and according to
        arguments passed by the user. It creates a folder containing all the Voronoi bins that can
        then be used for the fitting procedure.
        Args:
            x_min_init: Minimal X value
            x_max_init: Maximal X value
            y_min_init: Minimal Y value
            y_max_init: Maximal Y value
            pixel_size: Pixel size of the image. For SITELLE use pixel_size = 0.0000436.
            stn_target: Signal-to-Noise target value for the Voronoi bins.
            roundness_crit: Roundness criteria for the pixel accretion into bins
            ToL: Convergence tolerance parameter for the SNR of the bins
            bkg: Background Spectrum (1D numpy array; default None)
        """
        print("#----------------WVT Algorithm----------------#")
        print("#----------------Creating SNR Map--------------#")
        Pixels = []
        self.create_snr_map(x_min_init, x_max_init, y_min_init, y_max_init, method=1, n_threads=1)
        print("#----------------Algorithm Part 1----------------#")
        start = time.time()
        SNR_map = fits.open(self.output_dir + '/' + self.object_name + '_SNR.fits')[0].data
        SNR_map = SNR_map[y_min_init:y_max_init, x_min_init:x_max_init]
        fits.writeto(self.output_dir + '/' + self.object_name + '_SNR.fits', SNR_map, overwrite=True)
        Pixels, x_min, x_max, y_min, y_max = read_in(self.output_dir + '/' + self.object_name + '_SNR.fits')
        Nearest_Neighbors(Pixels)
        Init_bins = Bin_Acc(Pixels, pixel_size, stn_target, roundness_crit)
        plot_Bins(Init_bins, x_min, x_max, y_min, y_max, stn_target, self.output_dir, "bin_acc")
        total_time = time.gmtime(float(time.time() - start))
        print('The first part of the algorithm took %s.' % (time.strftime("%H:%M:%S", total_time)))
        print("#----------------Algorithm Part 2----------------#")
        Final_Bins = WVT(Init_bins, Pixels, stn_target, ToL, pixel_size, self.output_dir)
        print("#----------------Algorithm Complete--------------#")
        plot_Bins(Final_Bins, x_min, x_max, y_min, y_max, stn_target, self.output_dir, "final")
        Bin_data(Final_Bins, Pixels, x_min, y_min, self.output_dir, "WVT_data")
        print("#----------------Bin Mapping--------------#")
        pixel_x = []
        pixel_y = []
        bins = []
        bin_map = np.zeros((x_max - x_min, y_max - y_min))
        j = 0
        i = 0
        with open(self.output_dir + '/WVT_data.txt', 'rt') as myfile:
            myfile = myfile.readlines()[3:]
            for myline in myfile:
                myline = myline.strip(' \n')
                data = [int(s) for s in myline.split() if s.isdigit()]
                pixel_x.append(data[0])
                pixel_y.append(data[1])
                bins.append(data[2])
        for pix_x, pix_y in zip(pixel_x, pixel_y):
            bin_map[pix_x, pix_y] = int(bins[i])
            i += 1
        # bin_map = np.rot90(bin_map)
        print("#----------------Numpy Bin Mapping--------------#")
        if not os.path.exists(self.output_dir + '/Numpy_Voronoi_Bins'):
            os.mkdir(self.output_dir + '/Numpy_Voronoi_Bins')
        if os.path.exists(self.output_dir + '/Numpy_Voronoi_Bins'):
            files = glob.glob(self.output_dir + '/Numpy_Voronoi_Bins/*.npy')
            for f in files:
                os.remove(f)
        for bin_num in tqdm(list(range(len(Final_Bins)))):
            bool_bin_map = np.zeros((2048, 2064), dtype=bool)
            for a, b in zip(np.where(bin_map == bin_num)[0][:], np.where(bin_map == bin_num)[1][:]):
                bool_bin_map[x_min_init + a, y_min_init + b] = True
            np.save(self.output_dir + '/Numpy_Voronoi_Bins/bool_bin_map_%i' % j, bool_bin_map)
            j += 1

    def fit_wvt(self, lines, fit_function, vel_rel, sigma_rel, bkg=None, bayes_bool=False, uncertainty_bool=False,
                n_threads=1, initial_values=[False]):
        """
        Function that takes the wvt mapping created using `self.create_wvt()` and fits the bins.
        Written by Benjamin Vigneron

        Args:
            lines: Lines to fit (e.x. ['Halpha', 'NII6583'])
            fit_function: Fitting function to use (e.x. 'gaussian')
            vel_rel: Constraints on Velocity/Position (must be list; e.x. [1, 2, 1])
            sigma_rel: Constraints on sigma (must be list; e.x. [1, 2, 1])
            bkg: Background Spectrum (1D numpy array; default None)
            bayes_bool: Boolean to determine whether or not to run Bayesian analysis
            uncertainty_bool: Boolean to determine whether or not to run the uncertainty analysis (default False)
            n_threads: Number of threads to use
            initial_values: Initial values of velocity and broadening for fitting specific lines (must be list)
        Return:
            Velocity, Broadening and Flux arrays (2d). Also return amplitudes array (3D) and header for saving
            figure.
        """
        x_min = 0
        x_max = self.cube_final.shape[0]
        y_min = 0
        y_max = self.cube_final.shape[1]
        chi2_fits = np.zeros((x_max - x_min, y_max - y_min), dtype=np.float32).T
        # First two dimensions are the X and Y dimensions.
        # The third dimension corresponds to the line in the order of the lines input parameter.
        ampls_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0, 2)
        flux_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0, 2)
        flux_errors_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0, 2)
        velocities_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0, 2)
        broadenings_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0, 2)
        velocities_errors_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0,
                                                                                                                  2)
        broadenings_errors_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0,
                                                                                                                   2)
        continuum_fits = np.zeros((x_max - x_min, y_max - y_min), dtype=np.float32).T
        if len(initial_values) == 2:
            # Obtain initial condition maps from files
            vel_init = fits.open(initial_values[0])[0].data
            broad_init = fits.open(initial_values[1])[0].data
        ct = 0
        set_num_threads(n_threads)
        if not os.path.exists(self.output_dir + '/' + self.object_name + '_deep.fits'):
            self.create_deep_image()
        wcs = WCS(self.header, naxis=2)
        cutout = Cutout2D(fits.open(self.output_dir + '/' + self.object_name + '_deep.fits')[0].data,
                          position=((x_max + x_min) / 2, (y_max + y_min) / 2), size=(x_max - x_min, y_max - y_min),
                          wcs=wcs)
        for bin_num in tqdm(list(range(len(os.listdir(self.output_dir + '/Numpy_Voronoi_Bins/'))))):
            bool_bin_map = self.output_dir + '/Numpy_Voronoi_Bins/bool_bin_map_%i.npy' % bin_num
            index = np.where(np.load(bool_bin_map) == True)
            initial_conditions = None
            for a, b in zip(index[0], index[1]):
                # TODO: PASS INITIAL CONDITIONS
                if initial_values is not False:  # If initial conditions were passed
                    initial_conditions = [vel_init[a, b], broad_init[a, b]]
                else:
                    initial_conditions = False
            bin_axis, bin_sky, bin_fit_dict = self.fit_spectrum_region(lines, fit_function, vel_rel, sigma_rel,
                                                                       region=bool_bin_map,
                                                                       initial_conditions=initial_conditions,
                                                                       bkg=bkg,
                                                                       bayes_bool=bayes_bool,
                                                                       uncertainty_bool=uncertainty_bool)
            for a, b in zip(index[0], index[1]):
                ampls_fits[a, b] = bin_fit_dict['amplitudes']
                flux_fits[a, b] = bin_fit_dict['fluxes']
                flux_errors_fits[a, b] = bin_fit_dict['flux_errors']
                broadenings_fits[a, b] = bin_fit_dict['sigmas']
                broadenings_errors_fits[a, b] = bin_fit_dict['sigmas_errors']
                chi2_fits[a, b] = bin_fit_dict['chi2']
                continuum_fits[a, b] = bin_fit_dict['continuum']
                velocities_fits[a, b] = bin_fit_dict['velocities']
                velocities_errors_fits[a, b] = bin_fit_dict['vels_errors']
        save_fits(self.output_dir, self.object_name, lines, ampls_fits, flux_fits, flux_errors_fits, velocities_fits,
                  broadenings_fits, velocities_errors_fits,
                  broadenings_errors_fits, chi2_fits, continuum_fits, cutout.wcs.to_header(),
                  binning=1, suffix='_wvt')
        return velocities_fits, broadenings_fits, flux_fits, chi2_fits, cutout.wcs.to_header()

    def wvt_fit_region(self, x_min_init, x_max_init, y_min_init, y_max_init, lines, fit_function, vel_rel, sigma_rel,
                       stn_target,
                       pixel_size=0.436, roundness_crit=0.3, ToL=1e-2, bkg=None,
                       bayes_bool=False, uncertainty_bool=False, n_threads=1):
        """
        Functionality to wrap-up the creation and fitting of weighted Voronoi bins.
        Args:
            x_min_init: Minimal X value
            x_max_init: Maximal X value
            y_min_init: Minimal Y value
            y_max_init: Maximal Y value
            lines: Lines to fit (e.x. ['Halpha', 'NII6583'])
            fit_function: Fitting function to use (e.x. 'gaussian')
            vel_rel: Constraints on Velocity/Position (must be list; e.x. [1, 2, 1])
            sigma_rel: Constraints on sigma (must be list; e.x. [1, 2, 1])
            stn_target: Signal-to-Noise target value for the Voronoi bins.
            pixel_size: Pixel size of the image. For SITELLE use pixel_size = 0.0000436.
            roundness_crit: Roundness criteria for the pixel accretion into bins
            ToL: Convergence tolerance parameter for the SNR of the bins
            bkg: Background Spectrum (1D numpy array; default None)
            bayes_bool: Boolean to determine whether or not to run Bayesian analysis
            uncertainty_bool: Boolean to determine whether or not to run the uncertainty analysis (default False)
            n_threads: Number of threads to use
            initial_values: Initial values of velocity and broadening for fitting specific lines (must be list;
            e.x. [velocity, broadening])
        Return:
            Velocity, Broadening and Flux arrays (2d). Also return amplitudes array (3D).
        """
        # Call create wvt function to create the WVT map and numpy files corresponding to each bin
        self.create_wvt(x_min_init, x_max_init, y_min_init, y_max_init, pixel_size, stn_target, roundness_crit, ToL)
        print("#----------------WVT Fitting--------------#")
        # Fit the bins
        velocities_fits, broadenings_fits, flux_fits, chi2_fits, header = self.fit_wvt(lines,
                                                                                       fit_function, vel_rel, sigma_rel,
                                                                                       bkg=bkg, bayes_bool=bayes_bool,
                                                                                       uncertainty_bool=uncertainty_bool,
                                                                                       n_threads=n_threads,
                                                                                       initial_values=False)
        output_name = self.object_name + '_wvt_1'  # Add the '_1' because the binning is set to 1
        for line_ in lines:
            amp = fits.open(self.output_dir + '/Amplitudes/' + output_name + '_' + line_ + '_Amplitude.fits')[0].data.T
            flux = fits.open(self.output_dir + '/Fluxes/' + output_name + '_' + line_ + '_Flux.fits')[0].data.T
            flux_err = fits.open(self.output_dir + '/Fluxes/' + output_name + '_' + line_ + '_Flux_err.fits')[0].data.T
            vel = fits.open(self.output_dir + '/Velocity/' + output_name + '_' + line_ + '_velocity.fits')[0].data.T
            broad = fits.open(self.output_dir + '/Broadening/' + output_name + '_' + line_ + '_broadening.fits')[
                0].data.T
            vel_err = fits.open(self.output_dir + '/Velocity/' + output_name + '_' + line_ + '_velocity_err.fits')[
                0].data.T
            broad_err = \
                fits.open(self.output_dir + '/Broadening/' + output_name + '_' + line_ + '_broadening_err.fits')[
                    0].data.T
            chi2 = fits.open(self.output_dir + '/' + output_name + '_Chi2.fits')[0].data.T
            cont = fits.open(self.output_dir + '/' + output_name + '_continuum.fits')[0].data.T
            fits.writeto(self.output_dir + '/Amplitudes/' + output_name + '_' + line_ + '_Amplitude.fits',
                         amp, header, overwrite=True)
            fits.writeto(self.output_dir + '/Fluxes/' + output_name + '_' + line_ + '_Flux.fits', flux,
                         header, overwrite=True)
            fits.writeto(self.output_dir + '/Fluxes/' + output_name + '_' + line_ + '_Flux_err.fits',
                         flux_err, header, overwrite=True)
            fits.writeto(self.output_dir + '/Velocity/' + output_name + '_' + line_ + '_velocity.fits',
                         vel, header, overwrite=True)
            fits.writeto(self.output_dir + '/Broadening/' + output_name + '_' + line_ + '_broadening.fits',
                         broad, header, overwrite=True)
            fits.writeto(self.output_dir + '/Velocity/' + output_name + '_' + line_ + '_velocity_err.fits',
                         vel_err, header, overwrite=True)
            fits.writeto(self.output_dir + '/Broadening/' + output_name + '_' + line_ + '_broadening_err.fits',
                         broad_err, header, overwrite=True)
            fits.writeto(self.output_dir + '/' + output_name + '_Chi2.fits', chi2, header, overwrite=True)
            fits.writeto(self.output_dir + '/' + output_name + '_continuum.fits', cont, header, overwrite=True)
        return None

    def detection_map(self, x_min=None, x_max=None, y_min=None, y_max=None, n_threads=1):
        """
        Method to call the detection algorithm. The detection algorithm works as follows:
        For each pixel,
            1. Calculate the median spectrum for a 3x3 pixel region centered on the current pixel
            2. Calculate the median spectrum for a 9x9 pixel region centered on the current pixel
            3. Subtract the 9x9 spectrum from the 3x3 spectrum
            4. Take the maximum value of this subtracted spectrum as the detection map value

        In the end, we have a detection map of the maximum values.

        If no bounds are added, we calculate over the entire map

        Args:
            x_min: Lower bound in x
            x_max: Upper bound in x
            y_min: Lower bound in y
            y_max: Upper bound in y
            n_threads: Number of threads (default 1)
        """
        if x_min is None or x_max is None or y_min is None or y_max is None:
            # Set spatial bounds for entire cube
            x_min = 0 + 10
            x_max = self.cube_final.shape[0] - 10
            y_min = 0 + 10
            y_max = self.cube_final.shape[1] - 10
        # Initalize solution
        detection_map = np.zeros((x_max - x_min, y_max - y_min), dtype=np.float32).T
        # Correct header information
        if not os.path.exists(self.output_dir + '/' + self.object_name + '_deep.fits'):
            self.create_deep_image()

        wcs = WCS(self.header, naxis=2)
        cutout = Cutout2D(fits.open(self.output_dir + '/' + self.object_name + '_deep.fits')[0].data,
                          position=((x_max + x_min) / 2, (y_max + y_min) / 2), size=(x_max - x_min, y_max - y_min),
                          wcs=wcs)

        global value_calc

        def value_calc(i):
            y_pix = y_min + i
            detection_local = []
            for j in range(x_max - x_min):
                x_pix = x_min + j
                # Get the 3x3 bin group
                sky_1 = self.cube_final[x_pix - 1:x_pix + 1, y_pix - 1:y_pix + 1, :]
                sky_1 = np.nanmedian(sky_1, axis=0)  # Bin once
                sky_1 = np.nanmedian(sky_1, axis=0)  # Bin twice!
                good_sky_inds_1 = [~np.isnan(sky_1)]  # Clean up spectrum
                sky_1 = sky_1[good_sky_inds_1]
                # Get the 9x9 bin group
                sky_2 = self.cube_final[x_pix - 4:x_pix + 4, y_pix - 4:y_pix + 4, :]
                sky_2 = np.nanmedian(sky_2, axis=0)  # Bin once
                sky_2 = np.nanmedian(sky_2, axis=0)  # Bin twice!
                good_sky_inds_2 = [~np.isnan(sky_2)]  # Clean up spectrum
                sky_2 = sky_2[good_sky_inds_2]
                # Obtain the difference
                sky_diff = sky_1 - sky_2
                if len(sky_diff) > 0:
                    detection_val = np.nanmax(
                        sky_diff)  # The nan shouldn't be necessary -- I just put it so that it didn't feel left out
                    detection_local.append(detection_val)
                else:
                    detection_local.append(0)
            return i, detection_local

        pool = mp.Pool(n_threads)
        results = tqdm(pool.imap(value_calc, [row for row in (range(y_max - y_min))]), total=y_max - y_min)
        results = tuple(results)
        pool.close()
        for result in results:
            i, detection_local = result
            detection_map[i] = detection_local
        fits.writeto(self.output_dir + '/' + self.object_name + '_detection.fits', detection_map,
                     cutout.wcs.to_header(), overwrite=True)
        return detection_map

    def slicing(self, lines):
        """
        Slices the cube along the spectral axis around the specified emission line wavelengths. The input velocity dispersion
        serves as a rough estimate of the width of the spectral lines to make sure the slices cover a wide enough range on each side
        of the emission lines. The slices are saved in a new folder for each input emission line.

        Args:
            lines: Lines to fit (e.x. ['Halpha', 'NII6583'])
        """
        line_dict = {'Halpha': 656.280, 'NII6583': 658.341, 'NII6548': 654.803,
                     'SII6716': 671.647, 'SII6731': 673.085, 'OII3726': 372.603,
                     'OII3729': 372.882, 'OIII4959': 495.891, 'OIII5007': 500.684,
                     'Hbeta': 486.133, 'OH': 649.873, 'HalphaC4': 807.88068, 'NII6583C4': 810.417771,
                     'NII6548C4': 806.062493,
                     'OIII5007C2': 616.342, 'OIII4959C2': 610.441821, 'HbetaC2': 598.429723,
                     'OII3729C1': 459.017742, 'OII3726C1': 458.674293, }

        filter_line = {'SN1': ['OII3726', 'OII3729'], 'SN2': ['Hbeta', 'OIII4959', 'OIII5007'],
                       'SN3': ['Halpha', 'NII6583', 'NII6548', 'SII6716', 'SII6731']}

        spectral_axis = 1e7 / self.spectrum_axis  # Convert wavenumber in cm-1 to nm

        # Make sure the specified lines are part of the filter of the cube
        if all(line in (filter_line[self.filter]) for line in lines):

            wavelengths = np.array(
                list(map(line_dict.get, lines)))  # Get the rest wavelength values of the specified lines
            wavelength_redshift = (
                                              1 + self.redshift) * wavelengths  # Calculate the wavelength in the redshifted frame

            # Match the minimum and maximum wavelengths of each group of slices around the lines with the spectral axis of the cube
            def find_nearest(spectral_axis, line_lambda):
                indices = np.abs(np.subtract.outer(spectral_axis, line_lambda)).argmin(0)
                return indices

            # Loop for every emission line
            for i in range(len(wavelength_redshift)):

                delta_lambda = 200 * wavelength_redshift[
                    i] / 299792  # In units of nm. The 200km/s is a reasonable maximum value for the velocity dispersion of galaxies
                lambda_range = np.array(
                    [wavelength_redshift[i] - 10 * delta_lambda, wavelength_redshift[i] + 10 * delta_lambda])
                idx_axis = find_nearest(spectral_axis, lambda_range)[::-1]

                directory = self.output_dir + '/Slice_{}'.format(lines[i])

                if not os.path.exists(directory):
                    os.makedirs(directory)

                # Loop for every slice
                for j in tqdm(range(idx_axis[0], idx_axis[1] + 1)):
                    cube_slice = self.cube_final[:, :, j]
                    hdu = fits.PrimaryHDU(cube_slice, header=self.header)
                    hdu.writeto(directory + '/slice_{}.fits'.format(j - (idx_axis[0] - 1)), overwrite=True)

                print('')
                print("#######################################################################")
                print("Wavelength of the {} line in the redshifted frame: {} nm".format(lines[i], np.round(
                    wavelength_redshift[i], 2)))
                print("Wavelength of the last slice: " + str(np.round(spectral_axis[idx_axis[1]],
                                                                      2)) + " nm")  # .format() was not giving the right number of decimals..
                print("Wavelength of the first slice: " + str(np.round(spectral_axis[idx_axis[0]],
                                                                       2)) + " nm")  # so not the cleanest way to do it but still it works
                print('')

        else:
            print("The specified lines are not in the wavelength range covered by the filter of this cube")

    '''def update_astrometry(self, api_key):
            """
            Use astronomy.net to update the astrometry in the header using the deep image.
            If astronomy.net successfully finds the corrected astrononmy, the self.header is updated. Otherwise,
            the header is not updated and an exception is thrown.

            This automatically updates the deep images header! If you want the header to be binned, then you can bin it
            using the standard creation mechanisms (for this example binning at 2x2) and then run this code:

            >>> cube.create_deep_image(binning=2)
            >>> cube.update_astrometry(api_key)

            Args:
                api_key: Astronomy.net user api key
            """
            # Initiate Astronomy Net
            ast = AstrometryNet()
            ast.key = api_key
            ast.api_key = api_key
            try_again = True
            submission_id = None
            # Check that deep image exists. Otherwise make one
            if not os.path.exists(self.output_dir + '/' + self.object_name + '_deep.fits'):
                self.create_deep_image()
            # Now submit to astronomy.net until the value is found
            while try_again:
                if not submission_id:
                    try:
                        wcs_header = ast.solve_from_image(self.output_dir + '/' + self.object_name + '_deep.fits',
                                                          submission_id=submission_id,
                                                          solve_timeout=300)  # , use_sextractor=True, center_ra=float(ra), center_dec=float(dec))
                    except Exception as e:
                        print("Timedout")
                        submission_id = e.args[1]
                    else:
                        # got a result, so terminate
                        print("Result")
                        try_again = False
                else:
                    try:
                        wcs_header = ast.monitor_submission(submission_id, solve_timeout=300)
                    except Exception as e:
                        print("Timedout")
                        submission_id = e.args[1]
                    else:
                        # got a result, so terminate
                        print("Result")
                        try_again = False

            if wcs_header:
                # Code to execute when solve succeeds
                # update deep image header
                deep = fits.open(self.output_dir + '/' + self.object_name + '_deep.fits')
                deep[0].header.update(wcs_header)
                deep.close()
                # Update normal header
                self.header = wcs_header

            else:
                # Code to execute when solve fails
                print('Astronomy.net failed to solve. This astrometry has not been updated!')'''
