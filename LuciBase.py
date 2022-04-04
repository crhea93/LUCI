import h5py
import glob
import pandas
from astropy.wcs import WCS
import astropy.units as u
from tqdm import tqdm
import keras
from LUCI.LuciComponentCalculations import calculate_components_in_region_function, create_component_map_function
from LUCI.LuciConvenience import reg_to_mask
from LUCI.LuciFit import Fit
from astropy.nddata import Cutout2D
from astroquery.astrometry_net import AstrometryNet
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from numba import jit, set_num_threads
from LUCI.LuciNetwork import create_MDN_model, negative_loglikelihood
import numpy.ma as ma
from LUCI.LuciWVT import read_in, Nearest_Neighbors, Bin_Acc, plot_Bins, WVT, Bin_data
from LUCI.LuciSNR import create_snr_map_function, calculate_snr_region_function
from LUCI.LuciUtility import save_fits, get_quadrant_dims, get_interferometer_angles, update_header, \
    read_in_reference_spectrum, read_in_transmission, check_luci_path, spectrum_axis_func, bin_cube_function


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
        self.Luci_path = check_luci_path(Luci_path)  # Make sure the path is correctly written
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
        if ML_bool is True:
            if not self.mdn:
                if self.filter in ['SN1', 'SN2', 'SN3', 'C4']:
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
        self.cube_final = self.cube_final  # .transpose(1, 0, 2)
        self.header, self.hdr_dict = update_header(file)
        self.interferometer_theta = get_interferometer_angles(file, self.hdr_dict)

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

    def fit_cube(self, lines, fit_function, vel_rel, sigma_rel,
                 x_min, x_max, y_min, y_max, bkg=None, binning=None,
                 bayes_bool=False, bayes_method='emcee',
                 uncertainty_bool=False, n_threads=1, nii_cons=True, initial_values=False):
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
            initial_values: List of files containing initial conditions (default False)
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
        initial_conditions = False
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
        set_num_threads(n_threads)
        # Initialize initiatl conditions for velocity and broadening as False --> Assuming we don't have them
        vel_init = False
        broad_init = False
        # TODO: ALLOW BINNING OF INITIAL CONDITIONS
        if initial_values is not False:
            # Obtain initial condition maps from files
            vel_init = fits.open(initial_values[0])[0].data
            broad_init = fits.open(initial_values[0])[0].data

        @jit(nopython=False)
        def fit_calc(i, ampls_fit, flux_fit, flux_errs_fit, vels_fit, vels_errs_fit, broads_fit, broads_errs_fit,
                     chi2_fit, corr_fit, step_fit, continuum_fit, initial_conditions=initial_conditions):

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
                axis = self.spectrum_axis[good_sky_inds]  # Clean up axis  accordingly
                # TODO: PASS INITIAL CONDITIONS
                if vel_init is not False and broad_init is not False:  # If initial conditions were passed
                    initial_conditions = [vel_init[x_pix, y_pix], broad_init[x_pix, y_pix]]
                else:
                    initial_conditions = False
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
                              mdn=self.mdn, nii_cons=nii_cons, initial_values=initial_conditions
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
            return i, ampls_fit, flux_fit, flux_errs_fit, vels_fit, vels_errs_fit, broads_fit, broads_errs_fit, chi2_fit, corr_fit, step_fit, continuum_fit

        # Write outputs (Velocity, Broadening, and Amplitudes)
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
        for step_i in tqdm(range(y_max - y_min)):
            fit_calc(step_i, ampls_fits, flux_fits, flux_errors_fits, velocities_fits, velocities_errors_fits,
                     broadenings_fits, broadenings_errors_fits, chi2_fits, corr_fits, step_fits, continuum_fits)
        save_fits(self.output_dir, self.object_name, lines, ampls_fits, flux_fits, flux_errors_fits, velocities_fits,
                  broadenings_fits,
                  velocities_errors_fits, broadenings_errors_fits, chi2_fits, continuum_fits,
                  cutout.wcs.to_header(), binning)

        return velocities_fits, broadenings_fits, flux_fits, chi2_fits

    def fit_region(self, lines, fit_function, vel_rel, sigma_rel, region,
                   bkg=None, binning=None, bayes_bool=False, bayes_method='emcee',
                   output_name=None, uncertainty_bool=False, n_threads=1, nii_cons=True):
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
            # x_min = int(x_min/binning) ; y_min = int(y_min/binning) ; x_max = int(x_max/binning) ;  y_max = int(y_max/binning)
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
            # r = pyregion.open(region).as_imagecoord(header)  # Obtain pyregion region
            # mask = r.get_mask(shape=shape).T  # Calculate mask from pyregion region
            mask = reg_to_mask(region, header)
        elif '.npy' in region:
            mask = np.load(region)
        else:
            print("At the moment, we only support '.reg' and '.npy' files for masks.")
            print("Terminating Program!")
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
        velocities_errors_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0,
                                                                                                                  2)
        broadenings_errors_fits = np.zeros((x_max - x_min, y_max - y_min, len(lines)), dtype=np.float32).transpose(1, 0,
                                                                                                                   2)
        continuum_fits = np.zeros((x_max - x_min, y_max - y_min), dtype=np.float32).T
        ct = 0
        set_num_threads(n_threads)

        @jit(nopython=False)
        def fit_calc(i, ampls_fit, flux_fit, flux_errs_fit, vels_fit, vels_errs_fit, broads_fit, broads_errs_fit,
                     chi2_fit, continuum_fit):
            # for i in tqdm(range(y_max-y_min)):
            y_pix = y_min + i
            ampls_local = []
            flux_local = []
            flux_errs_local = []
            vels_local = []
            broads_local = []
            vels_errs_local = []
            broads_errs_local = []
            chi2_local = []
            continuum_local = []
            for j in range(x_max - x_min):
                x_pix = x_min + j
                # Check if pixel is in the mask or not
                # If so, fit as normal. Else, set values to zero
                if mask[x_pix, y_pix] == True:
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
                    sky = sky[good_sky_inds]
                    axis = self.spectrum_axis[good_sky_inds]
                    # Call fit!
                    fit = Fit(sky, axis, self.wavenumbers_syn, fit_function, lines, vel_rel, sigma_rel,
                              self.model_ML, trans_filter=self.transmission_interpolated,
                              theta=self.interferometer_theta[x_pix, y_pix],
                              delta_x=self.hdr_dict['STEP'], n_steps=self.step_nb,
                              zpd_index=self.zpd_index,
                              filter=self.hdr_dict['FILTER'],
                              bayes_bool=bayes_bool, bayes_method=bayes_method,
                              uncertainty_bool=uncertainty_bool,
                              mdn=self.mdn, nii_cons=nii_cons)
                    fit_dict = fit.fit()
                    # Save local list of fit values
                    ampls_local.append(fit_dict['amplitudes'])
                    flux_local.append(fit_dict['fluxes'])
                    flux_errs_local.append(fit_dict['flux_errors'])
                    vels_local.append(fit_dict['velocities'])
                    broads_local.append(fit_dict['sigmas'])
                    vels_errs_local.append(fit_dict['vels_errors'])
                    broads_errs_local.append(fit_dict['sigmas_errors'])
                    chi2_local.append(fit_dict['chi2'])
                    continuum_local.append(fit_dict['continuum'])
                else:  # If outside of mask set to zero
                    ampls_local.append([0] * len(lines))
                    flux_local.append([0] * len(lines))
                    flux_errs_local.append([0] * len(lines))
                    vels_local.append([0] * len(lines))
                    broads_local.append([0] * len(lines))
                    vels_errs_local.append([0] * len(lines))
                    broads_errs_local.append([0] * len(lines))
                    chi2_local.append(0)
                    continuum_local.append(0)
            ampls_fits[i] = ampls_local
            flux_fits[i] = flux_local
            flux_errors_fits[i] = flux_errs_local
            velocities_fits[i] = vels_local
            broadenings_fits[i] = broads_local
            velocities_errors_fits[i] = vels_errs_local
            broadenings_errors_fits[i] = broads_errs_local
            chi2_fits[i] = chi2_local
            continuum_fits[i] = continuum_local
            return i, ampls_fit, flux_fit, flux_errs_fit, vels_fit, vels_errs_fit, broads_fit, broads_errs_fit, chi2_fit, continuum_fit

        # Write outputs (Velocity, Broadening, and Amplitudes)
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
        for step_i in tqdm(range(y_max - y_min)):
            fit_calc(step_i, ampls_fits, flux_fits, flux_errors_fits, velocities_fits, velocities_errors_fits,
                     broadenings_fits, broadenings_errors_fits, chi2_fits, continuum_fits)
        save_fits(self.output_dir, self.object_name, lines, ampls_fits, flux_fits, flux_errors_fits, velocities_fits,
                  broadenings_fits,
                  velocities_errors_fits, broadenings_errors_fits, chi2_fits, continuum_fits,
                  cutout.wcs.to_header(), binning)
        return velocities_fits, broadenings_fits, flux_fits, chi2_fits, mask

    def fit_pixel(self, lines, fit_function, vel_rel, sigma_rel,
                  pixel_x, pixel_y, bin=None, bkg=None,
                  bayes_bool=False, bayes_method='emcee',
                  uncertainty_bool=False,
                  nii_cons=True):
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
                  mdn=self.mdn, nii_cons=nii_cons)
        fit_dict = fit.fit()
        return axis, sky, fit_dict

    def bin_cube(self, cube_final, header, binning, x_min, x_max, y_min, y_max):
        self.header_binned, self.cube_binned = bin_cube_function(cube_final, header, binning, x_min, x_max, y_min, y_max)


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
            # shape = (2064, 2048)  # (self.header["NAXIS1"], self.header["NAXIS2"])  # Get the shape
            ##r = pyregion.open(region).as_imagecoord(self.header)  # Obtain pyregion region
            # mask = r.get_mask(shape=shape).T  # Calculate mask from pyregion region
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

    def fit_spectrum_region(self, lines, fit_function, vel_rel, sigma_rel, initial_conditions,
                            region, bkg=None,
                            bayes_bool=False, bayes_method='emcee',
                            uncertainty_bool=False, mean=False, nii_cons=True
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
            initial_conditions : Initial conditions of velocity and broadening for fitting specific lines
            region: Name of ds9 region file (e.x. 'region.reg'). You can also pass a boolean mask array.
            bkg: Background Spectrum (1D numpy array; default None)
            bayes_bool: Boolean to determine whether or not to run Bayesian analysis
            bayes_method: Bayesian Inference method. Options are '[emcee', 'dynesty'] (default 'emcee')
            uncertainty_bool: Boolean to determine whether or not to run the uncertainty analysis (default False)
            mean: Boolean to determine whether or not the mean spectrum is taken. This is used for calculating background spectra.
            nii_cons: Boolean to turn on or off NII doublet ratio constraint (default True)

        Return:
            X-axis and spectral axis of region.

        """
        # Create mask
        mask = None  # Initialize
        if '.reg' in region:
            mask = reg_to_mask(region, self.header)
            # shape = (2064, 2048)  # (self.header["NAXIS1"], self.header["NAXIS2"])  # Get the shape
            # r = pyregion.open(region).as_imagecoord(self.header)  # Obtain pyregion region
            # mask = r.get_mask(shape=shape).T  # Calculate mask from pyregion region
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
            integrated_spectrum -= bkg  # Subtract background spectrum
        good_sky_inds = [~np.isnan(integrated_spectrum)]  # Clean up spectrum
        sky = integrated_spectrum[good_sky_inds]
        axis = self.spectrum_axis[good_sky_inds]
        # Call fit!
        if initial_conditions != False:
            fit = Fit(sky, axis, self.wavenumbers_syn, fit_function, lines, vel_rel, sigma_rel,
                  ML_model=None, trans_filter=self.transmission_interpolated,
                  theta=self.interferometer_theta[x_pix, y_pix],
                  delta_x=self.hdr_dict['STEP'], n_steps=self.step_nb,
                  zpd_index=self.zpd_index,
                  filter=self.hdr_dict['FILTER'],
                  bayes_bool=bayes_bool, bayes_method=bayes_method,
                  uncertainty_bool=uncertainty_bool, nii_cons=nii_cons,
                  mdn=self.mdn, initial_values=initial_conditions)
        else:
            fit = Fit(sky, axis, self.wavenumbers_syn, fit_function, lines, vel_rel, sigma_rel,
                  self.model_ML, trans_filter=self.transmission_interpolated,
                  theta=self.interferometer_theta[x_pix, y_pix],
                  delta_x=self.hdr_dict['STEP'], n_steps=self.step_nb,
                  zpd_index=self.zpd_index,
                  filter=self.hdr_dict['FILTER'],
                  bayes_bool=bayes_bool, bayes_method=bayes_method,
                  uncertainty_bool=uncertainty_bool, nii_cons=nii_cons,
                  mdn=self.mdn, initial_values=initial_conditions)
        fit_dict = fit.fit()
        return axis, sky, fit_dict

    def create_snr_map(self, x_min=0, x_max=2048, y_min=0, y_max=2064, method=1, position=1, n_threads=2, bkg=None):
        """
        Create signal-to-noise ratio (SNR) map of a given region. If no bounds are given,
        a map of the entire cube is calculated.

        Args:
            x_min: Minimal X value (default 0)
            x_max: Maximal X value (default 2048)
            y_min: Minimal Y value (default 0)
            y_max: Maximal Y value (default 2064)
            method: Method used to calculate SNR (default 1; options 1 or 2)
            position: Spectral range used to calculate SNR (default 1; options 1 [NII and Halpha group] or 2[SII])
            n_threads: Number of threads to use
            bkg: Background Spectrum (1D numpy array; default None)
        Return:
            snr_map: Signal-to-Noise ratio map

        """
        # Calculate bounds for SNR calculation
        create_snr_map_function(self.cube_final, self.spectrum_axis, self.hdr_dict, self.output_dir, self.object_name, self.header, x_min, x_max, y_min, y_max, method, n_threads)
        return None

    def calculate_snr_region(self, region, bkg, method):
        snr = calculate_snr_region_function(self.cube_final, self.spectrum_axis, self.hdr_dict, self.header, region=region, bkg=bkg, method=method)
        return snr

    def update_astrometry(self, api_key):
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
            print('Astronomy.net failed to solve. This astrometry has not been updated!')

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
        # Read in sky lines
        sky_lines_df = pandas.read_csv('Data/sky_lines.csv', skiprows=2)
        sky_lines = sky_lines_df['Wavelength']  # Get wavelengths
        sky_lines = [sky_line / 10 for sky_line in sky_lines]  # Convert from angstroms to nanometers
        # Create skyline dictionary
        sky_line_dict = {}  # {OH_num: wavelength in nm}
        for line_ct, line_wvl in enumerate(sky_lines):
            sky_line_dict['OH_%i' % line_ct] = line_wvl
        # Calculate grid
        x_min = 0
        x_max = self.cube_final.shape[0]
        x_step = int(
            (x_max - x_min) / n_grid)  # Calculate step size based on min and max values and the number of grid points
        y_min = 0
        y_max = self.cube_final.shape[1]
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
                fit = Fit(sky, axis, self.wavenumbers_syn, 'sinc', ['OH_%i' % num for num in len(sky_lines)], [1], [1],
                          self.model_ML, trans_filter=self.transmission_interpolated,
                          theta=self.interferometer_theta[x_center, y_center],
                          delta_x=self.hdr_dict['STEP'], n_steps=self.step_nb,
                          zpd_index=self.zpd_index,
                          filter=self.hdr_dict['FILTER'], bayes_bool=True, bayes_method='emcee', sky_lines=sky_line_dict
                          )

                velocity, fit_vector = fit.fit(sky_line=True)
                vel_grid[x_grid, y_grid] = float(velocity)
        # Now that we have the grid, we need to reproject it onto the original pixel grid
        print(vel_grid)
        vel_grid_final = np.zeros((x_max, y_max))
        for x_grid in range(n_grid):  # Step through x steps
            for y_grid in range(n_grid):  # Step through y steps
                # Collect spectrum in 10x10 region
                x_center = x_min + int(0.5 * (x_step) * (x_grid + 1))
                y_center = y_min + int(0.5 * (y_step) * (y_grid + 1))
                vel_grid_final[x_center - x_step:x_center + x_step, y_center - y_step:y_center + y_step] = vel_grid[
                    x_grid, y_grid]
        fits.writeto(self.output_dir + '/velocity_correction.fits', vel_grid, self.header, overwrite=True)

    def calculate_component_map(self, x_min=0, x_max=2048, y_min=0, y_max=2064, bkg=None, n_threads=2, region=None):
        # TODO: ADD Documentation and example
        return create_component_map_function(self.header, self.hdr_dict, self.Luci_path, self.resolution, self.filter,
                                      self.cube_final, self.spectrum_axis,
                                      self.wavenumbers_syn_full, self.output_dir, self.object_name, x_min, x_max, y_min,
                                      y_max, bkg,
                                      n_threads, region)

    def calculate_components_in_region(self, region, bkg):
        # TODO: ADD Documentation and example
        return calculate_components_in_region_function(self.header, self.hdr_dict, self.Luci_path, self.resolution, self.filter,
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

    def check_luci_path(self):
        """
        Functionality to check that the user has included the trailing "/" to Luci_path.
        If they have not, we add it.
        TODO: TEST
        """
        if not self.Luci_path.endswith('/'):
            self.Luci_path += '/'
            print("We have added a trailing '/' to your Luci_path variable.\n")
            print("Please add this in the future.\n")

    def create_wvt(self, x_min_init, x_max_init, y_min_init, y_max_init, pixel_size, StN_target, roundness_crit, ToL, bkg=None):

        """
        Functionality to create a weighted Voronoi tesselation map from a region and according to 
        arguments passed by the user. It creates a folder containing all the Voronoi bins that can
        then be used for the fitting procedure. 

        Args:
            x_min_init: Minimal X value
            x_max_init: Maximal X value
            y_min_init: Minimal Y value
            y_max_init: Maximal Y value
            pixel_size: Pixel size of the image. For SITELLE use pixel_size = 0.0000436.
            StN_target: Signal-to-Noise target value for the Voronoi bins.
            roundness_crit: Roundness criteria for the pixel accretion into bins
            ToL: Convergence tolerance parameter for the SNR of the bins
            bkg: Background Spectrum (1D numpy array; default None)
        """
        print("#----------------WVT Algorithm----------------#")
        Pixels = []
        self.create_snr_map(x_min_init, x_max_init, y_min_init, y_max_init, method=2, position=1, n_threads=4, bkg=bkg)
        print("#----------------Algorithm Part 1----------------#")
        SNR_map = fits.open(self.output_dir + '/' + self.object_name + '_SNR.fits')[0].data
        SNR_map = SNR_map[y_min_init:y_max_init, x_min_init:x_max_init]
        fits.writeto(self.output_dir + '/' + self.object_name + '_SNR.fits', SNR_map, overwrite=True)
        Pixels, x_min, x_max, y_min, y_max = read_in(self.output_dir + '/' + self.object_name + '_SNR.fits')
        Nearest_Neighbors(Pixels)
        Init_bins = Bin_Acc(Pixels, pixel_size, StN_target, roundness_crit)
        plot_Bins(Init_bins, x_min, x_max, y_min, y_max, StN_target, self.output_dir, "bin_acc")
        print("#----------------Algorithm Part 2----------------#")
        Final_Bins = WVT(Init_bins, Pixels, StN_target, ToL, pixel_size, self.output_dir)
        print("#----------------Algorithm Complete--------------#")
        plot_Bins(Final_Bins, x_min, x_max, y_min, y_max, StN_target, self.output_dir, "final")
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
                mean=False, n_threads=1):
        """
        Function that takes the wvt mapping created using `self.create_wvt()` and fits the bins.

        Args:

    def fit_wvt(self, x_min, x_max, y_min, y_max, lines, fit_function, vel_rel, sigma_rel, bkg=None, bayes_bool=False, uncertainty_bool=False, mean=False, n_threads=1, initial_values = False):
        """
        Functionality to fit the weighted Voronoi tesselation bins created with the create_wvt() function.

        Args:
            x_min_init: Minimal X value
            x_max_init: Maximal X value
            y_min_init: Minimal Y value
            y_max_init: Maximal Y value
            lines: Lines to fit (e.x. ['Halpha', 'NII6583'])
            fit_function: Fitting function to use (e.x. 'gaussian')
            vel_rel: Constraints on Velocity/Position (must be list; e.x. [1, 2, 1])
            sigma_rel: Constraints on sigma (must be list; e.x. [1, 2, 1])
            bkg: Background Spectrum (1D numpy array; default None)
            bayes_bool: Boolean to determine whether or not to run Bayesian analysis
            uncertainty_bool: Boolean to determine whether or not to run the uncertainty analysis (default False)
            mean: Boolean to determine whether or not the mean spectrum is taken. This is used for calculating background spectra.
            n_treads: Number of threads to use
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
        component_fits = np.zeros((x_max - x_min, y_max - y_min), dtype=np.float32).T
        component_prob_fits = np.zeros((x_max - x_min, y_max - y_min), dtype=np.float32).T
        # First two dimensions are the X and Y dimensions.
        #The third dimension corresponds to the line in the order of the lines input parameter.
        ampls_fits = np.zeros((x_max-x_min, y_max-y_min, len(lines)), dtype=np.float32).transpose(1,0,2)
        flux_fits = np.zeros((x_max-x_min, y_max-y_min, len(lines)), dtype=np.float32).transpose(1,0,2)
        flux_errors_fits = np.zeros((x_max-x_min, y_max-y_min, len(lines)), dtype=np.float32).transpose(1,0,2)
        velocities_fits = np.zeros((x_max-x_min, y_max-y_min, len(lines)), dtype=np.float32).transpose(1,0,2)
        broadenings_fits = np.zeros((x_max-x_min, y_max-y_min, len(lines)), dtype=np.float32).transpose(1,0,2)
        velocities_errors_fits = np.zeros((x_max-x_min, y_max-y_min, len(lines)), dtype=np.float32).transpose(1,0,2)
        broadenings_errors_fits = np.zeros((x_max-x_min, y_max-y_min, len(lines)), dtype=np.float32).transpose(1,0,2)
        continuum_fits = np.zeros((x_max-x_min, y_max-y_min), dtype=np.float32).T
        # TODO: ALLOW BINNING OF INITIAL CONDITIONS
        if initial_values is not False:
            # Obtain initial condition maps from files
            vel_init = fits.open(initial_values[0])[0].data
            broad_init = fits.open(initial_values[1])[0].data

        ct = 0
        set_num_threads(n_threads)
        if not os.path.exists(self.output_dir + '/' + self.object_name + '_deep.fits'):
            self.create_deep_image()
        wcs = WCS(self.header, naxis=2)
        cutout = Cutout2D(fits.open(self.output_dir+'/'+self.object_name+'_deep.fits')[0].data, position=((x_max+x_min)/2, (y_max+y_min)/2), size=(x_max-x_min, y_max-y_min), wcs=wcs)
        for bin_num in tqdm(list(range(len(os.listdir(self.output_dir+'/Numpy_Voronoi_Bins/'))))):
            #print("We're at bin number : ", bin_num)
            bool_bin_map = self.output_dir+'/Numpy_Voronoi_Bins/bool_bin_map_%i.npy'%bin_num            
            index = np.where(np.load(bool_bin_map) == True)
            for a, b in zip(index[0], index[1]):
                # TODO: PASS INITIAL CONDITIONS
                if initial_values is not False:  # If initial conditions were passed
                    initial_conditions = [vel_init[a, b], broad_init[a, b]]
                else:
                    initial_conditions = False
            bin_axis, bin_sky, bin_fit_dict = self.fit_spectrum_region(lines, fit_function, vel_rel, sigma_rel, initial_conditions, region= bool_bin_map, bkg=bkg, bayes_bool=False, uncertainty_bool=False, mean=True)
            for a, b in zip(index[0], index[1]):
                ampls_fits[a,b] = bin_fit_dict['amplitudes']
                flux_fits[a,b] = bin_fit_dict['fluxes']
                flux_errors_fits[a,b] = bin_fit_dict['flux_errors']
                broadenings_fits[a,b] = bin_fit_dict['sigmas']
                broadenings_errors_fits[a,b] = bin_fit_dict['sigmas_errors']
                chi2_fits[a,b] = bin_fit_dict['chi2']
                continuum_fits[a,b] = bin_fit_dict['continuum']
                velocities_fits[a,b] = bin_fit_dict['velocities']
                velocities_errors_fits[a,b] = bin_fit_dict['vels_errors']
        self.save_fits(lines, ampls_fits, flux_fits, flux_errors_fits, velocities_fits, broadenings_fits, velocities_errors_fits, broadenings_errors_fits, chi2_fits, continuum_fits, cutout.wcs.to_header(), binning = 1)
        return velocities_fits, broadenings_fits, flux_fits, chi2_fits, cutout.wcs.to_header()

    def wvt_fit_region(self, x_min_init, x_max_init, y_min_init, y_max_init, lines, fit_function, vel_rel, sigma_rel, pixel_size, StN_target, roundness_crit, ToL, bkg=None, bayes_bool=False, uncertainty_bool=False, mean=False, n_threads=1, initial_values = False):
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
            pixel_size: Pixel size of the image. For SITELLE use pixel_size = 0.0000436.
            StN_target: Signal-to-Noise target value for the Voronoi bins.
            roundness_crit: Roundness criteria for the pixel accretion into bins
            ToL: Convergence tolerance parameter for the SNR of the bins
            bkg: Background Spectrum (1D numpy array; default None)
            bayes_bool: Boolean to determine whether or not to run Bayesian analysis
            uncertainty_bool: Boolean to determine whether or not to run the uncertainty analysis (default False)
            mean: Boolean to determine whether or not the mean spectrum is taken. This is used for calculating background spectra.
            n_treads: Number of threads to use
            initial_values: Initial values of velocity and broadening for fitting specific lines (must be list;
            e.x. [velocity, broadening])
        Return:
            Velocity, Broadening and Flux arrays (2d). Also return amplitudes array (3D).
        """
        self.create_wvt(x_min_init, x_max_init, y_min_init, y_max_init, pixel_size, StN_target, roundness_crit, ToL, bkg=bkg)
        print("#----------------WVT Fitting--------------#")
        velocities_fits, broadenings_fits, flux_fits, chi2_fits, header = self.fit_wvt(x_min_init, x_max_init, y_min_init, y_max_init, lines, fit_function, vel_rel, sigma_rel, bkg=None, bayes_bool=False, uncertainty_bool=False, mean=False, n_threads=1, initial_values)
        output_name = self.object_name
        for line_ in lines:
            amp = fits.open(self.output_dir + '/Amplitudes/' + output_name + '_' + line_ + '_Amplitude.fits')[0].data.T
            flux = fits.open(self.output_dir + '/Fluxes/' + output_name + '_' + line_ + '_Flux.fits')[0].data.T
            flux_err = fits.open(self.output_dir + '/Fluxes/' + output_name + '_' + line_ + '_Flux_err.fits')[0].data.T
            vel = fits.open(self.output_dir + '/Velocity/' + output_name + '_' + line_ + '_velocity.fits')[0].data.T
            broad = fits.open(self.output_dir + '/Broadening/' + output_name + '_' + line_ + '_broadening.fits')[0].data.T
            vel_err = fits.open(self.output_dir + '/Velocity/' + output_name + '_' + line_ + '_velocity_err.fits')[0].data.T
            broad_err = fits.open(self.output_dir + '/Broadening/' + output_name + '_' + line_ + '_broadening_err.fits')[0].data.T
            chi2 = fits.open(output_name + '_Chi2.fits')[0].data.T
            cont = fits.open(output_name + '_continuum.fits')[0].data.T
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
            fits.writeto(output_name + '_Chi2.fits', chi2, header, overwrite=True)
            fits.writeto(output_name + '_continuum.fits', cont, header, overwrite=True)
        return None
    
