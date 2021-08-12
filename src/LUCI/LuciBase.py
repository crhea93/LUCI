from astropy.io import fits
import h5py
import os
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from tqdm import tqdm
import numpy as np
import keras
import pyregion
from scipy import interpolate
import time
from joblib import Parallel, delayed
from LUCI.LuciFit import Fit
import matplotlib.pyplot as plt
from astropy.nddata import Cutout2D
from astroquery.astrometry_net import AstrometryNet
from astropy.io import fits


class Luci():
    """
    This is the primary class for the general purpose line fitting code LUCI. This contains
    all io/administrative functionality. The fitting functionality can be found in the
    Fit class (Lucifit.py).
    """
    def __init__(self, Luci_path, cube_path, output_dir, object_name, redshift, resolution, ML_bool=True):#ref_spec=None, model_ML_name=None):
        """
        Initialize our Luci class -- this acts similar to the SpectralCube class
        of astropy or spectral-cube.

        Args:
            Luci_path: Path to Luci
            cube_path: Full path to hdf5 cube with the hdf5 extension (e.x. '/user/home/M87.hdf5')
            output_dir: Full path to output directory
            object_name: Name of the object to fit. This is used for naming purposes. (e.x. 'M87')
            redshift: Redshift to the object. (e.x. 0.00428)
            resolution: Resolution requested of machine learning algorithm reference spectrum
            ML_bool: Boolean for applying machine learning; default=True
            ref_spec: Name of reference spectrum for machine learning algo (e.x. 'Reference-Spectrum-R5000')
            model_ML_name: Name of pretrained machine learning model
        """
        self.Luci_path = Luci_path
        self.cube_path = cube_path
        self.output_dir = output_dir+'/Luci'
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.object_name = object_name
        self.redshift = redshift
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
        self.hdr_dict = None
        self.interferometer_theta = None
        self.transmission_interpolated = None
        self.read_in_cube()
        self.spectrum_axis_func()
        if ML_bool is True:
            if self.hdr_dict['FILTER'] == 'SN1':
                self.ref_spec = self.Luci_path+'ML/Reference-Spectrum-R%i-SN1.fits'%(resolution)
                self.model_ML = keras.models.load_model(self.Luci_path+'ML/R%i-PREDICTOR-I-SN1'%(resolution))
            elif self.hdr_dict['FILTER'] == 'SN2':
                self.ref_spec = self.Luci_path+'ML/Reference-Spectrum-R%i-SN2.fits'%(resolution)
                self.model_ML = keras.models.load_model(self.Luci_path+'ML/R%i-PREDICTOR-I-SN2'%(resolution))
            elif self.hdr_dict['FILTER'] == 'SN3':
                self.ref_spec = self.Luci_path+'ML/Reference-Spectrum-R%i.fits'%(resolution)
                self.model_ML = keras.models.load_model(self.Luci_path+'ML/R%i-PREDICTOR-I'%(resolution))
            else:
                print('LUCI does not support machine learning parameter estimates for the filter you entered. Please set ML_bool=False.')
            self.read_in_reference_spectrum()
        else:
            self.model_ML = None
        self.read_in_transmission()


    def get_quadrant_dims(self, quad_number):
        """
        Calculate the x and y limits of a given quadrant in the HDF5 file. The
        data cube is saved in 9 individual arrays in the original HDF5 cube. This
        function gets the bouunds for each quadrant.

        Args:
            quad_number: Quadrant Number
        Return:
            x_min, x_max, y_min, y_max: Spatial bounds of quadrant
        """
        div_nb = int(np.sqrt(self.quad_nb))
        if (quad_number < 0) or (quad_number > self.quad_nb - 1):
            raise StandardError("quad_number out of bounds [0," + str(self.quad_nb- 1) + "]")
            return "SOMETHING FAILED"

        index_x = quad_number % div_nb
        index_y = (quad_number - index_x) / div_nb

        x_min = index_x * np.floor(self.dimx / div_nb)
        if (index_x != div_nb - 1):
            x_max = (index_x  + 1) * np.floor(self.dimx / div_nb)
        else:
            x_max = self.dimx

        y_min = index_y * np.floor(self.dimy / div_nb)
        if (index_y != div_nb - 1):
            y_max = (index_y  + 1) * np.floor(self.dimy / div_nb)
        else:
            y_max = self.dimy
        return int(x_min), int(x_max), int(y_min), int(y_max)


    def get_interferometer_angles(self, file):
        """
        Calculate the interferometer angle 2d array for the entire cube. We use
        the following equation:
        cos(theta) = lambda_ref/lambda
        where lambda_ref is the reference laser wavelength and lambda is the measured calibration laser wavelength.

        Args:
            file: hdf5 File object containing HDF5 file
        """
        calib_map = file['calib_map'][()]
        calib_ref = self.hdr_dict['CALIBNM']
        interferometer_cos_theta = calib_ref/calib_map#.T[::-1,::-1]
        # We need to convert to degree so bear with me here
        self.interferometer_theta = np.rad2deg(np.arccos(interferometer_cos_theta))


    def update_header(self, file):
        """
        Create a standard WCS header from the HDF5 header. To do this we clean up the
        header data (which is initially stored in individual arrays). We then create
        a new header dictionary with the old cleaned header info. Finally, we use
        astropy.wcs.WCS to create an updated WCS header for the 2 spatial dimensions.
        This is then saved to self.header while the header dictionary is saved
        as self.hdr_dict.

        Args:
            file: hdf5 File object containing HDF5 file
        """
        hdr_dict = {}
        header_cols = [str(val[0]).replace("'b",'').replace("'", "").replace("b",'') for val in list(file['header'][()])]
        header_vals = [str(val[1]).replace("'b",'').replace("'", "").replace("b",'') for val in list(file['header'][()])]
        header_types = [val[3] for val in list(file['header'][()])]
        for header_col, header_val, header_type in zip(header_cols,header_vals, header_types):
            if 'bool' in str(header_type):
                hdr_dict[header_col] = bool(header_val)
            if 'float' in str(header_type):
                hdr_dict[header_col] = float(header_val)
            if 'int' in str(header_type):
                hdr_dict[header_col] = int(header_val)
            else:
                try:
                    hdr_dict[header_col] = float(header_val)
                except:
                    hdr_dict[header_col] = str(header_val)
        hdr_dict['CTYPE3'] = 'WAVE-SIP'
        hdr_dict['CUNIT3'] = 'm'
        #hdr_dict['NAXIS1'] = 2064
        #hdr_dict['NAXIS2'] = 2048
        # Make WCS
        wcs_data = WCS(hdr_dict, naxis=2)
        self.header = wcs_data.to_header()
        self.header.insert('WCSAXES', ('SIMPLE', 'T'))
        self.header.insert('SIMPLE', ('NAXIS', 2), after=True)
        #self.header.insert('NAXIS', ('NAXIS1', 2064), after=True)
        #self.header.insert('NAXIS1', ('NAXIS2', 2048), after=True)
        self.hdr_dict = hdr_dict


    def create_deep_image(self, output_name=None):
        """
        Create deep image fits file of the cube. This takes the cube and sums
        the spectral axis. Then the deep image is saved as a fits file with the following
        naming convention: output_dir+'/'+object_name+'_deep.fits'
        """
        hdu = fits.PrimaryHDU()
        self.deep_image = np.sum(self.cube_final, axis=2).T
        if output_name == None:
            output_name = self.output_dir+'/'+self.object_name+'_deep.fits'
        fits.writeto(output_name, self.deep_image, self.header, overwrite=True)

    def read_in_cube(self):
        """
        Function to read the hdf5 data into a 3d numpy array (data cube). We also
        translate the header to standard wcs format by calling the update_header function.
        """
        print('Reading in data...')
        file =  h5py.File(self.cube_path+'.hdf5', 'r')  # Read in file
        self.quad_nb = file.attrs['quad_nb']  # Get the number of quadrants
        self.dimx = file.attrs['dimx']  # Get the dimensions in x
        self.dimy = file.attrs['dimy']  # Get the dimensions in y
        self.dimz = file.attrs['dimz']  # Get the dimensions in z (spectral axis)
        self.cube_final = np.zeros((self.dimx, self.dimy, self.dimz))  # Complete data cube
        for iquad in tqdm(range(self.quad_nb)):
            xmin,xmax,ymin,ymax = self.get_quadrant_dims(iquad)
            iquad_data = file['quad00%i'%iquad]['data'][:]  # Save data to intermediate array
            iquad_data[(np.isfinite(iquad_data) == False)]= 1e-22 # Modifs
            iquad_data[(iquad_data < -1e-16)]= -1e-22 # Modifs
            iquad_data[(iquad_data > 1e-9)]= 1e-22 # Modifs
            self.cube_final[xmin:xmax, ymin:ymax, :] = iquad_data  # Save to correct location in main cube
        self.cube_final = self.cube_final#.transpose(1, 0, 2)
        self.update_header(file)
        self.get_interferometer_angles(file)

    def spectrum_axis_func(self):
        """
        Create the x-axis for the spectra. We must construct this from header information
        since each pixel only has amplitudes of the spectra at each point.
        """
        len_wl = self.hdr_dict['STEPNB']    # Length of Spectral Axis
        start = self.hdr_dict['CRVAL3']  # Starting value of the spectral x-axis
        end = start + (len_wl)*self.hdr_dict['CDELT3']  # End
        step = self.hdr_dict['CDELT3']  # Step size
        self.spectrum_axis = np.array(np.linspace(start, end, len_wl)*(self.redshift+1), dtype=np.float32)  # Apply redshift correction
        self.spectrum_axis_unshifted = np.array(np.linspace(start, end, len_wl), dtype=np.float32)  # Do not apply redshift correction

        #min_ = 1e7  * (self.hdr_dict['ORDER'] / (2*self.hdr_dict['STEP']))# + 1e7  / (2*self.delta_x*self.n_steps)
        #max_ = 1e7  * ((self.hdr_dict['ORDER'] + 1) / (2*self.hdr_dict['STEP']))# - 1e7  / (2*self.delta_x*self.n_steps)
        #step_ = max_ - min_
        #axis = np.array([min_+j*step_/self.hdr_dict['STEPNB'] for j in range(self.hdr_dict['STEPNB'])])
        #self.spectrum_axis = axis#*(1+self.redshift)
        #self.spectrum_axis_unshifted = axis


    def read_in_reference_spectrum(self):
        """
        Read in the reference spectrum that will be used in the machine learning
        algorithm to interpolate the true spectra so that they
        wil all have the same size (required for our CNN). The reference spectrum
        will be saved as self.wavenumbers_syn [cm-1].
        """
        ref_spec = fits.open(self.ref_spec)[1].data
        channel = []
        counts = []
        for chan in ref_spec:  # Only want SN3 region
            channel.append(chan[0])
            counts.append(np.real(chan[1]))
        if self.hdr_dict['FILTER'] == 'SN3':
            min_ = np.argmin(np.abs(np.array(channel)-14700))
            max_ = np.argmin(np.abs(np.array(channel)-15600))
        elif self.hdr_dict['FILTER'] == 'SN2':
            min_ = np.argmin(np.abs(np.array(channel)-19000))
            max_ = np.argmin(np.abs(np.array(channel)-21000))
        elif self.hdr_dict['FILTER'] == 'SN1':
            min_ = np.argmin(np.abs(np.array(channel)-25500))
            max_ = np.argmin(np.abs(np.array(channel)-27500))
        else:
            print('We do not support this filter.')
            print('Terminating program!')
            exit()
        self.wavenumbers_syn = np.array(channel[min_:max_], dtype=np.float32)


    def read_in_transmission(self):
        """
        Read in the transmission spectrum for the filter. Then apply interpolation
        on it to make it have the same x-axis as the spectra.
        """
        transmission = np.loadtxt('%s/Data/%s_filter.dat'%(self.Luci_path, self.hdr_dict['FILTER']))  # first column - axis; second column - value
        f = interpolate.interp1d(transmission[:,0], [val/100 for val in transmission[:,1]], kind='slinear', fill_value="extrapolate")
        self.transmission_interpolated = f(self.spectrum_axis_unshifted)

    def bin_cube(self, binning, x_min, x_max, y_min, y_max):
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
        x_shape_new = int((x_max-x_min)/binning)
        y_shape_new = int((y_max-y_min)/binning)
        binned_cube = np.zeros((x_shape_new, y_shape_new, self.cube_final.shape[2]))
        for i in range(x_shape_new):
            for j in range(y_shape_new):
                summed_spec = self.cube_final[x_min+int(i*binning):x_min+int((i+1)*binning), y_min+int(j*binning):y_min+int((j+1)*binning), :]
                summed_spec = np.nansum(summed_spec, axis=0)
                summed_spec = np.nansum(summed_spec, axis=0)
                binned_cube[i,j] = summed_spec[:]
        self.header_binned = self.header
        self.header_binned['CRPIX1'] = self.header_binned['CRPIX1']/binning
        self.header_binned['CRPIX2'] = self.header_binned['CRPIX2']/binning
        self.header_binned['CDELT1'] = self.header_binned['CDELT1']*binning
        self.header_binned['CDELT2'] = self.header_binned['CDELT2']*binning
        self.cube_binned = binned_cube / (binning**2)





    def save_fits(self, lines, ampls_fits, flux_fits, velocities_fits, broadenings_fits, velocities_errors_fits, broadenings_errors_fits, chi2_fits, continuum_fits, header, binning):
        """
        Function to save the fits files returned from the fitting routine. We save the velocity, broadening,
        amplitude, flux, and chi-squared maps with the appropriate headers in the output directory
        defined when the cube is initiated.

        Args:
            lines: Lines to fit (e.x. ['Halpha', 'NII6583'])
            ampls_fits: 3D Numpy array of amplitude values
            flux_fis: 3D Numpy array of flux values
            velocities_fits: 3D Numpy array of velocity values
            broadenings_fits: 3D Numpy array of broadening values
            velocities_errors_fits: 3D Numpy array of velocity errors
            broadenings_errors_fits: 3D Numpy array of broadening errors
            chi2_fits: 2D Numpy array of chi-squared values
            continuum_fits: 2D Numpy array of continuum value
            header: Header object (either binned or unbinned)
            output_name: Output directory and naming convention
            binning: Value by which to bin (default None)

        """
        # Make sure output dirs exist for amps, flux, vel, and broad
        if not os.path.exists(self.output_dir+'/Amplitudes'):
            os.mkdir(self.output_dir+'/Amplitudes')
        if not os.path.exists(self.output_dir+'/Fluxes'):
            os.mkdir(self.output_dir+'/Fluxes')
        if not os.path.exists(self.output_dir+'/Velocity'):
            os.mkdir(self.output_dir+'/Velocity')
        if not os.path.exists(self.output_dir+'/Broadening'):
            os.mkdir(self.output_dir+'/Broadening')

        if binning is not None:
            output_name = self.object_name + "_" + str(binning)
        else:
            output_name = self.object_name
        for ct,line_ in enumerate(lines):  # Step through each line to save their individual amplitudes
            fits.writeto(self.output_dir + '/Amplitudes/'+ output_name+'_'+line_+'_Amplitude.fits', ampls_fits[:,:,ct], header, overwrite=True)
            fits.writeto(self.output_dir + '/Fluxes/'+ output_name +'_'+line_+'_Flux.fits', flux_fits[:,:,ct], header, overwrite=True)
            fits.writeto(self.output_dir + '/Velocity/' + output_name +'_' + line_ +'_velocity.fits', velocities_fits[:,:,ct], header, overwrite=True)
            fits.writeto(self.output_dir + '/Broadening/' + output_name +'_' + line_ +'_broadening.fits', broadenings_fits[:,:,ct], header, overwrite=True)
            fits.writeto(self.output_dir + '/Velocity/' + output_name +'_' + line_ + '_velocity_err.fits', velocities_errors_fits[:,:,ct], header, overwrite=True)
            fits.writeto(self.output_dir + '/Broadening/' + output_name +'_' + line_ + '_broadening_err.fits', broadenings_errors_fits[:,:,ct], header, overwrite=True)
        fits.writeto(output_name+'_Chi2.fits', chi2_fits, header, overwrite=True)
        fits.writeto(output_name+'_continuum.fits', continuum_fits, header, overwrite=True)


    def fit_entire_cube(self, lines, fit_function, vel_rel, sigma_rel):
        """
        Fit the entire cube (all spatial dimensions)
        Args:
            lines: Lines to fit (e.x. ['Halpha', 'NII6583'])
            fit_function: Fitting function to use (e.x. 'gaussian')
            vel_rel: Constraints on Velocity/Position (must be list; e.x. [1, 2, 1])
            sigma_rel: Constraints on sigma (must be list; e.x. [1, 2, 1])
        """
        x_min = 0
        x_max = self.cube_final.shape[0]
        y_min = 0
        y_max = self.cube_final.shape[1]
        self.fit_cube(lines, fit_function,vel_rel, sigma_rel, x_min, x_max, y_min, y_max)


    def fit_cube(self, lines, fit_function, vel_rel, sigma_rel, x_min, x_max, y_min, y_max, bkg=None, binning=None, bayes_bool=False, output_name=None, uncertainty_bool=False):
        """
        Primary fit call to fit rectangular regions in the data cube. This wraps the
        LuciFits.FIT().fit() call which applies all the fitting steps. This also
        saves the velocity and broadening fits files.

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
            output_name: User defined output path/name (default None)
            uncertainty_bool: Boolean to determine whether or not to run the uncertainty analysis (default False)
        Return:
            Velocity and Broadening arrays (2d). Also return amplitudes array (3D).

        Examples:
            As always, we must first have the cube initialized (see basic example).

            If we want to fit all five lines in SN3 with a sincgauss function and binning of 2
            over a rectangular region defined in image coordinates as 800<x<1500; 250<y<1250,
            we would run the following:

            >>> vel_map, broad_map, flux_map, chi2_fits = cube.fit_cube(['Halpha', 'NII6548', 'NII6583', 'SII6716', 'SII6731'], 'sincgauss', 800, 1500, 250, 750, binning=2)

        """
        # Initialize fit solution arrays
        if binning != None and binning > 1:
            self.bin_cube(binning, x_min, x_max, y_min, y_max)
            #x_min_bin = int(x_min/binning) ; y_min_bin = int(y_min/binning) ; x_max_bin = int(x_max/binning) ;  y_max_bin = int(y_max/binning)
            x_max = int((x_max-x_min)/binning) ;  y_max = int((y_max-y_min)/binning)
            x_min = 0 ; y_min = 0
        chi2_fits = np.zeros((x_max-x_min, y_max-y_min), dtype=np.float32).T
        corr_fits = np.zeros((x_max-x_min, y_max-y_min), dtype=np.float32).T
        step_fits = np.zeros((x_max-x_min, y_max-y_min), dtype=np.float32).T
        # First two dimensions are the X and Y dimensions.
        #The third dimension corresponds to the line in the order of the lines input parameter.
        ampls_fits = np.zeros((x_max-x_min, y_max-y_min, len(lines)), dtype=np.float32).transpose(1,0,2)
        flux_fits = np.zeros((x_max-x_min, y_max-y_min, len(lines)), dtype=np.float32).transpose(1,0,2)
        velocities_fits = np.zeros((x_max-x_min, y_max-y_min, len(lines)), dtype=np.float32).transpose(1,0,2)
        broadenings_fits = np.zeros((x_max-x_min, y_max-y_min, len(lines)), dtype=np.float32).transpose(1,0,2)
        velocities_errors_fits = np.zeros((x_max-x_min, y_max-y_min, len(lines)), dtype=np.float32).transpose(1,0,2)
        broadenings_errors_fits = np.zeros((x_max-x_min, y_max-y_min, len(lines)), dtype=np.float32).transpose(1,0,2)
        continuum_fits = np.zeros((x_max-x_min, y_max-y_min), dtype=np.float32).T
        #if output_name == None:
            #output_name = self.output_dir+'/'+self.object_name
        for i in tqdm(range(y_max-y_min)):
            y_pix = y_min + i
            ampls_local = []
            flux_local = []
            vels_local = []
            broads_local = []
            vels_errs_local = []
            broads_errs_local = []
            chi2_local = []
            corr_local = []
            step_local = []
            continuum_local = []
            for j in range(x_max-x_min):
                x_pix = x_min+j
                if binning is not None:
                    sky = self.cube_binned[x_pix, y_pix, :]
                else:
                    sky = self.cube_final[x_pix, y_pix, :]
                if bkg is not None:
                    if binning:
                        sky -= bkg * binning**2  # Subtract background spectrum
                    else:
                        sky -= bkg  # Subtract background spectrum
                good_sky_inds = [~np.isnan(sky)]  # Clean up spectrum
                sky = sky[good_sky_inds]
                axis = self.spectrum_axis[good_sky_inds]
                # Call fit!
                fit = Fit(sky, axis, self.wavenumbers_syn, fit_function, lines, vel_rel, sigma_rel,
                        self.model_ML, trans_filter = self.transmission_interpolated,
                        theta=self.interferometer_theta[x_pix, y_pix],
                        delta_x = self.hdr_dict['STEP'], n_steps = self.hdr_dict['STEPNB'],
                        filter = self.hdr_dict['FILTER'],
                        bayes_bool=bayes_bool, uncertainty_bool=uncertainty_bool)
                fit_dict = fit.fit()
                # Save local list of fit values
                ampls_local.append(fit_dict['amplitudes'])
                flux_local.append(fit_dict['fluxes'])
                vels_local.append(fit_dict['velocities'])
                broads_local.append(fit_dict['sigmas'])
                vels_errs_local.append(fit_dict['vels_errors'])
                broads_errs_local.append(fit_dict['sigmas_errors'])
                chi2_local.append(fit_dict['chi2'])
                corr_local.append(fit_dict['corr'])
                step_local.append(fit_dict['axis_step'])
                continuum_local.append(fit_dict['continuum'])
            # Update global array of fit values
            ampls_fits[i] = ampls_local
            flux_fits[i] = flux_local
            velocities_fits[i] = vels_local
            broadenings_fits[i] = broads_local
            velocities_errors_fits[i] = vels_errs_local
            broadenings_errors_fits[i] = broads_errs_local
            chi2_fits[i] = chi2_local
            corr_fits[i] = corr_local
            step_fits[i] = step_local
            continuum_fits[i] = continuum_local
        # Write outputs (Velocity, Broadening, and Amplitudes)
        if binning is not None:
            # Check if deep image exists: if not, create it
            if not os.path.exists(self.output_dir+'/'+self.object_name+'_deep.fits'):
                self.create_deep_image()
            wcs = WCS(self.header_binned)
        else:
            # Check if deep image exists: if not, create it
            if not os.path.exists(self.output_dir+'/'+self.object_name+'_deep.fits'):
                self.create_deep_image()
            wcs = WCS(self.header, naxis=2)
        cutout = Cutout2D(fits.open(self.output_dir+'/'+self.object_name+'_deep.fits')[0].data, position=((x_max+x_min)/2, (y_max+y_min)/2), size=(x_max-x_min, y_max-y_min), wcs=wcs)
        self.save_fits(lines, ampls_fits, flux_fits, velocities_fits, broadenings_fits, velocities_errors_fits, broadenings_errors_fits, chi2_fits, continuum_fits, cutout.wcs.to_header(), binning)

        return velocities_fits, broadenings_fits, flux_fits, chi2_fits

        #n_threads = 1
        #for i in range(x_max-x_min):
            #SNR_calc(VEL, BROAD, i)
        #Parallel(n_jobs=n_threads, backend="threading", batch_size=int((x_max-x_min)/n_threads))(delayed(SNR_calc)(VEL, BROAD, i) for i in range(x_max-x_min));
        #Parallel(n_jobs=n_threads, backend="threading")(delayed(SNR_calc)(VEL, BROAD, i) for i in tqdm(range(x_max-x_min)));

    def fit_region(self, lines, fit_function, vel_rel, sigma_rel, region, bkg=None, binning=None, bayes_bool=False, output_name=None, uncertainty_bool=False):
        """
        Fit the spectrum in a region. This is an extremely similar command to fit_cube except
        it works for ds9 regions. We first create a mask from the ds9 region file. Then
        we step through the cube and only fit the unmasked pixels. Although this may not
        be the most efficient method, it does ensure the fidelity of the wcs system.

        Args:
            lines: Lines to fit (e.x. ['Halpha', 'NII6583'])
            fit_function: Fitting function to use (e.x. 'gaussian')
            vel_rel: Constraints on Velocity/Position (must be list; e.x. [1, 2])
            sigma_rel: Constraints on sigma (must be list; e.x. [1, 2])
            region: Name of ds9 region file (e.x. 'region.reg'). You can also pass a boolean mask array.
            bkg: Background Spectrum (1D numpy array; default None)
            binning:  Value by which to bin (default None)
            bayes_bool: Boolean to determine whether or not to run Bayesian analysis (default False)
            output_name: User defined output path/name
            uncertainty_bool: Boolean to determine whether or not to run the uncertainty analysis (default False)
        Return:
            Velocity and Broadening arrays (2d). Also return amplitudes array (3D).

        Examples:
            As always, we must first have the cube initialized (see basic example).

            If we want to fit all five lines in SN3 with a gaussian function and no binning
            over a ds9 region called main.reg, we would run the following:

            >>> vel_map, broad_map, flux_map, chi2_fits = cube.fit_region(['Halpha', 'NII6548', 'NII6583', 'SII6716', 'SII6731'], 'gaussian', region='main.reg')

        """
        # Set spatial bounds for entire cube
        x_min = 0
        x_max = self.cube_final.shape[0]
        y_min = 0
        y_max = self.cube_final.shape[1]
        # Initialize fit solution arrays
        if binning != None and binning > 1:
            self.bin_cube(binning, x_min, x_max, y_min, y_max)
            #x_min = int(x_min/binning) ; y_min = int(y_min/binning) ; x_max = int(x_max/binning) ;  y_max = int(y_max/binning)
            x_max = int((x_max-x_min)/binning) ;  y_max = int((y_max-y_min)/binning)
            x_min = 0 ; y_min = 0
        # Create mask
        if '.reg' in region:
            shape = (2064, 2048)#(self.header["NAXIS1"], self.header["NAXIS2"])  # Get the shape
            if binning != None and binning > 1:
                header = self.header_binned
            else:
                header = self.header
            header.set('NAXIS1', 2064)
            header.set('NAXIS2', 2048)
            r = pyregion.open(region).as_imagecoord(header)  # Obtain pyregion region
            mask = r.get_mask(shape=shape).T  # Calculate mask from pyregion region
        else:
            try:  # If numpy .np file
                mask = np.load(region)
            except:
                mask = region
        # Clean up output name
        if isinstance(region, str):
            if len(region.split('/')) > 1:  # If region file is a path, just keep the name for output purposes
                region = region.split('/')[-1]
            if output_name == None:
                output_name = self.output_dir+'/'+self.object_name+'_'+region.split('.')[0]
        else:  # Passed mask not region file
            if output_name == None:
                output_name = self.output_dir+'/'+self.object_name+'_mask'

        chi2_fits = np.zeros((x_max-x_min, y_max-y_min), dtype=np.float32).T
        corr_fits = np.zeros((x_max-x_min, y_max-y_min), dtype=np.float32).T
        step_fits = np.zeros((x_max-x_min, y_max-y_min), dtype=np.float32).T
        # First two dimensions are the X and Y dimensions.
        #The third dimension corresponds to the line in the order of the lines input parameter.
        ampls_fits = np.zeros((x_max-x_min, y_max-y_min, len(lines)), dtype=np.float32).transpose(1,0,2)
        flux_fits = np.zeros((x_max-x_min, y_max-y_min, len(lines)), dtype=np.float32).transpose(1,0,2)
        velocities_fits = np.zeros((x_max-x_min, y_max-y_min, len(lines)), dtype=np.float32).transpose(1,0,2)
        broadenings_fits = np.zeros((x_max-x_min, y_max-y_min, len(lines)), dtype=np.float32).transpose(1,0,2)
        velocities_errors_fits = np.zeros((x_max-x_min, y_max-y_min, len(lines)), dtype=np.float32).transpose(1,0,2)
        broadenings_errors_fits = np.zeros((x_max-x_min, y_max-y_min, len(lines)), dtype=np.float32).transpose(1,0,2)
        continuum_fits = np.zeros((x_max-x_min, y_max-y_min), dtype=np.float32).T
        ct = 0
        for i in tqdm(range(y_max-y_min)):
            y_pix = y_min + i
            ampls_local = []
            flux_local = []
            vels_local = []
            broads_local = []
            vels_errs_local = []
            broads_errs_local = []
            chi2_local = []
            continuum_local = []
            for j in range(x_max-x_min):
                x_pix = x_min+j
                # Check if pixel is in the mask or not
                # If so, fit as normal. Else, set values to zero
                if mask[x_pix, y_pix] == True:
                    if binning is not None:
                        sky = self.cube_binned[x_pix, y_pix, :]
                    else:
                        sky = self.cube_final[x_pix, y_pix, :]
                    if bkg is not None:
                        if binning:
                            sky -= bkg * binning**2  # Subtract background spectrum
                        else:
                            sky -= bkg  # Subtract background spectrum
                    ct += 1
                    good_sky_inds = [~np.isnan(sky)]  # Clean up spectrum
                    sky = sky[good_sky_inds]
                    axis = self.spectrum_axis[good_sky_inds]
                    # Call fit!
                    fit = Fit(sky, axis, self.wavenumbers_syn, fit_function, lines, vel_rel, sigma_rel,
                            self.model_ML, trans_filter = self.transmission_interpolated,
                            theta = self.interferometer_theta[x_pix, y_pix],
                            delta_x = self.hdr_dict['CDELT3'], n_steps = self.hdr_dict['STEPNB'],
                            filter = self.hdr_dict['FILTER'],
                            bayes_bool=bayes_bool, uncertainty_bool=uncertainty_bool)
                    fit_dict = fit.fit()
                    # Save local list of fit values
                    ampls_local.append(fit_dict['amplitudes'])
                    flux_local.append(fit_dict['fluxes'])
                    vels_local.append(fit_dict['velocities'])
                    broads_local.append(fit_dict['sigmas'])
                    vels_errs_local.append(fit_dict['vels_errors'])
                    broads_errs_local.append(fit_dict['sigmas_errors'])
                    chi2_local.append(fit_dict['chi2'])
                    continuum_local.append(fit_dict['continuum'])
                else:  # If outside of mask set to zero
                    ampls_local.append([0]*len(lines))
                    flux_local.append([0]*len(lines))
                    vels_local.append([0]*len(lines))
                    broads_local.append([0]*len(lines))
                    vels_errs_local.append([0]*len(lines))
                    broads_errs_local.append([0]*len(lines))
                    chi2_local.append(0)
                    continuum_local.append(0)
            # Update global array of fit values
            ampls_fits[i] = ampls_local
            flux_fits[i] = flux_local
            velocities_fits[i] = vels_local
            broadenings_fits[i] = broads_local
            velocities_errors_fits[i] = vels_errs_local
            broadenings_errors_fits[i] = broads_errs_local
            chi2_fits[i] = chi2_local
            continuum_fits[i] = continuum_local
        # Write outputs (Velocity, Broadening, and Amplitudes)
        print(ct)
        if binning is not None:
            wcs = WCS(self.header_binned)
        else:
            wcs = WCS(self.header)
        cutout = Cutout2D(velocities_fits[:,:,0], position=((x_max+x_min)/2, (y_max+y_min)/2), size=(x_max-x_min, y_max-y_min), wcs=wcs)
        self.save_fits(lines, ampls_fits, flux_fits, velocities_fits, broadenings_fits, velocities_errors_fits, broadenings_errors_fits, chi2_fits, continuum_fits, cutout.wcs.to_header(), binning)


        return velocities_fits, broadenings_fits, flux_fits, chi2_fits, mask


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
            X-axis and spectral axis of region.

        """
        integrated_spectrum = np.zeros(self.cube_final.shape[2])
        spec_ct = 0
        # Initialize fit solution arrays
        if binning != None:
            self.bin_cube(binning, x_min, x_max, y_min, y_max)
            #x_min = int(x_min/binning) ; y_min = int(y_min/binning) ; x_max = int(x_max/binning) ;  y_max = int(y_max/binning)
            x_max = int((x_max-x_min)/binning) ;  y_max = int((y_max-y_min)/binning)
            x_min = 0 ; y_min = 0
        for i in tqdm(range(y_max-y_min)):
            y_pix = y_min + i
            vel_local = []
            broad_local = []
            ampls_local = []
            flux_local = []
            chi2_local = []
            for j in range(x_max-x_min):
                x_pix = x_min+j
                if binning is not None:
                    sky = self.cube_binned[x_pix, y_pix, :]
                else:
                    sky = self.cube_final[x_pix, y_pix, :]
                if bkg is not None:
                    if binning:
                        sky -= bkg * binning**2  # Subtract background spectrum
                    else:
                        sky -= bkg  # Subtract background spectrum
                good_sky_inds = [~np.isnan(sky)]  # Clean up spectrum
                integrated_spectrum += sky[good_sky_inds]
                if spec_ct == 0:
                    axis = self.spectrum_axis[good_sky_inds]
                    spec_ct +=1
        if mean == True:
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
            shape = (2064, 2048)#(self.header["NAXIS1"], self.header["NAXIS2"])  # Get the shape
            r = pyregion.open(region).as_imagecoord(self.header)  # Obtain pyregion region
            mask = r.get_mask(shape=shape)  # Calculate mask from pyregion region
        else:
            mask = region
        # Set spatial bounds for entire cube
        x_min = 0
        x_max = self.cube_final.shape[0]
        y_min = 0
        y_max = self.cube_final.shape[1]
        integrated_spectrum = np.zeros(self.cube_final.shape[2])
        spec_ct = 0
        for i in tqdm(range(y_max-y_min)):
            y_pix = y_min + i
            for j in range(x_max-x_min):
                x_pix = x_min+j
                # Check if pixel is in the mask or not
                if mask[x_pix, y_pix] == True:
                    integrated_spectrum += self.cube_final[x_pix, y_pix, :]
                    spec_ct += 1
                else:
                    pass
        if mean == True:
            integrated_spectrum /= spec_ct
        return self.spectrum_axis, integrated_spectrum



    def fit_spectrum_region(self, lines, fit_function, vel_rel, sigma_rel, region, bkg=None, bayes_bool=False, uncertainty_bool=False, mean=False):
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
            bkg: Background Spectrum (1D numpy array; default None)
            bayes_bool: Boolean to determine whether or not to run Bayesian analysis

        Return:
            X-axis and spectral axis of region.

        """
        # Create mask
        if '.reg' in region:
            shape = (2064, 2048)#(self.header["NAXIS1"], self.header["NAXIS2"])  # Get the shape
            r = pyregion.open(region).as_imagecoord(self.header)  # Obtain pyregion region
            mask = r.get_mask(shape=shape).T  # Calculate mask from pyregion region
        else:
            mask = region
        # Set spatial bounds for entire cube
        x_min = 0
        x_max = self.cube_final.shape[0]
        y_min = 0
        y_max = self.cube_final.shape[1]
        integrated_spectrum = np.zeros(self.cube_final.shape[2])
        spec_ct = 0
        for i in tqdm(range(y_max-y_min)):
            y_pix = y_min + i
            for j in range(x_max-x_min):
                x_pix = x_min+j
                # Check if pixel is in the mask or not
                if mask[x_pix, y_pix] == True:
                    integrated_spectrum += self.cube_final[x_pix, y_pix, :]
                    spec_ct += 1
                else:
                    pass
        if mean == True:
            integrated_spectrum /= spec_ct
        if bkg is not None:
            integrated_spectrum -= bkg  # Subtract background spectrum
        good_sky_inds = [~np.isnan(integrated_spectrum)]  # Clean up spectrum
        sky = integrated_spectrum[good_sky_inds]
        axis = self.spectrum_axis[good_sky_inds]
        # Call fit!
        fit = Fit(sky, axis, self.wavenumbers_syn, fit_function, lines, vel_rel, sigma_rel,
                self.model_ML, trans_filter = self.transmission_interpolated,
                theta = self.interferometer_theta[x_pix, y_pix],
                delta_x = self.hdr_dict['CDELT3'], n_steps = self.hdr_dict['STEPNB'],
                filter = self.hdr_dict['FILTER'],
                bayes_bool=bayes_bool, uncertainty_bool=uncertainty_bool)
        fit_dict = fit.fit()
        return axis, sky, fit_dict



    def create_snr_map(self, x_min=0, x_max=2048, y_min=0, y_max=2064, method=1):
        """
        Create signal-to-noise ratio (SNR) map of a given region. If no bounds are given,
        a map of the entire cube is calculated.

        Args:
            x_min: Minimal X value (default 0)
            x_max: Maximal X value (default 2048)
            y_min: Minimal Y value (default 0)
            y_max: Maximal Y value (default 2064)
            method: Method used to calculate SNR (default 1; options 1 or 2)
        Return:
            snr_map: Signal-to-Noise ratio map

        """
        # Calculate bounds for SNR calculation
        # Step through spectra
        SNR = np.zeros((x_max-x_min, y_max-y_min), dtype=np.float32)
        #start = time.time()
        #def SNR_calc(SNR, i):
        flux_min = 0 ; flux_max= 0; noise_min = 0; noise_max = 0  # Initializing bounds for flux and noise calculation regions
        if self.hdr_dict['FILTER'] == 'SN3':
            flux_min = 15150; flux_max = 15300; noise_min = 16000; noise_max = 16400
        elif self.hdr_dict['FILTER'] == 'SN1':
            flux_min = 26550; flux_max = 27550; noise_min = 25300; noise_max = 25700
        else:
            print('SNR Calculation for this filter has not been implemented')
        for i in range(y_max-y_min):
            y_pix = y_min + i
            snr_local = []
            for j in range(x_max-x_min):
                x_pix = x_min+j
                # Calculate SNR
                # Select spectral region around Halpha and NII complex
                min_ = np.argmin(np.abs(np.array(self.spectrum_axis)-flux_min))
                max_ = np.argmin(np.abs(np.array(self.spectrum_axis)-flux_max))
                flux_in_region = np.sum(self.cube_final[x_pix, y_pix, min_:max_])
                # Select distance region
                min_ = np.argmin(np.abs(np.array(self.spectrum_axis)-noise_min))
                max_ = np.argmin(np.abs(np.array(self.spectrum_axis)-noise_max))
                std_out_region = np.std(self.cube_final[x_pix, y_pix, min_:max_])
                if method == 1:
                    signal = np.max(flux_in_region)-np.median(flux_in_region)
                    noise = np.std(std_out_region)
                    snr = float(signal / np.sqrt(noise))
                    snr = snr/(np.sqrt(np.mean(flux_in_region)))
                else:
                    snr = float(flux_in_region/std_out_region)
                    snr = snr
                snr_local.append(snr)
            SNR[i] = snr_local
        #n_threads = 2
        #Parallel(n_jobs=n_threads, backend="threading", batch_size=int((x_max-x_min)/n_threads))(delayed(SNR_calc)(SNR,i) for i in tqdm(range(x_max-x_min)));
        #end = time.time()
        # Save
        fits.writeto(self.output_dir+'/'+self.object_name+'_SNR.fits', SNR.T, self.header, overwrite=True)



    def update_astrometry(self, api_key):
        """
        Use astronomy.net to update the astrometry in the header
        If astronomy.net successfully finds the corrected astrononmy, the self.header is updated. Otherwise,
        the header is not updated and an exception is thrown.

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
        if not os.path.exists(self.output_dir+'/'+self.object_name+'_deep.fits'):
            self.create_deep_image()
        # Now submit to astronomy.net until the value is found
        while try_again:
            if not submission_id:
                try:
                    wcs_header = ast.solve_from_image(self.output_dir+'/'+self.object_name+'_deep.fits', submission_id=submission_id, solve_timeout=300)#, use_sextractor=True, center_ra=float(ra), center_dec=float(dec))
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
            deep = fits.open(self.output_dir+'/'+self.object_name+'_deep.fits')
            deep.header.update(wcs_header)
            deep.close()
            # Update normal header
            self.header = wcs_header

        else:
            # Code to execute when solve fails
            print('Astronomy.net failed to solve. This astrometry has not been updated!')
