import os
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from scipy import interpolate


def check_luci_path(Luci_path):
    """
    Functionality to check that the user has included the trailing "/" to Luci_path.
    If they have not, we add it.
    """
    if not Luci_path.endswith('/'):
        Luci_path += '/'
        print("We have added a trailing '/' to your Luci_path variable.\n")
        print("Please add this in the future.\n")


def save_fits(output_dir, object_name, lines, ampls_fits, flux_fits, flux_errors_fits, velocities_fits,
              broadenings_fits,
              velocities_errors_fits, broadenings_errors_fits, chi2_fits, continuum_fits, header, binning, suffix=''):
    """
    Function to save the fits files returned from the fitting routine. We save the velocity, broadening,
    amplitude, flux, and chi-squared maps with the appropriate headers in the output directory
    defined when the cube is initiated.

    Args:
        lines: Lines to fit (e.x. ['Halpha', 'NII6583'])
        ampls_fits: 3D Numpy array of amplitude values
        flux_fis: 3D Numpy array of flux values
        flux_errors_fits 3D numpy array of flux errors
        velocities_fits: 3D Numpy array of velocity values
        broadenings_fits: 3D Numpy array of broadening values
        velocities_errors_fits: 3D Numpy array of velocity errors
        broadenings_errors_fits: 3D Numpy array of broadening errors
        chi2_fits: 2D Numpy array of chi-squared values
        continuum_fits: 2D Numpy array of continuum value
        header: Header object (either binned or unbinned)
        output_name: Output directory and naming convention
        binning: Value by which to bin (default None)
        suffix: Additional suffix to add (e.x. '_wvt')

    """
    # Make sure output dirs exist for amps, flux, vel, and broad
    if not os.path.exists(output_dir + '/Amplitudes'):
        os.mkdir(output_dir + '/Amplitudes')
    if not os.path.exists(output_dir + '/Fluxes'):
        os.mkdir(output_dir + '/Fluxes')
    if not os.path.exists(output_dir + '/Velocity'):
        os.mkdir(output_dir + '/Velocity')
    if not os.path.exists(output_dir + '/Broadening'):
        os.mkdir(output_dir + '/Broadening')
    output_name = object_name + suffix
    if binning is not None:
        output_name += "_" + str(binning)
    lines_fit = []  # List of lines which already have maps
    for ct, line_ in enumerate(lines):  # Step through each line to save their individual amplitudes
        if lines_fit.count(line_) >= 1:  # If the line is already present in the list of lines create
            # This means multiple components were fit of this line so they need to be nammed appropriately
            line_number = lines_fit.count(line_) + 1
            line_ += '_' + str(line_number)
        fits.writeto(output_dir + '/Amplitudes/' + output_name + '_' + line_ + '_Amplitude.fits',
                     ampls_fits[:, :, ct], header, overwrite=True)
        fits.writeto(output_dir + '/Fluxes/' + output_name + '_' + line_ + '_Flux.fits', flux_fits[:, :, ct],
                     header, overwrite=True)
        fits.writeto(output_dir + '/Fluxes/' + output_name + '_' + line_ + '_Flux_err.fits',
                     flux_errors_fits[:, :, ct], header, overwrite=True)
        fits.writeto(output_dir + '/Velocity/' + output_name + '_' + line_ + '_velocity.fits',
                     velocities_fits[:, :, ct], header, overwrite=True)
        fits.writeto(output_dir + '/Broadening/' + output_name + '_' + line_ + '_broadening.fits',
                     broadenings_fits[:, :, ct], header, overwrite=True)
        fits.writeto(output_dir + '/Velocity/' + output_name + '_' + line_ + '_velocity_err.fits',
                     velocities_errors_fits[:, :, ct], header, overwrite=True)
        fits.writeto(output_dir + '/Broadening/' + output_name + '_' + line_ + '_broadening_err.fits',
                     broadenings_errors_fits[:, :, ct], header, overwrite=True)
    fits.writeto(output_dir + '/' + output_name + '_Chi2.fits', chi2_fits, header, overwrite=True)
    fits.writeto(output_dir + '/' + output_name + '_continuum.fits', continuum_fits, header, overwrite=True)


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


def get_interferometer_angles(file, hdr_dict):
    """
    Calculate the interferometer angle 2d array for the entire cube. We use
    the following equation:
    cos(theta) = lambda_ref/lambda
    where lambda_ref is the reference laser wavelength and lambda is the measured calibration laser wavelength.

    Args:
        file: hdf5 File object containing HDF5 file
    """
    calib_map = file['calib_map'][()]
    calib_ref = hdr_dict['CALIBNM']
    interferometer_cos_theta = calib_ref / calib_map  # .T[::-1,::-1]
    # We need to convert to degree so bear with me here
    #del calib_map
    return np.rad2deg(np.arccos(interferometer_cos_theta))
    # self.interferometer_theta = np.rad2deg(np.arccos(interferometer_cos_theta))


def spectrum_axis_func(hdr_dict, redshift):
    """
    Create the x-axis for the spectra. We must construct this from header information
    since each pixel only has amplitudes of the spectra at each point.
    """
    len_wl = hdr_dict['STEPNB']  # Length of Spectral Axis
    start = hdr_dict['CRVAL3']  # Starting value of the spectral x-axis
    end = start + (len_wl) * hdr_dict['CDELT3']  # End
    step = hdr_dict['CDELT3']  # Step size
    spectrum_axis = np.array(np.linspace(start, end, len_wl) * (redshift + 1),
                             dtype=np.float32)  # Apply redshift correction
    spectrum_axis_unshifted = np.array(np.linspace(start, end, len_wl),
                                       dtype=np.float32)  # Do not apply redshift correction

    # min_ = 1e7  * (self.hdr_dict['ORDER'] / (2*self.hdr_dict['STEP']))# + 1e7  / (2*self.delta_x*self.n_steps)
    # max_ = 1e7  * ((self.hdr_dict['ORDER'] + 1) / (2*self.hdr_dict['STEP']))# - 1e7  / (2*self.delta_x*self.n_steps)
    # step_ = max_ - min_
    # axis = np.array([min_+j*step_/self.hdr_dict['STEPNB'] for j in range(self.hdr_dict['STEPNB'])])
    # self.spectrum_axis = axis#*(1+self.redshift)
    # self.spectrum_axis_unshifted = axis
    return spectrum_axis, spectrum_axis_unshifted


def update_header(file):
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
    header_cols = [str(val[0]).replace("'b", '').replace("'", "").replace("b", '') for val in
                   list(file['header'][()])]
    header_vals = [str(val[1]).replace("'b", '').replace("'", "").replace("b", '') for val in
                   list(file['header'][()])]
    header_types = [val[3] for val in list(file['header'][()])]
    for header_col, header_val, header_type in zip(header_cols, header_vals, header_types):
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
    # If NAXIS 1 does not exist we will add it
    if 'NAXIS1' not in hdr_dict.keys():
        hdr_dict['NAXIS1'] = 2048
        hdr_dict['NAXIS2'] = 2064
    # Make WCS
    wcs_data = WCS(hdr_dict, naxis=2)
    header = wcs_data.to_header()
    header.insert('WCSAXES', ('SIMPLE', 'T'))
    header.insert('SIMPLE', ('NAXIS', 2), after=True)
    hdr_dict = hdr_dict
    return header, hdr_dict


def read_in_reference_spectrum(ref_spec, hdr_dict):
    """
    Read in the reference spectrum that will be used in the machine learning
    algorithm to interpolate the true spectra so that they
    wil all have the same size (required for our CNN). The reference spectrum
    will be saved as self.wavenumbers_syn [cm-1].
    """
    ref_spec = fits.open(ref_spec)[1].data
    channel = []
    counts = []
    for chan in ref_spec:  # Only want SN3 region
        channel.append(chan[0])
        counts.append(np.real(chan[1]))
    if hdr_dict['FILTER'] == 'SN3':
        min_ = np.argmin(np.abs(np.array(channel) - 14700))
        max_ = np.argmin(np.abs(np.array(channel) - 15600))
    elif hdr_dict['FILTER'] == 'SN2':
        min_ = np.argmin(np.abs(np.array(channel) - 19000))
        max_ = np.argmin(np.abs(np.array(channel) - 21000))
    elif hdr_dict['FILTER'] == 'SN1':
        min_ = np.argmin(np.abs(np.array(channel) - 25500))
        max_ = np.argmin(np.abs(np.array(channel) - 27500))
    elif hdr_dict['FILTER'] == 'C3':
        min_ = np.argmin(np.abs(np.array(channel) - 17500))
        max_ = np.argmin(np.abs(np.array(channel) - 19500))
    elif hdr_dict['FILTER'] == 'C4':
        min_ = np.argmin(np.abs(np.array(channel) - 14700))
        max_ = np.argmin(np.abs(np.array(channel) - 15600))
    elif hdr_dict['FILTER'] == 'C2':
        min_ = np.argmin(np.abs(np.array(channel) - 18900))
        max_ = np.argmin(np.abs(np.array(channel) - 22500))
    elif hdr_dict['FILTER'] == 'C1':
        min_ = np.argmin(np.abs(np.array(channel) - 22000))
        max_ = np.argmin(np.abs(np.array(channel) - 32000))
    else:
        print('We do not support this filter.')
        print('Terminating program!')
        exit()
    print(min_, max_)
    wavenumbers_syn = np.array(channel[min_:max_], dtype=np.float32)
    wavenumbers_syn_full = np.array(channel, dtype=np.float32)
    return wavenumbers_syn, wavenumbers_syn_full


def read_in_transmission(Luci_path, hdr_dict, spectrum_axis_unshifted):
    """
    Read in the transmission spectrum for the filter. Then apply interpolation
    on it to make it have the same x-axis as the spectra.
    """
    transmission = np.loadtxt('%s/Data/%s_filter.dat' % (
        Luci_path, hdr_dict['FILTER']))  # first column - axis; second column - value
    f = interpolate.interp1d(transmission[:, 0], [val / 100 for val in transmission[:, 1]], kind='slinear',
                             fill_value="extrapolate")
    transmission_interpolated = f(spectrum_axis_unshifted)
    return transmission_interpolated


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
            summed_spec = cube_final[x_min + int(i * binning):x_min + int((i + 1) * binning),
                          y_min + int(j * binning):y_min + int((j + 1) * binning), :]
            summed_spec = np.nansum(summed_spec, axis=0)
            summed_spec = np.nansum(summed_spec, axis=0)
            binned_cube[i, j] = summed_spec[:]
    header_binned = header
    header_binned['CRPIX1'] = header_binned['CRPIX1'] / binning
    header_binned['CRPIX2'] = header_binned['CRPIX2'] / binning
    header_binned['CDELT1'] = header_binned['CDELT1'] * binning
    header_binned['CDELT2'] = header_binned['CDELT2'] * binning
    cube_binned = binned_cube / (binning ** 2)
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
    binned_mask = np.zeros((x_shape_new, y_shape_new))
    for i in range(x_shape_new):
        for j in range(y_shape_new):
            summed_spec = mask[x_min + int(i * binning):x_min + int((i + 1) * binning),
                          y_min + int(j * binning):y_min + int((j + 1) * binning)]
            summed_spec = np.nansum(summed_spec, axis=0)
            summed_spec = np.nansum(summed_spec, axis=0)
            binned_mask[i, j] = summed_spec[:]
    binned_mask = binned_mask / (binning ** 2)
    return binned_mask
