"""Translating SITELLE HDF5 headers into WCS and spectral axes."""

import numpy as np
from astropy.wcs import WCS


def get_interferometer_angles(file, hdr_dict):
    """
    Calculate the interferometer angle 2d array for the entire cube. We use
    the following equation:
    cos(theta) = lambda_ref/lambda
    where lambda_ref is the reference laser wavelength and lambda is the measured calibration laser wavelength.

    Args:
        file: hdf5 File object containing HDF5 file
    """

    calib_map = file["calib_map"][()].astype("float")
    try:
        calib_ref = hdr_dict["CALIBNM"]
    except KeyError:
        calib_ref = hdr_dict["nm_laser"]
    calib_ref = np.float32(calib_ref)
    interferometer_cos_theta = calib_ref / calib_map  # .T[::-1,::-1]
    # We need to convert to degree so bear with me here
    # del calib_map
    return np.rad2deg(np.arccos(interferometer_cos_theta))


def spectrum_axis_func(hdr_dict, redshift):
    """
    Create the x-axis for the spectra. We must construct this from header information
    since each pixel only has amplitudes of the spectra at each point.
    """

    len_wl = int(hdr_dict["STEPNB"])  # Length of Spectral Axis
    start = float(hdr_dict["CRVAL3"])  # Starting value of the spectral x-axis
    step = float(hdr_dict["CDELT3"])  # Step size
    end = start + (len_wl) * step  # End

    spectrum_axis = np.array(
        np.linspace(start, end, len_wl) * (redshift + 1), dtype=np.float32
    )  # Apply redshift correction
    spectrum_axis_unshifted = np.array(
        np.linspace(start, end, len_wl), dtype=np.float32
    )  # Do not apply redshift correction

    """min_ = 1e7  * (hdr_dict['ORDER'] / (2*hdr_dict['STEP']))# + 1e7  / (2*self.delta_x*self.n_steps)
    max_ = 1e7  * ((hdr_dict['ORDER'] + 1) / (2*hdr_dict['STEP']))# - 1e7  / (2*self.delta_x*self.n_steps)
    step_ = max_ - min_
    axis = np.array([min_+j*step_/hdr_dict['STEPNB'] for j in range(hdr_dict['STEPNB'])])
    spectrum_axis = axis*(1+redshift)
    spectrum_axis_unshifted = axis"""
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
    attribute_list = [attr for attr in list(file.attrs)]
    clean_hdr_dict = {}
    if "quad_nb" in attribute_list:  # Old HDF5
        header_cols = [
            str(val[0]).replace("'b", "").replace("'", "").replace("b", "") for val in list(file["header"][()])
        ]
        header_vals = [
            str(val[1]).replace("'b", "").replace("'", "").replace("b", "") for val in list(file["header"][()])
        ]
        header_types = [val[3] for val in list(file["header"][()])]
        for header_col, header_val, header_type in zip(header_cols, header_vals, header_types):
            if "bool" in str(header_type):
                hdr_dict[header_col] = bool(header_val)
            elif "float" in str(header_type):
                hdr_dict[header_col] = float(header_val)
            elif "int" in str(header_type):
                hdr_dict[header_col] = int(header_val)
            else:
                try:
                    hdr_dict[header_col] = float(header_val)
                except (TypeError, ValueError):
                    hdr_dict[header_col] = str(header_val)
        clean_hdr_dict = hdr_dict
    else:  # New HDF5
        header_cols = [attr for attr in list(file.attrs)]
        header_vals = [file.attrs[attr] for attr in list(file.attrs)]
        header_types = [type(file.attrs[attr]) for attr in list(file.attrs)]
        # Types are checked with issubclass/isinstance rather than `is np.<type>`.
        # The original compared against np.str and np.bool_ -- np.str was removed
        # in numpy 1.24, so on any modern numpy this raised AttributeError, which
        # the bare `except` below swallowed.  The result was that every string and
        # boolean keyword fell through to the fallback and clean_hdr_dict stayed
        # empty, producing a WCS with no axes for new-format cubes (bug B15).
        for header_col, header_val, header_type in zip(header_cols, header_vals, header_types):  # New HDF5 format
            try:
                if header_col == "flambda":
                    hdr_dict["flambda"] = header_val
                if issubclass(header_type, np.floating):
                    hdr_dict[header_col] = float(header_val)
                    clean_hdr_dict[header_col] = float(header_val)
                elif issubclass(header_type, np.bool_):
                    # Checked before np.integer: np.bool_ is not an integer type in
                    # numpy 2, but bool IS a subclass of int in plain Python, so a
                    # Python bool would otherwise be captured by the integer branch.
                    hdr_dict[header_col] = bool(header_val)
                    clean_hdr_dict[header_col] = bool(header_val)
                elif issubclass(header_type, (np.integer, int)):
                    hdr_dict[header_col] = int(header_val)
                    clean_hdr_dict[header_col] = int(header_val)
                elif issubclass(header_type, (np.str_, str, bytes)):
                    value = header_val.decode() if isinstance(header_val, bytes) else str(header_val)
                    hdr_dict[header_col] = value
                    clean_hdr_dict[header_col] = value
                    if "path" in header_col:
                        hdr_dict[header_col] = ""
                        clean_hdr_dict[header_col] = ""
                elif issubclass(header_type, np.ndarray):
                    hdr_dict[header_col] = np.array(header_val)
            except Exception:
                hdr_dict[header_col] = str(header_val)
    hdr_dict["CTYPE3"] = "WAVE-SIP"
    hdr_dict["CUNIT3"] = "m"
    try:
        hdr_dict["A_ORDER"] = int(hdr_dict["A_ORDER"])
        hdr_dict["AP_ORDER"] = int(hdr_dict["AP_ORDER"])
    except KeyError:
        pass
    hdr_dict["NAXIS"] = int(hdr_dict["NAXIS"])
    # If NAXIS 1 does not exist we will add it
    if "NAXIS1" not in hdr_dict.keys():
        hdr_dict["NAXIS1"] = 2048
        hdr_dict["NAXIS2"] = 2064
    # Make WCS

    wcs_data = WCS(clean_hdr_dict, naxis=2)
    header = wcs_data.to_header()
    header.insert("WCSAXES", ("SIMPLE", "T"))
    header.insert("SIMPLE", ("NAXIS", 2), after=True)
    if "STEP_NB" in hdr_dict.keys():
        hdr_dict["STEPNB"] = int(hdr_dict["step_nb"])
    if "zpd_index" in hdr_dict.keys():
        hdr_dict["ZPDINDEX"] = int(hdr_dict["zpd_index"])
    if "filter_name" in hdr_dict.keys():
        hdr_dict["FILTER"] = str(hdr_dict["filter_name"])
    if "axis_min" in hdr_dict.keys():
        hdr_dict["CRVAL3"] = float(hdr_dict["axis_min"])
    if "axis_step" in hdr_dict.keys():
        hdr_dict["CDELT3"] = float(hdr_dict["axis_step"])
    hdr_dict = hdr_dict
    return header, hdr_dict
