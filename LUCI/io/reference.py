"""Reading the ML reference spectrum and the filter transmission curve."""

import numpy as np
from astropy.io import fits
from scipy import interpolate

from LUCI.instrument.filters import get_filter


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
    # Clip window comes from the filter registry instead of an if/elif chain
    # that used to call exit() from library code on an unknown filter.
    ref_lower, ref_upper = get_filter(hdr_dict["FILTER"]).reference_bounds()
    min_ = np.argmin(np.abs(np.array(channel) - ref_lower))
    max_ = np.argmin(np.abs(np.array(channel) - ref_upper))
    wavenumbers_syn = np.array(channel[min_:max_], dtype=np.float32)
    wavenumbers_syn_full = np.array(channel, dtype=np.float32)
    return wavenumbers_syn, wavenumbers_syn_full


def read_in_transmission(Luci_path, hdr_dict, spectrum_axis_unshifted):
    """
    Read in the transmission spectrum for the filter. Then apply interpolation
    on it to make it have the same x-axis as the spectra.
    """
    transmission = np.loadtxt(
        "%s/Data/%s_filter.dat" % (Luci_path, hdr_dict["FILTER"])
    )  # first column - axis; second column - value
    f = interpolate.interp1d(
        transmission[:, 0], [val / 100 for val in transmission[:, 1]], kind="slinear", fill_value="extrapolate"
    )
    transmission_interpolated = f(spectrum_axis_unshifted)
    return transmission_interpolated
