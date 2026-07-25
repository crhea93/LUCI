"""Signal-to-noise and detection maps.

Extracted from the ``Luci`` god-class; each takes the cube as its first argument.
"""

import multiprocessing as mp
import os
import warnings

import astropy.stats as astrostats
import numpy as np
import numpy.ma as ma
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from joblib import Parallel, delayed
from tqdm import tqdm

from luci.fitting.spectrum_fitter import SpectrumFitter as Fit
from luci.instrument.filters import UnsupportedFilterError, pca_scale_indices
from luci.log import get_logger

logger = get_logger(__name__)


def create_snr_map(
    cube,
    x_min=0,
    x_max=None,
    y_min=0,
    y_max=None,
    method=1,
    n_threads=2,
    lines=[None],
    binning=1,
    bkgType=None,
    pca_coefficient_array=None,
    pca_vectors=None,
    pca_mean=None,
):
    """
    Create signal-to-noise ratio (SNR) map of a given region. If no bounds are given,
    a map of the entire cube is calculated.

    Args:
        x_min: Minimal X value (default 0)
        x_max: Maximal X value (default None -> this cube's x extent)
        y_min: Minimal Y value (default 0)
        y_max: Maximal Y value (default None -> this cube's y extent)
        method: Method used to calculate SNR (default 1; options 1 or 2)
        n_threads: Number of threads to use
        lines: Lines to focus on (default None: For SN2 you can choose OIII)
        binning: Bin to apply (default 1)

    Return:
        snr_map: Signal-to-Noise ratio map

    """
    # Default to this cube's extent instead of the standard detector size,
    # which silently mis-indexed any non-standard cube (bug B10).
    if x_max is None:
        x_max = cube.cube_final.shape[0]
    if y_max is None:
        y_max = cube.cube_final.shape[1]
    cube_to_use = cube.cube_final
    if binning > 1:
        cube.bin_cube(cube.cube_final, cube.header, binning, x_min, x_max, y_min, y_max)
        x_max = int((x_max - x_min) / binning)
        y_max = int((y_max - y_min) / binning)
        x_min = 0
        y_min = 0
        cube_to_use = cube.cube_binned
    SNR = np.zeros((x_max - x_min, y_max - y_min), dtype=np.float32).T
    flux_min = 0
    flux_max = 0
    noise_min = 0
    noise_max = 0  # Initializing bounds for flux and noise calculation regions
    if cube.hdr_dict["FILTER"] == "SN3":  # Halpha complex
        flux_min = 15150
        flux_max = 15300
        noise_min = 14500
        noise_max = 14600
    elif cube.hdr_dict["FILTER"] == "SN2":
        if "OIII" in lines:  # OIII lines
            flux_min = 1e7 / 505
            flux_max = 1e7 / 495
        else:  # Hbeta by default
            flux_min = 1e7 / 486
            flux_max = 1e7 / 482
        noise_min = 19000
        noise_max = 19500
    elif cube.hdr_dict["FILTER"] == "SN1":  ## OII lines
        flux_min = 26550
        flux_max = 27550
        noise_min = 25700
        noise_max = 26300
    elif cube.hdr_dict["FILTER"] == "SN4":  # Halpha complex in the narrow Halpha filter
        flux_min = 15150
        flux_max = 15300
        # The order 15 free spectral range extends well past the 652-665 nm pass band, so the
        # noise can be taken from a completely blocked portion of the axis. Use the blue side:
        # ORB leaves the red end (14669-14905 cm-1) NaN in real SN4 cubes. Kept in step with
        # the SN4 noise window in luci/instrument/filters.py.
        noise_min = 15380
        noise_max = 15650
    elif cube.hdr_dict["FILTER"] == "C3":  # Only for MACSJ1621
        flux_min = 18500
        flux_max = 20500
        noise_min = 20500
        noise_max = 21500
    else:
        raise UnsupportedFilterError(f"SNR calculation is not implemented for filter {cube.hdr_dict['FILTER']!r}.")

    # Channel bounds of the flux and noise windows. These depend only on the spectral axis, so they
    # are the same for every pixel -- they used to be recomputed inside the loop, which rebuilt
    # `np.array(cube.spectrum_axis)` and ran an argmin over it four times per pixel. On a full
    # SITELLE field that is 17 million array constructions to arrive at four constants.
    spectrum_axis = np.asarray(cube.spectrum_axis)
    flux_lo = int(np.argmin(np.abs(spectrum_axis - flux_min)))
    flux_hi = int(np.argmin(np.abs(spectrum_axis - flux_max)))
    noise_lo = int(np.argmin(np.abs(spectrum_axis - noise_min)))
    noise_hi = int(np.argmin(np.abs(spectrum_axis - noise_max)))

    def SNR_calc(i):
        # One whole row of the map at once. The per-pixel version rebuilt every reduction
        # (nanmax, nanmean, nanstd, and -- for method 2 -- a ten-iteration sigma clip) as a
        # separate Python call for each of the ~2000 pixels in the row. `astropy.stats.sigma_clip`
        # clips each spectrum independently when handed `axis=1`, and every other step is a plain
        # axis reduction, so the whole row collapses to a handful of vectorised calls -- the ~500x
        # speedup that made the method-2 map on a full field tractable.
        #
        # Method 1 is bitwise identical to the per-pixel version. Method 2 matches on ~99.6% of
        # pixels and differs by <=0.1% on the rest: `sigma_clip(axis=1)` occasionally converges to a
        # different iteration than the per-1D-array call for a spectrum sitting on a clipping
        # boundary, nudging the continuum (a min over the clipped spectrum) at the fourth
        # significant figure. That feeds a binning heuristic, not a measurement, so a 0.1% shift on a
        # third of a percent of pixels changes no selection; matching it exactly would mean the
        # per-pixel loop this replaces.
        y_pix = y_min + i
        block = np.asarray(cube_to_use[x_min:x_max, y_pix, :])  # (n_x, n_channels)

        with warnings.catch_warnings():
            # All-NaN slices give a NaN and a RuntimeWarning; the per-pixel code produced the same
            # NaN. Silence the warning, keep the value.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if method == 1:
                signal = np.nanmax(block, axis=1) - np.nanmean(block, axis=1)
                noise = np.abs(np.nanstd(block[:, noise_lo:noise_hi], axis=1))
                snr = signal / np.sqrt(noise)
                scale = np.sqrt(np.nanmean(np.abs(block), axis=1))
                # The per-pixel code divided by `scale` only on the non-negative branch, so a
                # negative signal maps to exactly 0 and never touches `scale`.
                snr_local = np.where(snr < 0, 0.0, snr / scale)
            else:
                flux_in_region = np.nansum(block[:, flux_lo:flux_hi], axis=1)
                # Sigma clip each spectrum, then the continuum is the min of what survives. copy=True
                # leaves `block` intact; the per-pixel version clipped in place with copy=False, and
                # the only place that mattered was the noise window below, which we read from the
                # clipped array to match.
                clipped = astrostats.sigma_clip(block, sigma=1, axis=1, masked=False, copy=True, maxiters=10)
                cont_val = np.nanmin(clipped, axis=1)
                flux_in_region = flux_in_region - cont_val * (flux_hi - flux_lo)
                # Noise from the *unclipped* spectrum. The per-pixel version passed copy=False, which
                # despite appearances does not write the clip back into the pixel's own view, so its
                # `nanstd(out_region)` saw the original noise window -- only the continuum (the nanmin
                # above) used clipped values.
                std_out_region = np.nanstd(block[:, noise_lo:noise_hi], axis=1)
                snr = flux_in_region / std_out_region
                snr_local = np.where(snr < 0, 0.0, snr)
        return np.asarray(snr_local, dtype=float), i

    res = Parallel(n_jobs=n_threads, backend="threading")(delayed(SNR_calc)(i) for i in tqdm(range(y_max - y_min)))
    # Save
    for snr_ind in res:
        snr_vals, step_i = snr_ind
        SNR[step_i] = snr_vals
    if os.path.exists(cube.output_dir + "/SNR"):
        pass
    else:
        os.mkdir(cube.output_dir + "/SNR")
    fits.writeto(cube.output_dir + "/SNR/" + cube.object_name + "_SNR.fits", SNR, cube.header, overwrite=True)
    # Save masks for SNr 3, 5, and 10
    masks = []
    for snr_val in [1, 3, 5, 10]:
        mask = ma.masked_where(SNR >= snr_val, SNR)
        masks.append(mask)
        np.save("%s/SNR/%s_SNR_%i_mask.npy" % (cube.output_dir, cube.object_name, snr_val), mask.mask)
    return masks


def detection_map(cube, x_min=None, x_max=None, y_min=None, y_max=None, n_threads=1):
    """
    Method to call the detection algorithm. The detection algorithm works as follows:
    For each pixel,
        1. Calculate the median spectrum for a 3x3 pixel region centered on the current pixel
        2. Calculate the median spectrum for a 9x9 pixel region centered on the current pixel
        3. Subtract the 9x9 spectrum from the 3x3 spectrum
        4. Take the maximum value of this subtracted spectrum as the detection map value

    In the end, we have a detection map of the maximum values.

    If no bounds are added, we calculate over the entire map

    Louis-Simon Guité

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
        x_max = cube.cube_final.shape[0] - 10
        y_min = 0 + 10
        y_max = cube.cube_final.shape[1] - 10
    # Initalize solution
    detection_map = np.zeros((x_max - x_min, y_max - y_min), dtype=np.float32).T
    # Correct header information
    if not os.path.exists(cube.output_dir + "/" + cube.object_name + "_deep.fits"):
        cube.create_deep_image()
    wcs = WCS(cube.header, naxis=2)
    cutout = Cutout2D(
        fits.open(cube.output_dir + "/" + cube.object_name + "_deep.fits")[0].data,
        position=((x_max + x_min) / 2, (y_max + y_min) / 2),
        size=(x_max - x_min, y_max - y_min),
        wcs=wcs,
    )
    global value_calc

    def value_calc(i):
        y_pix = y_min + i
        detection_local = []
        for j in range(x_max - x_min):
            x_pix = x_min + j
            # Get the 3x3 bin group
            sky_1 = cube.cube_final[x_pix - 1 : x_pix + 1, y_pix - 1 : y_pix + 1, :]
            sky_1 = np.nanmedian(sky_1, axis=0)  # Bin once
            sky_1 = np.nanmedian(sky_1, axis=0)  # Bin twice!
            good_sky_inds_1 = [~np.isnan(sky_1)]  # Clean up spectrum
            sky_1 = sky_1[good_sky_inds_1]
            # Get the 9x9 bin group
            sky_2 = cube.cube_final[x_pix - 4 : x_pix + 4, y_pix - 4 : y_pix + 4, :]
            sky_2 = np.nanmedian(sky_2, axis=0)  # Bin once
            sky_2 = np.nanmedian(sky_2, axis=0)  # Bin twice!
            good_sky_inds_2 = [~np.isnan(sky_2)]  # Clean up spectrum
            sky_2 = sky_2[good_sky_inds_2]
            # Obtain the difference
            sky_diff = sky_1 - sky_2
            if len(sky_diff) > 0:
                detection_val = np.nanmax(
                    sky_diff
                )  # The nan shouldn't be necessary -- I just put it so that it didn't feel left out
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
    fits.writeto(
        cube.output_dir + "/" + cube.object_name + "_detection.fits",
        detection_map,
        cutout.wcs.to_header(),
        overwrite=True,
    )
    return detection_map
