import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed
from scipy import interpolate
from tqdm import tqdm

from luci.engine.selection import resolve_mask
from luci.instrument.filters import UnsupportedFilterError
from luci.log import get_logger

logger = get_logger(__name__)


def _load_components_model(path):
    """Load the component-counting network (TF imported lazily; fitting uses ONNX)."""
    try:
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Counting spectral components requires TensorFlow, which is an "
            "optional dependency. Install it with:  pip install 'luci-sitelle[ml-legacy]'"
        ) from exc
    return keras.models.load_model(path)


def create_component_map_function(
    header,
    hdr_dict,
    Luci_path,
    resolution,
    filter_,
    cube_final,
    spectrum_axis,
    wavenumbers_syn_full,
    output_dir,
    object_name,
    x_min=0,
    x_max=2048,
    y_min=0,
    y_max=2064,
    bkg=None,
    n_threads=2,
    region=None,
):
    """
    Create component map of a given region following our third paper. If no bounds are given,
    a map of the entire cube is calculated.

    Args:
        x_min: Minimal X value (default 0)
        x_max: Maximal X value (default 2048)
        y_min: Minimal Y value (default 0)
        y_max: Maximal Y value (default 2064)
        bkg: Background Spectrum (1D numpy array; default None)
        n_threads: Number of threads to use
        region: Name of ds9 region file (e.x. 'region.reg'). You can also pass a boolean mask array.
    Return:
        component_map: component map

    """
    # Calculate bounds for SNR calculation
    x_size = x_max - x_min
    y_size = y_max - y_min
    mask = resolve_mask(region, header, cube_final.shape) if region is not None else None
    # Step through spectra
    # Sized from the cube rather than the standard detector dimensions (B10).
    Comps = np.zeros(cube_final.shape[:2], dtype=np.float32).T
    Preds = np.zeros(cube_final.shape[:2], dtype=np.float32).T
    if hdr_dict["FILTER"] == "SN3":
        # Read in machine learning algorithm
        comps_model = _load_components_model(Luci_path + "ML/R%i-COMPONENTS-%s.h5" % (resolution, filter_))
    else:
        raise UnsupportedFilterError(f"Component calculation is only implemented for SN3, not {hdr_dict['FILTER']!r}.")

    def component_calc(i):
        y_pix = y_min + i
        comps_local = np.zeros(x_size)
        preds_local = np.zeros(x_size)
        for j in range(x_max - x_min):
            x_pix = x_min + j
            if mask and mask[x_pix, y_pix] == False:  # If there is a mask and we are not in the mask break
                break
            # Calculate how many components are present in SN3 using a convolutional neural network
            sky = cube_final[x_pix, y_pix, :]
            good_sky_inds = [~np.isnan(sky)]  # Clean up spectrum
            if bkg is not None:
                sky = sky[good_sky_inds] - bkg[good_sky_inds]
            else:
                sky = sky[good_sky_inds]
            axis = spectrum_axis[good_sky_inds]
            # Interpolate
            f = interpolate.interp1d(axis, sky, kind="slinear", fill_value="extrapolate")
            spectrum_interpolated = f(wavenumbers_syn_full[2:-2])
            spectrum_scaled = spectrum_interpolated / np.max(spectrum_interpolated)
            Spectrum = spectrum_scaled.reshape(1, spectrum_scaled.shape[0], 1)
            predictions = comps_model(Spectrum, training=False)
            max_ind = np.argmax(predictions[0])  # ID of outcome (0 -> single; 1 -> double)
            comp = max_ind + 1  # 1 -> single; 2 -> double
            comps_local[x_pix] = comp
            # print(predictions)
            preds_local[x_pix] = predictions[0][comp - 1]
        return comps_local, preds_local, i

    res = Parallel(n_jobs=n_threads, backend="threading")(delayed(component_calc)(i) for i in tqdm(range(y_size)))
    # Save
    for comp_ind in res:
        comp_vals, preds_vals, step_i = comp_ind
        Comps[y_min + step_i] = comp_vals
        Preds[y_min + step_i] = preds_vals

    fits.writeto(output_dir + "/" + object_name + "_comps.fits", Comps, header, overwrite=True)
    fits.writeto(output_dir + "/" + object_name + "_comps_probs.fits", Preds, header, overwrite=True)


def calculate_components_in_region_function(
    header, hdr_dict, Luci_path, resolution, filter_, cube_final, spectrum_axis, wavenumbers_syn_full, region, bkg
):
    """
    Primary fit call to fit a single pixel in the data cube. This wraps the
    LuciFits.FIT().fit() call which applies all the fitting steps.

    Args:
        lines: Lines to fit (e.x. ['Halpha', 'NII6583'])
        region:
        bkg: Background Spectrum (1D numpy array; default None)

    Return:
        Returns the x-axis (redshifted), sky, and fit dictionary


    """
    mask = None  # Initialize
    comps_model = None
    mask = resolve_mask(region, header, cube_final.shape)
    if hdr_dict["FILTER"] == "SN3":
        # Read in machine learning algorithm
        comps_model = _load_components_model(Luci_path + "ML/R%i-COMPONENTS-%s.h5" % (resolution, filter_))
    else:
        raise UnsupportedFilterError(f"Component calculation is only implemented for SN3, not {hdr_dict['FILTER']!r}.")
    # Set spatial bounds for entire cube
    x_min = 0
    x_max = cube_final.shape[0]
    y_min = 0
    y_max = cube_final.shape[1]
    integrated_spectrum = np.zeros(cube_final.shape[2])
    spec_ct = 0
    for i in tqdm(range(y_max - y_min)):
        y_pix = y_min + i
        for j in range(x_max - x_min):
            x_pix = x_min + j
            # Check if pixel is in the mask or not
            if mask[x_pix, y_pix]:
                integrated_spectrum += cube_final[x_pix, y_pix, :]
                spec_ct += 1
            else:
                pass
    if bkg is not None:
        integrated_spectrum -= bkg  # Subtract background spectrum
    sky = cube_final[x_pix, y_pix, :]
    good_sky_inds = [~np.isnan(sky)]  # Clean up spectrum
    if bkg is not None:
        sky = sky[good_sky_inds] - bkg[good_sky_inds]
    else:
        sky = sky[good_sky_inds]
    axis = spectrum_axis[good_sky_inds]
    # Interpolate
    f = interpolate.interp1d(axis, sky, kind="slinear", fill_value="extrapolate")
    spectrum_interpolated = f(wavenumbers_syn_full[2:-2])
    spectrum_scaled = spectrum_interpolated / np.max(spectrum_interpolated)
    Spectrum = spectrum_scaled.reshape(1, spectrum_scaled.shape[0], 1)
    predictions = comps_model(Spectrum, training=False)
    max_ind = np.argmax(predictions[0])  # ID of outcome (0 -> single; 1 -> double)
    comp = max_ind + 1  # 1 -> single; 2 -> double
    comp_prob = predictions[0][comp - 1]  # Probability of classification
    return {"components": comp, "component_probability": comp_prob}
