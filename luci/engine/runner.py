"""
Shared orchestration for cube fits: run the per-slice fits and collect them.

The entry points (``fit_cube``, ``fit_region``, ...) differ only in how they
choose pixels and what they name the output. Everything between -- resolving
initial-condition maps, building the WCS cutout, fanning out over y-slices, and
scattering results -- is the same, and lives here.
"""

from __future__ import annotations

import os

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from joblib import Parallel, delayed
from tqdm import tqdm

from luci.engine.maps import FitMaps


def resolve_initial_values(initial_values):
    """
    Turn the ``initial_values`` argument into a (velocity, broadening) map pair.

    Accepts either paths to FITS maps or arrays from a previous fit; returns
    ``(False, False)`` when no initial conditions were supplied.
    """
    if len(initial_values) != 2:
        return False, False
    try:
        return fits.open(initial_values[0])[0].data, fits.open(initial_values[1])[0].data
    except (OSError, TypeError, ValueError):
        # Already arrays rather than file paths.
        return initial_values[0], initial_values[1]


def deep_image_cutout(cube, x_min, x_max, y_min, y_max, binning):
    """WCS cutout of the deep image matching the fitted region, creating it if absent."""
    deep_path = os.path.join(cube.output_dir, cube.object_name + "_deep.fits")
    if not os.path.exists(deep_path):
        cube.create_deep_image()
    if binning is not None and binning > 1:
        wcs = WCS(cube.header_binned)
    else:
        wcs = WCS(cube.header, naxis=2)
    return Cutout2D(
        fits.open(deep_path)[0].data,
        position=((x_max + x_min) / 2, (y_max + y_min) / 2),
        size=(x_max - x_min, y_max - y_min),
        wcs=wcs,
    )


def run_fit(
    cube,
    cube_to_slice,
    lines,
    fit_function,
    vel_rel,
    sigma_rel,
    x_min,
    x_max,
    y_min,
    y_max,
    n_threads=1,
    mask=None,
    **fit_kwargs,
) -> FitMaps:
    """
    Fit every y-slice of the selected region in parallel and collect the maps.

    ``fit_kwargs`` are forwarded verbatim to ``Luci.fit_calc``, so a new fit
    option only has to be threaded through once.
    """
    maps = FitMaps.allocate(
        x_max - x_min, y_max - y_min, len(lines), absorption=bool(fit_kwargs.get("absorption_bool"))
    )
    results = Parallel(n_jobs=n_threads)(
        delayed(cube.fit_calc)(
            sl,
            x_min,
            x_max,
            y_min,
            fit_function,
            lines,
            vel_rel,
            sigma_rel,
            cube_slice=cube_to_slice[:, y_min + sl, :],
            spectrum_axis=cube.spectrum_axis,
            wavenumbers_syn=cube.wavenumbers_syn,
            transmission_interpolated=cube.transmission_interpolated,
            interferometer_theta=cube.interferometer_theta,
            hdr_dict=cube.hdr_dict,
            step_nb=cube.step_nb,
            zpd_index=cube.zpd_index,
            mdn=cube.mdn,
            ML_bool=cube.ML_bool,
            resolution=cube.resolution,
            Luci_path=cube.Luci_path,
            mask=mask,
            **fit_kwargs,
        )
        for sl in tqdm(range(y_max - y_min))
    )
    for result in results:
        maps.scatter(result)
    return maps
