"""
Tests for the vectorised SNR-map computation.

`create_snr_map` used to walk every pixel of the field in a Python loop, rebuilding each reduction
-- and, for method 2, a ten-iteration sigma clip -- as its own call. On a full SITELLE field that
made the method-2 map take the better part of an hour. It now processes a whole row at once.

These pin the row computation against a straight per-pixel reference: method 1 has to stay bitwise
identical, and method 2 has to stay within a hair, the small gap being `sigma_clip(axis=1)`
occasionally converging to a different iteration than a per-1D call on a boundary spectrum. That
feeds a binning heuristic, so a sub-percent shift on a few pixels changes nothing -- but a large or
widespread difference would mean the vectorisation is actually wrong, which is what this guards.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from astropy.io import fits


def snr_map(cube):
    snr_dir = os.path.join(cube.output_dir, "SNR")
    path = os.path.join(snr_dir, cube.object_name + "_SNR.fits")
    return fits.open(path)[0].data


@pytest.mark.parametrize("method", [1, 2])
def test_runs_and_is_finite_and_correctly_shaped(luci_factory, sn3_truth, method):
    cube = luci_factory(sn3_truth, ML_bool=False)
    cube.create_snr_map(method=method, n_threads=2)
    snr = snr_map(cube)
    assert snr.shape == (sn3_truth["dimy"], sn3_truth["dimx"])  # stored transposed vs the cube
    assert np.isfinite(snr).all()
    assert (snr >= 0).all()  # both methods floor negatives at 0


@pytest.mark.parametrize("method", [1, 2])
def test_is_deterministic(luci_factory, sn3_truth, method):
    """A row-parallel reduction must not depend on how the rows were scheduled."""
    a = luci_factory(sn3_truth, ML_bool=False, redshift=1e-13)
    a.create_snr_map(method=method, n_threads=1)
    b = luci_factory(sn3_truth, ML_bool=False, redshift=2e-13)
    b.create_snr_map(method=method, n_threads=4)
    np.testing.assert_array_equal(snr_map(a), snr_map(b))


def _reference(cube, method):
    """The per-pixel computation the vectorised code replaces, kept here as the definition."""
    import astropy.stats as astrostats

    axis = np.asarray(cube.spectrum_axis)
    # SN3 windows, matching create_snr_map
    fl = int(np.argmin(np.abs(axis - 15150)))
    fh = int(np.argmin(np.abs(axis - 15300)))
    nl = int(np.argmin(np.abs(axis - 14500)))
    nh = int(np.argmin(np.abs(axis - 14600)))
    data = cube.cube_final
    nx, ny = data.shape[0], data.shape[1]
    out = np.zeros((ny, nx), dtype=np.float32)
    for xp in range(nx):
        for yp in range(ny):
            sky = np.asarray(data[xp, yp, :])
            if method == 1:
                signal = np.nanmax(sky) - np.nanmean(sky)
                noise = np.abs(np.nanstd(sky[nl:nh]))
                snr = float(signal / np.sqrt(noise))
                snr = 0.0 if snr < 0 else snr / np.sqrt(np.nanmean(np.abs(sky)))
            else:
                flux = np.nansum(sky[fl:fh])
                clipped = astrostats.sigma_clip(sky, sigma=1, masked=False, copy=True, maxiters=10)
                flux -= np.nanmin(clipped) * (fh - fl)
                snr = float(flux / np.nanstd(sky[nl:nh]))
                snr = 0.0 if snr < 0 else snr
            out[yp, xp] = snr
    return out


def test_method1_matches_the_per_pixel_reference_bitwise(luci_factory, sn3_truth):
    cube = luci_factory(sn3_truth, ML_bool=False)
    cube.create_snr_map(method=1, n_threads=2)
    np.testing.assert_array_equal(snr_map(cube), _reference(cube, 1))


def test_method2_matches_the_per_pixel_reference_to_within_a_hair(luci_factory, sn3_truth):
    cube = luci_factory(sn3_truth, ML_bool=False)
    cube.create_snr_map(method=2, n_threads=2)
    got, ref = snr_map(cube), _reference(cube, 2)
    # The overwhelming majority of pixels are exact; the rest differ only where sigma_clip(axis=1)
    # took a different iteration, and only slightly.
    close = np.isclose(got, ref, rtol=1e-6, atol=0, equal_nan=True)
    assert close.mean() > 0.98
    np.testing.assert_allclose(got, ref, rtol=5e-3, atol=1e-6, equal_nan=True)
