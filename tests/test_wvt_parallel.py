"""
Tests for fanning the WVT bin fits out across workers.

`fit_wvt` fitted bins one at a time, which on a full field was ~81,000 serial fits -- about 3.3 of
the 4 hours of a full-cube run, with one core busy and the rest idle. Extraction needs the cube
(~8 GB, so it cannot be shipped to a worker) but fitting needs only a few kB per bin, so extraction
stays in the parent and the fits fan out.

The result must not depend on how the work was divided, which is what these tests check.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from luci.analysis.wvt import load_bin_regions

LINES = ["Halpha", "NII6583"]
PRODUCTS = [
    ("Fluxes", "Flux"),
    ("Velocity", "velocity"),
    ("Broadening", "broadening"),
]


def read_products(cube, stn):
    """Every per-line map of a finished WVT run, keyed by (line, product)."""
    out = {}
    for line in LINES:
        for subdir, name in PRODUCTS:
            path = "%s/%s/%s_wvt_%d_1_%s_%s.fits" % (cube.output_dir, subdir, cube.object_name, stn, line, name)
            out[(line, name)] = fits.open(path)[0].data
    return out


@pytest.fixture(scope="module")
def truth(cube_truth_factory):
    return cube_truth_factory("SN3", dimx=9, dimy=11)


def run_wvt(cube, truth, n_threads):
    cube.wvt_fit_region(
        0, truth["dimx"], 0, truth["dimy"], LINES, "sincgauss",
        [1, 1], [1, 1], stn_target=3, n_threads=n_threads,
    )
    return read_products(cube, 3)


def test_parallel_and_serial_agree(luci_factory, truth):
    """
    One worker and several must give the same maps.

    A fit that behaved differently under parallelism -- through shared state, or through a bin's
    result landing on the wrong pixels -- would show up here as a mismatch.
    """
    serial = run_wvt(luci_factory(truth, ML_bool=False), truth, n_threads=1)
    parallel = run_wvt(luci_factory(truth, ML_bool=False, redshift=1e-12), truth, n_threads=4)
    for key, expected in serial.items():
        np.testing.assert_allclose(
            parallel[key], expected, rtol=1e-6, atol=0, equal_nan=True,
            err_msg="%s %s differs between serial and parallel" % key,
        )


def test_every_binned_pixel_is_filled(luci_factory, truth):
    """Chunking must not drop a bin: each bin's pixels have to share one non-zero fit."""
    cube = luci_factory(truth, ML_bool=False, redshift=2e-12)
    maps = run_wvt(cube, truth, n_threads=4)
    velocity = maps[("Halpha", "velocity")]
    regions = load_bin_regions(cube.output_dir, cube.cube_final.shape)
    assert len(regions) > 1
    for xs, ys in regions:
        if xs.size == 0:
            continue
        values = velocity[ys, xs]  # maps are [y, x]
        assert np.allclose(values, values[0], equal_nan=True)


def test_worker_count_does_not_change_the_chunk_boundaries(luci_factory, truth):
    """
    Chunk size is derived from n_threads, so two worker counts use different boundaries.

    If a bin's result were ever paired with the wrong bin's pixels, a change of chunking would move
    which bins are affected and the two runs would disagree.
    """
    a = run_wvt(luci_factory(truth, ML_bool=False, redshift=3e-12), truth, n_threads=2)
    b = run_wvt(luci_factory(truth, ML_bool=False, redshift=4e-12), truth, n_threads=8)
    for key, expected in a.items():
        np.testing.assert_allclose(b[key], expected, rtol=1e-6, equal_nan=True, err_msg="%s %s" % key)
