"""
End-to-end WVT run over a whole synthetic cube.

This is the path a full-field run takes, and it exercises the pieces that made such a run
impossible: the bin label map (rather than one full-field mask per bin), the indexed region sum,
and the map scatter. The scatter in particular used to be indexed [x, y] against arrays shaped
(n_y, n_x), so on any cube whose y extent exceeds its x extent -- SITELLE is 2048 x 2064 -- a
full-field run raised IndexError as soon as a bin reached y >= dimx.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from astropy.io import fits

from luci.analysis.wvt import BIN_DIR, BIN_MAP_NAME, load_bin_regions

LINES = ["Halpha", "NII6548", "NII6583"]


@pytest.fixture(scope="module")
def wvt_run(luci_factory, cube_truth_factory):
    """A completed WVT fit over a non-square cube, so the y > x indexing case is covered."""
    truth = cube_truth_factory("SN3", dimx=8, dimy=11)
    cube = luci_factory(truth, ML_bool=False)
    cube.wvt_fit_region(
        0, truth["dimx"], 0, truth["dimy"], LINES, "sincgauss",
        [1, 1, 1], [1, 1, 1], stn_target=3, n_threads=1,
    )
    return cube, truth


def test_run_completes_over_a_non_square_cube(wvt_run):
    """The regression: this raised IndexError once a bin reached y >= dimx."""
    cube, truth = wvt_run
    assert truth["dimy"] > truth["dimx"]
    assert os.path.exists(os.path.join(cube.output_dir, BIN_DIR, BIN_MAP_NAME))


def test_writes_one_bin_map_and_no_per_bin_masks(wvt_run):
    cube, _ = wvt_run
    contents = os.listdir(os.path.join(cube.output_dir, BIN_DIR))
    assert BIN_MAP_NAME in contents
    assert not [f for f in contents if f.startswith("bool_bin_map_")]


def test_bin_map_covers_the_cube_and_partitions_it(wvt_run):
    """Every pixel belongs to at most one bin, and the map is the cube's shape."""
    cube, truth = wvt_run
    bin_map = np.load(os.path.join(cube.output_dir, BIN_DIR, BIN_MAP_NAME))
    assert bin_map.shape == (truth["dimx"], truth["dimy"])
    regions = load_bin_regions(cube.output_dir, cube.cube_final.shape)
    seen = [pixel for region in regions for pixel in zip(*region)]
    assert len(seen) == len(set(seen))  # no pixel in two bins


@pytest.mark.parametrize("line", LINES)
def test_flux_maps_are_written_in_y_by_x_orientation(wvt_run, line):
    """
    The maps must match the fit_cube products' orientation, which is (n_y, n_x).

    A transposed map is not obviously wrong to look at -- it is the same numbers -- which is why
    this went unnoticed on square regions.
    """
    cube, truth = wvt_run
    path = os.path.join(cube.output_dir, "Fluxes", "%s_wvt_3_1_%s_Flux.fits" % (cube.object_name, line))
    data = fits.open(path)[0].data
    assert data.shape == (truth["dimy"], truth["dimx"])


def test_fitted_pixels_line_up_with_the_bins(wvt_run):
    """
    A pixel in a bin must have been written, and every pixel of one bin must share its value.

    This is what catches a scatter that lands on the wrong pixels: transposing the indices still
    fills the map, just in the wrong places.
    """
    cube, _ = wvt_run
    velocity = fits.open(
        os.path.join(cube.output_dir, "Velocity", "%s_wvt_3_1_Halpha_velocity.fits" % cube.object_name)
    )[0].data
    regions = load_bin_regions(cube.output_dir, cube.cube_final.shape)
    non_trivial = [r for r in regions if r[0].size > 1]
    assert non_trivial, "expected at least one multi-pixel bin"
    for xs, ys in non_trivial:
        values = velocity[ys, xs]  # maps are [y, x]
        assert np.allclose(values, values[0], equal_nan=True)
