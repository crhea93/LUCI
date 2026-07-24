"""
Tests for summing the spectra of a masked region.

Both region extractors used to walk every pixel of the cube in Python to sum a mask that might hold
a dozen pixels, costing the same ~4.2 million iterations per call regardless. `fit_wvt` makes one
call per bin, so a full-field WVT run spent about 18 hours in that loop alone. These tests pin the
behaviour the indexed version has to keep: the same total, the same pixel count, and NaN still
poisoning any channel it appears in.
"""

from __future__ import annotations

import numpy as np
import pytest


def reference_sum(cube, mask):
    """The original double loop, kept as the definition of a correct region sum."""
    total = np.zeros(cube.shape[2])
    count = 0
    for i in range(cube.shape[1]):
        for j in range(cube.shape[0]):
            if mask[j, i]:
                total += cube[j, i, :]
                count += 1
    return total, count


@pytest.fixture
def masks(sn3_truth):
    shape = (sn3_truth["dimx"], sn3_truth["dimy"])

    def _mask(*boxes):
        mask = np.zeros(shape, dtype=bool)
        for x0, x1, y0, y1 in boxes:
            mask[x0:x1, y0:y1] = True
        return mask

    return _mask


def test_matches_the_reference_loop(sn3_cube_noml, masks):
    mask = masks((4, 8, 4, 8))
    expected, count = reference_sum(sn3_cube_noml.cube_final, mask)
    _, summed = sn3_cube_noml.extract_spectrum_region(mask)
    assert count == 16
    np.testing.assert_allclose(summed, expected, rtol=1e-12)


def test_matches_the_reference_loop_for_a_scattered_mask(sn3_cube_noml, masks):
    """A WVT bin is not a rectangle, so the disjoint case has to agree too."""
    mask = masks((1, 3, 1, 2), (7, 9, 6, 9), (4, 5, 8, 9))
    expected, count = reference_sum(sn3_cube_noml.cube_final, mask)
    _, summed = sn3_cube_noml.extract_spectrum_region(mask)
    assert count == mask.sum()
    np.testing.assert_allclose(summed, expected, rtol=1e-12)


def test_single_pixel_mask_equals_that_pixel(sn3_cube_noml, masks):
    mask = masks((5, 6, 5, 6))
    _, summed = sn3_cube_noml.extract_spectrum_region(mask)
    np.testing.assert_allclose(summed, sn3_cube_noml.cube_final[5, 5, :], rtol=1e-12)


def test_mean_divides_by_the_pixel_count(sn3_cube_noml, masks):
    mask = masks((4, 8, 4, 8))
    _, summed = sn3_cube_noml.extract_spectrum_region(mask)
    _, averaged = sn3_cube_noml.extract_spectrum_region(mask, mean=True)
    np.testing.assert_allclose(averaged, summed / 16.0, rtol=1e-12)


def test_accumulates_in_float64(sn3_cube_noml, masks):
    """The cube is float32; summing in float32 would lose precision on a large bin."""
    _, summed = sn3_cube_noml.extract_spectrum_region(masks((0, 10, 0, 10)))
    assert summed.dtype == np.float64


def test_nan_in_one_pixel_still_poisons_the_channel(sn3_cube_noml, masks):
    """
    The old loop used plain `+=`, so a NaN channel in any masked pixel made that channel NaN.

    This is load-bearing for SN4: ORB leaves ~60 channels per pixel NaN, and the background is
    built from a region sum, so whether NaN propagates decides which channels survive.
    """
    cube = sn3_cube_noml.cube_final
    original = cube[5, 5, 3]
    try:
        cube[5, 5, 3] = np.nan
        _, summed = sn3_cube_noml.extract_spectrum_region(masks((4, 8, 4, 8)))
        assert np.isnan(summed[3])
        assert np.isfinite(summed[0])
    finally:
        cube[5, 5, 3] = original


def test_empty_mask_sums_to_zero_without_dividing_by_zero(sn3_cube_noml, sn3_truth):
    """A dropped WVT bin can present an all-False mask; it must not raise."""
    mask = np.zeros((sn3_truth["dimx"], sn3_truth["dimy"]), dtype=bool)
    _, summed = sn3_cube_noml.extract_spectrum_region(mask)
    assert np.all(summed == 0)
