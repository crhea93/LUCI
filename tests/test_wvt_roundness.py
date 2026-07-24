"""
Tests for the WVT roundness criterion.

`Roundness` compares the bin's maximum centroid distance against the radius of a circle of equal
area. Both are lengths, so the ratio is a pure shape measure and must not depend on the physical
size of a pixel. It used to: `rad_max` was in pixel indices while `rad_equiv` carried a factor of
`pixel_length`, so roundness scaled as 1/pixel_length. At the `wvt_fit_region` default of
pixel_size=0.436 a two-pixel bin scored 0.437, above the 0.3 default criterion, and no bin could
ever accrete a second pixel -- fatal for any cube whose per-pixel S/N is below the target.
"""

from __future__ import annotations

import pytest

from luci.analysis.wvt import Roundness

ROUNDNESS_CRIT = 0.3  # wvt_fit_region default


class FakePixel:
    def __init__(self, x, y):
        self.pix_x = x
        self.pix_y = y


class FakeBin:
    def __init__(self, coords):
        self.pixels = [FakePixel(x, y) for x, y in coords]


@pytest.mark.parametrize("pixel_length", [0.0000436, 0.436, 1.0, 2.5, 100.0])
def test_roundness_is_independent_of_pixel_size(pixel_length):
    """It is a shape measure -- rescaling the pixel must not change it."""
    bin_ = FakeBin([(0, 0)])
    reference = Roundness(FakeBin([(0, 0)]), FakePixel(1, 0), 1.0)
    assert Roundness(bin_, FakePixel(1, 0), pixel_length) == pytest.approx(reference)


@pytest.mark.parametrize("pixel_length", [0.0000436, 0.436, 1.0])
def test_two_adjacent_pixels_are_round_enough_to_merge(pixel_length):
    """The regression: at pixel_size=0.436 this scored 0.437 and blocked all accretion."""
    roundness = Roundness(FakeBin([(0, 0)]), FakePixel(1, 0), pixel_length)
    assert roundness < ROUNDNESS_CRIT


def test_compact_bin_beats_a_straggly_one():
    """The criterion must still do its job: reject strung-out bins."""
    compact = Roundness(FakeBin([(0, 0), (1, 0), (0, 1)]), FakePixel(1, 1), 1.0)
    strung_out = Roundness(FakeBin([(0, 0), (1, 0), (2, 0)]), FakePixel(3, 0), 1.0)
    assert compact < strung_out
    assert compact < ROUNDNESS_CRIT
    assert strung_out > ROUNDNESS_CRIT


def test_a_long_line_of_pixels_is_rejected():
    bin_ = FakeBin([(i, 0) for i in range(8)])
    assert Roundness(bin_, FakePixel(8, 0), 1.0) > ROUNDNESS_CRIT
