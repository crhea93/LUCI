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

import os

import numpy as np
import pytest
from astropy.io import fits

from luci.analysis.wvt import (
    Bin,
    Bin_data,
    Nearest_Neighbors,
    Pixel,
    Potential_SN,
    Roundness,
    adjacency,
    bin_num_pixel,
    closest_node,
    dist,
    read_in,
)

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


def test_bin_tracks_signal_to_noise_centroid_area_and_membership():
    pixels = [Pixel(0, 0, 0, 3.0), Pixel(1, 1, 0, 4.0)]
    bin_ = Bin(7)
    bin_.add_pixel(pixels[0])
    bin_.add_pixel(pixels[1])
    bin_.CalcCentroid()
    bin_.CalcArea(0.5)

    assert bin_.StN[0] == pytest.approx(7 / np.sqrt(2))
    assert bin_.centroidx[0] == pytest.approx(0.5)
    assert bin_.centroidy[0] == pytest.approx(0.0)
    assert bin_.Area[0] == pytest.approx(0.5)
    assert bin_num_pixel([bin_], pixels[1]) == 0

    bin_.remove_pixel(pixels[0])
    assert bin_.StN[0] == pytest.approx(4.0)
    bin_.clear_pixels()
    assert bin_.pixels == [] and bin_.StN[0] == 0


def test_pixel_state_and_nearest_neighbors_on_a_three_by_three_grid():
    pixels = [Pixel(i, x, y, 1.0) for i, (x, y) in enumerate((x, y) for x in range(3) for y in range(3))]
    Nearest_Neighbors(pixels)
    center = pixels[4]
    assert {(pixel.pix_x, pixel.pix_y) for pixel in center.neighbors} == {(0, 1), (1, 0), (1, 2), (2, 1)}

    bin_ = Bin(1)
    center.add_to_bin(bin_)
    assert center.assigned_to_bin and center.assigned_bin is bin_
    center.clear_bin()
    assert not center.assigned_to_bin and center.assigned_bin is None


def test_wvt_distance_adjacency_closest_node_and_potential_snr():
    origin = Pixel(0, 0, 0, 3.0)
    neighbor = Pixel(1, 1, 0, 4.0)
    farther = Pixel(2, 3, 0, 5.0)
    origin.add_neighbor(neighbor, 1, 0)
    bin_ = Bin(0)
    bin_.add_pixel(origin)
    bin_.CalcCentroid()

    assert dist(0, 0, 3, 4) == pytest.approx(5.0)
    assert closest_node(bin_, [neighbor, farther]) is neighbor
    assert adjacency(bin_, neighbor)
    assert not adjacency(bin_, farther)
    assert Potential_SN(bin_, neighbor) == pytest.approx(7 / np.sqrt(2))
    assert bin_.pixels == [origin]


def test_read_and_write_wvt_boundary_files(tmp_path):
    snr_path = tmp_path / "snr.fits"
    fits.writeto(snr_path, np.arange(9, dtype=float).reshape(3, 3))
    pixels, x_min, x_max, y_min, y_max = read_in(snr_path)
    assert (x_min, x_max, y_min, y_max) == (0, 3, 0, 3)
    assert len(pixels) == 9
    assert pixels[-1].StN == 8

    bin_ = Bin(0)
    bin_.add_pixel(pixels[0])
    bin_.CalcCentroid()
    bin_.scale_length[0] = 1.25
    Bin_data([bin_], [], 0, 0, str(tmp_path), "bins")
    assert os.path.exists(tmp_path / "bins.txt")
    assert os.path.exists(tmp_path / "bins_bins.txt")
