"""
Tests for skipping blank sky when building the WVT pixel list.

Most of a SITELLE field is empty. Binning it is not merely wasted effort -- both accretion and
refinement scale with the pixel count -- it is wasted on bins that never reach the S/N target and
are discarded by the `0.5 * StN_Target` test regardless. An S/N floor drops those pixels up front.

Pixels below the floor must be left *unfitted*, not cropped away: the maps stay full-field so they
still overlay the deep image and the fit_cube products.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from luci.analysis.wvt import read_in


@pytest.fixture
def snr_map(tmp_path):
    """A 10x8 S/N map (stored y, x) with a bright 3x3 patch and a NaN."""
    snr = np.full((8, 10), 2.0, dtype=np.float32)
    snr[2:5, 3:6] = 25.0
    snr[0, 0] = np.nan
    path = str(tmp_path / "snr.fits")
    fits.writeto(path, snr, overwrite=True)
    return path, snr


def test_no_floor_keeps_every_pixel(snr_map):
    path, snr = snr_map
    pixels, _, x_max, _, y_max = read_in(path)
    assert len(pixels) == snr.size
    assert (x_max, y_max) == (10, 8)


def test_floor_keeps_only_the_bright_pixels(snr_map):
    path, _ = snr_map
    pixels, _, _, _, _ = read_in(path, snr_floor=10.0)
    assert len(pixels) == 9
    assert {(p.pix_x, p.pix_y) for p in pixels} == {(x, y) for x in range(3, 6) for y in range(2, 5)}


def test_floor_reports_the_full_map_extent(snr_map):
    """The bounds must stay full-field, or the plots and the bin map get the wrong shape."""
    path, _ = snr_map
    _, x_min, x_max, y_min, y_max = read_in(path, snr_floor=10.0)
    assert (x_min, x_max, y_min, y_max) == (0, 10, 0, 8)


def test_floor_drops_nan_pixels(snr_map):
    """A NaN S/N is not a detection. `SNR > floor` is False for NaN, which is the intent."""
    path, _ = snr_map
    pixels, _, _, _, _ = read_in(path, snr_floor=1.0)
    assert (0, 0) not in {(p.pix_x, p.pix_y) for p in pixels}
    assert len(pixels) == 79  # 80 pixels above 1.0, minus the NaN


def test_pixels_keep_their_snr_so_bins_still_accrete_correctly(snr_map):
    path, _ = snr_map
    pixels, _, _, _, _ = read_in(path, snr_floor=10.0)
    assert all(p.StN == pytest.approx(25.0) for p in pixels)


def test_pixel_numbering_stays_contiguous_after_filtering(snr_map):
    path, _ = snr_map
    pixels, _, _, _, _ = read_in(path, snr_floor=10.0)
    assert [p.pix_number for p in pixels] == list(range(len(pixels)))


def test_a_floor_above_the_whole_map_is_an_error_not_an_empty_run(snr_map):
    """Silently binning nothing would look like a completed run that produced no bins."""
    path, _ = snr_map
    with pytest.raises(ValueError, match="nothing to bin"):
        read_in(path, snr_floor=1000.0)
