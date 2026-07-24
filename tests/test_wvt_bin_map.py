"""
Tests for how WVT bin membership is stored.

Every bin used to be saved as its own full-field boolean mask: 4.2 MB on a SITELLE field to record
the dozen pixels the bin holds. An 8,992-bin run cost 36 GB, and a full-field run at ~341,000 bins
would have needed about 1.4 TB -- more disk than the machine has, so the run could not finish at
all. A single int32 label map is 17 MB whatever the bin count.

Runs made before the label map still have to be fittable, so the legacy directory is read too.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from luci.analysis.wvt import BIN_MAP_UNASSIGNED, BIN_DIR, load_bin_regions, save_bin_map


@pytest.fixture
def bin_map():
    """A 6x5 field: bin 0 twice, bin 1 three times, bin 2 once, the rest unclaimed."""
    m = np.full((6, 5), BIN_MAP_UNASSIGNED, dtype=np.int32)
    m[0, 0] = 0
    m[3, 4] = 0
    m[1, 1] = 1
    m[1, 2] = 1
    m[5, 0] = 1
    m[2, 3] = 2
    return m


def test_round_trips_bin_membership(tmp_path, bin_map):
    save_bin_map(str(tmp_path), bin_map)
    regions = load_bin_regions(str(tmp_path))
    assert len(regions) == 3
    assert set(zip(*regions[0])) == {(0, 0), (3, 4)}
    assert set(zip(*regions[1])) == {(1, 1), (1, 2), (5, 0)}
    assert set(zip(*regions[2])) == {(2, 3)}


def test_unassigned_pixels_belong_to_no_bin(tmp_path, bin_map):
    """
    -1 has to mean 'no bin', not bin 0.

    The map used to be initialised to zeros, and 0 is a real bin number, so every pixel no bin
    claimed was silently folded into bin 0 and fitted along with it.
    """
    save_bin_map(str(tmp_path), bin_map)
    regions = load_bin_regions(str(tmp_path))
    claimed = {pixel for region in regions for pixel in zip(*region)}
    assert len(claimed) == 6  # not 30
    assert (0, 1) not in claimed


def test_indices_can_be_used_directly_on_a_cube_shaped_array(tmp_path, bin_map):
    """The indices are (xs, ys) in cube order, so they must index a (dimx, dimy) array."""
    save_bin_map(str(tmp_path), bin_map)
    field = np.zeros((6, 5))
    for value, (xs, ys) in enumerate(load_bin_regions(str(tmp_path)), start=1):
        field[xs, ys] = value
    assert field[0, 0] == 1 and field[3, 4] == 1
    assert field[1, 1] == 2 and field[5, 0] == 2
    assert field[2, 3] == 3
    assert field[4, 4] == 0


def test_shape_mismatch_is_caught_rather_than_mis_indexed(tmp_path, bin_map):
    save_bin_map(str(tmp_path), bin_map)
    with pytest.raises(ValueError, match="different cube or region"):
        load_bin_regions(str(tmp_path), cube_shape=(10, 10, 300))


def test_saving_clears_masks_from_an_earlier_run(tmp_path, bin_map):
    """Stale per-bin masks left beside a new map would be read as the bins of this run."""
    bin_dir = tmp_path / BIN_DIR
    bin_dir.mkdir()
    for i in range(3):
        np.save(bin_dir / ("bool_bin_map_%d.npy" % i), np.zeros((6, 5), dtype=bool))
    save_bin_map(str(tmp_path), bin_map)
    assert not [f for f in os.listdir(bin_dir) if f.startswith("bool_bin_map_")]
    assert len(load_bin_regions(str(tmp_path))) == 3


def test_legacy_per_bin_masks_are_still_readable(tmp_path, bin_map):
    """Output from a run predating the label map must stay fittable."""
    bin_dir = tmp_path / BIN_DIR
    bin_dir.mkdir()
    n_bins = int(bin_map.max()) + 1
    for b in range(n_bins):
        np.save(bin_dir / ("bool_bin_map_%d.npy" % b), bin_map == b)
    regions = load_bin_regions(str(tmp_path))
    assert len(regions) == n_bins
    assert set(zip(*regions[1])) == {(1, 1), (1, 2), (5, 0)}


def test_legacy_masks_are_ordered_numerically_not_lexically(tmp_path):
    """bool_bin_map_10 must not sort before bool_bin_map_2, or every bin is mislabelled."""
    bin_dir = tmp_path / BIN_DIR
    bin_dir.mkdir()
    for b in range(12):
        mask = np.zeros((20, 4), dtype=bool)
        mask[b, 0] = True
        np.save(bin_dir / ("bool_bin_map_%d.npy" % b), mask)
    regions = load_bin_regions(str(tmp_path))
    for b, (xs, ys) in enumerate(regions):
        assert (xs[0], ys[0]) == (b, 0)


def test_missing_bin_map_is_an_explicit_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="create_wvt"):
        load_bin_regions(str(tmp_path))


def test_label_map_is_far_smaller_than_per_bin_masks(tmp_path):
    """The point of the change: size must not scale with the number of bins."""
    shape = (256, 256)
    n_bins = 400
    m = np.full(shape, BIN_MAP_UNASSIGNED, dtype=np.int32)
    flat = m.ravel()
    flat[:n_bins] = np.arange(n_bins)
    path = save_bin_map(str(tmp_path), m)
    label_map_bytes = os.path.getsize(path)
    per_bin_bytes = n_bins * shape[0] * shape[1]  # one bool per pixel per bin
    assert label_map_bytes < per_bin_bytes / 50
