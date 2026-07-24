"""
Tests for the weighted Voronoi assignment step.

`Rebin_Pixels` assigns every pixel to the bin minimising distance / scale_length. Written as a
Python list comprehension over all bins for every pixel it is O(N_pixels * N_bins), which for a
full SITELLE field works out to about 18 days across the five WVT iterations. It is now chunked
array arithmetic, so these tests pin it against the brute-force definition -- including the
tie-breaking rule, because which bin wins a tie decides the tessellation.
"""

from __future__ import annotations

import numpy as np
import pytest

from luci.analysis.wvt import Bin_Acc, Nearest_Neighbors, Pixel, Rebin_Pixels, dist, nearest_weighted_bin


def brute_force(pix_x, pix_y, cent_x, cent_y, scale_length):
    """The original list comprehension, kept as the definition of the assignment."""
    out = []
    for px, py in zip(pix_x, pix_y):
        distances = [dist(px, py, cx, cy) / s for cx, cy, s in zip(cent_x, cent_y, scale_length)]
        out.append(distances.index(min(distances)))
    return np.array(out)


@pytest.mark.parametrize("seed", range(5))
def test_matches_brute_force_on_random_configurations(seed):
    rng = np.random.default_rng(seed)
    pix_x = rng.uniform(0, 60, 400)
    pix_y = rng.uniform(0, 60, 400)
    cent_x = rng.uniform(0, 60, 25)
    cent_y = rng.uniform(0, 60, 25)
    scale = rng.uniform(0.5, 3.0, 25)
    np.testing.assert_array_equal(
        nearest_weighted_bin(pix_x, pix_y, cent_x, cent_y, scale),
        brute_force(pix_x, pix_y, cent_x, cent_y, scale),
    )


def test_scale_length_actually_weights_the_choice():
    """A more distant bin wins if its scale length is large enough -- that is the 'weighted' part."""
    pix_x, pix_y = np.array([0.0]), np.array([0.0])
    cent_x, cent_y = np.array([1.0, 4.0]), np.array([0.0, 0.0])
    # Unweighted, bin 0 is closer
    assert nearest_weighted_bin(pix_x, pix_y, cent_x, cent_y, np.array([1.0, 1.0]))[0] == 0
    # Weighted so that 4/5 < 1/1, bin 1 takes it
    assert nearest_weighted_bin(pix_x, pix_y, cent_x, cent_y, np.array([1.0, 5.0]))[0] == 1


def test_ties_go_to_the_lowest_bin_index():
    """`list.index(min(...))` took the first minimum, and so must the array version."""
    pix_x, pix_y = np.array([0.0]), np.array([0.0])
    cent_x, cent_y = np.array([2.0, -2.0, 2.0]), np.array([0.0, 0.0, 0.0])
    scale = np.array([1.0, 1.0, 1.0])
    assert nearest_weighted_bin(pix_x, pix_y, cent_x, cent_y, scale)[0] == 0


def test_chunking_does_not_change_the_result():
    """The memory ceiling is an implementation detail; every chunk size must agree."""
    rng = np.random.default_rng(7)
    pix_x, pix_y = rng.uniform(0, 40, 300), rng.uniform(0, 40, 300)
    cent_x, cent_y = rng.uniform(0, 40, 17), rng.uniform(0, 40, 17)
    scale = rng.uniform(0.5, 2.0, 17)
    reference = nearest_weighted_bin(pix_x, pix_y, cent_x, cent_y, scale)
    for chunk_bytes in (8, 136, 1024, 1 << 20):
        np.testing.assert_array_equal(
            nearest_weighted_bin(pix_x, pix_y, cent_x, cent_y, scale, chunk_bytes=chunk_bytes),
            reference,
        )


def test_no_bins_is_an_error_not_an_empty_answer():
    empty = np.array([])
    with pytest.raises(ValueError, match="no bins"):
        nearest_weighted_bin(np.array([0.0]), np.array([0.0]), empty, empty, empty)


@pytest.mark.parametrize("scale_range", [(0.9, 1.1), (0.5, 3.0), (0.05, 20.0), (1.0, 1.0)])
def test_tree_path_is_exact_for_any_spread_of_scale_lengths(scale_range):
    """
    The k-d tree only narrows candidates; the answer must still equal brute force.

    The spread of scale lengths is what decides how far the search has to reach, so a wide spread is
    the case that would expose a bound that is too tight. A tessellation quietly differing from the
    definition is exactly the failure this has to rule out.
    """
    rng = np.random.default_rng(11)
    n_bins = 300
    pix_x, pix_y = rng.uniform(0, 200, 2000), rng.uniform(0, 200, 2000)
    cent_x, cent_y = rng.uniform(0, 200, n_bins), rng.uniform(0, 200, n_bins)
    scale = rng.uniform(*scale_range, n_bins)
    np.testing.assert_array_equal(
        nearest_weighted_bin(pix_x, pix_y, cent_x, cent_y, scale),
        brute_force(pix_x, pix_y, cent_x, cent_y, scale),
    )


def test_brute_force_fallback_agrees_with_the_tree():
    """Squeezing max_candidates forces the fallback branch; it must not change the answer."""
    rng = np.random.default_rng(5)
    n_bins = 200
    pix_x, pix_y = rng.uniform(0, 100, 800), rng.uniform(0, 100, 800)
    cent_x, cent_y = rng.uniform(0, 100, n_bins), rng.uniform(0, 100, n_bins)
    scale = rng.uniform(0.05, 20.0, n_bins)  # wide, so the bound often fails
    expected = brute_force(pix_x, pix_y, cent_x, cent_y, scale)
    for max_candidates in (1, 2, 8, 1000):
        np.testing.assert_array_equal(
            nearest_weighted_bin(pix_x, pix_y, cent_x, cent_y, scale, max_candidates=max_candidates),
            expected,
        )


def test_degenerate_scale_lengths_do_not_crash_the_tree_path():
    """A bin with zero scale length scores inf and can never win, as in the original."""
    rng = np.random.default_rng(2)
    n_bins = 50
    pix_x, pix_y = rng.uniform(0, 50, 200), rng.uniform(0, 50, 200)
    cent_x, cent_y = rng.uniform(0, 50, n_bins), rng.uniform(0, 50, n_bins)
    scale = rng.uniform(0.5, 2.0, n_bins)
    scale[7] = 0.0
    result = nearest_weighted_bin(pix_x, pix_y, cent_x, cent_y, scale)
    assert 7 not in set(result.tolist())
    assert np.all((result >= 0) & (result < n_bins))


def test_ties_go_to_the_lowest_bin_index_on_the_tree_path():
    """The tree returns candidates ordered by distance, so tie-breaking needs explicit handling."""
    # A ring of equidistant, equally weighted centroids around the origin, plus filler so the
    # bin count clears the tree threshold.
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    cent_x = np.concatenate([np.cos(angles) * 5, np.arange(100, 140, dtype=float)])
    cent_y = np.concatenate([np.sin(angles) * 5, np.zeros(40)])
    scale = np.ones(cent_x.size)
    pix = np.zeros(40)
    result = nearest_weighted_bin(pix, pix, cent_x, cent_y, scale)
    expected = brute_force(pix, pix, cent_x, cent_y, scale)
    np.testing.assert_array_equal(result, expected)


def make_field(side, seed=0):
    rng = np.random.default_rng(seed)
    snr = rng.normal(4.2, 0.8, (side, side)).clip(0.5, 12.5)
    pixels = [Pixel(r * side + c, c, r, snr[r][c]) for r in range(side) for c in range(side)]
    Nearest_Neighbors(pixels)
    return pixels


def test_rebin_pixels_assigns_every_pixel_exactly_once():
    pixels = make_field(30)
    bins = Bin_Acc(pixels, 0.436, 20.0, 0.3)
    successful = Rebin_Pixels(bins, pixels, 0.436, 20.0)
    assigned = sum(len(b.pixels) for b in successful)
    assert assigned == len(pixels)
    assert all(p.assigned_to_bin for p in pixels)


def test_rebin_pixels_agrees_with_the_brute_force_assignment():
    """End to end: the bin each pixel lands in must be the brute-force winner."""
    pixels = make_field(30, seed=3)
    bins = Bin_Acc(pixels, 0.436, 20.0, 0.3)
    cent_x = [b.centroidx_prev[0] for b in bins]
    cent_y = [b.centroidy_prev[0] for b in bins]
    scale = [b.scale_length_prev[0] for b in bins]
    expected = brute_force([p.pix_x for p in pixels], [p.pix_y for p in pixels], cent_x, cent_y, scale)
    Rebin_Pixels(bins, pixels, 0.436, 20.0)
    actual = np.array([bins.index(p.assigned_bin) for p in pixels])
    np.testing.assert_array_equal(actual, expected)
