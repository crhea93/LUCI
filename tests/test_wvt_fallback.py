"""
Tests for the nearest-unassigned fallback in bin accretion.

When a bin cannot grow through its neighbours, accretion needs the nearest still-unassigned pixel
anywhere in the field to seed the next bin. That used to be a scan over every unassigned pixel --
O(n) per bin, O(n^2) over a run, which on a 200k-pixel field was most of the accretion time. It is
now a k-d tree query (`_nearest_unassigned`).

The tree must return *exactly* what the scan did, tie-breaking included: early bin centroids are
simple fractions, so several grid pixels sit at the same distance, and the original always took the
lowest-indexed one. A different choice reseeds a bin, and seeds chain, so a single wrong tie cascades
into a different tessellation -- which is precisely what happened before the tie rule was made
explicit. These pin the fallback against the linear scan it replaces, with ties deliberately forced.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from luci.analysis.wvt import Pixel, _linear_nearest_unassigned, _nearest_unassigned


def build(coords, assigned=()):
    pixels = [Pixel(i, x, y, 1.0) for i, (x, y) in enumerate(coords)]
    for i in assigned:
        pixels[i].add_to_bin(0)
    tree = cKDTree(np.array(coords, dtype=float))
    return pixels, tree


def same_pixel(a, b):
    return a is b or (a.pix_x, a.pix_y, a.pix_number) == (b.pix_x, b.pix_y, b.pix_number)


@pytest.mark.parametrize("seed", range(8))
def test_matches_the_linear_scan_on_random_fields(seed):
    rng = np.random.default_rng(seed)
    coords = [(int(x), int(y)) for x, y in rng.integers(0, 40, size=(300, 2))]
    # unique the coords so indices are unambiguous
    coords = list(dict.fromkeys(coords))
    n = len(coords)
    assigned = rng.choice(n, size=n // 2, replace=False).tolist()
    pixels, tree = build(coords, assigned)
    for _ in range(30):
        px, py = rng.uniform(0, 40), rng.uniform(0, 40)
        got = _nearest_unassigned(px, py, pixels, tree)
        ref = _linear_nearest_unassigned(px, py, pixels)
        assert same_pixel(got, ref), f"mismatch at ({px:.3f},{py:.3f})"


def test_ties_go_to_the_lowest_index():
    """
    A half-integer query point is equidistant from the two pixels bracketing it. The scan takes the
    lower index; the tree must too, whatever order it returns tied points in.
    """
    coords = [(0, 0), (0, 1), (0, 2), (0, 3)]  # a column
    pixels, tree = build(coords)
    # (0, 0.5) is distance 0.5 from both pixel 0 (0,0) and pixel 1 (0,1)
    got = _nearest_unassigned(0.0, 0.5, pixels, tree)
    assert (got.pix_x, got.pix_y) == (0, 0)  # lower index of the tied pair
    assert same_pixel(got, _linear_nearest_unassigned(0.0, 0.5, pixels))


def test_ties_skip_assigned_pixels():
    """The lowest-index rule applies only among the *unassigned* tied pixels."""
    coords = [(0, 0), (0, 1)]
    pixels, tree = build(coords, assigned=[0])  # the lower-index one is taken
    got = _nearest_unassigned(0.0, 0.5, pixels, tree)
    assert (got.pix_x, got.pix_y) == (0, 1)


def test_finds_a_far_lone_survivor():
    """
    When the whole neighbourhood is assigned, the query has to widen until it reaches the one
    unassigned pixel far away -- the case that makes a fixed small k wrong.
    """
    coords = [(i % 20, i // 20) for i in range(400)]  # a 20x20 block
    lone = 399
    assigned = [i for i in range(400) if i != lone]
    pixels, tree = build(coords, assigned)
    got = _nearest_unassigned(0.0, 0.0, pixels, tree)  # query from the opposite corner
    assert got.pix_number == lone
    assert same_pixel(got, _linear_nearest_unassigned(0.0, 0.0, pixels))


def test_all_assigned_returns_none():
    coords = [(0, 0), (1, 1)]
    pixels, tree = build(coords, assigned=[0, 1])
    assert _nearest_unassigned(0.0, 0.0, pixels, tree) is None


def test_low_cap_forces_the_linear_fallback_but_same_answer():
    """Squeezing k_cap makes the tree give up and scan; the result must not change."""
    rng = np.random.default_rng(3)
    coords = list(dict.fromkeys((int(x), int(y)) for x, y in rng.integers(0, 30, size=(200, 2))))
    n = len(coords)
    assigned = rng.choice(n, size=n - 3, replace=False).tolist()  # only 3 left, scattered
    pixels, tree = build(coords, assigned)
    for _ in range(20):
        px, py = rng.uniform(0, 30), rng.uniform(0, 30)
        got = _nearest_unassigned(px, py, pixels, tree, k_cap=1)
        ref = _linear_nearest_unassigned(px, py, pixels)
        assert same_pixel(got, ref)
