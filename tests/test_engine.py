"""Tests for the shared cube-fit orchestration helpers."""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from luci.engine.runner import deep_image_cutout, resolve_initial_values


def test_resolve_initial_values_returns_false_without_a_map_pair():
    assert resolve_initial_values([False]) == (False, False)


def test_resolve_initial_values_accepts_arrays():
    velocity = np.arange(9).reshape(3, 3)
    broadening = velocity + 10

    actual_velocity, actual_broadening = resolve_initial_values([velocity, broadening])

    assert actual_velocity is velocity
    assert actual_broadening is broadening


def test_resolve_initial_values_loads_fits_maps(tmp_path):
    velocity = np.arange(9).reshape(3, 3)
    broadening = velocity + 10
    velocity_path = tmp_path / "velocity.fits"
    broadening_path = tmp_path / "broadening.fits"
    fits.writeto(velocity_path, velocity)
    fits.writeto(broadening_path, broadening)

    actual_velocity, actual_broadening = resolve_initial_values([str(velocity_path), str(broadening_path)])

    np.testing.assert_array_equal(actual_velocity, velocity)
    np.testing.assert_array_equal(actual_broadening, broadening)


def test_deep_image_cutout_creates_missing_image(sn3_cube_noml):
    cutout = deep_image_cutout(sn3_cube_noml, 3, 9, 4, 12, binning=None)

    assert cutout.data.shape == (8, 6)
    expected = np.nansum(sn3_cube_noml.cube_final, axis=2).T[4:12, 3:9]
    np.testing.assert_allclose(cutout.data, expected)
