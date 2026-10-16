"""Tests for background modelling and derived analysis products."""

from __future__ import annotations

import os

import numpy as np
import pytest
from astropy.io import fits

from luci.background.detection import find_background_pixels
from luci.background.pca import create_background_subspace
from luci.background.subtraction import pca_background, subtract_pca, subtract_standard
from luci.instrument.filters import UnsupportedFilterError


def _slice_bounds(cube, wavelength_nm):
    spectral_axis_nm = 1e7 / cube.spectrum_axis
    shifted = (1 + cube.redshift) * wavelength_nm
    delta = 200 * shifted / 299792
    limits = np.array([shifted - 10 * delta, shifted + 10 * delta])
    return np.abs(np.subtract.outer(spectral_axis_nm, limits)).argmin(0)[::-1]


def test_standard_and_pca_background_subtraction(sn3_cube_noml):
    sky = np.arange(6.0)
    background = np.full(6, 0.25)
    assert subtract_standard(sky, None) is sky
    np.testing.assert_allclose(subtract_standard(sky, background), sky - background)
    np.testing.assert_allclose(subtract_standard(sky, background, binning=2), sky - 4 * background)

    coefficients = np.array([2.0, -0.5])
    vectors = np.array([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]])
    mean = np.array([0.5, 0.5, 0.5])
    np.testing.assert_allclose(pca_background(coefficients, vectors, mean), [2.5, -0.5, 2.5])

    spectrum = np.array(sn3_cube_noml.cube_final[5, 5], copy=True)
    unit_background = np.ones_like(spectrum)
    subtracted = subtract_pca(spectrum, unit_background, sn3_cube_noml.spectrum_axis, "SN3")
    assert subtracted.shape == spectrum.shape
    assert np.all(subtracted <= spectrum)


def test_threshold_background_detection_partitions_pixels(tmp_path):
    image = np.full((60, 60), 100_000.0)
    image[10:20, 30:40] = 300_000.0

    background, source = find_background_pixels(
        image, str(tmp_path), bkg_algo="threshold", plot_mask=False
    )

    assert len(background) == 3500
    assert len(source) == 100
    assert [15, 35] in source.tolist()


def test_pca_background_subspace_handles_a_small_cube(sn3_cube_noml, monkeypatch):
    region_shape = (5, 5)
    source = np.array([[2, 2]])
    background = np.array(
        [(x, y) for x in range(region_shape[0]) for y in range(region_shape[1]) if (x, y) != (2, 2)]
    )
    monkeypatch.setattr(
        "luci.background.pca.find_background_pixels",
        lambda *args, **kwargs: (background, source),
    )

    transformed, pca, interpolated, idx_bkg, idx_src, coefficients = create_background_subspace(
        sn3_cube_noml,
        x_min=0,
        x_max=5,
        y_min=0,
        y_max=5,
        n_components=2,
        n_components_keep=2,
        interpolation="nearest",
    )

    assert transformed.shape == (24, 2)
    assert pca.components_.shape == (2, sn3_cube_noml.cube_final.shape[2])
    assert interpolated.shape == (1, 2)
    np.testing.assert_array_equal(idx_bkg, background)
    np.testing.assert_array_equal(idx_src, source)
    assert coefficients.shape == (*sn3_cube_noml.cube_final.shape[:2], 2)
    assert np.all(np.isfinite(coefficients[2, 2]))
    assert os.path.exists(os.path.join(sn3_cube_noml.output_dir, "pca_SN3.pkl"))


def test_slicing_writes_independent_sum_for_each_line(sn3_cube_noml):
    lines = {"Halpha": 656.280, "NII6583": 658.341}

    sn3_cube_noml.slicing(list(lines))

    for line, wavelength in lines.items():
        lower, upper = _slice_bounds(sn3_cube_noml, wavelength)
        expected = np.sum(sn3_cube_noml.cube_final[:, :, lower : upper + 1], axis=2)
        path = os.path.join(sn3_cube_noml.output_dir, f"Slice_{line}", "slice_sum.fits")
        np.testing.assert_allclose(fits.getdata(path), expected, rtol=1e-6)


def test_slicing_subtracts_a_supplied_background(sn3_cube_noml):
    background = np.full(sn3_cube_noml.cube_final.shape[2], 1e-20)
    lower, upper = _slice_bounds(sn3_cube_noml, 656.280)

    sn3_cube_noml.slicing(["Halpha"], bkg=background)

    path = os.path.join(sn3_cube_noml.output_dir, "Slice_Halpha", "slice_sum.fits")
    expected = np.sum(sn3_cube_noml.cube_final[:, :, lower : upper + 1] - background[lower : upper + 1], axis=2)
    np.testing.assert_allclose(fits.getdata(path), expected, rtol=1e-6)


def test_snr_rejects_an_unsupported_filter(sn3_cube_noml, monkeypatch):
    header = dict(sn3_cube_noml.hdr_dict)
    header["FILTER"] = "UNKNOWN"
    monkeypatch.setattr(sn3_cube_noml, "hdr_dict", header)

    with pytest.raises(UnsupportedFilterError, match="UNKNOWN"):
        sn3_cube_noml.create_snr_map(x_max=2, y_max=2, n_threads=1)


def test_snr_method_two_supports_binning(sn3_cube_noml):
    masks = sn3_cube_noml.create_snr_map(
        x_min=0, x_max=8, y_min=0, y_max=8, method=2, binning=2, n_threads=1
    )

    assert len(masks) == 4
    assert all(mask.shape == (4, 4) for mask in masks)
    written = fits.getdata(os.path.join(sn3_cube_noml.output_dir, "SNR", "TESTOBJ_SNR.fits"))
    assert written.shape == (4, 4)
