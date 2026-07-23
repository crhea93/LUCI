"""
Tests for background subtraction and the region / single-pixel fit entry points.

These cover a cluster of bugs that all shared a cause: ``fit_cube``,
``fit_region`` and ``fit_pixel`` each re-implemented the same orchestration by
hand, so a fix or a parameter added to one silently never reached the others.

  * **B3** -- ``fit_region`` forwarded ``bkg`` but not ``bkgType``, and the
    consumer gates subtraction on ``bkgType``, so the background was ignored
    entirely. A silently wrong answer, not a crash.
  * **B5** -- the unbinned PCA branch of ``fit_pixel`` referenced ``x_pix`` /
    ``y_pix``, which do not exist in that scope (``NameError``).
  * **B6** -- ``fit_pixel`` computed ``bkg * binning ** 2`` with ``binning``
    defaulting to ``None`` (``TypeError``), so its own documented defaults could
    not be used together.
  * **B9** -- ``fit_region`` did not forward ``fit_function`` to ``save_fits``,
    so it wrote different filenames than ``fit_cube`` for the same fit.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from LUCI.instrument.filters import PCABackgroundUnsupportedError, pca_scale_indices

SN3_LINES = ["Halpha", "NII6548", "NII6583", "SII6716", "SII6731"]


def _small_mask(truth: dict) -> np.ndarray:
    mask = np.zeros((truth["dimx"], truth["dimy"]), dtype=bool)
    mask[9:11, 9:11] = True
    return mask


# --------------------------------------------------------------------------
# B3 -- fit_region must actually subtract the background
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_fit_region_subtracts_the_background(sn3_cube_noml, sn3_truth):
    """
    Fitting with a background must not give the same answer as fitting without.

    Before the fix ``bkgType`` never reached the fit, so passing ``bkg`` changed
    nothing at all and the two fits below were bit-identical.

    Compared on the *returned* arrays rather than the written FITS files: both
    calls write to the same output directory, and ``fits.open`` memmaps by
    default, so a previously "read" map silently reflects the later write.
    """
    mask = _small_mask(sn3_truth)
    # Every spaxel in the fixture carries the same lines, so this "background"
    # contains real signal -- subtracting it must visibly change the fit.
    _, background = sn3_cube_noml.extract_spectrum(2, 6, 2, 6, mean=True)

    _, _, plain_flux, _, _ = sn3_cube_noml.fit_region(SN3_LINES, "sincgauss", [1] * 5, [1] * 5, mask, n_threads=1)
    plain_flux = np.array(plain_flux, copy=True)

    _, _, subtracted_flux, _, _ = sn3_cube_noml.fit_region(
        SN3_LINES, "sincgauss", [1] * 5, [1] * 5, mask, bkg=background, n_threads=1
    )

    # atol=0 matters: fluxes are ~1e-16, so numpy's default atol=1e-8 would call
    # any two of these arrays "close" and the assertion would never fire.
    assert not np.allclose(
        plain_flux, subtracted_flux, rtol=1e-6, atol=0.0
    ), "passing bkg made no difference -- background was ignored (B3)"
    # Concretely: subtracting a background that contains the lines themselves
    # must strip most of the fitted flux at a fitted pixel.
    assert subtracted_flux[9, 9, 0] < 0.5 * plain_flux[9, 9, 0]


@pytest.mark.slow
def test_fit_region_names_outputs_like_fit_cube(sn3_cube_noml, sn3_truth):
    """B9: fit_function must appear in fit_region's filenames, as it does for fit_cube."""
    mask = _small_mask(sn3_truth)
    sn3_cube_noml.fit_region(SN3_LINES, "sincgauss", [1] * 5, [1] * 5, mask, n_threads=1)
    written = os.listdir(os.path.join(sn3_cube_noml.output_dir, "Velocity"))
    assert any(
        "sincgauss" in name for name in written
    ), f"fit_function missing from fit_region output names: {written[:5]}"


# --------------------------------------------------------------------------
# B5 / B6 -- fit_pixel's own defaults must work
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_fit_pixel_works_with_default_arguments(sn3_cube_noml, sn3_truth):
    """
    B6: the documented defaults (binning=None, bkg=None) raised TypeError from
    ``bkg * binning ** 2`` before anything was fitted.
    """
    axis, sky, result = sn3_cube_noml.fit_pixel(SN3_LINES, "sincgauss", [1] * 5, [1] * 5, 10, 10)
    assert axis.shape == sky.shape
    assert np.isfinite(result["velocities"][0])
    assert result["velocities"][0] == pytest.approx(sn3_truth["velocity_kms"], abs=15.0)


@pytest.mark.slow
def test_fit_pixel_subtracts_a_standard_background(sn3_cube_noml, sn3_truth):
    """With a background supplied, the fitted continuum should drop."""
    _, background = sn3_cube_noml.extract_spectrum(2, 6, 2, 6, mean=True)
    _, _, plain = sn3_cube_noml.fit_pixel(SN3_LINES, "sincgauss", [1] * 5, [1] * 5, 10, 10)
    _, _, subtracted = sn3_cube_noml.fit_pixel(SN3_LINES, "sincgauss", [1] * 5, [1] * 5, 10, 10, bkg=background)
    assert subtracted["continuum"] < plain["continuum"]


def test_fit_pixel_rejects_an_unknown_background_type(sn3_cube_noml):
    """An unrecognised bkgType must raise rather than silently do nothing."""
    with pytest.raises(ValueError, match="bkgType"):
        sn3_cube_noml.fit_pixel(["Halpha"], "sincgauss", [1], [1], 10, 10, bkgType="nonsense")


# --------------------------------------------------------------------------
# The PCA scaling window, now shared instead of copy-pasted three times
# --------------------------------------------------------------------------


def test_pca_scale_indices_bracket_the_expected_wavelengths(sn3_cube_noml):
    """SN3's PCA scaling window is 670-675 nm; the indices must bracket it."""
    lower, upper = pca_scale_indices("SN3", sn3_cube_noml.spectrum_axis)
    axis = np.asarray(sn3_cube_noml.spectrum_axis)
    assert 1e7 / axis[lower] == pytest.approx(675.0, abs=1.0)
    assert 1e7 / axis[upper] == pytest.approx(670.0, abs=1.0)
    # Wavelength decreases as wavenumber increases, so the slice is ordered.
    assert lower < upper


def test_pca_scale_indices_raises_for_an_uncharacterised_filter(sn3_cube_noml):
    """
    C-filters have no PCA background window. The three original copies of this
    logic called quit() here, killing the interpreter from library code.
    """
    with pytest.raises(PCABackgroundUnsupportedError):
        pca_scale_indices("C1", sn3_cube_noml.spectrum_axis)
