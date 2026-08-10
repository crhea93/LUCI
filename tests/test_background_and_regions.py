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
  * **B28** -- ``output_name`` reached ``save_fits`` from nowhere: ``fit_region``
    computed one and dropped it, and ``fit_entire_cube`` forwarded one to a
    ``fit_cube`` that did not accept it (``TypeError`` on every call).
"""

from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

from luci.background.subtraction import combine_pca_coefficients, pca_background, subtract_pca
from luci.instrument.filters import PCABackgroundUnsupportedError, pca_scale_indices

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
# B28 -- output_name must reach save_fits
# --------------------------------------------------------------------------


def test_fit_entire_cube_forwards_only_what_fit_cube_accepts():
    """
    B28: ``fit_entire_cube`` forwarded ``output_name=`` to ``fit_cube``, which had no such
    parameter, so *every* call to it raised ``TypeError``. The whole entry point was unusable
    and nothing noticed, because nothing called it.

    Asserted over the signatures rather than by fitting: this is the wrapper-drift class that
    produced B3, B9 and B27, and a signature comparison catches the next one instantly instead
    of after a ten-minute whole-cube fit.
    """
    from luci.cube import SitelleCube

    wrapper = set(inspect.signature(SitelleCube.fit_entire_cube).parameters) - {"self"}
    target = set(inspect.signature(SitelleCube.fit_cube).parameters) - {"self"}
    assert wrapper <= target, f"fit_entire_cube forwards parameters fit_cube rejects: {sorted(wrapper - target)}"


@pytest.mark.slow
def test_output_name_renames_the_products(sn3_cube_noml, sn3_truth):
    """
    B28: ``fit_region``'s ``output_name`` was computed into a local and never passed on, so a
    caller-supplied name was silently ignored and a region fit's maps overwrote a whole-cube
    fit's -- both derived their filenames from the object name alone.
    """
    mask = _small_mask(sn3_truth)
    sn3_cube_noml.fit_region(SN3_LINES, "sincgauss", [1] * 5, [1] * 5, mask, output_name="CUSTOMNAME", n_threads=1)

    written = os.listdir(os.path.join(sn3_cube_noml.output_dir, "Velocity"))
    assert any(name.startswith("CUSTOMNAME") for name in written), f"output_name ignored: {written[:5]}"
    # The decorations still apply, so runs remain distinguishable by fit function.
    assert any(name.startswith("CUSTOMNAME") and "sincgauss" in name for name in written)
    # And the object name is not silently dropped from a default-named run: see
    # test_fit_region_names_outputs_like_fit_cube, which runs without output_name.


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


# --------------------------------------------------------------------------
# B29 -- a group of pixels' PCA coefficients combine by mean, not sum
# --------------------------------------------------------------------------


def _fake_pca_model(n_x, n_y, n_channels, axis):
    """A one-component PCA model with a distinct coefficient per pixel."""
    vectors = np.array([np.sin(np.linspace(0, 3, n_channels)) + 2.0])
    pca_mean = np.cos(np.linspace(0, 2, n_channels)) + 3.0
    coefficients = np.arange(n_x * n_y, dtype=float).reshape(n_x, n_y, 1) / (n_x * n_y)
    return vectors, pca_mean, coefficients


def test_combined_coefficients_reproduce_the_summed_background_exactly():
    """
    B29: the invariant that decides mean vs sum.

    A pixel's background is ``pca_mean + sum_i c_i v_i``, so for N summed pixels the
    background is ``N * pca_mean + sum_i (sum_p c_ip) v_i``. Rebuilding from the
    *mean* coefficients and multiplying by N reproduces that exactly. Rebuilding
    from the *summed* coefficients does not -- it under-weights ``pca_mean`` by N,
    changing the background's shape, which `subtract_pca`'s rescaling cannot undo.
    """
    n_channels = 40
    axis = np.linspace(14700.0, 15800.0, n_channels)
    vectors, pca_mean, _ = _fake_pca_model(2, 2, n_channels, axis)
    per_pixel = np.array([[0.3], [0.7], [1.1], [2.9]])  # 4 pixels, 1 component

    truth = sum(pca_background(c, vectors, pca_mean) for c in per_pixel)
    combined = pca_background(combine_pca_coefficients(per_pixel), vectors, pca_mean)

    np.testing.assert_allclose(len(per_pixel) * combined, truth, rtol=1e-12)
    # The old summed form is not merely differently scaled -- it is a different shape,
    # so no constant could rescale it onto the truth.
    summed = pca_background(np.nansum(per_pixel, axis=0), vectors, pca_mean)
    assert not np.allclose(summed / np.max(summed), truth / np.max(truth), rtol=1e-3)


def test_combined_coefficients_are_a_no_op_for_a_single_pixel():
    """A one-pixel group must come back unchanged, or unbinned fits would shift."""
    one = np.array([[0.3, -1.2, 4.0]])
    np.testing.assert_allclose(combine_pca_coefficients(one), one[0], rtol=1e-12)


# --------------------------------------------------------------------------
# B27 follow-up -- the PCA background must reach the region / WVT paths
# --------------------------------------------------------------------------


def test_extract_region_for_fit_subtracts_a_pca_background(sn3_cube_noml, sn3_truth):
    """
    Until now ``extract_region_for_fit`` -- and so ``fit_spectrum_region``, ``fit_wvt``
    and ``wvt_fit_region`` -- accepted only a `standard` background. Those paths could
    apply an absorption template but could not run behind the PCA background the
    template was built for.
    """
    mask = _small_mask(sn3_truth)
    xs, ys = sn3_cube_noml.region_indices(mask)
    axis = np.asarray(sn3_cube_noml.spectrum_axis)
    vectors, pca_mean, coefficients = _fake_pca_model(sn3_truth["dimx"], sn3_truth["dimy"], len(axis), axis)

    sky, _, _, _ = sn3_cube_noml.extract_region_for_fit(
        mask,
        bkgType="pca",
        pca_coefficient_array=coefficients,
        pca_vectors=vectors,
        pca_mean=pca_mean,
    )

    raw = sn3_cube_noml.cube_final[xs, ys, :].sum(axis=0, dtype=np.float64)
    background = pca_background(combine_pca_coefficients(coefficients[xs, ys, :]), vectors, pca_mean)
    expected = subtract_pca(raw, background, axis, "SN3")
    np.testing.assert_allclose(sky, expected[~np.isnan(expected)], rtol=1e-10)
    # And it is doing something: the raw sum is not already the answer. atol=0 for the reason
    # given in test_fit_region_subtracts_the_background -- these are ~1e-16 fluxes.
    assert not np.allclose(sky, raw[~np.isnan(expected)], rtol=1e-6, atol=0.0)


def test_an_unknown_bkgtype_raises_in_region_extraction(sn3_cube_noml, sn3_truth):
    with pytest.raises(ValueError, match="must be 'standard', 'pca', or None"):
        sn3_cube_noml.extract_region_for_fit(_small_mask(sn3_truth), bkgType="PCA")


def test_a_mean_region_spectrum_subtracts_the_background_once(sn3_cube_noml, sn3_truth):
    """
    B30: the standard background was scaled by the pixel count regardless of ``mean``,
    so ``mean=True`` over-subtracted it by that count -- the same mistake as B2, which
    was fixed in ``extract_spectrum`` and left standing here.

    A mean spectrum holds one spaxel's worth of flux, so one background comes off.
    """
    mask = _small_mask(sn3_truth)
    _, background = sn3_cube_noml.extract_spectrum(2, 6, 2, 6, mean=True)

    sky, _, _, _ = sn3_cube_noml.extract_region_for_fit(mask, bkg=background, mean=True)

    xs, ys = sn3_cube_noml.region_indices(mask)
    raw_mean = sn3_cube_noml.cube_final[xs, ys, :].sum(axis=0, dtype=np.float64) / xs.size
    expected = raw_mean - background
    np.testing.assert_allclose(sky, expected[~np.isnan(expected)], rtol=1e-10)


# --------------------------------------------------------------------------
# B10 -- cube dimensions must come from the cube
# --------------------------------------------------------------------------


def test_snr_map_defaults_to_this_cubes_extent(sn3_cube_noml, sn3_truth):
    """
    B10: ``create_snr_map`` defaulted to x_max=2048, y_max=2064 -- the standard
    SITELLE detector size -- so on any other cube (a trimmed one, a test fixture,
    a future detector) the defaults indexed far past the data.
    """
    from astropy.io import fits

    # Calling with default bounds is the whole test: before the fix this walked
    # off the end of a 20x20 cube.
    sn3_cube_noml.create_snr_map(method=1, n_threads=1)

    snr_dir = os.path.join(sn3_cube_noml.output_dir, "SNR")
    written = [f for f in os.listdir(snr_dir) if f.endswith(".fits")]
    assert written, "no SNR map written"
    snr = fits.open(os.path.join(snr_dir, written[0]))[0].data
    # SNR maps are stored transposed relative to the cube.
    assert snr.shape == (sn3_truth["dimy"], sn3_truth["dimx"])
    assert np.all(np.isfinite(snr))


def test_no_hardcoded_detector_dimensions_remain():
    """
    Guards against the pattern coming back.

    2048x2064 is the standard SITELLE detector; baking it into defaults or array
    shapes is what B10 was. Comments may still mention it.
    """
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Scans LUCI/cube.py: LuciBase.py is now only a re-export shim, so pointing
    # this at it would pass vacuously.
    source = open(os.path.join(root, "luci", "cube.py")).read()
    offenders = [
        line.strip()
        for line in source.splitlines()
        if re.search(r"\b(2048|2064)\b", line) and not line.strip().startswith("#")
    ]
    assert not offenders, f"hardcoded detector dimensions reintroduced: {offenders}"


def test_pca_scale_indices_raises_for_an_uncharacterised_filter(sn3_cube_noml):
    """
    C-filters have no PCA background window. The three original copies of this
    logic called quit() here, killing the interpreter from library code.
    """
    with pytest.raises(PCABackgroundUnsupportedError):
        pca_scale_indices("C1", sn3_cube_noml.spectrum_axis)


# --------------------------------------------------------------------------
# B23 -- pixel-list selection
# --------------------------------------------------------------------------


def test_pixel_list_selects_only_the_listed_pixels():
    """
    B23: the pixel-list branch started from ``np.ones`` -- every pixel already
    selected -- and then set the listed pixels True, so ``pixel_list=True``
    silently fitted the entire cube instead of the handful of pixels asked for.
    """
    from luci.engine.selection import resolve_mask

    mask = resolve_mask([(2, 3), (4, 5)], header=None, cube_shape=(10, 10), pixel_list=True)
    assert mask.dtype == bool
    assert mask.sum() == 2, "expected exactly the two listed pixels"
    assert mask[2, 3] and mask[4, 5]


def test_resolve_mask_passes_through_a_boolean_array():
    from luci.engine.selection import resolve_mask

    given = np.zeros((6, 6), dtype=bool)
    given[1, 1] = True
    out = resolve_mask(given, header=None, cube_shape=(6, 6))
    np.testing.assert_array_equal(out, given)


def test_resolve_mask_rejects_an_unknown_region_type():
    """An unrecognised value used to print a message and carry on with no mask."""
    from luci.engine.selection import resolve_mask

    with pytest.raises(ValueError, match="Unrecognised region file"):
        resolve_mask("region.txt", header=None, cube_shape=(4, 4))
    with pytest.raises(ValueError, match="No region given"):
        resolve_mask(None, header=None, cube_shape=(4, 4))


def test_fit_cube_subtracts_a_background_given_without_bkgtype(sn3_cube_noml, sn3_truth):
    """
    B26: ``fit_cube(bkg=...)`` with no ``bkgType`` silently ignored the background.

    This is the same defect as B3 but on the other entry point, and it is what
    ``Examples/BasicExample.ipynb`` does -- it extracts a background, plots it,
    then passes it to ``fit_cube`` with no ``bkgType``, so the flagship tutorial
    never actually subtracted anything.
    """
    background = np.ones(sn3_cube_noml.cube_final.shape[2]) * 1e-17

    plain = sn3_cube_noml.fit_cube(SN3_LINES, "sincgauss", [1] * 5, [1] * 5, 5, 8, 5, 8, n_threads=1)
    subtracted = sn3_cube_noml.fit_cube(
        SN3_LINES, "sincgauss", [1] * 5, [1] * 5, 5, 8, 5, 8, bkg=background, n_threads=1
    )
    # flux maps are index 2 of the returned tuple
    assert not np.allclose(
        plain[2], subtracted[2], rtol=1e-9, atol=0.0
    ), "passing bkg to fit_cube made no difference -- background was ignored (B26)"


def test_fit_entire_cube_forwards_its_arguments(sn3_cube_noml, monkeypatch):
    """
    B27: ``fit_entire_cube`` accepted bkg/binning/bayes_bool/output_name/
    uncertainty_bool/n_threads and then called ``fit_cube`` with none of them.
    Every one was silently discarded.
    """
    captured = {}

    def fake_fit_cube(*args, **kwargs):
        captured.update(kwargs)
        return "sentinel"

    monkeypatch.setattr(sn3_cube_noml, "fit_cube", fake_fit_cube)
    background = np.ones(sn3_cube_noml.cube_final.shape[2]) * 1e-17
    result = sn3_cube_noml.fit_entire_cube(
        SN3_LINES, "sincgauss", [1] * 5, [1] * 5, bkg=background, binning=2, n_threads=7
    )
    assert result == "sentinel", "fit_entire_cube must return the fit result, not None"
    assert captured.get("binning") == 2
    assert captured.get("n_threads") == 7
    assert captured.get("bkg") is background
