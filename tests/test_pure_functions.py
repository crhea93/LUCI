"""
Unit tests for the side-effect-free helpers.

These are the cheapest coverage available: no cube, no ML model, no I/O.  They
pin the numerical contracts that the fitting code is built on, so Phase 3 can
move these functions between modules with confidence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from LUCI.LuciFitParameters import (
    calculate_broad,
    calculate_broad_err,
    calculate_flux,
    calculate_flux_err,
    calculate_vel,
    calculate_vel_err,
)
from LUCI.LuciFunctions import Gaussian, Sinc, SincGauss
from LUCI.LuciUtility import bin_cube_function, bin_mask, get_quadrant_dims, hessianComp

SPEED_OF_LIGHT = 299792.0
LINE_DICT = {"Halpha": 656.280, "NII6583": 658.341, "NII6548": 654.803}


# --------------------------------------------------------------------------
# Quadrant geometry
# --------------------------------------------------------------------------


@pytest.mark.parametrize("quad_nb", [1, 4, 9])
def test_quadrants_tile_the_cube_exactly(quad_nb):
    """Every pixel must belong to exactly one quadrant, with no gaps or overlap."""
    dimx, dimy = 2048, 2064
    covered = np.zeros((dimx, dimy), dtype=int)
    for iquad in range(quad_nb):
        x0, x1, y0, y1 = get_quadrant_dims(iquad, quad_nb, dimx, dimy)
        covered[x0:x1, y0:y1] += 1
    assert covered.min() == 1
    assert covered.max() == 1


def test_quadrant_out_of_range_raises():
    with pytest.raises(Exception):
        get_quadrant_dims(9, 9, 2048, 2064)


@pytest.mark.parametrize("dimx,dimy", [(20, 20), (100, 100), (2048, 2064)])
def test_quadrants_cover_non_square_and_small_cubes(dimx, dimy):
    covered = np.zeros((dimx, dimy), dtype=int)
    for iquad in range(4):
        x0, x1, y0, y1 = get_quadrant_dims(iquad, 4, dimx, dimy)
        covered[x0:x1, y0:y1] += 1
    assert np.all(covered == 1)


# --------------------------------------------------------------------------
# Velocity / broadening / flux conversions
# --------------------------------------------------------------------------


def test_calculate_vel_roundtrips_a_known_velocity():
    """A line placed at the position implied by v must read back as v."""
    velocity = 137.0
    rest = LINE_DICT["Halpha"]
    position = 1e7 / (rest * (1.0 + velocity / SPEED_OF_LIGHT))
    fit_sol = np.array([1.0, position, 1.0, 0.0])
    got = calculate_vel(0, ["Halpha"], fit_sol, LINE_DICT)
    assert got == pytest.approx(velocity, rel=1e-6)


def test_calculate_vel_is_signed():
    """Blueshift must come back negative, not as a magnitude."""
    rest = LINE_DICT["Halpha"]
    position = 1e7 / (rest * (1.0 - 250.0 / SPEED_OF_LIGHT))
    fit_sol = np.array([1.0, position, 1.0, 0.0])
    assert calculate_vel(0, ["Halpha"], fit_sol, LINE_DICT) == pytest.approx(-250.0, rel=1e-6)


def test_calculate_broad_roundtrips_a_known_broadening():
    broadening = 42.0
    axis_step = 2.0
    # calculate_broad converts sigma (in channels) back to km/s.
    position = 15200.0
    sigma = position * broadening / SPEED_OF_LIGHT
    fit_sol = np.array([1.0, position, sigma, 0.0])
    got = calculate_broad(0, fit_sol, axis_step)
    assert got == pytest.approx(broadening, rel=1e-3)


def test_calculate_vel_err_scales_linearly_with_uncertainty():
    rest = LINE_DICT["Halpha"]
    position = 1e7 / (rest * (1.0 + 100.0 / SPEED_OF_LIGHT))
    fit_sol = np.array([1.0, position, 1.0, 0.0])
    small = calculate_vel_err(0, ["Halpha"], fit_sol, LINE_DICT, np.array([0.0, 0.1, 0.0, 0.0]))
    large = calculate_vel_err(0, ["Halpha"], fit_sol, LINE_DICT, np.array([0.0, 0.2, 0.0, 0.0]))
    assert large == pytest.approx(2.0 * small, rel=1e-6)


def test_calculate_broad_err_is_non_negative():
    fit_sol = np.array([1.0, 15200.0, 1.5, 0.0])
    err = calculate_broad_err(0, fit_sol, 2.0, np.array([0.0, 0.0, 0.05, 0.0]))
    assert err >= 0.0


@pytest.mark.parametrize("model", ["gaussian", "sinc", "sincgauss"])
def test_flux_is_proportional_to_amplitude(model):
    """Doubling the amplitude must double the flux for every model."""
    sinc_width = 2.5
    f1 = calculate_flux(1.0, 1.5, model, sinc_width)
    f2 = calculate_flux(2.0, 1.5, model, sinc_width)
    assert f2 == pytest.approx(2.0 * f1, rel=1e-9)


def test_gaussian_flux_uses_lucis_sinc_scaled_normalisation():
    """
    LUCI's Gaussian flux is *not* the textbook integral amp*sigma*sqrt(2*pi).

    It carries an extra factor of 1.20671/FWHM_COEFF ~= 0.5124, i.e. the sinc
    FWHM coefficient divided by the Gaussian one, even for a pure Gaussian
    model (LuciFitParameters.calculate_flux).  Whether that is physically
    intended is a separate question -- this test exists to pin the published
    behaviour so the refactor cannot change it silently.
    """
    amp, sigma = 3.0, 1.7
    fwhm_coeff = 2.0 * math.sqrt(2.0 * math.log(2.0))
    expected = (1.20671 / fwhm_coeff) * math.sqrt(2.0 * math.pi) * amp * sigma
    assert calculate_flux(amp, sigma, "gaussian", 2.5) == pytest.approx(expected, rel=1e-9)
    # And it is roughly half the textbook value -- stated explicitly so the
    # discrepancy is impossible to miss when reading the suite.
    textbook = amp * sigma * math.sqrt(2.0 * math.pi)
    assert calculate_flux(amp, sigma, "gaussian", 2.5) == pytest.approx(0.5124 * textbook, rel=1e-3)


def test_flux_err_is_non_negative():
    fit_sol = np.array([2.0, 15200.0, 1.5, 0.0])
    unc = np.array([0.1, 0.01, 0.05, 0.0])
    assert calculate_flux_err(0, fit_sol, unc, "gaussian", 2.5) >= 0.0


# --------------------------------------------------------------------------
# Line-shape models
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model_cls,extra", [(Gaussian, ()), (Sinc, (2.5,)), (SincGauss, (2.5,))])
def test_models_peak_at_the_line_position(model_cls, extra):
    axis = np.linspace(15100.0, 15350.0, 2001)
    position = 15232.0
    theta = [1.0, position, 1.2]
    values = model_cls().evaluate(axis, theta, 1, *extra)
    assert axis[int(np.argmax(values))] == pytest.approx(position, abs=1.0)


@pytest.mark.parametrize("model_cls,extra", [(Gaussian, ()), (SincGauss, (2.5,))])
def test_models_are_additive_across_lines(model_cls, extra):
    """Evaluating two lines together equals the sum of evaluating them apart."""
    axis = np.linspace(15100.0, 15350.0, 501)
    a = [1.0, 15232.0, 1.2]
    b = [0.4, 15195.0, 1.2]
    together = model_cls().evaluate(axis, a + b, 2, *extra)
    apart = model_cls().evaluate(axis, a, 1, *extra) + model_cls().evaluate(axis, b, 1, *extra)
    np.testing.assert_allclose(together, apart, rtol=1e-9, atol=0.0)


def test_gaussian_amplitude_is_the_peak_value():
    axis = np.linspace(15200.0, 15270.0, 4001)
    values = Gaussian().evaluate(axis, [2.5, 15232.0, 1.2], 1)
    assert values.max() == pytest.approx(2.5, rel=1e-4)


def test_sincgauss_is_symmetric_about_the_line_centre():
    """The sinc-Gauss profile is an even function of (channel - position)."""
    sinc_width = 2.5
    position = 15232.0
    offsets = np.linspace(0.5, 20.0, 200)
    left = SincGauss().evaluate(position - offsets, [1.0, position, 1.2], 1, sinc_width)
    right = SincGauss().evaluate(position + offsets, [1.0, position, 1.2], 1, sinc_width)
    np.testing.assert_allclose(left, right, rtol=1e-8, atol=1e-12)


def test_sincgauss_is_finite_and_bounded_for_small_nonzero_sigma():
    """
    Corrects an earlier mischaracterisation (see B14 in REFACTOR_BUGS.md).

    The sinc-Gauss was thought to be numerically unstable as sigma -> 0.  It is
    not: for any *non-zero* sigma, down to 1e-8, the profile stays finite and
    bounded by its amplitude.  (The earlier test compared it against LUCI's
    separate Sinc model, which uses a different width convention, so a constant
    ~0.2 offset was misread as instability.)
    """
    axis = np.linspace(15200.0, 15270.0, 2001)
    sinc_width = 2.5
    for sigma in (1.0, 0.1, 1e-3, 1e-5, 1e-8):
        v = np.asarray(SincGauss().function(axis, [1.0, 15232.0, sigma], sinc_width))
        assert np.all(np.isfinite(v)), f"non-finite at sigma={sigma}"
        assert np.nanmax(np.abs(v)) <= 1.0 + 1e-9


def test_sincgauss_at_exactly_zero_sigma_is_guarded():
    """
    The one genuinely singular input: sigma == 0 makes the Dawson form divide by
    zero and return NaN for every channel.  The guard in SincGauss.function
    (B14) substitutes a tiny floor so the result stays finite.  This is the
    degenerate initial guess behind B1; a converged fit never reaches it.
    """
    axis = np.linspace(15200.0, 15270.0, 51)
    v = np.asarray(SincGauss().function(axis, [1.0, 15232.0, 0.0], 2.5))
    assert np.all(np.isfinite(v))
    assert np.nanmax(np.abs(v)) <= 1.0 + 1e-9


# --------------------------------------------------------------------------
# Binning
# --------------------------------------------------------------------------


def test_bin_cube_sums_flux_and_shrinks_shape():
    rng = np.random.default_rng(0)
    cube = rng.random((20, 20, 8))
    from astropy.io import fits

    header = fits.Header()
    for key, value in (
        ("CRPIX1", 10.0),
        ("CRPIX2", 10.0),
        ("CDELT1", -1e-4),
        ("CDELT2", 1e-4),
        ("CRVAL1", 24.0),
        ("CRVAL2", 15.0),
        ("PC1_1", 1.0),
        ("PC1_2", 0.0),
        ("PC2_1", 0.0),
        ("PC2_2", 1.0),
    ):
        header[key] = value

    _, binned = bin_cube_function(cube, header, 2, 0, 20, 0, 20)
    assert binned.shape == (10, 10, 8)
    # Total flux is conserved: binning sums, it does not average.
    assert binned.sum() == pytest.approx(cube.sum(), rel=1e-9)
    # A single bin equals the sum of its four contributing spaxels.
    np.testing.assert_allclose(binned[0, 0, :], cube[0:2, 0:2, :].sum(axis=(0, 1)), rtol=1e-9)


def test_bin_mask_marks_a_bin_containing_any_true_pixel():
    """A bin is selected when any contributing pixel is selected (B8 fixed)."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[3, 5] = True
    binned = bin_mask(mask, 2, 0, 20, 0, 20)
    assert binned.shape == (10, 10)
    assert binned[1, 2]
    assert binned.sum() == 1


def test_bin_mask_returns_a_boolean_mask():
    """
    B8: bin_mask used to divide the finished boolean mask by binning**2 -- a
    line copy-pasted from the flux-averaging path in bin_cube_function -- which
    turned True into 0.25 and the result into a float array.  It survived only
    because the one consumer tests `if mask[x, y]:` and 0.25 is truthy; anything
    that summed the mask or used it for boolean indexing got wrong answers.
    """
    mask = np.zeros((20, 20), dtype=bool)
    mask[3, 5] = True
    binned = bin_mask(mask, 2, 0, 20, 0, 20)
    assert binned.dtype == bool
    assert binned.sum() == 1
    # Usable directly for boolean indexing, which the float version was not.
    assert np.arange(100).reshape(10, 10)[binned].tolist() == [12]


def test_bin_mask_of_all_false_is_all_false():
    binned = bin_mask(np.zeros((20, 20), dtype=bool), 2, 0, 20, 0, 20)
    assert not binned.any()


# --------------------------------------------------------------------------
# Numerical Hessian
# --------------------------------------------------------------------------


def test_hessian_of_a_quadratic_is_its_constant_curvature():
    """For f(x) = 3x0^2 + 5x1^2 the Hessian is diag(6, 10) everywhere."""

    def f(x):
        return 3.0 * x[0] ** 2 + 5.0 * x[1] ** 2

    got = hessianComp(f, np.array([1.0, 1.0]), delta=1e-3)
    np.testing.assert_allclose(np.diag(got), [6.0, 10.0], rtol=1e-4)
    assert got[0, 1] == pytest.approx(0.0, abs=1e-5)


def test_hessian_is_symmetric_for_a_coupled_function():
    def f(x):
        return x[0] ** 2 + 2.0 * x[0] * x[1] + 4.0 * x[1] ** 2

    got = hessianComp(f, np.array([0.5, -0.3]), delta=1e-3)
    np.testing.assert_allclose(got, got.T, rtol=1e-6, atol=1e-8)
    assert got[0, 1] == pytest.approx(2.0, rel=1e-3)
