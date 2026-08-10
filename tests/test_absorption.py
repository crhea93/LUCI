"""
Tests for stellar absorption continuum removal.

Nothing in this area had any coverage: the ``absorp`` hook existed on ``fit_calc``
and ``fit_pixel``, was never reachable from ``fit_region`` or the WVT paths
(**B27**), and the template it consumes could only be built by a module whose
functions referenced ``self`` and so raised ``NameError`` on any call.

  * ``subtract_absorption`` is pinned on its two invariants -- a flat template is
    exactly a no-op, and the continuum *level* survives the subtraction -- plus
    the length check that is the one way to misuse it silently.
  * ``build_absorption_template`` is pinned on the property that motivates it:
    de-redshifting each spaxel before stacking keeps the absorption trough as
    deep as it is in one spaxel, where a plain average over a velocity spread
    smears it into something shallower and wider.
  * And on the ordering that makes it usable next to a background-subtracted fit:
    the sky comes off each spaxel *before* the Doppler shift, so what is stacked
    is the stellar continuum alone rather than the stellar continuum plus a sky
    the fit will remove again.
"""

from __future__ import annotations

import numpy as np
import pytest

from luci.background.absorption import SPEED_OF_LIGHT, build_absorption_template, subtract_absorption
from luci.background.subtraction import pca_background, subtract_pca
from luci.config import LINE_DICT
from luci.fitting.absorption import (
    emission_mask,
    fit_absorption,
    measure_absorption_width,
    resolve_absorption_width,
)

SN3_LINES = ["Halpha", "NII6548", "NII6583", "SII6716", "SII6731"]


class StubCube:
    """
    The attributes ``build_absorption_template`` reads off a cube.

    A real cube cannot be used for the shift tests: the synthetic fixture has no
    stellar absorption in it, and injecting a trough means writing a cube. This
    stub takes the ``(xs, ys)`` branch of ``SitelleCube.region_indices``, which is
    the same path ``fit_wvt`` uses. ``hdr_dict`` is read only on the PCA path,
    where the filter picks the background's scaling window.
    """

    def __init__(self, cube_final, spectrum_axis, filter_name="SN3"):
        self.cube_final = cube_final
        self.spectrum_axis = spectrum_axis
        self.hdr_dict = {"FILTER": filter_name}

    def region_indices(self, region):
        return np.asarray(region[0], dtype=np.intp), np.asarray(region[1], dtype=np.intp)


def _trough_cube(velocities, trough_center=15237.0, trough_width=10.0, depth=0.4):
    """
    An N x 1 cube whose every spaxel is a flat continuum with one absorption
    trough, each spaxel's trough redshifted by its own velocity.

    The trough is 10 cm^-1 wide, which is about what 5 A of stellar Halpha
    absorption comes to at this wavenumber -- narrow enough that a few hundred
    km/s of velocity spread genuinely smears it.

    Returns the stub cube, the (xs, ys) selection, and the velocity map.
    """
    axis = np.linspace(14700.0, 15800.0, 800)
    velocities = np.asarray(velocities, dtype=float)
    n = len(velocities)
    cube = np.zeros((n, 1, len(axis)))
    for i, vel in enumerate(velocities):
        beta = vel / SPEED_OF_LIGHT
        # Where this spaxel's rest-frame trough is observed.
        observed_center = trough_center * np.sqrt((1 - beta) / (1 + beta))
        cube[i, 0, :] = 1.0 - depth * np.exp(-0.5 * ((axis - observed_center) / trough_width) ** 2)
    xs = np.arange(n)
    ys = np.zeros(n, dtype=int)
    # LUCI's maps are (n_y, n_x), indexed [y, x].
    vel_map = velocities.reshape(1, n)
    return StubCube(cube, axis), (xs, ys), vel_map


# --------------------------------------------------------------------------
# subtract_absorption
# --------------------------------------------------------------------------


def test_no_template_leaves_the_spectrum_alone():
    sky = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert subtract_absorption(sky, None) is sky


def test_a_flat_template_is_exactly_a_no_op():
    """
    A template with no shape carries no absorption, so it must remove nothing.

    Exact, not approximate: ``template / median(template)`` is 1.0 in every
    channel, so the expression collapses to ``sky - level + level``.
    """
    rng = np.random.default_rng(0)
    sky = 10.0 + rng.normal(0, 0.1, 200)
    np.testing.assert_allclose(subtract_absorption(sky, np.full(200, 3.7)), sky)


def test_the_continuum_level_survives_the_subtraction():
    """
    The point of adding the median back: the fit still needs a continuum to fit.

    A subtraction that left the spectrum near zero would take the continuum map
    with it, and every amplitude is measured relative to that continuum.
    """
    axis = np.linspace(14700.0, 15800.0, 800)
    template = 1.0 - 0.4 * np.exp(-0.5 * ((axis - 15237.0) / 25.0) ** 2)
    sky = 50.0 * template  # A spectrum that is pure absorbed continuum

    cleaned = subtract_absorption(sky, template)

    edge = int(len(sky) * 0.25)
    assert np.nanmedian(cleaned[edge:-edge]) == pytest.approx(np.nanmedian(sky[edge:-edge]), rel=1e-6)


def test_the_trough_is_removed_but_the_emission_line_is_not():
    """The whole point: absorption flattens out, emission survives."""
    axis = np.linspace(14700.0, 15800.0, 2201)  # 0.5 cm^-1, fine enough to sample the line
    template = 1.0 - 0.4 * np.exp(-0.5 * ((axis - 15237.0) / 25.0) ** 2)
    line = 5.0 * np.exp(-0.5 * ((axis - 15237.0) / 3.0) ** 2)
    sky = 50.0 * template + line

    cleaned = subtract_absorption(sky, template)

    # Away from the line, the spectrum is now flat: the trough is gone.
    trough_wing = np.abs(axis - 15237.0) < 60
    off_line = trough_wing & (np.abs(axis - 15237.0) > 15)
    assert np.ptp(cleaned[off_line]) < 0.02 * np.ptp(sky[off_line])
    # The line is still there, and still about as tall as it was.
    assert cleaned.max() - np.nanmedian(cleaned) == pytest.approx(line.max(), rel=0.05)


def test_a_template_of_the_wrong_length_raises():
    """
    The one misuse that would otherwise be silent.

    numpy broadcasts nothing here -- a 469-channel template against a
    468-channel spectrum is a plain shape error -- but a template built on a
    *rebinned* axis of the right length would quietly subtract the wrong
    wavelength from every channel, which is why the check names the axis it wants.
    """
    with pytest.raises(ValueError, match="must be on the cube's full spectral axis"):
        subtract_absorption(np.ones(400), np.ones(399))


# --------------------------------------------------------------------------
# build_absorption_template
# --------------------------------------------------------------------------


def test_zero_velocities_reproduce_the_plain_mean_spectrum():
    """With nothing to de-redshift, the stack is just the average."""
    cube, region, vel_map = _trough_cube([0.0, 0.0, 0.0])

    template = build_absorption_template(cube, region, vel_map)

    expected = cube.cube_final[region[0], region[1], :].mean(axis=0)
    np.testing.assert_allclose(template, expected, rtol=1e-12)


def test_de_redshifting_keeps_the_trough_deep():
    """
    The reason the shift exists.

    Three spaxels spread over 800 km/s put their troughs ~20 cm^-1 apart, twice
    the trough's own width. Stacked as-is, that is a shallower, wider trough --
    and subtracting it would over-correct the wings and under-correct the core.
    De-redshifted first, the stack is as deep as a single spaxel.
    """
    cube, region, vel_map = _trough_cube([-400.0, 0.0, 400.0], depth=0.4)

    template = build_absorption_template(cube, region, vel_map)
    naive = cube.cube_final[region[0], region[1], :].mean(axis=0)

    template_depth = np.nanmax(template) - np.nanmin(template)
    naive_depth = np.nanmax(naive) - np.nanmin(naive)
    assert template_depth == pytest.approx(0.4, rel=0.05)
    assert naive_depth < 0.5 * template_depth


def test_the_template_sits_at_the_regions_mean_velocity():
    """
    Not at rest: ``subtract_absorption`` works channel by channel against spectra
    observed at roughly the region's velocity, so the template has to be in that
    frame. A rest-frame template would be offset by the mean velocity -- 300 km/s
    is ~15 cm^-1 at Halpha, several channels.
    """
    mean_vel = 300.0
    cube, region, vel_map = _trough_cube([mean_vel - 100, mean_vel, mean_vel + 100], trough_center=15237.0)

    template = build_absorption_template(cube, region, vel_map)

    beta = mean_vel / SPEED_OF_LIGHT
    expected_center = 15237.0 * np.sqrt((1 - beta) / (1 + beta))
    found_center = cube.spectrum_axis[np.nanargmin(template)]
    axis_step = float(np.diff(cube.spectrum_axis)[0])
    assert abs(found_center - expected_center) <= axis_step
    # And that really is displaced from the rest position, i.e. the test can fail.
    assert abs(expected_center - 15237.0) > 5 * axis_step


def test_spaxels_beyond_vel_max_are_skipped():
    """A wild velocity is a failed fit, not a fast star; it must not enter the stack."""
    cube, region, vel_map = _trough_cube([0.0, 0.0, 90000.0])

    template = build_absorption_template(cube, region, vel_map, vel_max=500)

    # Identical to stacking only the two good spaxels.
    good = build_absorption_template(cube, (region[0][:2], region[1][:2]), vel_map)
    np.testing.assert_allclose(template, good, rtol=1e-12)


def test_nan_velocities_are_skipped_rather_than_poisoning_the_stack():
    cube, region, vel_map = _trough_cube([0.0, 0.0, 0.0])
    vel_map = vel_map.copy()
    vel_map[0, 2] = np.nan

    template = build_absorption_template(cube, region, vel_map)

    assert np.isfinite(template).all()


def test_nan_channels_are_averaged_over_not_propagated():
    """
    One spaxel's dead channel must not blank that channel of the template.

    The channel is averaged over the spaxels that do have it -- the count is
    per-channel, not per-spaxel.
    """
    cube, region, vel_map = _trough_cube([0.0, 0.0, 0.0])
    cube.cube_final[1, 0, 200] = np.nan

    template = build_absorption_template(cube, region, vel_map)

    assert np.isfinite(template[200])
    assert template[200] == pytest.approx(cube.cube_final[[0, 2], 0, 200].mean())


# --------------------------------------------------------------------------
# The background has to come off before the stack
# --------------------------------------------------------------------------


def test_the_sky_is_removed_before_the_shift_not_after():
    """
    The ordering that makes a template usable alongside a background-subtracted fit.

    Every spaxel carries the same sky line at the same *observed* wavenumber, while
    their stellar troughs sit at different ones. Removing the sky in the observed
    frame cancels it exactly, leaving a template of the stellar continuum alone --
    identical to the one built from a sky-free cube. Remove it after the Doppler
    shift instead and each spaxel's sky line lands somewhere else, so it survives
    the stack as three residual spikes.
    """
    velocities = [-400.0, 0.0, 400.0]
    clean, region, vel_map = _trough_cube(velocities)
    sky_line = 0.6 * np.exp(-0.5 * ((clean.spectrum_axis - 15100.0) / 4.0) ** 2)
    dirty = StubCube(clean.cube_final + sky_line, clean.spectrum_axis)

    template = build_absorption_template(dirty, region, vel_map, bkg=sky_line, bkgType="standard")

    np.testing.assert_allclose(template, build_absorption_template(clean, region, vel_map), rtol=1e-12)
    # And the removal is doing real work: left in, the sky line survives the stack as three spikes
    # of about a third its height (0.6 / 3 spaxels), where the corrected template is flat continuum.
    raw = build_absorption_template(dirty, region, vel_map)
    assert np.nanmax(raw) == pytest.approx(1.0 + 0.6 / 3, rel=0.05)
    assert np.nanmax(template) == pytest.approx(1.0, rel=1e-6)


def test_a_bkg_without_a_bkgtype_is_taken_as_standard():
    """Matches `fit_calc`'s allowance (B26), so the template is built as the fit will run."""
    clean, region, vel_map = _trough_cube([0.0, 0.0])
    sky_line = 0.5 * np.exp(-0.5 * ((clean.spectrum_axis - 15100.0) / 4.0) ** 2)
    dirty = StubCube(clean.cube_final + sky_line, clean.spectrum_axis)

    implicit = build_absorption_template(dirty, region, vel_map, bkg=sky_line)
    explicit = build_absorption_template(dirty, region, vel_map, bkg=sky_line, bkgType="standard")

    np.testing.assert_allclose(implicit, explicit, rtol=1e-12)


def test_the_pca_background_is_removed_per_spaxel():
    """
    On the PCA path each spaxel has its *own* background, rebuilt from its own
    coefficients -- so the coefficients have to be indexed [x, y] like the cube,
    not [y, x] like the maps. Distinct coefficients per spaxel are what makes a
    transposed read give a different answer.
    """
    cube, region, vel_map = _trough_cube([0.0, 0.0, 0.0])
    axis = cube.spectrum_axis
    vectors = np.array([np.exp(-0.5 * ((axis - 15100.0) / 4.0) ** 2)])
    pca_mean = np.zeros_like(axis)
    # cube_final is (x, y, channel), so the coefficient array is indexed the same way.
    coefficients = np.zeros((cube.cube_final.shape[0], 1, 1))
    coefficients[:, 0, 0] = [0.1, 0.2, 0.3]
    cube.cube_final = cube.cube_final + np.array([[0.1], [0.2], [0.3]])[:, :, None] * vectors[0]

    template = build_absorption_template(
        cube,
        region,
        vel_map,
        bkgType="pca",
        pca_coefficient_array=coefficients,
        pca_vectors=vectors,
        pca_mean=pca_mean,
    )

    # Same operation, spelled out with the public helpers, at zero velocity so no shift intervenes.
    expected = np.mean(
        [
            subtract_pca(cube.cube_final[i, 0, :], pca_background(coefficients[i, 0], vectors, pca_mean), axis, "SN3")
            for i in range(3)
        ],
        axis=0,
    )
    np.testing.assert_allclose(template, expected, rtol=1e-12)
    assert not np.allclose(template, build_absorption_template(cube, region, vel_map))


def test_pca_without_its_model_says_which_arrays_are_missing():
    """
    One message at the call site, not a TypeError per spaxel a minute into the stack.
    """
    cube, region, vel_map = _trough_cube([0.0, 0.0])

    with pytest.raises(ValueError, match="pca_coefficient_array, pca_vectors, pca_mean"):
        build_absorption_template(cube, region, vel_map, bkgType="pca")


def test_standard_without_a_bkg_raises():
    cube, region, vel_map = _trough_cube([0.0, 0.0])

    with pytest.raises(ValueError, match="needs a bkg spectrum"):
        build_absorption_template(cube, region, vel_map, bkgType="standard")


def test_an_unknown_bkgtype_raises():
    cube, region, vel_map = _trough_cube([0.0, 0.0])

    with pytest.raises(ValueError, match="must be 'standard', 'pca', or None"):
        build_absorption_template(cube, region, vel_map, bkgType="PCA")


def test_no_usable_velocity_raises_rather_than_returning_nan():
    cube, region, vel_map = _trough_cube([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="finite velocity"):
        build_absorption_template(cube, region, np.full_like(vel_map, np.nan))


def test_an_empty_region_raises():
    cube, _, vel_map = _trough_cube([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="selects no pixels"):
        build_absorption_template(cube, (np.array([], dtype=int), np.array([], dtype=int)), vel_map)


# --------------------------------------------------------------------------
# Against a real cube
# --------------------------------------------------------------------------


def test_velocity_map_orientation_is_checked(cube_truth_factory, luci_factory):
    """
    A map passed as [x, y] must raise, not silently read the wrong velocities.

    This matters because the deleted ``LuciAbsorp`` indexed its velocity map
    [x, y] while every map LUCI writes is [y, x], so the natural thing to pass is
    the wrong way round -- and reading velocities from the transpose is silently
    wrong, not an error.

    Deliberately on a non-square cube: a shape check is all that stands between
    the two orientations, so on a square field (like the default fixture, and
    unlike a real 2048 x 2064 SITELLE cube) a transposed map is undetectable.
    The error message names the orientation for that reason.
    """
    truth = cube_truth_factory("SN3", dimx=20, dimy=30)
    cube = luci_factory(truth, ML_bool=False)
    mask = np.zeros((truth["dimx"], truth["dimy"]), dtype=bool)
    mask[9:11, 9:11] = True
    wrong_way = np.zeros((truth["dimx"], truth["dimy"]))  # [x, y] instead of [y, x]

    with pytest.raises(ValueError, match=r"expected .*n_y, n_x"):
        cube.build_absorption_template(mask, wrong_way)


def test_template_from_a_real_cube_is_on_the_spectral_axis(sn3_cube_noml, sn3_truth):
    """
    The contract that makes the template feedable to ``absorp=``: same length as
    ``cube.spectrum_axis``, so ``subtract_absorption`` accepts it.
    """
    mask = np.zeros((sn3_truth["dimx"], sn3_truth["dimy"]), dtype=bool)
    mask[8:12, 8:12] = True
    vel_map = np.full((sn3_truth["dimy"], sn3_truth["dimx"]), 100.0)

    template = sn3_cube_noml.build_absorption_template(mask, vel_map)

    assert template.shape == sn3_cube_noml.spectrum_axis.shape
    # Feeding it straight back to the consumer must not raise on length.
    subtract_absorption(sn3_cube_noml.cube_final[10, 10, :], template)


@pytest.mark.slow
def test_fit_region_applies_the_absorption_template(sn3_cube_noml, sn3_truth):
    """
    B27 -- ``fit_region`` did not accept ``absorp`` at all, so a region fit could
    not have its stellar continuum removed while ``fit_cube`` could.

    Compared on the returned arrays rather than the written FITS products, for the
    reason spelled out in ``test_fit_region_subtracts_the_background``: both calls
    write to the same output directory and ``fits.open`` memmaps.
    """
    mask = np.zeros((sn3_truth["dimx"], sn3_truth["dimy"]), dtype=bool)
    mask[9:11, 9:11] = True
    # A trough sitting right under the fixture's Halpha, which is where a stellar one would be.
    beta = sn3_truth["velocity_kms"] / SPEED_OF_LIGHT
    halpha = 1e7 / LINE_DICT["Halpha"] * np.sqrt((1 - beta) / (1 + beta))
    axis = np.asarray(sn3_cube_noml.spectrum_axis, dtype=float)
    template = 1.0 - 0.3 * np.exp(-0.5 * ((axis - halpha) / 30.0) ** 2)

    _, _, plain_flux, _, _ = sn3_cube_noml.fit_region(SN3_LINES, "sincgauss", [1] * 5, [1] * 5, mask, n_threads=1)
    plain_flux = np.array(plain_flux, copy=True)
    _, _, corrected_flux, _, _ = sn3_cube_noml.fit_region(
        SN3_LINES, "sincgauss", [1] * 5, [1] * 5, mask, absorp=template, n_threads=1
    )

    # The selection is square, so the fitted pixels are [9:11, 9:11] in the (y, x) maps too.
    # atol=0 matters: fluxes are ~1e-16, so numpy's default atol=1e-8 would call any two of
    # these arrays "close" and the assertion would never fire.
    assert not np.allclose(
        plain_flux[9:11, 9:11], corrected_flux[9:11, 9:11], rtol=1e-6, atol=0.0
    ), "passing absorp made no difference -- the template never reached the fit (B27)"
    # Directional, not just different: filling in an absorption trough under the line gives back
    # flux, so the corrected Halpha must be the brighter of the two.
    assert corrected_flux[9, 9, 0] > plain_flux[9, 9, 0]


# --------------------------------------------------------------------------
# The fitted component (B22): measuring the trough instead of templating it
# --------------------------------------------------------------------------


def _trough_spectrum(depth=0.3, broadening_kms=200.0, continuum=1.0, noise=0.0, seed=0):
    """A continuum with one Gaussian trough at Halpha, optionally with noise."""
    axis = np.linspace(14700.0, 15800.0, 800)
    position = 1e7 / LINE_DICT["Halpha"]
    sigma = position * broadening_kms / SPEED_OF_LIGHT
    spectrum = continuum - depth * continuum * np.exp(-0.5 * ((axis - position) / sigma) ** 2)
    if noise:
        spectrum = spectrum + np.random.default_rng(seed).normal(0, noise, len(axis))
    return axis, spectrum, position


def test_fit_absorption_recovers_a_clean_trough():
    """
    With no emission in the way and the width supplied correctly, depth and velocity
    come back essentially exactly -- so any error later is the emission's doing, not
    the estimator's.
    """
    axis, spectrum, _ = _trough_spectrum(depth=0.3)

    result = fit_absorption(axis, spectrum, 1e-3, LINE_DICT["Halpha"], [], 30.0)

    assert result.success
    assert result.depth == pytest.approx(0.3, rel=0.02)
    assert result.velocity == pytest.approx(0.0, abs=5.0)


def test_the_broadening_is_echoed_not_fitted():
    """
    The reported broadening is the input. It cannot be measured from this spectrum --
    the trough's centre is where the emission line is -- so it is reported to record
    what produced the depth, and this test pins that it is not silently a fit result.
    """
    axis, spectrum, _ = _trough_spectrum(depth=0.3, broadening_kms=200.0)

    told_wrong = fit_absorption(axis, spectrum, 1e-3, LINE_DICT["Halpha"], [], 30.0, absorption_broadening_kms=350.0)

    assert told_wrong.broadening == pytest.approx(350.0, rel=0.01)
    # And being told the wrong width biases the depth, which is exactly why it is an input
    # the caller has to get right rather than something to leave at the default and trust.
    assert told_wrong.depth != pytest.approx(0.3, rel=0.05)


def test_no_trough_gives_no_depth():
    axis, spectrum, _ = _trough_spectrum(depth=0.0, noise=1e-3)

    result = fit_absorption(axis, spectrum, 1e-3, LINE_DICT["Halpha"], [], 30.0)

    assert result.depth < 0.01


def test_too_few_channels_fails_soft():
    """
    A window that cannot constrain the fit returns success=False and a zero profile, so
    the caller's correction is a no-op rather than a crash or a wild trough.
    """
    axis = np.linspace(15230.0, 15240.0, 6)  # Far narrower than one core

    result = fit_absorption(axis, np.ones(6), 1e-3, LINE_DICT["Halpha"], [], 30.0)

    assert not result.success
    np.testing.assert_allclose(result.profile(axis), np.zeros(6))


def test_emission_mask_scales_with_the_line_width():
    """The mask is sized off the *emission* broadening, with a floor so a zero prior still masks."""
    axis = np.linspace(14700.0, 15800.0, 800)
    position = 1e7 / LINE_DICT["Halpha"]

    narrow = emission_mask(axis, [position], 10.0)
    broad = emission_mask(axis, [position], 200.0)

    assert broad.sum() < narrow.sum()
    # The floor: a zero broadening estimate must still mask the core.
    assert emission_mask(axis, [position], 0.0).sum() < len(axis)


@pytest.mark.slow
def test_fitted_absorption_recovers_an_injected_trough_end_to_end(cube_truth_factory, luci_factory):
    """
    The whole feature through `fit_pixel`: a 0.30 trough injected into the cube comes back
    as a depth, the emission velocity is unharmed, and the maps' broadening echoes the input.

    The tolerance is a measurement, not a guess, and it is symmetric because the estimator is
    unbiased here: over 400 noise realisations the mean recovered depth matches the injected one
    to within 0.004 for troughs of 0.2 and deeper. What it carries is a scatter of about 0.07 per
    spaxel at this cube's 10% continuum noise -- set by the noise and by how many channels survive
    the emission mask, not by the trough -- so this allows +/-0.22, about three sigma, on a single
    spaxel. Tighten it by averaging, not by wishing: the error goes as 1/sqrt(N).

    An earlier version of this test asserted 70-100% of truth on the theory that emission flux
    filling the masked core biased the depth low. That was wrong twice over: the recovered depth
    does not move at all with emission from 0 to 30x the continuum, and the apparent shortfall
    was one noise realisation seen at several depths -- every fixture cube shares a seed, so the
    same draw produced the same absolute error each time and looked like a constant bias. The
    old band would also have failed for any draw landing high, which this fixture's happens not
    to do.
    """
    truth = cube_truth_factory("SN3", absorption_depth=0.3)
    cube = luci_factory(truth, ML_bool=False)
    lines = ["Halpha", "NII6548", "NII6583"]

    _, _, plain = cube.fit_pixel(lines, "sincgauss", [1] * 3, [1] * 3, 10, 10)
    _, _, corrected = cube.fit_pixel(lines, "sincgauss", [1] * 3, [1] * 3, 10, 10, absorption_bool=True)

    assert corrected["absorption_depth"] == pytest.approx(0.3, abs=0.22)  # ~3 sigma of 0.07
    assert corrected["absorption_velocity"] == pytest.approx(truth["velocity_kms"], abs=50.0)
    assert corrected["absorption_broadening"] == pytest.approx(200.0, rel=0.01)  # The supplied width
    # Filling the trough in gives back flux, and leaves the kinematics where they were.
    assert corrected["fluxes"][0] > plain["fluxes"][0]
    assert corrected["velocities"][0] == pytest.approx(truth["velocity_kms"], abs=15.0)


@pytest.mark.slow
def test_a_cube_without_absorption_is_left_alone(sn3_cube_noml):
    """
    Enabling the option on a spectrum with no trough must not invent one, or every fit in
    a field would be quietly altered wherever the stars are faint.
    """
    lines = ["Halpha", "NII6548", "NII6583"]
    _, _, plain = sn3_cube_noml.fit_pixel(lines, "sincgauss", [1] * 3, [1] * 3, 10, 10)
    _, _, corrected = sn3_cube_noml.fit_pixel(lines, "sincgauss", [1] * 3, [1] * 3, 10, 10, absorption_bool=True)

    assert corrected["absorption_depth"] < 0.02
    assert corrected["fluxes"][0] == pytest.approx(plain["fluxes"][0], rel=1e-3)


@pytest.mark.slow
def test_absorption_maps_are_written_only_when_asked(cube_truth_factory, luci_factory):
    """
    The three extra products appear when the option is on and not otherwise: downstream
    scripts glob these directories, so an ordinary fit's output layout must not change.
    """
    import glob
    import os

    truth = cube_truth_factory("SN3", absorption_depth=0.25)
    cube = luci_factory(truth, ML_bool=False)
    lines = ["Halpha", "NII6548"]

    cube.fit_cube(lines, "sincgauss", [1] * 2, [1] * 2, 8, 12, 8, 12, n_threads=1)
    assert not glob.glob(os.path.join(cube.output_dir, "*absorption*"))

    cube.fit_cube(lines, "sincgauss", [1] * 2, [1] * 2, 8, 12, 8, 12, absorption_bool=True, n_threads=1)
    written = sorted(os.path.basename(p) for p in glob.glob(os.path.join(cube.output_dir, "*absorption*")))
    assert [name.split("absorption_")[-1] for name in written] == [
        "broadening.fits",
        "depth.fits",
        "velocity.fits",
    ]


# --------------------------------------------------------------------------
# Measuring the width from a template, so the caller does not have to know it
# --------------------------------------------------------------------------


def _template(width_kms, depth=0.3, scale=1.0, n=842):
    axis = np.linspace(14400.0, 15600.0, n)
    rest = 1e7 / LINE_DICT["Halpha"]
    sigma = rest * width_kms / SPEED_OF_LIGHT
    return axis, scale * (1.0 - depth * np.exp(-0.5 * ((axis - rest) / sigma) ** 2))


@pytest.mark.parametrize("width", [120.0, 200.0, 310.0, 450.0])
def test_the_width_can_be_measured_from_a_template(width):
    """
    The measurement `fit_absorption` cannot make. It works here for one reason: a template has
    no emission in it, so nothing is masked and the trough's own peak is observed -- which is
    exactly what breaks the per-spaxel case.
    """
    axis, template = _template(width)

    measured = measure_absorption_width(axis, template)

    assert measured.success
    assert measured.broadening == pytest.approx(width, rel=0.02)
    assert measured.depth == pytest.approx(0.3, rel=0.02)


@pytest.mark.parametrize("scale", [1.0, 1e-17, 1e3])
def test_measuring_the_width_is_independent_of_flux_scale(scale):
    """
    Regression: real templates are ~1e-17, which makes the sum of squares ~1e-32 -- small enough
    that L-BFGS-B's convergence test passes at the starting point and it returns the initial
    guess without moving. That silently gave every real template the 200 km/s default. The fit
    normalises first; depth and width are both scale-free, so nothing is lost.
    """
    axis, template = _template(310.0, scale=scale)

    measured = measure_absorption_width(axis, template)

    assert measured.success
    assert measured.broadening == pytest.approx(310.0, rel=0.02)


def test_a_template_with_no_trough_reports_no_width():
    """
    Otherwise the amplitude sits on its zero bound and the width is whatever the optimiser
    happened to hold -- a meaningless number that would then bias every depth in the field.
    """
    axis, flat = _template(310.0, depth=0.0)

    assert not measure_absorption_width(axis, flat).success


def test_the_width_is_taken_from_the_caller_then_the_template_then_the_default():
    """`resolve_absorption_width`'s precedence, which is what makes the width automatic."""
    axis, template = _template(310.0)

    assert resolve_absorption_width(axis, None, 275.0) == pytest.approx(275.0)
    assert resolve_absorption_width(axis, template, None) == pytest.approx(310.0, rel=0.02)
    assert resolve_absorption_width(axis, None, None) == pytest.approx(200.0)
    # An explicit width wins even when a template is present.
    assert resolve_absorption_width(axis, template, 275.0) == pytest.approx(275.0)


@pytest.mark.slow
def test_a_measured_width_beats_the_blind_default(cube_truth_factory, luci_factory):
    """
    The point of the whole mechanism. The fixture's stellar width is 310 km/s, not the 200 km/s
    default, and the depth scales roughly as 1/width -- so guessing costs you a third of the
    answer while measuring lands inside the per-spaxel scatter.
    """
    truth = cube_truth_factory(
        "SN3", absorption_depth=0.30, absorption_broadening_kms=310.0, amplitude=3.0e-17, continuum=1.0e-17
    )
    stellar = cube_truth_factory(
        "SN3", absorption_depth=0.30, absorption_broadening_kms=310.0, amplitude=0.0, continuum=1.0e-17
    )
    cube = luci_factory(truth, ML_bool=False)
    stellar_cube = luci_factory(stellar, ML_bool=False)
    lines = ["Halpha", "NII6548", "NII6583"]

    n_x, n_y = stellar_cube.cube_final.shape[:2]
    region = np.zeros((n_x, n_y), dtype=bool)
    region[2:18, 2:18] = True
    template = stellar_cube.build_absorption_template(region, np.full((n_y, n_x), 100.0))

    width = stellar_cube.measure_absorption_width(template)
    assert width == pytest.approx(310.0, rel=0.10)

    _, _, measured = cube.fit_pixel(
        lines, "sincgauss", [1] * 3, [1] * 3, 10, 10, absorption_bool=True, absorption_broadening_kms=width
    )
    _, _, blind = cube.fit_pixel(lines, "sincgauss", [1] * 3, [1] * 3, 10, 10, absorption_bool=True)

    # Within ~3 sigma (0.07) with the measured width; badly over-deep with the default. The gap
    # between the two is systematic, not noise: both read the same spaxel and the same draw.
    assert measured["absorption_depth"] == pytest.approx(0.30, abs=0.22)
    assert blind["absorption_depth"] > measured["absorption_depth"] + 0.10


def test_a_single_velocity_can_be_passed_instead_of_a_map():
    """
    The safe fallback when no *stellar* velocity map exists, and the reason it is supported:
    without it, the only 2D velocity array most users have to hand is the emission-line map,
    which is the gas's velocity field and is unconstrained over a starlight region anyway.
    A scalar smears the trough by the internal stellar spread but never shifts a spaxel the
    wrong way.
    """
    cube, region, vel_map = _trough_cube([100.0, 100.0, 100.0])

    from_scalar = build_absorption_template(cube, region, 100.0)
    from_map = build_absorption_template(cube, region, vel_map)

    np.testing.assert_allclose(from_scalar, from_map, rtol=1e-12)
