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
"""

from __future__ import annotations

import numpy as np
import pytest

from luci.background.absorption import SPEED_OF_LIGHT, build_absorption_template, subtract_absorption
from luci.config import LINE_DICT

SN3_LINES = ["Halpha", "NII6548", "NII6583", "SII6716", "SII6731"]


class StubCube:
    """
    The three attributes ``build_absorption_template`` reads off a cube.

    A real cube cannot be used for the shift tests: the synthetic fixture has no
    stellar absorption in it, and injecting a trough means writing a cube. This
    stub takes the ``(xs, ys)`` branch of ``SitelleCube.region_indices``, which is
    the same path ``fit_wvt`` uses.
    """

    def __init__(self, cube_final, spectrum_axis):
        self.cube_final = cube_final
        self.spectrum_axis = spectrum_axis

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
