"""
Regression tests for how initial parameter estimates reach the optimiser.

These exist because of a defect found while building the Phase 0 safety net:
disabling the machine-learning priors does not fall back to a sensible guess,
it silently destroys the kinematics.  The tests below pin the current behaviour
so it cannot get quietly worse, and state the intended behaviour as strict
xfails that will flip to passing when the fallback is fixed.
"""

from __future__ import annotations

import numpy as np
import pytest

from LUCI.LuciFit import Fit


def _make_fit(cube, model="sincgauss", lines=("Halpha",), ml_bool=True, pixel=(9, 9), **kwargs):
    x, y = pixel
    sky = np.copy(cube.cube_final[x, y, :])
    return Fit(
        sky,
        cube.spectrum_axis,
        cube.wavenumbers_syn,
        model,
        list(lines),
        [1] * len(lines),
        [1] * len(lines),
        trans_filter=cube.transmission_interpolated,
        theta=cube.interferometer_theta[x, y],
        delta_x=cube.hdr_dict["STEP"],
        n_steps=cube.step_nb,
        zpd_index=cube.zpd_index,
        filter=cube.hdr_dict["FILTER"],
        ML_bool=ml_bool,
        resolution=cube.resolution,
        Luci_path=cube.Luci_path,
        **kwargs,
    )


# --------------------------------------------------------------------------
# The defect
# --------------------------------------------------------------------------


def test_ml_disabled_leaves_the_prior_estimates_at_zero(sn3_cube_noml):
    """
    Fit.__init__ initialises vel_ml = 0.0 and broad_ml = 0.0, and nothing ever
    updates them when ML_bool is False.
    """
    fit = _make_fit(sn3_cube_noml, ml_bool=False)
    assert fit.vel_ml == 0.0
    assert fit.broad_ml == 0.0


def test_ml_disabled_yields_a_zero_initial_broadening(sn3_cube_noml):
    """
    line_vals_estimate computes line_broad_est = line_pos * broad_ml / c.

    With broad_ml == 0 that is exactly zero, which is a singular point of both
    the Gaussian and the sinc-Gauss line models.
    """
    fit = _make_fit(sn3_cube_noml, ml_bool=False)
    _, _, sigma_est = fit.line_vals_estimate("Halpha")
    assert sigma_est == 0.0


@pytest.mark.parametrize("model", ["sincgauss", "gaussian"])
def test_ml_disabled_returns_exactly_zero_kinematics(sn3_cube_noml, model):
    """
    CURRENT behaviour, pinned: the optimiser cannot move off a singular start,
    so it returns its initial vector and every velocity and broadening is
    exactly 0.0 -- regardless of what is actually in the spectrum.

    The amplitude and continuum *are* fitted, so the flux maps look entirely
    reasonable.  That is what makes this dangerous: nothing about the output
    announces that the kinematics are fabricated.
    """
    fit = _make_fit(sn3_cube_noml, model=model, ml_bool=False)
    result = fit.fit()
    assert result["velocities"] == [0.0]
    assert result["sigmas"] == [0.0]
    # ... while the amplitude is a genuine, plausible-looking number.
    assert result["amplitudes"][0] > 0.0
    assert np.isfinite(result["continuum"])


def test_ml_disabled_still_recovers_velocity_for_the_sinc_model(sn3_cube_noml, sn3_truth):
    """
    The defect is model-dependent, which is worth stating explicitly.

    A pure sinc has no sigma in its profile -- its width is the fixed
    instrumental sinc_width -- so a zero initial sigma is not a singular point
    and the optimiser can still move the line position.  Velocity comes back
    correct; the broadening is still stuck at zero because nothing constrains it.

    So of the three fit functions, ML_bool=False silently breaks two.
    """
    fit = _make_fit(sn3_cube_noml, model="sinc", ml_bool=False)
    result = fit.fit()
    assert result["velocities"][0] == pytest.approx(sn3_truth["velocity_kms"], abs=15.0)
    assert result["sigmas"][0] == 0.0


@pytest.mark.xfail(strict=True, reason="ML_bool=False has no working prior fallback; fix in Phase 3")
def test_ml_disabled_should_still_recover_the_injected_velocity(sn3_cube_noml, sn3_truth):
    """
    The intended contract.

    Disabling the ML priors should fall back to estimating the line position
    from the data (or to a documented default broadening), not to a guess that
    is guaranteed to be singular.  The docs actively steer users here: LuciFit
    prints "Please set ML_bool=False" for any filter without a trained
    predictor.
    """
    fit = _make_fit(sn3_cube_noml, ml_bool=False)
    result = fit.fit()
    assert result["velocities"][0] == pytest.approx(sn3_truth["velocity_kms"], abs=15.0)
    assert result["sigmas"][0] == pytest.approx(sn3_truth["broadening_kms"], abs=10.0)


# --------------------------------------------------------------------------
# The working path, for contrast
# --------------------------------------------------------------------------


@pytest.mark.ml
def test_ml_enabled_recovers_the_injected_velocity(sn3_cube, sn3_truth):
    pytest.importorskip("keras")
    fit = _make_fit(sn3_cube, ml_bool=True)
    result = fit.fit()
    assert result["velocities"][0] == pytest.approx(sn3_truth["velocity_kms"], abs=15.0)
    assert result["sigmas"][0] == pytest.approx(sn3_truth["broadening_kms"], abs=10.0)


@pytest.mark.ml
def test_ml_prior_lands_near_the_truth_before_fitting(sn3_cube, sn3_truth):
    """
    The predictor itself, independent of the optimiser that follows it.

    Tolerances are wide on purpose.  The predictors were trained on real SITELLE
    spectra, and this cube is synthetic, so the prior is only expected to land
    in the right neighbourhood -- roughly 45 km/s against an injected 100 km/s.
    That is fine: its only job is to give SLSQP a non-singular starting point,
    and the full fit converges to ~99.5 km/s from there.  The test guards
    against the prior becoming garbage or NaN, not against it being imprecise.
    """
    pytest.importorskip("keras")
    fit = _make_fit(sn3_cube, ml_bool=True)
    fit.interpolate_spectrum()
    fit.estimate_priors_ML()
    assert np.isfinite(fit.vel_ml)
    assert np.isfinite(fit.broad_ml)
    assert fit.vel_ml == pytest.approx(sn3_truth["velocity_kms"], abs=100.0)
    assert fit.broad_ml > 0.0, "a zero prior broadening is the singular case"
    assert fit.broad_ml == pytest.approx(sn3_truth["broadening_kms"], abs=30.0)


@pytest.mark.ml
def test_freezing_supplies_priors_without_the_ml_model(sn3_cube_noml, sn3_truth):
    """
    The `initial_values` path is the one working way to fit without ML.

    line_vals_estimate copies initial_values into vel_ml/broad_ml when frozen,
    so the optimiser gets a non-singular start.  This is effectively the
    fallback that ML_bool=False should have had all along.
    """
    fit = _make_fit(
        sn3_cube_noml,
        ml_bool=False,
        initial_values=[sn3_truth["velocity_kms"], sn3_truth["broadening_kms"]],
    )
    assert fit.freeze is True
    result = fit.fit()
    # Velocity and broadening are held at the supplied values, not refit.
    assert result["velocities"][0] == pytest.approx(sn3_truth["velocity_kms"], abs=1.0)
    assert result["sigmas"][0] == pytest.approx(sn3_truth["broadening_kms"], abs=5.0)
