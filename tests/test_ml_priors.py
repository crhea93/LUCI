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

from LUCI.LuciFit import DEFAULT_BROADENING_KMS, Fit


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
# The non-ML data-driven fallback (B1 fix)
# --------------------------------------------------------------------------
#
# History: ML_bool=False used to leave vel_ml/broad_ml at their 0.0
# initialisers, so line_vals_estimate handed the optimiser an initial sigma of
# exactly zero -- a singular point of the Gaussian and sinc-Gauss models -- and
# SLSQP returned its starting vector unchanged.  Every velocity and broadening
# came back as exactly 0.0 while the amplitude/continuum still fit, so the
# output looked plausible.  estimate_priors_data() now seeds the optimiser from
# the data instead.  The tests below assert the fix.


def test_ml_disabled_seeds_priors_from_the_data_during_fit(sn3_cube_noml):
    """
    The priors start at their 0.0 initialisers and fit() replaces them, via
    estimate_priors_data(), with a data-driven velocity and a non-singular
    default broadening.
    """
    fit = _make_fit(sn3_cube_noml, ml_bool=False)
    assert fit.vel_ml == 0.0 and fit.broad_ml == 0.0  # before fit()
    fit.fit()
    assert fit.broad_ml == pytest.approx(DEFAULT_BROADENING_KMS)  # no longer singular
    assert fit.vel_ml != 0.0  # estimated from the brightest peak


@pytest.mark.parametrize("model", ["sincgauss", "gaussian"])
def test_ml_disabled_now_recovers_kinematics(sn3_cube_noml, sn3_truth, model):
    """
    The core of the B1 fix: what used to return exactly (0.0, 0.0) now recovers
    both the injected velocity and broadening for the two models that were
    silently broken.
    """
    fit = _make_fit(sn3_cube_noml, model=model, ml_bool=False)
    result = fit.fit()
    assert result["velocities"][0] == pytest.approx(sn3_truth["velocity_kms"], abs=15.0)
    assert result["sigmas"][0] == pytest.approx(sn3_truth["broadening_kms"], abs=10.0)
    assert result["amplitudes"][0] > 0.0
    assert np.isfinite(result["continuum"])


def test_ml_disabled_recovers_velocity_for_the_sinc_model(sn3_cube_noml, sn3_truth):
    """
    A pure sinc has no sigma in its profile (its width is the fixed instrumental
    sinc_width), so the fitted 'broadening' is unconstrained and simply stays
    near the seed value -- but the velocity is recovered, as it always was for
    this model.  Documented separately so the sinc broadening's meaninglessness
    is explicit rather than surprising.
    """
    fit = _make_fit(sn3_cube_noml, model="sinc", ml_bool=False)
    result = fit.fit()
    assert result["velocities"][0] == pytest.approx(sn3_truth["velocity_kms"], abs=15.0)
    assert result["sigmas"][0] == pytest.approx(DEFAULT_BROADENING_KMS, abs=15.0)


# --------------------------------------------------------------------------
# The working path, for contrast
# --------------------------------------------------------------------------


@pytest.mark.ml
def test_ml_enabled_recovers_the_injected_velocity(sn3_cube, sn3_truth):
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
