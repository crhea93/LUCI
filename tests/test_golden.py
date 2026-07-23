"""
Characterization ("golden") tests.

These do not assert that LUCI is *correct*.  They assert that LUCI produces
exactly the numbers it produced before a refactor touched it.  That is the
safety net the whole restructuring plan rests on: any change to fit values shows
up here as a diff, and every such diff must be explainable.

Record a fresh baseline with::

    pytest tests/test_golden.py --record

Review the resulting diff in tests/golden/ before committing it.  Re-recording
to make a failure disappear defeats the entire purpose -- if a fix is *meant* to
change values, re-record in the same commit as the fix so the numeric change is
visible in review.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")

# Tight enough to catch a genuine change in the maths, loose enough to survive
# SLSQP converging a fraction of an ULP differently.  These baselines are
# recorded and compared on the same machine during a refactor; treat a failure
# as a real signal, not as noise to be widened away.
RTOL = 1e-6
ATOL = 1e-30  # fluxes are ~1e-16, so absolute tolerance must be far below that

# Fields captured from the fit.  Keep this list stable -- adding a field
# invalidates every stored baseline.
CAPTURED = (
    "velocities",
    "broadenings",
    "fluxes",
    "amplitudes",
)


class Config:
    """One point in the golden matrix."""

    def __init__(self, ident, filter_name, model, ml, lines, binning=None, box=(8, 10, 8, 10), resolution=None):
        self.id = ident
        self.filter_name = filter_name
        self.model = model
        self.ml = ml
        self.lines = lines
        self.binning = binning
        self.box = box
        self.resolution = resolution


SN3_LINES = ["Halpha", "NII6548", "NII6583", "SII6716", "SII6731"]
SN2_LINES = ["Hbeta", "OIII4959", "OIII5007"]
SN1_LINES = ["OII3726", "OII3729"]

# The matrix is deliberately ML-*on* dominant.
#
# ML_bool=False does not merely skip the priors -- it leaves vel_ml and broad_ml
# at their 0.0 initialisers, so line_vals_estimate() hands the optimiser an
# initial sigma of exactly zero, the line model is singular there, and SLSQP
# returns the starting vector untouched.  Every velocity and broadening comes
# back as exactly 0.0.  Baselines recorded down that path would pin a column of
# zeros and prove nothing about the fitting maths, so ML-off appears here only
# as explicit documentation of the defect (see the *_broken_zeros entries and
# tests/test_ml_priors.py).
CONFIGS = [
    # SN3: every model.  This is the path the overwhelming majority of users take.
    Config("sn3_sincgauss_ml", "SN3", "sincgauss", True, SN3_LINES),
    Config("sn3_gaussian_ml", "SN3", "gaussian", True, SN3_LINES),
    Config("sn3_sinc_ml", "SN3", "sinc", True, SN3_LINES),
    # Binning changes the cube, the header and the WCS cutout, so it needs its
    # own baseline.  Kept to an 8x8 box because ML fits cost seconds per pixel.
    Config("sn3_sincgauss_ml_bin2", "SN3", "sincgauss", True, SN3_LINES, binning=2, box=(0, 8, 0, 8)),
    # Single-line fit: exercises the path where no inter-line constraints are
    # constructed at all.
    Config("sn3_halpha_only_ml", "SN3", "sincgauss", True, ["Halpha"]),
    # NII doublet present -> NII_constraints() adds the amplitude-ratio bound.
    Config("sn3_nii_doublet_ml", "SN3", "sincgauss", True, ["Halpha", "NII6548", "NII6583"]),
    # Other filters pick up different wavelength bounds and reference spectra.
    Config("sn2_sincgauss_ml", "SN2", "sincgauss", True, SN2_LINES),
    Config("sn1_sincgauss_ml", "SN1", "sincgauss", True, SN1_LINES),
    # Documents the ML-off defect. When it is fixed these baselines MUST change;
    # re-record them in the same commit as the fix.
    Config("sn3_sincgauss_noml_broken_zeros", "SN3", "sincgauss", False, SN3_LINES),
    Config("sn3_gaussian_noml_broken_zeros", "SN3", "gaussian", False, SN3_LINES),
]


def _run_fit(config, luci_factory, cube_truth_factory) -> dict:
    truth = cube_truth_factory(config.filter_name)
    cube = luci_factory(truth, ML_bool=config.ml, resolution=config.resolution)
    x_min, x_max, y_min, y_max = config.box
    n = len(config.lines)
    vel, broad, flux, ampl = cube.fit_cube(
        config.lines,
        config.model,
        [1] * n,
        [1] * n,
        x_min,
        x_max,
        y_min,
        y_max,
        binning=config.binning,
        n_threads=1,
    )
    return {
        "velocities": vel,
        "broadenings": broad,
        "fluxes": flux,
        "amplitudes": ampl,
    }


def _to_jsonable(arrays: dict) -> dict:
    return {
        key: {
            "shape": list(np.asarray(arrays[key]).shape),
            "values": np.asarray(arrays[key], dtype=np.float64).ravel().tolist(),
        }
        for key in CAPTURED
    }


def _golden_path(config) -> str:
    return os.path.join(GOLDEN_DIR, f"{config.id}.json")


@pytest.mark.slow
@pytest.mark.parametrize("config", CONFIGS, ids=[c.id for c in CONFIGS])
def test_golden_fit_values(config, luci_factory, cube_truth_factory, record_golden):
    if config.ml:
        pytest.importorskip("keras", reason="ML priors need the Keras predictors")

    result = _run_fit(config, luci_factory, cube_truth_factory)
    payload = _to_jsonable(result)
    path = _golden_path(config)

    if record_golden:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
        pytest.skip(f"recorded baseline -> {os.path.relpath(path)}")

    if not os.path.exists(path):
        pytest.fail(
            f"no baseline at {os.path.relpath(path)}; "
            f"run `pytest tests/test_golden.py --record` and review the result"
        )

    with open(path) as handle:
        expected = json.load(handle)

    for key in CAPTURED:
        got = np.asarray(payload[key]["values"], dtype=np.float64)
        want = np.asarray(expected[key]["values"], dtype=np.float64)
        assert payload[key]["shape"] == expected[key]["shape"], (
            f"{config.id}: {key} changed shape " f"{expected[key]['shape']} -> {payload[key]['shape']}"
        )
        np.testing.assert_allclose(
            got,
            want,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{config.id}: {key} drifted from the recorded baseline",
        )


@pytest.mark.slow
def test_golden_matrix_covers_every_fit_function():
    """Guard against a model quietly losing its baseline coverage."""
    covered = {c.model for c in CONFIGS}
    assert covered == {"gaussian", "sinc", "sincgauss"}


@pytest.mark.slow
def test_golden_baselines_recover_the_injected_physics(luci_factory, cube_truth_factory):
    """
    A guard on the guards.

    Golden files only prove "unchanged", never "right".  If the fixture or the
    fitter drifted far enough to make every baseline meaningless, this would
    still catch it: the recovered Halpha velocity and broadening must stay close
    to what was injected into the cube.
    """
    pytest.importorskip("keras", reason="ML priors need the Keras predictors")
    truth = cube_truth_factory("SN3")
    cube = luci_factory(truth, ML_bool=True)
    vel, broad, _, _ = cube.fit_cube(SN3_LINES, "sincgauss", [1] * 5, [1] * 5, 8, 12, 8, 12, n_threads=1)
    assert vel[:, :, 0].mean() == pytest.approx(truth["velocity_kms"], abs=5.0)
    assert broad[:, :, 0].mean() == pytest.approx(truth["broadening_kms"], abs=5.0)
