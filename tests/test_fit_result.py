"""
Tests for FitResult, the typed replacement for the legacy 22-key fit dict.

The whole point of FitResult is that it can be introduced without touching a
single downstream caller, so these tests focus on read-compatibility with the
old dict alongside the new typed access.
"""

from __future__ import annotations

import numpy as np
import pytest

from luci.fitting.result import FitResult

# The exact key set the pre-refactor Fit.fit() returned.  If a refactor drops or
# renames one of these, a caller indexing the old dict breaks -- so pin it.
LEGACY_KEYS = [
    "fit_sol",
    "fit_uncertainties",
    "amplitudes",
    "fluxes",
    "flux_errors",
    "chi2",
    "velocities",
    "sigmas",
    "vels_errors",
    "sigmas_errors",
    "axis_step",
    "corr",
    "continuum",
    "continuum_error",
    "scale",
    "flat_samples",
    "vel_ml",
    "vel_ml_sigma",
    "broad_ml",
    "broad_ml_sigma",
    "fit_vector",
    "fit_axis",
]


def _sample() -> FitResult:
    return FitResult(
        fit_sol=np.array([1.0, 2.0, 3.0, 0.1]),
        fit_uncertainties=np.array([0.1, 0.2, 0.3, 0.01]),
        amplitudes=[1.0],
        fluxes=[2.0],
        flux_errors=[0.2],
        chi2=1.23,
        velocities=[100.0],
        sigmas=[30.0],
        vels_errors=[5.0],
        sigmas_errors=[2.0],
        axis_step=2.0,
        corr=1.01,
        continuum=0.1,
        continuum_error=0.01,
        scale=1e-16,
        flat_samples=None,
        vel_ml=98.0,
        vel_ml_sigma=10.0,
        broad_ml=28.0,
        broad_ml_sigma=3.0,
        fit_vector=np.zeros(5),
        fit_axis=np.linspace(15000, 15400, 5),
    )


def test_exposes_exactly_the_legacy_keys_in_order():
    assert _sample().keys() == LEGACY_KEYS


def test_dict_style_access_matches_attribute_access():
    r = _sample()
    for key in LEGACY_KEYS:
        assert r[key] is getattr(r, key)


def test_typed_attribute_access():
    r = _sample()
    assert r.velocities == [100.0]
    assert r.chi2 == 1.23


def test_as_dict_reproduces_the_legacy_mapping():
    r = _sample()
    d = r.as_dict()
    assert list(d.keys()) == LEGACY_KEYS
    assert d["velocities"] == [100.0]
    # A plain consumer that only reads by key cannot tell it isn't a dict.
    assert d["continuum"] == r["continuum"]


def test_missing_key_raises_keyerror_not_attributeerror():
    """Callers that catch KeyError on a dict must keep working."""
    with pytest.raises(KeyError):
        _ = _sample()["does_not_exist"]


def test_contains_and_get_behave_like_a_dict():
    r = _sample()
    assert "velocities" in r
    assert "nope" not in r
    assert r.get("velocities") == [100.0]
    assert r.get("nope", "default") == "default"


def test_iterates_over_keys_like_a_dict():
    assert list(iter(_sample())) == LEGACY_KEYS
