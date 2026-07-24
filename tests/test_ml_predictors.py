"""
Tests for the ONNX parameter predictors.

These run against the real converted models in ``ML/onnx/`` and need only
onnxruntime -- no TensorFlow -- which is the whole point of the migration.

The MDN tests exist because of a bug caught during the ONNX work: the
``IndependentNormal`` head's scale output needs a **softplus**, but the first
model inspected (R5000-MDN-SN3) emits only large positive raw scales, where
``softplus(x) == x`` in float32, so identity looked correct.  The SN2 MDNs emit
*negative* raw scales, where identity yields a negative standard deviation --
off by up to 35 km/s.  The positivity assertion below is the invariant that
makes the mistake impossible to reintroduce.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from luci.ml.onnx_backend import OnnxMDNPredictor, OnnxPredictor, _run
from luci.ml.registry import get_predictor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_DIR = os.path.join(REPO_ROOT, "ML", "onnx")


def _onnx(name: str) -> str:
    path = os.path.join(ONNX_DIR, f"{name}.onnx")
    if not os.path.exists(path):
        pytest.skip(f"{name}.onnx not converted; run tools/convert_models_to_onnx.py")
    return path


def _spectrum(length: int) -> np.ndarray:
    """A normalised spectrum-shaped input, like Fit.spectrum_interp_norm."""
    return np.random.default_rng(0).random(length).astype(np.float32)


# --------------------------------------------------------------------------
# Standard predictors
# --------------------------------------------------------------------------


def test_standard_predictor_returns_finite_velocity_and_broadening():
    predictor = OnnxPredictor(_onnx("R5000-PREDICTOR-I-SN3"))
    estimate = predictor.predict(_spectrum(460))
    assert np.isfinite(estimate.velocity)
    assert np.isfinite(estimate.broadening)
    # Standard predictors carry no distribution, matching the historical
    # vel_ml_sigma = 0 behaviour.
    assert estimate.velocity_sigma == 0.0
    assert estimate.broadening_sigma == 0.0


def test_standard_predictor_is_deterministic():
    predictor = OnnxPredictor(_onnx("R5000-PREDICTOR-I-SN3"))
    spectrum = _spectrum(460)
    first, second = predictor.predict(spectrum), predictor.predict(spectrum)
    assert first == second


def test_sn4_predictor_loads_and_predicts():
    """SN4 support is actively used; its conversion needed a special fix."""
    predictor = OnnxPredictor(_onnx("R4800-PREDICTOR-I-SN4"))
    estimate = predictor.predict(_spectrum(_input_length("R4800-PREDICTOR-I-SN4")))
    assert np.isfinite(estimate.velocity)
    assert np.isfinite(estimate.broadening)


def _input_length(name: str) -> int:
    import onnxruntime as ort

    return int(ort.InferenceSession(_onnx(name)).get_inputs()[0].shape[1])


# --------------------------------------------------------------------------
# MDN predictors -- the softplus transform
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["R5000-PREDICTOR-I-MDN-SN3", "R1000-PREDICTOR-I-MDN-SN2", "R4800-PREDICTOR-I-MDN-SN4"],
)
def test_mdn_sigmas_are_strictly_positive(name):
    """
    A standard deviation cannot be negative.

    This is the invariant the identity transform violated: the SN2 MDNs emit
    negative raw scale outputs, so passing them through unchanged produced
    negative sigmas.  softplus maps them to small positive numbers.
    """
    path = _onnx(name)
    estimate = OnnxMDNPredictor(path).predict(_spectrum(_input_length(name)))
    assert estimate.velocity_sigma > 0.0
    assert estimate.broadening_sigma > 0.0
    assert np.isfinite(estimate.velocity_sigma)
    assert np.isfinite(estimate.broadening_sigma)


@pytest.mark.parametrize("name", ["R5000-PREDICTOR-I-MDN-SN3", "R1000-PREDICTOR-I-MDN-SN2"])
def test_mdn_applies_softplus_to_the_scale_half(name):
    """The sigmas must equal softplus of the raw Dense(4) scale outputs."""
    path = _onnx(name)
    spectrum = _spectrum(_input_length(name))
    raw = _run(path, spectrum)[0]
    estimate = OnnxMDNPredictor(path).predict(spectrum)

    assert estimate.velocity_sigma == pytest.approx(np.logaddexp(0.0, raw[2]), rel=1e-6)
    assert estimate.broadening_sigma == pytest.approx(np.logaddexp(0.0, raw[3]), rel=1e-6)
    # ... while the means are the loc half verbatim, with no transform.
    assert estimate.velocity == pytest.approx(float(raw[0]), rel=1e-9)
    assert estimate.broadening == pytest.approx(float(raw[1]), rel=1e-9)


def test_mdn_softplus_differs_from_identity_where_raw_scale_is_negative():
    """
    Pins the actual failure mode rather than just the fix.

    For a model with negative raw scales the two transforms disagree materially;
    a test that only used R5000-MDN-SN3 would pass under either.
    """
    name = "R1000-PREDICTOR-I-MDN-SN2"
    path = _onnx(name)
    length = _input_length(name)
    rng = np.random.default_rng(1)
    saw_negative = False
    for _ in range(20):
        spectrum = rng.random(length).astype(np.float32)
        raw = _run(path, spectrum)[0]
        if raw[2] < 0 or raw[3] < 0:
            saw_negative = True
            estimate = OnnxMDNPredictor(path).predict(spectrum)
            # softplus keeps it positive; identity would not have.
            assert estimate.velocity_sigma > 0.0 and estimate.broadening_sigma > 0.0
            break
    assert saw_negative, "expected this SN2 MDN to emit a negative raw scale"


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_registry_resolves_and_caches_a_predictor(luci_path):
    first = get_predictor(5000, "SN3", False, luci_path)
    second = get_predictor(5000, "SN3", False, luci_path)
    assert first is not None
    assert first is second, "predictor must be cached (B13: no reload per pixel)"


def test_registry_selects_the_mdn_variant(luci_path):
    standard = get_predictor(5000, "SN3", False, luci_path)
    mdn = get_predictor(5000, "SN3", True, luci_path)
    assert isinstance(standard, OnnxPredictor)
    assert isinstance(mdn, OnnxMDNPredictor)


def test_registry_returns_none_for_an_unknown_configuration(luci_path):
    """A missing model degrades to data-driven priors rather than crashing."""
    assert get_predictor(1234, "SN3", False, luci_path) is None
