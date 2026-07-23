"""
Resolve ``(resolution, filter, mdn)`` to a :class:`ParameterPredictor`.

Predictors are cached per identity so a process loads each one at most once
(part of the B13 fix -- see ``onnx_backend``).  ONNX artifacts live in
``ML/onnx/`` next to the legacy Keras models; the naming mirrors the originals:

    ML/onnx/R5000-PREDICTOR-I-SN3.onnx
    ML/onnx/R5000-PREDICTOR-I-MDN-SN3.onnx

``get_predictor`` returns ``None`` when no converted model exists for the request
(e.g. an unsupported filter or a resolution that was never trained); the caller
then falls back to the data-driven priors (``Fit.estimate_priors_data``), so a
missing model degrades gracefully instead of crashing.
"""

from __future__ import annotations

import os

from LUCI.ml.base import ParameterPredictor
from LUCI.ml.onnx_backend import OnnxMDNPredictor, OnnxPredictor

_CACHE: dict[tuple, ParameterPredictor | None] = {}


def _onnx_path(luci_path: str, resolution: int, filter_name: str, mdn: bool) -> str:
    kind = "PREDICTOR-I-MDN" if mdn else "PREDICTOR-I"
    name = f"R{resolution}-{kind}-{filter_name}.onnx"
    return os.path.join(luci_path, "ML", "onnx", name)


def get_predictor(resolution: int, filter_name: str, mdn: bool, luci_path: str) -> ParameterPredictor | None:
    """Return the cached predictor for this configuration, or ``None``."""
    key = (resolution, filter_name, bool(mdn))
    if key in _CACHE:
        return _CACHE[key]

    path = _onnx_path(luci_path, resolution, filter_name, mdn)
    if os.path.exists(path):
        predictor: ParameterPredictor | None = OnnxMDNPredictor(path) if mdn else OnnxPredictor(path)
    else:
        predictor = None
    _CACHE[key] = predictor
    return predictor
