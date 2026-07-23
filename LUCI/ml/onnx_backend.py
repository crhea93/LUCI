"""
ONNX Runtime inference backend for the parameter predictors.

Two predictor types mirror the two model families:

* :class:`OnnxPredictor` wraps a converted standard predictor whose two outputs
  are ``[velocity, broadening]`` km/s.
* :class:`OnnxMDNPredictor` wraps a converted MDN predictor, exported truncated
  at its ``Dense(4)`` layer.  The four outputs are ``[loc_v, loc_b, scale_v,
  scale_b]``; the distribution is reproduced by splitting the vector -- mean is
  the first half, standard deviation the second (verified during conversion).

Sessions are cached per file path at module scope.  This is what fixes B13: the
old code called ``keras.models.load_model`` inside ``Fit.__init__``, i.e. once
per pixel (measured at ~5.7 s/pixel).  Here the first fit in a process builds the
ONNX session and every subsequent fit reuses it, so joblib workers pay the load
cost once each instead of once per spectrum.
"""

from __future__ import annotations

import numpy as np

from LUCI.ml.base import PriorEstimate

_SESSIONS: dict[str, object] = {}


def _session(path: str):
    session = _SESSIONS.get(path)
    if session is None:
        import onnxruntime as ort

        # CPU only and single-threaded: fits already run under joblib
        # parallelism, so intra-op threads would oversubscribe the cores.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
        _SESSIONS[path] = session
    return session


def _run(path: str, spectrum: np.ndarray) -> np.ndarray:
    session = _session(path)
    x = np.asarray(spectrum, dtype=np.float32).reshape(1, -1, 1)
    return np.asarray(session.run(None, {session.get_inputs()[0].name: x})[0])


class OnnxPredictor:
    """Standard predictor: outputs velocity and broadening directly."""

    def __init__(self, path: str) -> None:
        self.path = path

    def predict(self, spectrum: np.ndarray) -> PriorEstimate:
        out = _run(self.path, spectrum)
        return PriorEstimate(velocity=float(out[0][0]), broadening=float(out[0][1]))


class OnnxMDNPredictor:
    """
    MDN predictor: the Dense(4) output is ``[loc_v, loc_b, scale_v, scale_b]``.

    The mean is the loc half verbatim; the standard deviation is **softplus** of
    the scale half, which is the positivity transform ``IndependentNormal``
    applies.  ``np.logaddexp(0, x)`` is a numerically stable softplus.

    Getting this wrong is subtle: for predictors whose raw scales are all large
    and positive, ``softplus(x) == x`` in float32 and identity looks right.  The
    SN2 MDNs emit negative raw scales, where identity is off by up to 35 km/s.
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def predict(self, spectrum: np.ndarray) -> PriorEstimate:
        out = _run(self.path, spectrum)[0]
        return PriorEstimate(
            velocity=float(out[0]),
            broadening=float(out[1]),
            velocity_sigma=float(np.logaddexp(0.0, out[2])),
            broadening_sigma=float(np.logaddexp(0.0, out[3])),
        )
