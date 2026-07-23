"""Backend-independent types for ML parameter prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class PriorEstimate:
    """A predictor's estimate of a spectrum's kinematics, in km/s.

    ``velocity_sigma`` / ``broadening_sigma`` are non-zero only for the MDN
    predictors, which output a distribution; the standard predictors leave them
    at 0.0 (matching the historical ``vel_ml_sigma = 0`` behaviour).
    """

    velocity: float
    broadening: float
    velocity_sigma: float = 0.0
    broadening_sigma: float = 0.0


@runtime_checkable
class ParameterPredictor(Protocol):
    """Anything that turns a prepared spectrum into a :class:`PriorEstimate`.

    The spectrum passed to ``predict`` is the interpolated, normalised spectrum
    on the reference-spectrum axis (``Fit.spectrum_interp_norm``) -- the same
    input the original Keras models received.
    """

    def predict(self, spectrum: np.ndarray) -> PriorEstimate: ...
