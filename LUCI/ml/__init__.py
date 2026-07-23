"""
Machine-learning parameter prediction.

LUCI seeds each fit's velocity and broadening from a small neural network.
Historically the network was a Keras model that ``Fit`` loaded from disk **on
every pixel** (bug B13) and ran through TensorFlow.  This package puts a thin
interface (``ParameterPredictor``) in front of that so the fitting code no longer
imports Keras directly, the model is loaded once per process, and inference runs
on ONNX Runtime instead of TensorFlow.

Public surface:
  * ``PriorEstimate``     -- the (velocity, broadening) prediction + its sigmas
  * ``ParameterPredictor``-- the protocol every backend implements
  * ``get_predictor``     -- resolve (resolution, filter, mdn) -> predictor or None
"""

from LUCI.ml.base import ParameterPredictor, PriorEstimate
from LUCI.ml.registry import get_predictor

__all__ = ["ParameterPredictor", "PriorEstimate", "get_predictor"]
