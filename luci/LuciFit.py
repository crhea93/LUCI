"""Backward-compatibility shim. The fitter moved to ``LUCI.fitting.spectrum_fitter``."""

from luci.fitting.spectrum_fitter import (  # noqa: F401
    DEFAULT_BROADENING_KMS,
    SPEED_OF_LIGHT,
    Fit,
    SpectrumFitter,
)
