"""
Backward-compatibility shim.

The emission-line models and their helpers moved to ``LUCI.fitting.models``.
This module re-exports them so existing imports
(``from LUCI.LuciFunctions import Gaussian, Sinc, SincGauss``) keep working.
Prefer importing from ``LUCI.fitting.models`` in new code.
"""

from LUCI.fitting.models import (  # noqa: F401
    FWHM_COEFF,
    FWHM_SINC_COEFF,
    SINCGAUSS_SIGMA_FLOOR,
    SPEED_OF_LIGHT,
    Gaussian,
    Sinc,
    SincGauss,
    frozen_values,
)
