"""
Backward-compatibility shim.

The velocity / broadening / flux conversions moved to
``LUCI.fitting.parameters``.  This module re-exports them so existing imports
keep working.  Prefer importing from ``LUCI.fitting.parameters`` in new code.
"""

from LUCI.fitting.parameters import (  # noqa: F401
    FWHM_COEFF,
    SPEED_OF_LIGHT,
    calculate_broad,
    calculate_broad_err,
    calculate_flux,
    calculate_flux_err,
    calculate_vel,
    calculate_vel_err,
)
