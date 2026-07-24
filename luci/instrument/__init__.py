"""Instrument-level descriptions: SITELLE filters and header handling."""

from luci.instrument.filters import (
    FILTERS,
    SUPPORTED_FILTERS,
    FilterSpec,
    PCABackgroundUnsupportedError,
    UnsupportedFilterError,
    get_filter,
    pca_scale_indices,
)
from luci.instrument.flux import flux_calibration_vector, is_flux_calibrated

__all__ = [
    "FILTERS",
    "SUPPORTED_FILTERS",
    "FilterSpec",
    "PCABackgroundUnsupportedError",
    "UnsupportedFilterError",
    "flux_calibration_vector",
    "get_filter",
    "is_flux_calibrated",
    "pca_scale_indices",
]
