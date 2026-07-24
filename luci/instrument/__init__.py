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

__all__ = [
    "FILTERS",
    "SUPPORTED_FILTERS",
    "FilterSpec",
    "PCABackgroundUnsupportedError",
    "UnsupportedFilterError",
    "get_filter",
    "pca_scale_indices",
]
