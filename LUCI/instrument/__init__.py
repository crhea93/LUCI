"""Instrument-level descriptions: SITELLE filters and header handling."""

from LUCI.instrument.filters import (
    FILTERS,
    SUPPORTED_FILTERS,
    FilterSpec,
    UnsupportedFilterError,
    get_filter,
)

__all__ = [
    "FILTERS",
    "SUPPORTED_FILTERS",
    "FilterSpec",
    "UnsupportedFilterError",
    "get_filter",
]
