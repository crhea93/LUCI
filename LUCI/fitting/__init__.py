"""Fitting: line models, parameter conversions, and the fit result type."""

from LUCI.fitting.models import Gaussian, Sinc, SincGauss, frozen_values
from LUCI.fitting.parameters import (
    calculate_broad,
    calculate_broad_err,
    calculate_flux,
    calculate_flux_err,
    calculate_vel,
    calculate_vel_err,
)
from LUCI.fitting.result import FitResult
from LUCI.fitting.spectrum_fitter import SpectrumFitter

__all__ = [
    "FitResult",
    "SpectrumFitter",
    "Gaussian",
    "Sinc",
    "SincGauss",
    "frozen_values",
    "calculate_vel",
    "calculate_vel_err",
    "calculate_broad",
    "calculate_broad_err",
    "calculate_flux",
    "calculate_flux_err",
]
