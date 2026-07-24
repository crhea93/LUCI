"""
Flag unreliable fits so an empty field does not masquerade as a measurement.

Every pixel in a cube gets fit, whether or not there is a line in it, and a fit to noise still
returns a velocity and a broadening. Over a full SITELLE field most of the frame is blank sky, so
an unmasked map is mostly meaningless numbers -- and because those numbers are drawn from the
whole allowed parameter range, they dominate any percentile-based colour scale and hide the real
structure.

This is worse after adaptive binning. A WVT bin that never reached its S/N target still produces
a fit, and it covers many pixels, so a single bad bin paints a large patch of the map.

`fit_quality_mask` combines the checks below into one boolean mask, True where the fit is
trustworthy. Each is independently switchable, and `fit_quality_report` says how many pixels each
one removed -- a filter that silently drops 90% of the field is worth noticing.
"""

from __future__ import annotations

import numpy as np

from luci.log import get_logger

logger = get_logger(__name__)

#: Sigma is bounded at 10 cm-1 in `luci.fitting.constraints.sigma_bounds`. Converted to km/s that
#: is ~197 km/s in SN4 and ~200 in SN3, and a fit sitting on the bound has not converged -- it is
#: a lower limit, not a measurement.
SIGMA_BOUND_CM1 = 10.0
SPEED_OF_LIGHT = 299792.458


def broadening_bound_kms(line_position_cm1):
    """
    The sigma bound expressed in km/s at a given position on the spectral axis.

    Args:
        line_position_cm1: Position of the line in cm-1 (e.x. 15253 for Halpha in SN4)

    Return:
        The upper bound on the broadening in km/s
    """
    return SPEED_OF_LIGHT * SIGMA_BOUND_CM1 / float(line_position_cm1)


def fit_quality_mask(
    flux=None,
    flux_err=None,
    velocity=None,
    broadening=None,
    chi2=None,
    snr=None,
    snr_min=3.0,
    max_flux_err_ratio=0.5,
    velocity_range=None,
    chi2_max=None,
    broadening_max=None,
    require_positive_flux=True,
):
    """
    Boolean mask of trustworthy fits, True where every requested check passes.

    All maps must broadcast against each other. Any argument left as None is skipped, so this
    works with whatever subset of the output maps you have to hand.

    Args:
        flux: Flux map
        flux_err: Flux error map, used for the relative-error cut
        velocity: Velocity map [km/s]
        broadening: Broadening map [km/s]
        chi2: Chi-squared map
        snr: Signal-to-noise map
        snr_min: Minimum S/N (default 3.0). The single most effective cut on an empty field.
        max_flux_err_ratio: Maximum flux_err/flux (default 0.5, i.e. a 2-sigma detection)
        velocity_range: (low, high) in km/s; fits outside are dropped. A fit to noise scatters
            across the whole allowed range, so this catches what S/N alone misses.
        chi2_max: Maximum chi-squared
        broadening_max: Maximum broadening [km/s]. Pass `broadening_bound_kms(...)` minus a small
            margin to drop fits pinned to the sigma bound.
        require_positive_flux: Drop non-finite and non-positive fluxes (default True)

    Return:
        Boolean numpy array, True where the fit is good

    Example:
        >>> good = fit_quality_mask(flux=flux, snr=snr, velocity=vel, snr_min=5,
        ...                         velocity_range=(-800, 800))
        >>> vel_clean = np.where(good, vel, np.nan)
    """
    shapes = [np.shape(m) for m in (flux, flux_err, velocity, broadening, chi2, snr) if m is not None]
    if not shapes:
        raise ValueError("fit_quality_mask needs at least one map to work from")
    mask = np.ones(np.broadcast_shapes(*shapes), dtype=bool)

    if flux is not None:
        flux = np.asarray(flux, dtype=float)
        mask &= np.isfinite(flux)
        if require_positive_flux:
            mask &= flux > 0
    if flux_err is not None and flux is not None and max_flux_err_ratio is not None:
        flux_err = np.asarray(flux_err, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(flux_err) / np.abs(flux)
        mask &= np.isfinite(ratio) & (ratio <= max_flux_err_ratio)
    if snr is not None and snr_min is not None:
        snr = np.asarray(snr, dtype=float)
        mask &= np.isfinite(snr) & (snr >= snr_min)
    if velocity is not None:
        velocity = np.asarray(velocity, dtype=float)
        mask &= np.isfinite(velocity)
        if velocity_range is not None:
            mask &= (velocity >= velocity_range[0]) & (velocity <= velocity_range[1])
    if broadening is not None:
        broadening = np.asarray(broadening, dtype=float)
        mask &= np.isfinite(broadening)
        if broadening_max is not None:
            mask &= broadening <= broadening_max
    if chi2 is not None:
        chi2 = np.asarray(chi2, dtype=float)
        mask &= np.isfinite(chi2)
        if chi2_max is not None:
            mask &= chi2 <= chi2_max
    return mask


def fit_quality_report(log=True, **kwargs):
    """
    Apply each cut on its own and report how much of the field it removes.

    Use this to choose thresholds: a cut that removes everything, or nothing, is not doing what
    you think it is.

    Args:
        log: Emit the report through the logger (default True)
        **kwargs: Exactly as `fit_quality_mask`

    Return:
        (mask, report) where report maps a cut name to the fraction of pixels it alone rejects
    """
    combined = fit_quality_mask(**kwargs)
    individual_cuts = {
        "snr": ("snr_min",),
        "flux": ("require_positive_flux",),
        "flux_err": ("max_flux_err_ratio",),
        "velocity": ("velocity_range",),
        "broadening": ("broadening_max",),
        "chi2": ("chi2_max",),
    }
    report = {}
    for name, keys in individual_cuts.items():
        if kwargs.get(name) is None or all(kwargs.get(k) is None for k in keys):
            continue
        only = {k: v for k, v in kwargs.items() if k in (name, *keys)}
        # A single-map call still needs its own map present
        rejected = ~fit_quality_mask(**only)
        report[name] = float(np.mean(rejected))
    report["combined"] = float(np.mean(~combined))
    if log:
        logger.info("Fit quality: %.1f%% of pixels rejected overall", 100 * report["combined"])
        for name, fraction in report.items():
            if name != "combined":
                logger.info("    %-12s rejects %.1f%%", name, 100 * fraction)
    return combined, report


def apply_quality_mask(maps, mask, fill=np.nan):
    """
    Blank the rejected pixels in a set of maps.

    Args:
        maps: Dict of {name: 2D array}
        mask: Boolean array from `fit_quality_mask`, True where the fit is good
        fill: Value written where the mask is False (default NaN)

    Return:
        Dict of the same keys with masked copies
    """
    return {name: np.where(mask, np.asarray(array, dtype=float), fill) for name, array in maps.items()}
