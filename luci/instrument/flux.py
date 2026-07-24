"""
Flux calibration for ORB/SITELLE cubes.

LUCI's flux formulas (``docs/source/howLuciWorks.rst``) are amplitude x width, so the fitted
fluxes come out in whatever units the cube holds. That is fine for a cube ORBS already
calibrated, but ORB's newer "level 3" cubes are stored **in counts** and carry the calibration
as a header vector instead of applying it. Fitting one of those and reading the result as
erg/cm2/s/A -- which is what every LUCI axis label says -- is wrong by ~20 orders of magnitude,
with nothing in the output to indicate it.

The convention is documented in ``HDFCube.get_level`` in ORB's ``orb/cube.py``::

    * level 1: old hdf5 architecture, real output, unit in erg/cm2/s/A
    * level 2: new hdf5 architecture, real output, unit in erg/cm2/s/A
    * level 3: new hdf5 architecture, complex output, unit in counts, data can be calibrated
      via flambda parameter: spectrum *= cube.params.flambda / cube.dimz / cube.exposure_time
    * level 2.5: CFHT version, similar to level3 but calibrated data is hard written.

So a level-3 cube is calibrated by the *mean count rate* -- counts divided by the total
integration time ``dimz * exposure_time`` -- times ``flambda``.

Note that ORB itself is inconsistent here: ``HDFCube.to_fits`` multiplies by ``flambda`` alone,
without the ``/dimz/exposure_time``, which for a typical SN4 cube is 17822x too bright. The
docstring form is the one that reproduces published surface brightnesses (checked against M86's
stripped Halpha filament, ~1e-17 erg/s/cm2/arcsec2), so that is what LUCI uses.
"""

from __future__ import annotations

import numpy as np

from luci.log import get_logger

logger = get_logger(__name__)

#: Header keys that mean "these data are already in erg/cm2/s/A".
_CALIBRATED_FLAGS = ("flux_calibration", "WAVCALIB")


def is_flux_calibrated(hdr_dict) -> bool:
    """
    Whether the cube's data are already in erg/cm2/s/A.

    ORB writes ``flux_calibration`` on new-format cubes; the older DR1 FITS-style headers say
    ``BUNIT = 'FLUX'`` instead. Absent both, assume the data are calibrated -- that is how every
    LUCI cube behaved before this module existed, so it keeps old cubes reducing unchanged.

    Args:
        hdr_dict: Header dictionary from `update_header`

    Return:
        True if no calibration should be applied
    """
    if "flux_calibration" in hdr_dict:
        return bool(hdr_dict["flux_calibration"])
    if str(hdr_dict.get("BUNIT", "")).strip().upper() == "FLUX":
        return True
    return "flambda" not in hdr_dict


def flux_calibration_vector(hdr_dict, n_channels: int):
    """
    Per-channel factor converting stored counts to erg/cm2/s/A, or None if none is needed.

    Args:
        hdr_dict: Header dictionary from `update_header`
        n_channels: Length of the spectral axis

    Return:
        1D numpy array of length `n_channels`, or None when the cube is already calibrated or
        carries no usable `flambda`
    """
    if is_flux_calibrated(hdr_dict):
        return None

    flambda = hdr_dict.get("flambda")
    if flambda is None:
        logger.warning(
            "Cube reports flux_calibration=False but has no flambda vector, so fluxes stay in "
            "counts. Treat the flux and amplitude maps as instrumental units."
        )
        return None

    flambda = np.atleast_1d(np.asarray(flambda, dtype=float))
    if flambda.size == 1:
        flambda = np.full(n_channels, float(flambda[0]))
    elif flambda.size != n_channels:
        logger.warning(
            "flambda has %i entries but the spectral axis has %i channels, so no flux "
            "calibration was applied. Fluxes stay in counts.",
            flambda.size,
            n_channels,
        )
        return None

    # Total integration time: one exposure per step of the interferogram
    n_steps = hdr_dict.get("step_nb", hdr_dict.get("STEPNB"))
    exposure_time = hdr_dict.get("exposure_time")
    if not n_steps or not exposure_time:
        logger.warning(
            "Cube is uncalibrated but is missing step_nb/exposure_time, so no flux calibration "
            "was applied. Fluxes stay in counts."
        )
        return None

    return flambda / float(n_steps) / float(exposure_time)
