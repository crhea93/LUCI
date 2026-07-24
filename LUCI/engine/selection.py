"""Turning region definitions into pixel masks."""

from __future__ import annotations

import numpy as np
import pyregion


def reg_to_mask(region, header):
    """
    Convert a ds9 ``.reg`` file into a boolean mask the fitting code can use.

    The shape comes from the header's NAXIS keywords; it used to be hardcoded to
    the standard SITELLE detector (2064, 2048), which silently produced a
    wrong-sized mask for any other cube (B10).
    """
    try:
        shape = (int(header["NAXIS1"]), int(header["NAXIS2"]))
    except KeyError as exc:
        raise KeyError("reg_to_mask needs NAXIS1/NAXIS2 on the header to size the mask") from exc
    r = pyregion.open(region).as_imagecoord(header)
    return r.get_mask(shape=shape).T


def mask_from_pixel_list(pixels, shape) -> np.ndarray:
    """Boolean mask selecting an explicit list of (x, y) pixel pairs."""
    mask = np.zeros(shape, dtype=bool)
    for pair in pixels:
        mask[pair] = True
    return mask
