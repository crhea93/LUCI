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
    """
    Boolean mask selecting an explicit list of (x, y) pixel pairs.

    Starts from all-False. The original started from ``np.ones`` -- every pixel
    already selected -- and then set the listed pixels True, so ``pixel_list=True``
    silently fitted the whole cube instead of the requested pixels (B23).
    """
    mask = np.zeros(shape, dtype=bool)
    for pair in pixels:
        mask[pair] = True
    return mask


def resolve_mask(region, header, cube_shape, pixel_list=False) -> np.ndarray:
    """
    Resolve any supported region specification to a boolean mask.

    ``region`` may be a ds9 ``.reg`` path, a ``.npy`` path, a boolean array, or
    -- with ``pixel_list=True`` -- a sequence of (x, y) pairs. Previously each
    fit entry point re-implemented this chain, and an unrecognised value merely
    printed a message and carried on with ``mask`` unset.
    """
    if pixel_list:
        return mask_from_pixel_list(region, cube_shape)
    if isinstance(region, str):
        if region.endswith(".reg"):
            header = header.copy()
            # NAXIS from the cube, not the standard detector size (B10).
            header.set("NAXIS1", cube_shape[1])
            header.set("NAXIS2", cube_shape[0])
            return reg_to_mask(region, header)
        if region.endswith(".npy"):
            return np.load(region)
        raise ValueError(f"Unrecognised region file {region!r}: expected a .reg or .npy path.")
    if region is None:
        raise ValueError(
            "No region given. Pass a .reg path, a .npy path, a boolean array, " "or a pixel list with pixel_list=True."
        )
    return np.asarray(region, dtype=bool)
