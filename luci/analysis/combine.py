"""
Co-add the per-line flux maps of a blended complex into one total-flux map.

A fit writes one flux map per line, but for a complex whose lines are blended and fit with a tied
velocity and broadening -- Halpha + the NII doublet in SN4, say -- the individual maps each carry
only a fraction of the emitted photons. Adding them gives a single map of the whole complex, which
shows faint structure the brightest line alone does not.

The lines are added straight, not averaged: the result is the total flux of the complex in the same
units as the inputs. A NaN in any input line propagates to the total, because a pixel where one
line's fit failed has no meaningful total.

Note on the NII doublet. `fit_cube` defaults to `nii_cons=True`, which holds NII6548 at exactly one
third the flux of NII6583, so the doublet is a single free parameter. That does not change the sum
-- the saved maps already obey the ratio, so adding all three lines gives the right total -- but it
does mean the two NII maps are not independent measurements, and an uncertainty on the total must
use (4/3) * sigma(NII6583) rather than adding the two errors in quadrature.
"""

from __future__ import annotations

import os

import numpy as np
from astropy.io import fits

from luci.log import get_logger

logger = get_logger(__name__)


def line_flux_path(output_dir, prefix, line, suffix="_Flux"):
    """Path of a single line's flux map, following `luci.io.outputs.save_fits` naming."""
    return os.path.join(output_dir, "Fluxes", "%s_%s%s.fits" % (prefix, line, suffix))


def combine_line_fluxes(
    output_dir,
    prefix,
    lines,
    out_name=None,
    bunit=None,
    overwrite=True,
):
    """
    Sum the flux maps of `lines` into a single total-complex flux map and write it to FITS.

    The output lands beside the inputs in `<output_dir>/Fluxes/` and inherits the WCS header of
    the first line's map, so it overlays the per-line maps and the deep image without regridding.

    Args:
        output_dir: Directory holding the `Fluxes/` subdirectory (the cube's output dir)
        prefix: Output-name prefix the fit used (e.x. 'M86_2_sincgauss', 'M86_wvt_20_1')
        lines: Lines to add (e.x. ['Halpha', 'NII6548', 'NII6583'])
        out_name: Basename of the output map, without '.fits'. Defaults to
            '<prefix>_<joined lines>_Flux' (e.x. 'M86_2_sincgauss_Halpha+NII_Flux').
        bunit: Value for the BUNIT card. Leave None to omit it -- an uncalibrated cube's fluxes
            are in counts, not ergs/s/cm2, and a wrong BUNIT is worse than none.
        overwrite: Overwrite an existing output map (default True)

    Return:
        (total, path): the combined flux map and where it was written
    """
    if len(lines) < 2:
        raise ValueError("Combining needs at least two lines, got %r." % (lines,))

    total = None
    header = None
    for line in lines:
        path = line_flux_path(output_dir, prefix, line)
        if not os.path.exists(path):
            raise FileNotFoundError("No flux map for %s at %s" % (line, path))
        with fits.open(path) as hdul:
            data = hdul[0].data.astype(np.float64)
            if total is None:
                total, header = data, hdul[0].header.copy()
            elif data.shape != total.shape:
                raise ValueError(
                    "%s is %s but %s is %s -- the maps are on different grids, so they cannot be "
                    "added. Were they written by runs with different binning or regions?"
                    % (line, data.shape, lines[0], total.shape)
                )
            else:
                total = total + data

    # Drop any card that would describe the inputs rather than the sum
    for card in ("BUNIT", "LINE", "EXTNAME"):
        header.remove(card, ignore_missing=True)
    header["NCOMB"] = (len(lines), "Number of line flux maps co-added")
    header["COMBLINE"] = (",".join(lines), "Lines co-added into this map")
    for i, line in enumerate(lines):
        header["CLINE%d" % i] = (line, "Co-added line %d" % i)
    header.add_comment("Total flux of the line complex: sum of the per-line flux maps.")
    header.add_comment("NaN where any contributing line's fit produced NaN.")
    if bunit is not None:
        header["BUNIT"] = bunit

    if out_name is None:
        out_name = "%s_%s_Flux" % (prefix, "+".join(lines))
    path = os.path.join(output_dir, "Fluxes", out_name + ".fits")
    fits.writeto(path, total.astype(np.float32), header, overwrite=overwrite)
    return total, path


def combination_summary(output_dir, prefix, lines, total, mask=None, top_percent=1.0):
    """
    Describe what co-adding bought, as a dict of diagnostics.

    The point of a combined map is more signal, so it is worth checking that it actually arrived.
    `gain_vs_brightest` is the combined flux over that of the strongest single line; a value near 1
    means the other lines contributed nothing and the combination is not helping.

    All of this is measured over the *detected* pixels only -- by default the brightest `top_percent`
    of the combined map, or `mask` if you have a quality mask to hand. Measuring over the whole map
    gives numbers that describe the noise rather than the source: most of a SITELLE field is blank
    sky, a fit to blank sky returns a flux of either sign, and those fluxes are uncorrelated between
    lines, so summing them makes the noise partially cancel. Done over the full field the statistic
    reports a *loss* of flux from co-adding, which is an artefact of the empty 90% of the frame.

    Args:
        output_dir: Directory holding the `Fluxes/` subdirectory
        prefix: Output-name prefix the fit used
        lines: Lines that were co-added
        total: The combined map returned by `combine_line_fluxes`
        mask: Boolean map, True where the fit is trustworthy (e.x. from `fit_quality_mask`).
            Defaults to None, which selects the brightest `top_percent` of the combined map.
        top_percent: Percentage of the brightest pixels to measure over when `mask` is None
            (default 1.0)

    Return:
        Dict of diagnostics, including per-line flux over the detected pixels and the gain over the
        brightest line
    """
    per_line = {}
    for line in lines:
        with fits.open(line_flux_path(output_dir, prefix, line)) as hdul:
            per_line[line] = hdul[0].data.astype(np.float64)

    finite = np.isfinite(total)
    if mask is None:
        if not finite.any():
            raise ValueError("The combined map is entirely NaN, so there is nothing to summarise.")
        threshold = np.percentile(total[finite], 100.0 - top_percent)
        detected = finite & (total >= threshold)
    else:
        detected = finite & np.asarray(mask, dtype=bool)
        if not detected.any():
            raise ValueError("The supplied mask selects no finite pixel of the combined map.")

    fluxes = {line: float(np.nansum(arr[detected])) for line, arr in per_line.items()}
    brightest = max(fluxes, key=fluxes.get)
    ref = fluxes[brightest]
    gain = float(np.nansum(total[detected])) / ref if ref > 0 else float("nan")

    lost = int((~np.isfinite(total)).sum() - (~np.isfinite(per_line[brightest])).sum())
    return {
        "detected_flux": fluxes,
        "n_detected": int(detected.sum()),
        "brightest_line": brightest,
        "gain_vs_brightest": gain,
        "n_pixels": int(total.size),
        "n_nan": int((~np.isfinite(total)).sum()),
        "n_nan_beyond_brightest": max(lost, 0),
        "n_negative": int(np.sum(finite & (total < 0))),
    }
