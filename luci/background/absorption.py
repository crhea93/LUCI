"""
Removing a stellar absorption continuum from a spectrum.

An old stellar population puts a broad Halpha absorption trough under the
emission line. In an early-type galaxy the stellar continuum is far brighter than
the line-emitting gas, so the trough eats a real fraction of the line flux and
the fit reports an Halpha that is too faint -- and, because the trough is not
centred on the emission line's velocity, kinematics that are pulled with it.

Two halves:

* ``build_absorption_template`` -- stack the spectra of a region into a single
  template, de-redshifting each spaxel by its own velocity first so the trough
  does not smear out across the stack.
* ``subtract_absorption`` -- scale that template onto an observed spectrum's own
  continuum and subtract it, leaving the continuum *level* where it was.

Both halves existed before and neither was reachable. The subtraction was
written out longhand in ``fit_calc`` and again in ``fit_pixel`` (so only
``fit_cube`` and ``fit_pixel`` could use it -- see B27), and the template builder
lived in a top-level ``LuciAbsorp.py`` whose module-level functions referenced
``self``, so it raised ``NameError`` however it was called.
"""

from __future__ import annotations

import numpy as np

# km/s. The same value `luci.fitting.spectrum_fitter` uses, so a velocity map
# produced by a fit round-trips through the shift below unchanged.
SPEED_OF_LIGHT = 299792

# Pixels per block when reading spectra out of the cube. Fancy-indexing the whole
# selection at once would materialise n_pixels x n_channels floats; a region can
# be most of a 2048 x 2064 field.
_PIXEL_BLOCK = 4096


def _doppler_factor(velocity):
    """
    Relativistic factor taking an observed wavenumber to its rest wavenumber.

    ``sigma_rest = sigma_observed * doppler_factor(v)`` for a source receding at
    ``v`` km/s: recession lowers the observed wavenumber, so the factor is > 1.
    """
    beta = np.asarray(velocity, dtype=float) / SPEED_OF_LIGHT
    return np.sqrt((1 + beta) / (1 - beta))


def subtract_absorption(sky, template, edge_fraction=0.25):
    """
    Scale an absorption template onto this spectrum's continuum and subtract it.

    The template is normalised by its own median and multiplied by the median of
    the observed spectrum, so only its *shape* matters -- an absolute scale, and
    therefore the difference between a single spaxel and a summed region, drops
    out. The observed median is added back so the subtraction removes the trough
    without also removing the continuum the fit needs.

    Both medians are taken over the middle of the spectrum, away from the filter
    edges where transmission falls off and the flux is meaningless.

    Args:
        sky: Observed spectrum, before the NaN channels are dropped
        template: Absorption template on the same spectral axis as ``sky``
        edge_fraction: Fraction of the spectrum to exclude at each end when
            measuring the continuum level (default 0.25)

    Return:
        The spectrum with the absorption continuum removed. ``sky`` unchanged if
        ``template`` is None.
    """
    if template is None:
        return sky
    template = np.asarray(template, dtype=float)
    sky = np.asarray(sky)
    if template.shape != sky.shape:
        raise ValueError(
            f"absorp has {template.shape} channels but the spectrum has {sky.shape}. The template "
            "must be on the cube's full spectral axis (`cube.spectrum_axis`), before NaN channels "
            "are dropped -- `build_absorption_template` returns it that way."
        )
    edge = int(len(sky) * edge_fraction)
    # A spectrum shorter than 1/edge_fraction channels would make sky[edge:-edge]
    # empty, and nanmedian of an empty slice is NaN -- which would silently turn
    # the whole spectrum into NaN rather than raise.
    middle = sky[edge:-edge] if edge > 0 else sky
    level = np.nanmedian(middle)
    return sky - template / np.nanmedian(template) * level + level


def build_absorption_template(cube, region, vel_map, vel_max=500.0, mean=True):
    """
    Stack a region's spectra into one absorption template.

    Each spaxel is shifted to the region's *mean* velocity before it is added to
    the stack, so the stellar trough of every spaxel lands at the same
    wavenumber instead of being smeared over the region's velocity spread. The
    result is left at the mean velocity rather than at rest, because that is the
    frame ``subtract_absorption`` needs: it subtracts the template channel by
    channel from spectra observed at roughly that velocity.

    The velocity map is the one a previous fit produced -- typically a low
    signal-to-noise fit of the same field, or a fit of the stellar continuum
    itself. Spaxels with no velocity (NaN) or an implausible one are skipped.

    Args:
        cube: The ``SitelleCube`` to read spectra from
        region: Anything ``cube.region_indices`` accepts -- a ds9 ``.reg`` path,
            a ``.npy`` path, a boolean mask, or an ``(xs, ys)`` index pair
        vel_map: Velocity map in km/s, shaped ``(n_y, n_x)`` and indexed
            ``[y, x]``, matching what a fit writes out
        vel_max: Skip spaxels whose \\|velocity\\| exceeds this, in km/s (default 500)
        mean: Return the per-spaxel mean spectrum rather than the sum (default
            True). Only the shape matters to ``subtract_absorption``, which
            normalises the template by its own median.

    Return:
        The template, on ``cube.spectrum_axis``. Channels no spaxel could
        contribute to -- the ends, which shift out of range -- are NaN.

    Examples:
        >>> vel = fits.open('Luci_outputs/M86_velocity_Halpha.fits')[0].data
        >>> absorp = cube.build_absorption_template('stellar.reg', vel)
        >>> cube.fit_region(['Halpha'], 'sincgauss', [1], [1], 'gas.reg', absorp=absorp)
    """
    xs, ys = cube.region_indices(region)
    if xs.size == 0:
        raise ValueError("The region selects no pixels, so there is nothing to build a template from.")

    vel_map = np.asarray(vel_map, dtype=float)
    n_x, n_y = cube.cube_final.shape[0], cube.cube_final.shape[1]
    if vel_map.ndim != 2:
        raise ValueError(
            f"vel_map must be a 2D velocity map, got {vel_map.ndim} dimensions. A fit returns "
            "velocities shaped (n_y, n_x, n_line); index the line you want, e.g. vel_map[:, :, 0]."
        )
    if vel_map.shape != (n_y, n_x):
        raise ValueError(
            f"vel_map has shape {vel_map.shape}, expected {(n_y, n_x)} -- (n_y, n_x), indexed [y, x], "
            "which is how LUCI writes its maps. Pass vel_map.T if yours is [x, y]."
        )

    velocities = vel_map[ys, xs]
    keep = np.isfinite(velocities) & (np.abs(velocities) <= vel_max)
    if not keep.any():
        raise ValueError(
            f"No pixel in the region has a finite velocity within +/-{vel_max} km/s, so every "
            "spaxel was skipped. Check that vel_map covers this region and is in km/s."
        )
    xs, ys, velocities = xs[keep], ys[keep], velocities[keep]

    axis = np.asarray(cube.spectrum_axis, dtype=float)
    # Each spaxel is shifted by its velocity *relative to the region mean*: the two doppler factors
    # compose into one, which is what leaves the stack at the mean velocity rather than at rest.
    factors = _doppler_factor(velocities) / _doppler_factor(np.mean(velocities))

    total = np.zeros(len(axis))
    count = np.zeros(len(axis), dtype=np.int64)
    n_used = 0
    for start in range(0, xs.size, _PIXEL_BLOCK):
        block = slice(start, start + _PIXEL_BLOCK)
        spectra = cube.cube_final[xs[block], ys[block], :]
        for spectrum, factor in zip(spectra, factors[block]):
            finite = np.isfinite(spectrum)
            if finite.sum() < 2:  # Nothing to interpolate between
                continue
            # `np.interp` holds its endpoints flat by default, which would invent continuum in the
            # channels that shifted out of range; NaN marks them as uncovered instead.
            shifted = np.interp(axis, axis[finite] * factor, spectrum[finite], left=np.nan, right=np.nan)
            covered = np.isfinite(shifted)
            total[covered] += shifted[covered]
            count[covered] += 1
            n_used += 1

    if n_used == 0:
        raise ValueError("Every spaxel in the region is entirely NaN, so there is no template to build.")
    with np.errstate(invalid="ignore"):
        template = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    if not mean:
        template = template * n_used
    return template
