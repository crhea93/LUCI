"""
Removing a stellar absorption continuum from a spectrum.

An old stellar population puts a broad Halpha absorption trough under the
emission line. In an early-type galaxy the stellar continuum is far brighter than
the line-emitting gas, so the trough eats a real fraction of the line flux and
the fit reports an Halpha that is too faint -- and, because the trough is not
centred on the emission line's velocity, kinematics that are pulled with it.

Two halves:

* ``build_absorption_template`` -- stack the spectra of a region into a single
  template, removing the sky from each spaxel and de-redshifting it by its own
  velocity first, so the template is of the stellar continuum alone and the
  trough does not smear out across the stack.
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

from luci.background.subtraction import pca_background, subtract_pca, subtract_standard
from luci.log import get_logger

logger = get_logger(__name__)

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


def _resolve_background(bkg, bkgType, pca_coefficient_array, pca_vectors, pca_mean):
    """
    Validate the background arguments and return the scheme to apply per spaxel.

    Fails here rather than inside the stacking loop, so a missing PCA model is one
    message at the call site instead of a ``TypeError`` per spaxel.
    """
    # A caller who passes bkg= plainly means "subtract this background"; `fit_calc` makes the same
    # allowance (B26), and the template must be built the same way the fit will be run.
    if bkg is not None and bkgType is None:
        bkgType = "standard"
    if bkgType is None:
        return None
    if bkgType == "standard":
        if bkg is None:
            raise ValueError("bkgType='standard' needs a bkg spectrum to subtract.")
        return "standard"
    if bkgType == "pca":
        missing = [
            name
            for name, value in (
                ("pca_coefficient_array", pca_coefficient_array),
                ("pca_vectors", pca_vectors),
                ("pca_mean", pca_mean),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"bkgType='pca' needs {', '.join(missing)}. These are the same arrays you pass to "
                "fit_cube(bkgType='pca'), from cube.create_background_subspace()."
            )
        return "pca"
    raise ValueError("bkgType must be 'standard', 'pca', or None; got %r" % (bkgType,))


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


def build_absorption_template(
    cube,
    region,
    vel_map,
    vel_max=500.0,
    mean=True,
    bkg=None,
    bkgType=None,
    pca_coefficient_array=None,
    pca_vectors=None,
    pca_mean=None,
):
    """
    Stack a region's spectra into one absorption template.

    Each spaxel is shifted to the region's *mean* velocity before it is added to
    the stack, so the stellar trough of every spaxel lands at the same
    wavenumber instead of being smeared over the region's velocity spread. The
    result is left at the mean velocity rather than at rest, because that is the
    frame ``subtract_absorption`` needs: it subtracts the template channel by
    channel from spectra observed at roughly that velocity.

    **The velocity map must be the stars', not the gas'.** This shift is applied to
    stellar continuum, so it needs stellar kinematics. Do *not* reach for the velocity
    map a normal LUCI fit writes out: that is the emission-line velocity of the gas,
    which in an early-type galaxy is routinely counter-rotating or in a misaligned
    disk -- a different velocity field, not a noisy version of the same one. And the
    region you stack here is chosen precisely because it has no gas in it, so an
    emission-line fit over it is returning noise, not a velocity.

    What to pass, in order of preference:

    1. A stellar velocity map -- ``absorption_velocity`` from an
       ``absorption_bool=True`` fit is one, as is anything external.
    2. **A single number**: the region's systemic velocity. A scalar is accepted and
       broadcast. This de-redshifts nothing, so the trough smears by whatever the
       internal stellar spread is, but it never shifts a spaxel the *wrong* way --
       which is what a gas map does. Keep the region small and this costs little.

    The two compose, if you want them to: stack with a scalar to get a width, fit with
    that width to get an ``absorption_velocity`` map, then restack with it.

    Spaxels with no velocity (NaN) or an implausible one are skipped.

    **Pass the same background arguments you will pass to the fit.** The sky is
    removed from each spaxel *before* it is stacked, for the same reason
    ``fit_calc`` removes it before applying the template: what is wanted is a
    template of the *stellar* continuum, and what is left after the sky comes off
    is the stellar continuum. Build the template from raw spectra and it carries
    the sky as well -- which then gets subtracted a second time, from a spectrum
    the fit has already cleaned. With ``bkgType='pca'`` that second subtraction is
    of a per-pixel background averaged over a region, so it does not even cancel
    where it is wrong.

    Removal happens before the Doppler shift, not after: the sky sits at fixed
    observed wavenumbers, so it has to come off in the observed frame.

    Args:
        cube: The ``SitelleCube`` to read spectra from
        region: Anything ``cube.region_indices`` accepts -- a ds9 ``.reg`` path,
            a ``.npy`` path, a boolean mask, or an ``(xs, ys)`` index pair
        vel_map: **Stellar** velocity in km/s -- either a map shaped ``(n_y, n_x)`` and
            indexed ``[y, x]`` (the orientation LUCI writes), or a single number applied
            to the whole region. Not the gas velocity; see above.
        vel_max: Skip spaxels whose \\|velocity\\| exceeds this, in km/s (default 500)
        mean: Return the per-spaxel mean spectrum rather than the sum (default
            True). Only the shape matters to ``subtract_absorption``, which
            normalises the template by its own median.
        bkg: Background spectrum to remove from each spaxel (default None)
        bkgType: 'standard', 'pca', or None (default None -- no removal). Passing
            ``bkg`` without ``bkgType`` is taken as 'standard', as in ``fit_calc``.
        pca_coefficient_array: Per-pixel PCA coefficients, for ``bkgType='pca'``
        pca_vectors: PCA eigenspectra, for ``bkgType='pca'``
        pca_mean: PCA mean spectrum, for ``bkgType='pca'``

    Return:
        The template, on ``cube.spectrum_axis``. Channels no spaxel could
        contribute to -- the ends, which shift out of range -- are NaN.

    Examples:
        Building the template the way a PCA-background fit will consume it:

        >>> _, pca, _, _, _, coeff = cube.create_background_subspace(...)
        >>> vel = fits.open('Luci_outputs/M86_velocity_Halpha.fits')[0].data
        >>> absorp = cube.build_absorption_template(
        ...     'stellar.reg', vel, bkgType='pca',
        ...     pca_coefficient_array=coeff, pca_vectors=pca.components_, pca_mean=pca.mean_)
        >>> cube.fit_cube(['Halpha'], 'sincgauss', [1], [1], 0, 100, 0, 100, absorp=absorp,
        ...               bkgType='pca', pca_coefficient_array=coeff,
        ...               pca_vectors=pca.components_, pca_mean=pca.mean_)
    """
    scheme = _resolve_background(bkg, bkgType, pca_coefficient_array, pca_vectors, pca_mean)
    xs, ys = cube.region_indices(region)
    if xs.size == 0:
        raise ValueError("The region selects no pixels, so there is nothing to build a template from.")

    vel_map = np.asarray(vel_map, dtype=float)
    n_x, n_y = cube.cube_final.shape[0], cube.cube_final.shape[1]
    if vel_map.ndim == 0:
        # A single systemic velocity for the region. The safe option when no stellar map
        # exists, and the one that keeps callers from substituting a gas velocity map
        # because it happens to be the 2D array they have.
        vel_map = np.full((n_y, n_x), float(vel_map))
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

    filter_name = cube.hdr_dict["FILTER"] if scheme == "pca" else None
    logger.info("Stacking %d spaxels into an absorption template (background: %s)", xs.size, scheme or "none removed")

    total = np.zeros(len(axis))
    count = np.zeros(len(axis), dtype=np.int64)
    n_used = 0
    for start in range(0, xs.size, _PIXEL_BLOCK):
        block = slice(start, start + _PIXEL_BLOCK)
        # `filled` because a cube read from HDF5 can come back masked, and `np.isfinite` on a masked
        # array returns a masked result whose `.sum()` does not count what this loop needs it to.
        spectra = np.asarray(np.ma.filled(cube.cube_final[xs[block], ys[block], :], np.nan), dtype=float)
        for spectrum, x_pix, y_pix, factor in zip(spectra, xs[block], ys[block], factors[block]):
            # Before the shift: the sky sits at fixed observed wavenumbers.
            if scheme == "standard":
                spectrum = subtract_standard(spectrum, bkg)
            elif scheme == "pca":
                background = pca_background(pca_coefficient_array[x_pix, y_pix], pca_vectors, pca_mean)
                spectrum = subtract_pca(spectrum, background, axis, filter_name)
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
