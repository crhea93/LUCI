"""
Fitting the stellar absorption trough that sits under an emission line.

This is the *fitted* counterpart to `luci.background.absorption`, which removes a
stacked template measured elsewhere in the field. Here the trough is measured on
the spectrum itself, per fit, and reported as three numbers -- fractional depth,
velocity, broadening -- alongside the emission-line parameters.

**The width is an input because the data cannot choose it.** Not because the problem is
formally degenerate -- with no noise, chi-squared picks the right width sharply (5e-07 at
the true 200 km/s against 0.02 at 150). The trouble is the scale of that penalty. The
core carries ~17 channels, so at 10% continuum noise the chi-squared floor is ~0.17 --
five times the penalty for getting the width wrong by half again. At any realistic
signal-to-noise the width is therefore unconstrained, and depth slides along a ridge
with it: at a true depth of 0.30, the best fit is 0.55 if the width is taken as 100 km/s,
0.30 at 200, and 0.21 at 300. Left free, noise picks a point on that ridge and the depth
follows; in practice sigma ran to whichever bound it was given.

So the stellar velocity dispersion is *supplied* and only the depth and velocity are
fitted. Get it from a template fit, from the literature for the galaxy type, or leave the
200 km/s default for an old population -- and treat it as the dominant systematic on the
depth, because it is.

**What it is worth: it depends on depth over noise, not depth.** With the width supplied
correctly and no noise the estimator is exact. With noise, the scatter on the depth is set
by the continuum noise and by how many channels survive the emission mask -- not by the
trough -- and follows a simple rule, measured to hold from 1% to 10% noise:

    sigma_depth ~= 0.7 * (continuum noise / continuum)

So what matters is whether your trough clears that. A shallow trough is not a problem in
itself; a shallow trough in a noisy continuum is. Injected 0.10, measured over 400 noise
realisations:

    continuum S/N    10      20      50      100
    recovered      0.125   0.098   0.099   0.099
    scatter        0.201   0.036   0.015   0.007

At S/N 10 it is unusable and *biased high*: the scatter has overtaken the signal, and since
a depth cannot be negative the distribution piles against zero and drags the mean up. Two
sigma above the noise and that bias is gone -- at S/N 20 the same trough returns 0.098.

Practical requirements, per spaxel:

* a 3-sigma detection of a trough of depth d needs continuum S/N > ~2 / d
* measuring d to 20% needs continuum S/N > ~3.5 / d

Both relax as sqrt(N) when you average, so a WVT bin of 100 spaxels turns S/N 10 into an
effective 100. Since the error is scatter rather than bias (once clear of the zero bound),
binning genuinely buys precision.

Emission brightness does *not* affect it: the recovered depth is unchanged with lines
from 0 to 30 times the continuum, so the mask is doing its job. The supplied width
does: told 300 km/s when the truth is 200, a 0.30 trough reads 0.235; told 100, 0.358.

`luci.background.absorption` -- a template stacked over a stellar region -- measures
the trough on starlight instead of under an emission line, and averages many spaxels
while doing it. Prefer it where the field gives you somewhere to build one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from luci.log import get_logger

logger = get_logger(__name__)

SPEED_OF_LIGHT = 299792  # km/s, matching luci.fitting.parameters

# Default stellar velocity dispersion, km/s -- an old population sits around 150-250. Held
# fixed during the fit, for the reason in the module docstring; override per fit with
# `absorption_broadening_kms`.
DEFAULT_BROADENING_KMS = 200.0

# How far the trough's centre may wander from the rest position of the line, km/s.
POSITION_BOUND_KMS = 1000.0

# Emission cores are masked to +/- MASK_SIGMAS * the emission broadening, but never
# less than MIN_MASK_KMS -- a broadening prior of 0 would otherwise mask nothing and
# let the emission line pull the trough's centre.
MASK_SIGMAS = 3.0
MIN_MASK_KMS = 75.0

# The trough is fitted in a window this many sigmas wide either side of the line, not over
# the whole fit range. Over a full SITELLE window the continuum is nothing like flat -- the
# transmission correction leaves a pronounced curve towards the filter edges -- and a
# straight-line continuum asked to cover all of it is fitted by a wide shallow Gaussian
# instead, halving the depth. Eight sigmas is narrow enough that a line is a fair description
# of the continuum and wide enough to leave sidebands to measure it in.
WINDOW_SIGMAS = 8.0

# Inside this many guess-sigmas is "core" -- where the trough is fitted. Beyond it, out to
# WINDOW_SIGMAS, is sideband, where the continuum is measured. At three sigmas the trough is
# down to about 1% of its depth, so the sidebands are continuum to well within the noise.
CORE_SIGMAS = 3.0

# Below this many channels in either the core or the sidebands, nothing is constrained. Sized
# against what SN3 actually provides: it samples at ~2 cm^-1, so a three-sigma core spans ~30
# channels and the emission masks take about half, leaving ~17 for three parameters.
MIN_CHANNELS = 10


@dataclass
class AbsorptionFit:
    """
    One spectrum's stellar absorption measurement.

    ``depth`` is the fraction of the continuum absorbed at the trough's centre --
    0.2 meaning "20% of the continuum is eaten". Fractional because it is the
    scale-free quantity: it does not depend on the normalisation the fit ran in, so
    it means the same thing for a single spaxel and a summed region.
    """

    depth: float = 0.0
    velocity: float = 0.0
    # Echoed back, not measured: this is the width the fit was told to assume. Reported so
    # the maps record what produced the depths rather than leaving it implicit.
    broadening: float = 0.0
    continuum: float = 0.0
    amplitude: float = 0.0  # In the units of the spectrum fitted; negative
    position: float = 0.0  # cm^-1
    sigma: float = 0.0  # cm^-1
    success: bool = False

    def profile(self, axis):
        """The trough alone (no continuum), evaluated on ``axis``. Zeros if the fit failed."""
        if not self.success:
            return np.zeros_like(np.asarray(axis, dtype=float))
        return absorption_profile(axis, self.amplitude, self.position, self.sigma)


def absorption_profile(axis, amplitude, position, sigma):
    """A Gaussian trough: negative ``amplitude``, so this is <= 0 everywhere."""
    axis = np.asarray(axis, dtype=float)
    return amplitude * np.exp(-0.5 * ((axis - position) / sigma) ** 2)


def emission_mask(axis, line_positions, broadening_kms):
    """
    Channels usable for the absorption fit: everything outside the emission cores.

    Args:
        axis: Spectral axis in cm^-1
        line_positions: Observed positions of the emission lines, cm^-1
        broadening_kms: Emission broadening estimate, km/s (the ML or data prior)

    Return:
        Boolean array, True where no emission line is within the mask half-width
    """
    axis = np.asarray(axis, dtype=float)
    halfwidth_kms = max(MASK_SIGMAS * abs(broadening_kms), MIN_MASK_KMS)
    keep = np.ones(axis.shape, dtype=bool)
    for position in line_positions:
        # A velocity half-width is a *fractional* one in wavenumber, so it converts
        # against the line's own position rather than a global scale.
        halfwidth = position * halfwidth_kms / SPEED_OF_LIGHT
        keep &= np.abs(axis - position) > halfwidth
    return keep


def fit_absorption(
    axis,
    spectrum,
    noise,
    rest_wavelength_nm,
    line_positions,
    broadening_kms,
    absorption_broadening_kms=DEFAULT_BROADENING_KMS,
):
    """
    Fit the depth and velocity of a stellar absorption trough, at a supplied width.

    Args:
        axis: Spectral axis in cm^-1, restricted to the fit window
        spectrum: Spectrum on that axis, in whatever normalisation the caller uses
        noise: 1-sigma noise in the same units
        rest_wavelength_nm: Rest wavelength of the absorbing line (e.x. 656.280 for Halpha)
        line_positions: Observed emission-line positions in cm^-1, whose cores are masked
        broadening_kms: *Emission* broadening estimate, which sets the mask width
        absorption_broadening_kms: *Stellar* velocity dispersion, held fixed (default 200).
            Not fitted -- see the module docstring for why it cannot be.

    Return:
        An ``AbsorptionFit``. ``success=False`` -- and a zero profile, so the caller's
        correction becomes a no-op -- when there are too few channels to constrain either
        the continuum or the trough, the optimiser fails, or the result is not finite.
    """
    axis = np.asarray(axis, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    rest_position = 1e7 / rest_wavelength_nm
    sigma = rest_position * absorption_broadening_kms / SPEED_OF_LIGHT
    sigma_guess = sigma

    in_window = np.abs(axis - rest_position) <= WINDOW_SIGMAS * sigma_guess
    usable = in_window & emission_mask(axis, line_positions, broadening_kms) & np.isfinite(spectrum)
    if usable.sum() < MIN_CHANNELS:
        logger.debug("Absorption fit skipped: %d usable channels, need %d", int(usable.sum()), MIN_CHANNELS)
        return AbsorptionFit()

    # The continuum is measured in sidebands beyond the trough and then held fixed, rather
    # than fitted alongside it. Fitting both at once is degenerate over a window this size:
    # a Gaussian's wings look like a slope plus an offset, so the optimiser trades trough
    # width against continuum tilt. It shows up as a deep trough being fitted narrow -- an
    # injected depth of 0.5 came back as 0.76 with sigma pinned at its floor -- and the two
    # failure modes (too shallow with a flat continuum, too deep with a sloped one) are the
    # same degeneracy seen from either side. Sidebands break it by construction.
    core = np.abs(axis - rest_position) <= CORE_SIGMAS * sigma_guess
    sideband = usable & ~core
    core = usable & core
    if sideband.sum() < MIN_CHANNELS or core.sum() < MIN_CHANNELS:
        logger.debug(
            "Absorption fit skipped: %d sideband and %d core channels, need %d of each",
            int(sideband.sum()),
            int(core.sum()),
            MIN_CHANNELS,
        )
        return AbsorptionFit()
    slope, intercept = np.polyfit(axis[sideband] - rest_position, spectrum[sideband], 1)

    def continuum_at(x):
        return intercept + slope * (np.asarray(x, dtype=float) - rest_position)

    fit_axis = axis[core]
    fit_spectrum = spectrum[core] - continuum_at(fit_axis)
    sigma2 = float(noise) ** 2 if noise else 1.0
    continuum = float(continuum_at(rest_position))

    # Seed the depth from the deepest point rather than a fixed fraction: a guess on the
    # wrong side of zero would start the optimiser against its own bound.
    amplitude_guess = min(float(np.nanmin(fit_spectrum)), -1e-3 * abs(continuum) if continuum else 0.0)

    def chi2(theta):
        amplitude, position = theta
        return float(np.nansum((fit_spectrum - absorption_profile(fit_axis, amplitude, position, sigma)) ** 2) / sigma2)

    position_span = rest_position * POSITION_BOUND_KMS / SPEED_OF_LIGHT
    bounds = [
        (-2.0 * abs(continuum) if continuum else None, 0.0),  # Absorption only
        (rest_position - position_span, rest_position + position_span),
    ]
    solution = minimize(chi2, [amplitude_guess, rest_position], method="L-BFGS-B", bounds=bounds)
    amplitude, position = solution.x
    if not solution.success or not np.all(np.isfinite(solution.x)) or continuum <= 0 or position <= 0:
        logger.debug("Absorption fit did not converge: %s", getattr(solution, "message", "no message"))
        return AbsorptionFit()

    return AbsorptionFit(
        depth=float(abs(amplitude) / continuum),
        # Same conventions as luci.fitting.parameters, so absorption kinematics are
        # directly comparable with the emission lines'.
        velocity=float(SPEED_OF_LIGHT * (1e7 / position - rest_wavelength_nm) / rest_wavelength_nm),
        broadening=float(abs(SPEED_OF_LIGHT * sigma / position)),
        continuum=float(continuum),
        amplitude=float(amplitude),
        position=float(position),
        sigma=float(sigma),
        success=True,
    )


# A template with no trough in it has no width to measure: the fit puts the amplitude on its
# zero bound and the width is then whatever the optimiser last held, which is meaningless. Below
# this fractional depth, say so instead of returning a number.
MIN_TEMPLATE_DEPTH = 1e-3

# A template is a stack over many spaxels with no emission in it, so the trough's own peak is
# available and the window can be generous. This is the half-width used to fit one, in km/s --
# wide enough to bracket any plausible stellar trough and leave continuum either side.
TEMPLATE_WINDOW_KMS = 2000.0


def measure_absorption_width(axis, template, rest_wavelength_nm=None):
    """
    Fit a stacked absorption template to recover the trough's width.

    This is the measurement `fit_absorption` cannot make. There, the trough's centre is masked
    out with the emission line and the width is unconstrained at realistic noise, so it has to
    be supplied. Here neither applies: a template is starlight with no gas in it, stacked over
    however many spaxels the region held, so the peak is observed directly and there is nothing
    to mask. Amplitude, position and width separate cleanly, and the continuum can be fitted
    alongside them instead of from sidebands.

    So the intended workflow is: stack a template over a gas-free region, measure its width here,
    and hand that width to the per-spaxel fits. `resolve_absorption_width` does that for you when
    a fit is given a template.

    One caveat on interpretation: what comes back is the width of the *template*, which carries
    whatever velocity smearing survived the de-redshifted stack as well as the intrinsic trough.
    That is the right width to subtract, and slightly wider than a single spaxel's.

    Args:
        axis: Spectral axis in cm^-1, matching `template`
        template: A stacked absorption template, e.x. from `build_absorption_template`
        rest_wavelength_nm: Rest wavelength of the absorbing line (default Halpha)

    Return:
        An ``AbsorptionFit`` whose ``broadening`` is **measured** rather than echoed.
        ``success=False`` if there is too little to fit.
    """
    from luci.config import LINE_DICT  # Local: luci.config must not depend on the fitters.

    axis = np.asarray(axis, dtype=float)
    template = np.asarray(template, dtype=float)
    rest_wavelength_nm = rest_wavelength_nm or LINE_DICT["Halpha"]
    rest_position = 1e7 / rest_wavelength_nm

    span = rest_position * TEMPLATE_WINDOW_KMS / SPEED_OF_LIGHT
    usable = (np.abs(axis - rest_position) <= span) & np.isfinite(template)
    if usable.sum() < MIN_CHANNELS:
        logger.debug("Template width fit skipped: %d usable channels", int(usable.sum()))
        return AbsorptionFit()

    # Normalise before fitting. A template in real flux units sits around 1e-17, so the
    # sum-of-squares is ~1e-32 and L-BFGS-B's convergence test is satisfied at the starting
    # point -- it returns the initial guess without ever moving. Both quantities we want out of
    # here (a fractional depth and a width) are scale-free, so this costs nothing.
    scale = float(np.nanmedian(template[usable]))
    if not np.isfinite(scale) or scale == 0:
        logger.debug("Template width fit skipped: template median is %r", scale)
        return AbsorptionFit()
    template = template / scale

    fit_axis, fit_template = axis[usable], template[usable]
    continuum_guess = float(np.nanmedian(fit_template))
    dip = float(np.nanmin(fit_template)) - continuum_guess
    sigma_guess = rest_position * DEFAULT_BROADENING_KMS / SPEED_OF_LIGHT

    def chi2(theta):
        amplitude, position, sigma, continuum, slope = theta
        model = continuum + slope * (fit_axis - rest_position)
        model = model + absorption_profile(fit_axis, amplitude, position, sigma)
        return float(np.nansum((fit_template - model) ** 2))

    position_span = rest_position * POSITION_BOUND_KMS / SPEED_OF_LIGHT
    solution = minimize(
        chi2,
        [min(dip, -1e-6 * abs(continuum_guess)), rest_position, sigma_guess, continuum_guess, 0.0],
        method="L-BFGS-B",
        bounds=[
            (-2.0 * abs(continuum_guess) if continuum_guess else None, 0.0),
            (rest_position - position_span, rest_position + position_span),
            # Wide open, because here the data can actually choose.
            (rest_position * 10.0 / SPEED_OF_LIGHT, rest_position * 1500.0 / SPEED_OF_LIGHT),
            (0.0, None),
            (None, None),
        ],
    )
    amplitude, position, sigma, continuum, _slope = solution.x
    if not solution.success or not np.all(np.isfinite(solution.x)) or continuum <= 0 or position <= 0:
        logger.debug("Template width fit did not converge: %s", getattr(solution, "message", ""))
        return AbsorptionFit()
    if abs(amplitude) / continuum < MIN_TEMPLATE_DEPTH:
        logger.warning(
            "No absorption trough found in this template (depth < %g), so there is no width to "
            "measure. Is the region actually dominated by starlight?",
            MIN_TEMPLATE_DEPTH,
        )
        return AbsorptionFit()
    return AbsorptionFit(
        depth=float(abs(amplitude) / continuum),
        velocity=float(SPEED_OF_LIGHT * (1e7 / position - rest_wavelength_nm) / rest_wavelength_nm),
        broadening=float(abs(SPEED_OF_LIGHT * sigma / position)),
        # Back on the caller's scale, so these mean what they would have without the normalisation.
        continuum=float(continuum * scale),
        amplitude=float(amplitude * scale),
        position=float(position),
        sigma=float(sigma),
        success=True,
    )


def resolve_absorption_width(axis, absorp, absorption_broadening_kms, rest_wavelength_nm=None):
    """
    Decide what stellar width the per-spaxel fits should assume.

    The order is: what the caller asked for, else what the template says, else the default. The
    middle case is the one that matters -- a fit given a template no longer needs to be told the
    width, because the template can be measured. Getting that width wrong is the dominant
    systematic on the depth (roughly depth ~ 1/width), which is why guessing is a poor last resort
    and is logged when it happens.

    Args:
        axis: Spectral axis in cm^-1, matching `absorp`
        absorp: The absorption template being subtracted, or None
        absorption_broadening_kms: An explicit width in km/s, or None to determine one
        rest_wavelength_nm: Rest wavelength of the absorbing line (default Halpha)

    Return:
        Width in km/s.
    """
    if absorp is not None:
        # Both corrections are on. The template subtraction has already removed the trough, so
        # the per-spaxel fit will measure ~0 and change nothing -- harmless, but it means one of
        # the two is redundant. Use `cube.measure_absorption_width` to take the width from a
        # template without also subtracting it.
        logger.warning(
            "absorp= and absorption_bool=True are both set: the template has already removed the "
            "trough, so the fitted absorption will measure nothing. Pick one."
        )
    if absorption_broadening_kms is not None:
        return float(absorption_broadening_kms)
    if absorp is not None:
        measured = measure_absorption_width(axis, absorp, rest_wavelength_nm)
        if measured.success:
            logger.info(
                "Stellar width measured from the absorption template: %.0f km/s (depth %.2f)",
                measured.broadening,
                measured.depth,
            )
            return measured.broadening
        logger.warning(
            "Could not measure a width from the absorption template; using %.0f km/s", DEFAULT_BROADENING_KMS
        )
    else:
        logger.info(
            "No absorption template to measure a stellar width from; assuming %.0f km/s. Pass "
            "absorption_broadening_kms, or a template, if you know better -- the depth scales "
            "roughly as 1/width.",
            DEFAULT_BROADENING_KMS,
        )
    return DEFAULT_BROADENING_KMS
