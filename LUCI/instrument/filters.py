"""
The SITELLE filter registry.

Before this module, every band-dependent wavelength window in LUCI was an
``if filter == 'SN3' / elif 'SN2' / ...`` chain, and the same chain was
copy-pasted in five places with drifting numbers:

  * ``LuciFit.restrict_wavelength``   -- the fit window
  * ``LuciFit.calculate_noise``       -- the noise-estimation window
  * ``LuciUtility.read_in_reference_spectrum`` -- the reference-spectrum clip
  * the PCA background-scaling windows in ``LuciBase`` (routed in a later phase)
  * ``LuciSim.Spectrum``              -- delta_x / order per filter (separate concern)

Two of those copies had latent bugs that only a single source of truth can
prevent (see REFACTOR_BUGS.md):

  * **B4** -- the normal-C3 branch of ``calculate_noise`` misspelt
    ``bound_upper`` as ``buond_upper``, so ``bound_upper`` was never assigned.
    Because the window bounds were module-level ``global`` variables, C3
    inherited whatever window the previously-fitted spectrum left behind. Fixed
    here: normal C3 noise is ``(20000, 20250)`` cm^-1.
  * The C4 noise window was gated on ``'Halpha' in lines``; without Halpha the
    branch fell through to an error and stale globals. Here the C4 noise window
    is unconditional, so a C4 fit without Halpha gets a real noise estimate.

All bounds are in wavenumber (cm^-1).  A few filters need context:

  * C1 / C2 / C4 windows scale with ``(1 + obj_redshift)`` -- these filters are
    used for redshifted lines, so the window moves with the object.
  * C3 has a trigger line: when ``OII3726`` is being fit the object sits near
    z ~= 0.465 and C3 is treated as if it were SN1, so both windows shift.

The SN1/SN2/SN3/SN4 windows are plain unconditional pairs and are transcribed
byte-for-byte from the original code, because the golden baselines depend on
them being identical.
"""

from __future__ import annotations

from dataclasses import dataclass


class UnsupportedFilterError(ValueError):
    """Raised when a cube's FILTER keyword is not in the registry.

    The old code printed a message and either continued with stale ``global``
    bounds (silently wrong) or called ``quit()`` / ``exit()`` from library code
    (uncatchable).  An exception is both catchable and honest.
    """

    def __init__(self, filter_name: object) -> None:
        supported = ", ".join(SUPPORTED_FILTERS)
        super().__init__(f"Filter {filter_name!r} is not supported by LUCI. " f"Supported filters: {supported}.")
        self.filter_name = filter_name


@dataclass(frozen=True)
class BoundRule:
    """A wavelength window, possibly redshift-scaled or line-triggered.

    ``resolve`` returns the ``(lower, upper)`` pair in cm^-1, applying:

      * ``trigger_line``: if that line is present in the fit, use the trigger
        bounds instead of the defaults;
      * ``redshift_scaled``: multiply both bounds by ``redshift_corr``
        (``1 + obj_redshift``).
    """

    lower: float
    upper: float
    redshift_scaled: bool = False
    trigger_line: str | None = None
    trigger_lower: float | None = None
    trigger_upper: float | None = None

    def resolve(self, lines: object = (), redshift_corr: float = 1.0) -> tuple[float, float]:
        if self.trigger_line is not None and self.trigger_line in lines:
            lo, hi = self.trigger_lower, self.trigger_upper
        else:
            lo, hi = self.lower, self.upper
        if self.redshift_scaled:
            return lo * redshift_corr, hi * redshift_corr
        return lo, hi


@dataclass(frozen=True)
class FilterSpec:
    """Everything band-dependent about a single SITELLE filter."""

    name: str
    fit: BoundRule
    noise: BoundRule
    reference: BoundRule
    # Line-free window, in **nanometres**, used to scale a PCA background
    # eigenspectrum onto an observed spectrum.  Given as
    # (longer_wavelength, shorter_wavelength) because the code converts to
    # wavenumber (1e7 / nm), which reverses the ordering.  None where the PCA
    # background has not been characterised for that filter.
    pca_scale: tuple[float, float] | None = None

    def fit_bounds(self, lines: object = (), redshift_corr: float = 1.0) -> tuple[float, float]:
        """Window over which the fit is performed (``restrict_wavelength``)."""
        return self.fit.resolve(lines, redshift_corr)

    def noise_bounds(self, lines: object = (), redshift_corr: float = 1.0) -> tuple[float, float]:
        """Line-free window used to estimate the noise (``calculate_noise``)."""
        return self.noise.resolve(lines, redshift_corr)

    def reference_bounds(self) -> tuple[float, float]:
        """Clip window for the ML reference spectrum."""
        return self.reference.resolve()


# --------------------------------------------------------------------------
# The registry.  One entry replaces five hand-maintained if/elif chains.
# --------------------------------------------------------------------------
FILTERS: dict[str, FilterSpec] = {
    "SN3": FilterSpec(
        "SN3",
        fit=BoundRule(14750, 15400),
        noise=BoundRule(15600, 15800),
        reference=BoundRule(14700, 15600),
        pca_scale=(675.0, 670.0),
    ),
    "SN2": FilterSpec(
        "SN2",
        fit=BoundRule(19500, 20750),
        noise=BoundRule(18600, 19000),
        reference=BoundRule(19000, 21000),
        pca_scale=(505.0, 480.0),
    ),
    "SN1": FilterSpec(
        "SN1",
        fit=BoundRule(26000, 28000),
        noise=BoundRule(26000, 26200),
        reference=BoundRule(25500, 27500),
        pca_scale=(365.0, 360.0),
    ),
    "SN4": FilterSpec(
        "SN4",
        # Narrow Halpha filter (652-665 nm).  The order-15 free spectral range
        # is far wider than the pass band, so the noise window sits in a region
        # the filter blocks entirely.
        fit=BoundRule(15040, 15330),
        noise=BoundRule(14600, 14900),
        reference=BoundRule(15000, 15350),
        pca_scale=(664.5, 661.0),
    ),
    "C3": FilterSpec(
        "C3",
        # OII3726 present -> object near z ~= 0.465; treat C3 as SN1.
        fit=BoundRule(18100, 19500, trigger_line="OII3726", trigger_lower=26000, trigger_upper=29000),
        # B4 fix: normal-C3 upper was never assigned in the original code.
        noise=BoundRule(20000, 20250, trigger_line="OII3726", trigger_lower=26000, trigger_upper=26200),
        reference=BoundRule(17500, 19500),
    ),
    "C4": FilterSpec(
        "C4",
        # Redshifted Halpha (~z = 0.25).  Noise window was gated on 'Halpha' in
        # the original; here it is unconditional so a C4 fit without Halpha
        # still gets a valid noise estimate.
        fit=BoundRule(12150, 12550, redshift_scaled=True),
        noise=BoundRule(11800, 12150, redshift_scaled=True),
        reference=BoundRule(12100, 12600),
    ),
    "C2": FilterSpec(
        "C2",
        fit=BoundRule(15990, 17880, redshift_scaled=True),
        noise=BoundRule(15500, 15990, redshift_scaled=True),
        reference=BoundRule(15987, 17880),
    ),
    "C1": FilterSpec(
        "C1",
        fit=BoundRule(20408, 25974, redshift_scaled=True),
        noise=BoundRule(18000, 20665, redshift_scaled=True),
        reference=BoundRule(20408, 25974),
    ),
}

# Ordered for stable, readable error messages.
SUPPORTED_FILTERS: tuple[str, ...] = ("SN1", "SN2", "SN3", "SN4", "C1", "C2", "C3", "C4")


class PCABackgroundUnsupportedError(ValueError):
    """Raised when a filter has no characterised PCA background-scaling window."""

    def __init__(self, filter_name: object) -> None:
        supported = ", ".join(sorted(n for n, f in FILTERS.items() if f.pca_scale))
        super().__init__(
            f"PCA background subtraction is not implemented for filter {filter_name!r}. " f"Supported: {supported}."
        )
        self.filter_name = filter_name


def pca_scale_indices(filter_name: object, spectrum_axis) -> tuple[int, int]:
    """
    Index bounds of the line-free window used to scale a PCA background.

    ``spectrum_axis`` is in wavenumber (cm^-1); the stored window is in
    nanometres, so each edge is matched via ``1e7 / wavelength``.

    This replaces the same if/elif chain that was copy-pasted three times in
    ``LuciBase`` (``fit_calc``, ``fit_pixel``, ``create_background_subspace``),
    each of which called ``quit()`` on an unsupported filter -- killing the
    interpreter from library code.  This raises instead.
    """
    import numpy as _np

    spec = get_filter(filter_name)
    if spec.pca_scale is None:
        raise PCABackgroundUnsupportedError(filter_name)
    lower_nm, upper_nm = spec.pca_scale
    axis = _np.asarray(spectrum_axis)
    lower_idx = int(_np.argmin(_np.abs(1e7 / axis - lower_nm)))
    upper_idx = int(_np.argmin(_np.abs(1e7 / axis - upper_nm)))
    return lower_idx, upper_idx


def get_filter(name: object) -> FilterSpec:
    """Look up a :class:`FilterSpec`, raising :class:`UnsupportedFilterError`."""
    try:
        return FILTERS[name]  # type: ignore[index]
    except (KeyError, TypeError):
        raise UnsupportedFilterError(name) from None
