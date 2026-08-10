"""
Configuration objects for a fit.

``SpectrumFitter`` takes 28 positional/keyword arguments, which makes call sites
hard to read and means every new option has to be threaded through several
layers by hand. ``FitConfig`` groups the ones that describe *what to fit and
how* -- as opposed to the spectrum itself or the instrument it came from -- into
one validated object.

The fitter still accepts the individual keywords, so nothing existing breaks;
they are simply collected into a ``FitConfig`` internally. New code can build one
explicitly and reuse it across many spectra.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# Rest wavelengths in nm. The canonical list -- SpectrumFitter reads it from here.
LINE_DICT = {
    "Halpha": 656.280,
    "NII6583": 658.341,
    "NII6548": 654.803,
    "SII6716": 671.647,
    "SII6731": 673.085,
    "OII3726": 372.603,
    "OII3729": 372.882,
    "OIII4959": 495.891,
    "OIII5007": 500.684,
    "Hbeta": 486.133,
    "OH": 649.873,
    "HalphaC4": 807.881,
    "NII6583C4": 810.417,
    "NII6548C4": 804.7,
    "OIII5007C2": 616.342,
    "OIII4959C2": 610.441821,
    "HbetaC2": 598.429723,
    "OII3729C1": 459.017742,
    "OII3726C1": 458.674293,
    "OI6364": 636.3776,
    "FeXIV5303": 530.286,
    "NI5200": 520.026,
    "FeVII5158": 515.89,
    "HeII5411": 541.152,
    "FeXI6624": 662.43,
    "NiXV6703": 670.332,
}

AVAILABLE_MODELS = ("gaussian", "sinc", "sincgauss")


class InvalidFitConfig(ValueError):
    """Raised when a requested fit cannot be performed as described."""


@dataclass
class FitConfig:
    """
    What to fit and how.

    Validated on construction, so a bad request fails at the call site with a
    clear message instead of somewhere inside the optimiser.
    """

    lines: list[str]
    model_type: str = "sincgauss"
    vel_rel: list[int] = field(default_factory=list)
    sigma_rel: list[int] = field(default_factory=list)

    # Constraints
    nii_cons: bool = True

    # Priors
    ML_bool: bool = True
    mdn: bool = False
    initial_values: list = field(default_factory=lambda: [False])

    # Fit window and sampling
    spec_min: float | None = None
    spec_max: float | None = None
    obj_redshift: float = 0.0
    n_stoch: int = 1

    # Uncertainty / Bayesian
    uncertainty_bool: bool = False
    bayes_bool: bool = False
    bayes_method: str = "emcee"

    # Measure and remove the stellar absorption trough before fitting the lines, and the
    # stellar velocity dispersion (km/s) to assume for it -- which is an input, not a fitted
    # quantity; see luci.fitting.absorption.
    absorption_bool: bool = False
    absorption_broadening_kms: float | None = None

    def __post_init__(self) -> None:
        # 'gauss' was long accepted as an alias; normalise before validating.
        if self.model_type == "gauss":
            self.model_type = "gaussian"
        if not self.vel_rel:
            self.vel_rel = [1] * len(self.lines)
        if not self.sigma_rel:
            self.sigma_rel = [1] * len(self.lines)
        if self.absorption_broadening_kms is None:
            # None reaches here when nothing upstream determined a width -- either no template to
            # measure one from, or a direct SpectrumFitter call. Fall back to an old population.
            self.absorption_broadening_kms = 200.0
        self.validate()

    def validate(self) -> None:
        if self.model_type not in AVAILABLE_MODELS:
            raise InvalidFitConfig(
                f"Unknown fit function {self.model_type!r}. Available: {', '.join(AVAILABLE_MODELS)}."
            )
        unknown = [line for line in self.lines if line not in LINE_DICT]
        if unknown:
            raise InvalidFitConfig(f"Unknown line(s) {unknown}. Available: {', '.join(sorted(LINE_DICT))}.")
        if len(self.vel_rel) != len(self.lines):
            raise InvalidFitConfig(f"vel_rel has {len(self.vel_rel)} entries but there are {len(self.lines)} lines.")
        if len(self.sigma_rel) != len(self.lines):
            raise InvalidFitConfig(
                f"sigma_rel has {len(self.sigma_rel)} entries but there are {len(self.lines)} lines."
            )
        if self.bayes_method not in ("emcee", "dynesty"):
            raise InvalidFitConfig(f"bayes_method must be 'emcee' or 'dynesty', got {self.bayes_method!r}.")
        if self.absorption_broadening_kms <= 0:
            raise InvalidFitConfig(f"absorption_broadening_kms must be > 0, got {self.absorption_broadening_kms}.")
        if self.n_stoch < 1:
            raise InvalidFitConfig(f"n_stoch must be >= 1, got {self.n_stoch}.")

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    @property
    def freeze(self) -> bool:
        """Whether velocity and broadening are held at supplied initial values."""
        return bool(self.initial_values) and self.initial_values[0] is not False

    def replace(self, **changes) -> "FitConfig":
        """A copy with some fields changed, re-validated."""
        return replace(self, **changes)
