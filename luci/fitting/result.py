"""
The structured result of a single-spectrum fit.

Historically ``Fit.fit()`` returned a bare 22-key dict, which meant every caller
indexed it by string with no schema, no type information, and no protection
against a typo silently returning ``KeyError`` at runtime.  ``FitResult`` is that
schema, written down once.

It stays **read-compatible with the old dict** on purpose: ``result['velocities']``
still works, so existing callers (``LuciBase.fit_calc``, ``fit_pixel``, notebooks)
are untouched, and ``as_dict()`` reproduces the exact legacy mapping in the exact
legacy key order.  New code can use typed attributes (``result.velocities``).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class FitResult:
    """Outputs of fitting one spectrum.

    Field order matches the historical dict so ``as_dict()`` is order-identical.
    Per-line quantities (amplitudes, fluxes, velocities, ...) are lists with one
    entry per fitted line; the rest are scalars or arrays.
    """

    fit_sol: Any
    fit_uncertainties: Any
    amplitudes: list
    fluxes: list
    flux_errors: list
    chi2: float
    velocities: list
    sigmas: list
    vels_errors: list
    sigmas_errors: list
    axis_step: float
    corr: float
    continuum: float
    continuum_error: float
    scale: float
    flat_samples: Any
    vel_ml: float
    vel_ml_sigma: float
    broad_ml: float
    broad_ml_sigma: float
    fit_vector: Any
    fit_axis: Any
    # Stellar absorption, measured only when `absorption_bool` was set; all three are
    # 0.0 otherwise, so the schema does not change shape with the option. `depth` is the
    # fraction of the continuum absorbed at the trough's centre. Appended at the end so
    # `as_dict()` keeps its historical key order for everything before them.
    absorption_depth: float = 0.0
    absorption_velocity: float = 0.0
    absorption_broadening: float = 0.0

    # -- dict compatibility (read-only) ------------------------------------
    # The old return value was a plain dict indexed like fit_dict['velocities'].
    # These delegate to the dataclass fields so that access pattern keeps working
    # without anyone having to change downstream code.

    def as_dict(self) -> dict:
        """Return the exact legacy dict (same keys, same order)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __contains__(self, key: object) -> bool:
        return any(f.name == key for f in fields(self))

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return [f.name for f in fields(self)]

    def items(self):
        return self.as_dict().items()

    def values(self):
        return self.as_dict().values()

    def __iter__(self):
        return iter(self.keys())
