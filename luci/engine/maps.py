"""
The set of output maps a cube fit produces, and how results scatter into them.

``fit_cube``, ``fit_region``, ``fit_wvt`` and ``create_snr_map`` each used to
allocate these twelve arrays by hand, unpack the same thirteen-element tuple by
hand, and call ``save_fits`` by hand.  That duplication is what let B3, B5, B6
and B9 exist: a parameter added or a bug fixed in one entry point never reached
the others.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from luci.io.outputs import save_fits


@dataclass
class FitMaps:
    """Per-pixel fit products, shaped (y, x) for scalars and (y, x, line) per line."""

    amplitudes: np.ndarray
    fluxes: np.ndarray
    flux_errors: np.ndarray
    velocities: np.ndarray
    broadenings: np.ndarray
    velocities_errors: np.ndarray
    broadenings_errors: np.ndarray
    chi2: np.ndarray
    corr: np.ndarray
    step: np.ndarray
    continuum: np.ndarray
    continuum_error: np.ndarray
    # Stellar absorption. Allocated always, written only when the fit measured it, so the
    # dataclass does not change shape with the option; `absorption` gates the *output*
    # files, which downstream scripts glob for.
    absorption_depth: np.ndarray | None = None
    absorption_velocity: np.ndarray | None = None
    absorption_broadening: np.ndarray | None = None
    absorption: bool = False

    @classmethod
    def allocate(cls, n_x: int, n_y: int, n_lines: int, absorption: bool = False) -> "FitMaps":
        def scalar():
            return np.zeros((n_x, n_y), dtype=np.float32).T

        def per_line():
            return np.zeros((n_x, n_y, n_lines), dtype=np.float32).transpose(1, 0, 2)

        return cls(
            absorption_depth=scalar(),
            absorption_velocity=scalar(),
            absorption_broadening=scalar(),
            absorption=absorption,
            amplitudes=per_line(),
            fluxes=per_line(),
            flux_errors=per_line(),
            velocities=per_line(),
            broadenings=per_line(),
            velocities_errors=per_line(),
            broadenings_errors=per_line(),
            chi2=scalar(),
            corr=scalar(),
            step=scalar(),
            continuum=scalar(),
            continuum_error=scalar(),
        )

    def scatter(self, result) -> None:
        """Write one y-slice of results, as returned by ``Luci.fit_calc``."""
        (
            i,
            ampls,
            flux,
            flux_errs,
            vels,
            vels_errs,
            broads,
            broads_errs,
            chi2,
            corr,
            step,
            continuum,
            continuum_errs,
            absorption,
        ) = result
        self.amplitudes[i] = ampls
        self.fluxes[i] = flux
        self.flux_errors[i] = flux_errs
        self.velocities[i] = vels
        self.broadenings[i] = broads
        self.velocities_errors[i] = vels_errs
        self.broadenings_errors[i] = broads_errs
        self.chi2[i] = chi2
        self.corr[i] = corr
        self.step[i] = step
        self.continuum[i] = continuum
        self.continuum_error[i] = continuum_errs
        # (depth, velocity, broadening) per pixel; all zeros unless absorption was fitted.
        absorption = np.asarray(absorption, dtype=np.float32)
        self.absorption_depth[i] = absorption[:, 0]
        self.absorption_velocity[i] = absorption[:, 1]
        self.absorption_broadening[i] = absorption[:, 2]

    def save(
        self, output_dir, object_name, lines, header, binning, fit_function=None, suffix="", output_name=None
    ) -> None:
        save_fits(
            output_dir,
            object_name,
            lines,
            self.amplitudes,
            self.fluxes,
            self.flux_errors,
            self.velocities,
            self.broadenings,
            self.velocities_errors,
            self.broadenings_errors,
            self.chi2,
            self.continuum,
            self.continuum_error,
            header,
            binning,
            suffix=suffix,
            fit_function=fit_function,
            output_name=output_name,
            absorption_maps=(
                {
                    "absorption_depth": self.absorption_depth,
                    "absorption_velocity": self.absorption_velocity,
                    "absorption_broadening": self.absorption_broadening,
                }
                if self.absorption
                else None
            ),
        )
