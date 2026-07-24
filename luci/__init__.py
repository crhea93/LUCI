"""
LUCI: a general-purpose emission-line fitting pipeline for SITELLE IFU cubes.

The public API::

    from luci import SitelleCube, FitConfig, FitResult

    cube = SitelleCube(luci_path, cube_path, output_dir, object_name, redshift, resolution)
    cube.fit_cube(["Halpha", "NII6583"], "sincgauss", [1, 1], [1, 1], x_min, x_max, y_min, y_max)

``from LuciBase import Luci`` and ``from LUCI... import ...`` both still work --
see :mod:`LUCI` (the top-level alias module) and the shim modules named
``Luci*.py`` inside this package.

Names resolve lazily so ``import luci`` stays cheap; the fitting stack pulls in
scipy, astropy and onnxruntime, which is a slow import to pay for on a caller
that only wants a filter table.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("luci-sitelle")
except PackageNotFoundError:  # source tree with no install
    __version__ = "0.0.0.dev0"

# name -> (module, attribute)
_EXPORTS = {
    "SitelleCube": ("luci.cube", "SitelleCube"),
    "Luci": ("luci.cube", "SitelleCube"),
    "FitConfig": ("luci.config", "FitConfig"),
    "InvalidFitConfig": ("luci.config", "InvalidFitConfig"),
    "LINE_DICT": ("luci.config", "LINE_DICT"),
    "AVAILABLE_MODELS": ("luci.config", "AVAILABLE_MODELS"),
    "SpectrumFitter": ("luci.fitting.spectrum_fitter", "SpectrumFitter"),
    "FitResult": ("luci.fitting.result", "FitResult"),
    "FILTERS": ("luci.instrument.filters", "FILTERS"),
    "FilterSpec": ("luci.instrument.filters", "FilterSpec"),
    "UnsupportedFilterError": ("luci.instrument.filters", "UnsupportedFilterError"),
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value  # resolve once
    return value


def __dir__():
    return sorted(__all__)
