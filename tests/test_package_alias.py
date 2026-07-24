"""
Tests for the uppercase ``LUCI`` alias onto the lowercase ``luci`` package.

The package was renamed to satisfy PEP 8, but ``from LUCI.x import y`` appears in
every notebook and doc page and in unknown amounts of downstream code.  A shim
package is impossible on case-insensitive filesystems, so ``LUCI.py`` aliases the
modules through ``sys.modules``.

The thing that would actually hurt is a *duplicate* import -- ``LUCI.cube`` and
``luci.cube`` as two module objects with separate module-level state.  Most of
these tests exist to prove that doesn't happen.
"""

from __future__ import annotations

import sys

import pytest


def test_top_level_alias_resolves():
    import LUCI
    import luci

    assert LUCI.__version__ == luci.__version__


@pytest.mark.parametrize(
    "sub",
    [
        "cube",
        "config",
        "LuciFit",
        "LuciUtility",
        "fitting.models",
        "fitting.spectrum_fitter",
        "fitting.result",
        "instrument.filters",
        "engine.selection",
        "engine.runner",
        "ml.registry",
    ],
)
def test_submodule_is_the_same_object_under_both_names(sub):
    """Not merely importable -- literally the same module, so state is shared."""
    upper = __import__(f"LUCI.{sub}", fromlist=["_"])
    lower = __import__(f"luci.{sub}", fromlist=["_"])
    assert upper is lower
    assert sys.modules[f"LUCI.{sub}"] is sys.modules[f"luci.{sub}"]


def test_from_import_yields_identical_classes():
    from LUCI.cube import SitelleCube as Upper

    from luci.cube import SitelleCube as Lower

    assert Upper is Lower


def test_module_level_state_is_shared_not_duplicated():
    """
    The failure this guards against: a second copy of a module means a second copy
    of its globals, so a registry mutated through one name is invisible via the other.
    """
    import LUCI.instrument.filters as upper

    import luci.instrument.filters as lower

    sentinel = object()
    upper._alias_probe = sentinel
    try:
        assert getattr(lower, "_alias_probe", None) is sentinel
    finally:
        del upper._alias_probe


def test_unknown_submodule_still_raises():
    with pytest.raises(ImportError):
        __import__("LUCI.does_not_exist", fromlist=["_"])


def test_legacy_module_shims_work_through_the_alias():
    """The oldest documented spelling: `from LUCI.LuciFit import Fit`."""
    from LUCI.LuciFit import Fit

    from luci.fitting.spectrum_fitter import SpectrumFitter

    assert Fit is SpectrumFitter


def test_package_directory_is_lowercase_on_disk():
    import os

    import luci

    assert os.path.basename(os.path.dirname(luci.__file__)) == "luci"


def test_public_api_is_importable_from_the_package_root():
    """The documented entry point: `from luci import SitelleCube, FitConfig, FitResult`."""
    from luci import FitConfig, FitResult, SitelleCube
    from luci.config import FitConfig as ConfigFitConfig
    from luci.cube import SitelleCube as CubeSitelleCube

    assert SitelleCube is CubeSitelleCube
    assert FitConfig is ConfigFitConfig
    assert FitResult.__name__ == "FitResult"


def test_importing_the_package_does_not_pull_in_the_fitting_stack():
    """
    `import luci` must stay cheap -- the exports are lazy.

    Run in a subprocess because the rest of this suite has already imported
    everything into the parent interpreter.
    """
    import subprocess
    import sys as _sys

    code = "import sys, luci; print('luci.cube' in sys.modules, 'scipy' in sys.modules)"
    out = subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False False", out.stdout


def test_dunder_all_matches_what_is_actually_exported():
    import luci

    for name in luci.__all__:
        assert getattr(luci, name) is not None, name


def test_unknown_attribute_raises_attributeerror():
    import luci

    with pytest.raises(AttributeError):
        _ = luci.definitely_not_exported
