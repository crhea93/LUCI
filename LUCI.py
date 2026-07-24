"""
Compatibility alias: ``import LUCI...`` keeps working after the rename to ``luci``.

The package is now lowercase, but the uppercase spelling is what every notebook,
doc page and downstream script uses.  A shim *package* named ``LUCI`` is
impossible -- on macOS and Windows the filesystem cannot hold both ``LUCI/`` and
``luci/`` -- so this module registers the alias in ``sys.modules`` instead.
``LUCI.py`` and ``luci/`` are distinct names even case-insensitively, and
Python's finder matches the directory listing exactly, so ``import LUCI``
resolves here on every platform.

The finder goes at the front of ``sys.meta_path`` so it beats the path-based
finder, which would otherwise import ``luci/cube.py`` a second time under the
name ``LUCI.cube`` and give you two modules with independent state.
"""

from __future__ import annotations

import importlib
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

import luci

_PREFIX = "LUCI."


def _target(fullname: str) -> str:
    return "luci." + fullname[len(_PREFIX) :]


class _AliasLoader(Loader):
    def create_module(self, spec):
        return importlib.import_module(_target(spec.name))

    def exec_module(self, module):
        """Already executed under its real name."""


class _AliasFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_PREFIX):
            return None
        try:
            module = importlib.import_module(_target(fullname))
        except ImportError:
            return None
        spec = ModuleSpec(fullname, _AliasLoader())
        spec.submodule_search_locations = getattr(module, "__path__", None)
        return spec


sys.meta_path.insert(0, _AliasFinder())

# Required so `import LUCI.cube` accepts LUCI as a package; the finder above
# resolves the submodule before this path is ever searched.
__path__ = luci.__path__
__all__ = getattr(luci, "__all__", [])
__version__ = getattr(luci, "__version__", "")


def __getattr__(name):
    return getattr(luci, name)
