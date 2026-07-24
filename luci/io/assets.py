"""
Locating the data and model files LUCI ships with.

Every entry point used to take a ``Luci_path`` string with a mandatory trailing
slash -- ``check_luci_path`` exists purely to paper over people forgetting it --
and every example hardcoded an absolute path to someone else's home directory.
Now that LUCI installs as a package, that location is derivable, so
``Luci_path`` becomes optional: pass one to override, or let it resolve.
"""

from __future__ import annotations

import os
from pathlib import Path

from luci.log import get_logger

logger = get_logger(__name__)

#: Subdirectories that must exist for a directory to be a usable LUCI root.
REQUIRED = ("ML", "Data")


class LuciAssetsNotFound(FileNotFoundError):
    """Raised when the ML/ and Data/ directories cannot be located."""


def check_luci_path(Luci_path):
    """
    Functionality to check that the user has included the trailing "/" to Luci_path.
    If they have not, we add it.
    """
    if not Luci_path.endswith("/"):
        Luci_path += "/"
        logger.debug("Added a trailing '/' to Luci_path.")
    return Luci_path


def _looks_like_luci_root(path: Path) -> bool:
    return all((path / sub).is_dir() for sub in REQUIRED)


def default_luci_path() -> str:
    """
    Locate the directory holding ``ML/`` and ``Data/``.

    Order: ``$LUCI_DATA_DIR``, then the directory containing the installed
    package -- which is the repository root for a clone or editable install.
    """
    env = os.environ.get("LUCI_DATA_DIR")
    if env:
        candidate = Path(env).expanduser().resolve()
        if not _looks_like_luci_root(candidate):
            raise LuciAssetsNotFound(f"$LUCI_DATA_DIR is {env!r} but it does not contain {' and '.join(REQUIRED)}/.")
        return check_luci_path(str(candidate))

    root = Path(__file__).resolve().parent.parent.parent
    if _looks_like_luci_root(root):
        return check_luci_path(str(root))

    raise LuciAssetsNotFound(
        f"Could not locate LUCI's {'/, '.join(REQUIRED)}/ directories (looked in {root}). "
        "Set $LUCI_DATA_DIR or pass Luci_path explicitly."
    )


def resolve_luci_path(Luci_path: str | None = None) -> str:
    """Normalise an explicit ``Luci_path``, or work one out. Always ends in '/'."""
    if Luci_path:
        return check_luci_path(str(Luci_path))
    return default_luci_path()
