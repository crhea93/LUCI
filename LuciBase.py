"""Backward-compatibility shim. The cube class moved to ``LUCI.cube``.

Prefer ``from luci.cube import SitelleCube`` in new code; ``Luci`` remains an
alias so every existing script and notebook keeps working.
"""

from luci.cube import Luci, SitelleCube  # noqa: F401
