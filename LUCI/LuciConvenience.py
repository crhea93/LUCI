"""Backward-compatibility shim. Split into ``LUCI.fitting.components`` and ``LUCI.engine.selection``."""

from LUCI.engine.selection import reg_to_mask  # noqa: F401
from LUCI.fitting.components import get_individual_components  # noqa: F401
