"""Backward-compatibility shim. Split into ``LUCI.fitting.components`` and ``LUCI.engine.selection``."""

from luci.engine.selection import reg_to_mask  # noqa: F401
from luci.fitting.components import get_individual_components  # noqa: F401
