"""
Logging for LUCI.

The package used to talk to the user with bare ``print()``, which cannot be
silenced, redirected, or filtered by severity -- a problem when fitting a cube
emits a line per pixel into a notebook.

Libraries are normally expected to attach only a ``NullHandler`` and let the
application configure output.  LUCI is used interactively far more often than it
is embedded, so a silent-by-default logger would just look like the progress
messages had been deleted.  The compromise: attach a handler on first use, but
only if nothing else has configured logging.  Anyone who calls
``logging.basicConfig()`` (or ``configure()`` below) keeps full control.
"""

from __future__ import annotations

import logging

ROOT = "luci"
_configured = False


def configure(level: int | str = logging.INFO, fmt: str = "%(levelname)s %(name)s: %(message)s") -> logging.Logger:
    """Attach a stream handler to the ``luci`` logger. Idempotent."""
    global _configured
    logger = logging.getLogger(ROOT)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    _configured = True
    return logger


def _autoconfigure() -> None:
    global _configured
    if _configured:
        return
    _configured = True
    if logging.getLogger(ROOT).handlers or logging.getLogger().handlers:
        return  # the application is driving; don't fight it
    configure()


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Return the logger for a LUCI module.

    Pass ``__name__``; the ``luci.`` prefix is kept so every logger sits under
    one parent and ``logging.getLogger("luci").setLevel(...)`` governs them all.
    """
    _autoconfigure()
    if not name or name == ROOT:
        return logging.getLogger(ROOT)
    if not name.startswith(ROOT + "."):
        name = f"{ROOT}.{name}"
    return logging.getLogger(name)


def silence() -> None:
    """Suppress all LUCI output below CRITICAL."""
    get_logger().setLevel(logging.CRITICAL)


class LUCILog:
    """Deprecated: the old logger object. Use :func:`get_logger` instead."""

    def __init__(self):
        self.logger = get_logger()

    def warn(self, warning_message):
        self.logger.warning(warning_message)

    def info(self, info_message):
        self.logger.info(info_message)
