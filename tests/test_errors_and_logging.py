"""
Tests for the two Phase 6 behaviour changes: raising instead of exiting, and
logging instead of printing.

Library code used to respond to bad input by printing a message and calling
``exit()``/``quit()``.  In a script that kills the interpreter with no
traceback; in a notebook it kills the kernel.  Worse, several sites printed the
message and then *carried on* with a variable left unbound, turning a clear
input error into a confusing ``NameError`` further down.

Every case below is one of those sites.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from luci.fitting.parameters import calculate_flux_err
from luci.instrument.filters import UnsupportedFilterError


class TestRaisesInsteadOfExiting:
    def test_unknown_fit_model_raises(self):
        """
        The old branch evaluated the message as a bare expression -- it was never
        even printed -- and then called exit(). Silent interpreter death.
        """
        fit_sol = np.array([2.0, 15200.0, 1.5, 0.0])
        unc = np.array([0.1, 0.01, 0.05, 0.0])
        with pytest.raises(ValueError, match="Unknown fit model"):
            calculate_flux_err(0, fit_sol, unc, "not_a_model", 2.5)

    def test_known_fit_models_still_work(self):
        """Guard against the raise being reachable for valid input."""
        fit_sol = np.array([2.0, 15200.0, 1.5, 0.0])
        unc = np.array([0.1, 0.01, 0.05, 0.0])
        for model in ("gaussian", "sinc", "sincgauss"):
            assert calculate_flux_err(0, fit_sol, unc, model, 2.5) >= 0.0

    def test_unknown_background_algorithm_raises(self):
        from luci.background.detection import find_background_pixels

        rng = np.random.default_rng(0)
        deep = rng.normal(100.0, 5.0, size=(120, 120))
        with pytest.raises(ValueError, match="Unknown bkg_algo"):
            find_background_pixels(deep, bkg_algo="nonsense", plot_mask=False)

    def test_unsupported_filter_in_simulation_raises(self):
        """Also guards B25: luci.simulation must import without the optional ppxf."""
        from luci.simulation import Spectrum

        with pytest.raises(UnsupportedFilterError):
            Spectrum(["Halpha"], "sincgauss", [1.0], 100.0, 30.0, "NOT_A_FILTER", 5000, 100)

    def test_unrecognised_region_file_raises(self):
        from luci.engine.selection import resolve_mask

        with pytest.raises(ValueError, match="Unrecognised region file"):
            resolve_mask("mask.txt", None, (8, 8))


class TestWvtReassignTerminates:
    """
    B24: ``reassign_pixels`` looped until the pixel was assigned, and swallowed
    every exception with ``pass``.  With no candidate bins, ``min([])`` raised on
    each pass, so the function span forever instead of failing.
    """

    def test_no_successful_bins_returns_instead_of_hanging(self):
        from luci.analysis.wvt import reassign_pixels

        class FakePixel:
            def __init__(self):
                self.pix_x, self.pix_y = 0, 0
                self.assigned_to_bin = False

            def clear_bin(self):
                self.assigned_to_bin = False

        class FakeBin:
            pixels = [FakePixel()]

        # Before the fix this call never returned.
        assert reassign_pixels(FakeBin(), [], []) is None


class TestLogging:
    def test_messages_go_through_the_luci_logger(self, caplog):
        from luci.log import get_logger

        with caplog.at_level(logging.INFO, logger="luci"):
            get_logger("luci.test").info("hello")
        assert "hello" in caplog.text

    def test_child_loggers_are_governed_by_the_luci_parent(self):
        from luci.log import get_logger

        child = get_logger("luci.analysis.wvt")
        assert child.name.startswith("luci.")
        assert logging.getLogger("luci") in _ancestors(child)

    def test_get_logger_prefixes_bare_names(self):
        from luci.log import get_logger

        assert get_logger("cube").name == "luci.cube"
        assert get_logger("luci.cube").name == "luci.cube"

    def test_silence_suppresses_output(self):
        from luci.log import get_logger, silence

        previous = logging.getLogger("luci").level
        try:
            silence()
            # Every child inherits the parent's level, so nothing below CRITICAL
            # is emitted anywhere in the package.
            assert not get_logger("luci.quiet").isEnabledFor(logging.INFO)
            assert not get_logger("luci.analysis.wvt").isEnabledFor(logging.WARNING)
        finally:
            logging.getLogger("luci").setLevel(previous)
        assert logging.getLogger("luci").level == previous

    def test_autoconfigure_defers_to_an_application_that_configured_logging(self):
        """
        A library must not hijack logging an application already set up.  Under
        pytest the root logger has handlers, so LUCI attaches none of its own --
        which is why the `luci` logger sits at NOTSET here rather than INFO.
        """
        from luci.log import get_logger

        get_logger("luci.probe")
        assert logging.getLogger().handlers, "precondition: pytest configures root"
        assert not logging.getLogger("luci").handlers

    def test_package_emits_nothing_on_stdout(self, capsys):
        """The whole point: progress messages are no longer unfilterable prints."""
        from luci.log import get_logger

        get_logger("luci.stdout_check").info("routed through logging")
        assert capsys.readouterr().out == ""


def _ancestors(logger):
    out = []
    node = logger.parent
    while node is not None:
        out.append(node)
        node = node.parent
    return out
