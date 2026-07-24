"""
Tests for the counts -> erg/cm2/s/A conversion.

The numbers here come from a real ORB level-3 cube (M86, SN4): flambda ~7.359e-17 per channel,
step_nb 469, exposure_time 38 s, giving 4.13e-21 per count. Getting the convention wrong is not
a subtle error -- ORB's own `to_fits` omits the /dimz/exposure_time and lands 17822x high -- so
these pin the exact factor rather than just "some scaling happened".
"""

from __future__ import annotations

import numpy as np
import pytest

from luci.instrument.flux import flux_calibration_vector, is_flux_calibrated

N_CHANNELS = 469
FLAMBDA = 7.359e-17
EXPOSURE_TIME = 38.0


def level3_header(**overrides):
    """Header of an uncalibrated ORB level-3 cube."""
    hdr = {
        "flux_calibration": False,
        "flambda": np.full(N_CHANNELS, FLAMBDA),
        "step_nb": N_CHANNELS,
        "STEPNB": N_CHANNELS,
        "exposure_time": EXPOSURE_TIME,
        "FILTER": "SN4",
    }
    hdr.update(overrides)
    return hdr


def test_level3_cube_is_reported_uncalibrated():
    assert is_flux_calibrated(level3_header()) is False


def test_calibrated_cube_is_left_alone():
    assert is_flux_calibrated(level3_header(flux_calibration=True)) is True
    assert flux_calibration_vector(level3_header(flux_calibration=True), N_CHANNELS) is None


def test_old_dr1_header_is_recognised_via_bunit():
    """DR1-style cubes have no flux_calibration keyword; they say BUNIT = 'FLUX'."""
    hdr = {"BUNIT": "FLUX", "flambda": FLAMBDA}
    assert is_flux_calibrated(hdr) is True
    assert flux_calibration_vector(hdr, N_CHANNELS) is None


def test_header_without_flambda_is_assumed_calibrated():
    """Every cube LUCI handled before this existed had no flambda -- keep them unchanged."""
    assert is_flux_calibrated({"FILTER": "SN3"}) is True


def test_factor_divides_by_total_integration_time():
    """The ORB get_level convention: flambda / dimz / exposure_time, NOT flambda alone."""
    vector = flux_calibration_vector(level3_header(), N_CHANNELS)
    expected = FLAMBDA / N_CHANNELS / EXPOSURE_TIME
    assert vector.shape == (N_CHANNELS,)
    assert vector == pytest.approx(expected)
    # Guard against silently reverting to ORB's to_fits convention. atol=0 matters: these
    # numbers are ~1e-21, so the default atol=1e-8 would call any two of them "close".
    assert not np.isclose(vector[0], FLAMBDA, rtol=1e-6, atol=0.0)
    assert FLAMBDA / vector[0] == pytest.approx(N_CHANNELS * EXPOSURE_TIME)


def test_scalar_flambda_is_broadcast():
    vector = flux_calibration_vector(level3_header(flambda=FLAMBDA), N_CHANNELS)
    assert vector.shape == (N_CHANNELS,)
    assert vector == pytest.approx(FLAMBDA / N_CHANNELS / EXPOSURE_TIME)


def test_per_channel_flambda_is_preserved():
    """flambda varies across the band; the calibration must stay per channel."""
    flambda = np.linspace(6.9e-17, 7.9e-17, N_CHANNELS)
    vector = flux_calibration_vector(level3_header(flambda=flambda), N_CHANNELS)
    assert vector == pytest.approx(flambda / N_CHANNELS / EXPOSURE_TIME)


def test_mismatched_flambda_length_is_refused():
    """Better to leave the data in counts than to scale it by the wrong vector."""
    hdr = level3_header(flambda=np.full(100, FLAMBDA))
    assert flux_calibration_vector(hdr, N_CHANNELS) is None


@pytest.mark.parametrize("missing", ["step_nb", "exposure_time"])
def test_missing_integration_time_is_refused(missing):
    hdr = level3_header()
    hdr.pop(missing)
    if missing == "step_nb":
        hdr.pop("STEPNB")
    assert flux_calibration_vector(hdr, N_CHANNELS) is None


def test_uncalibrated_without_flambda_is_refused():
    hdr = level3_header()
    hdr.pop("flambda")
    assert flux_calibration_vector(hdr, N_CHANNELS) is None
