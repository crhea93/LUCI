"""
Tests for the SITELLE filter registry.

These pin the exact wavelength windows transcribed from the original five
if/elif chains, so the registry is provably a behaviour-preserving replacement
for the SN filters (which the golden baselines also cover) and a *deliberate*
fix for the C-filter bugs (which no golden covers).
"""

from __future__ import annotations

import pytest

from luci.instrument.filters import (
    FILTERS,
    SUPPORTED_FILTERS,
    UnsupportedFilterError,
    get_filter,
)

# The numbers below are transcribed straight from the pre-refactor code so the
# test is an independent record, not a mirror of the registry it checks.
SN_FIT = {"SN3": (14750, 15400), "SN2": (19500, 20750), "SN1": (26000, 28000), "SN4": (15040, 15330)}
SN_NOISE = {"SN3": (15600, 15800), "SN2": (18600, 19000), "SN1": (26000, 26200), "SN4": (14600, 14900)}
SN_REF = {"SN3": (14700, 15600), "SN2": (19000, 21000), "SN1": (25500, 27500), "SN4": (15000, 15350)}


@pytest.mark.parametrize("name", ["SN1", "SN2", "SN3", "SN4"])
def test_sn_fit_windows_are_byte_identical_to_the_original(name):
    """The golden baselines depend on these being exactly unchanged."""
    assert get_filter(name).fit_bounds() == SN_FIT[name]


@pytest.mark.parametrize("name", ["SN1", "SN2", "SN3", "SN4"])
def test_sn_noise_windows_are_byte_identical_to_the_original(name):
    assert get_filter(name).noise_bounds() == SN_NOISE[name]


@pytest.mark.parametrize("name", ["SN1", "SN2", "SN3", "SN4"])
def test_sn_reference_windows_are_byte_identical_to_the_original(name):
    assert get_filter(name).reference_bounds() == SN_REF[name]


def test_sn_windows_ignore_redshift_and_lines():
    """SN filters are unconditional -- context must not perturb them."""
    spec = get_filter("SN3")
    assert spec.fit_bounds(lines=["Halpha", "OII3726"], redshift_corr=1.5) == (14750, 15400)
    assert spec.noise_bounds(lines=["Halpha"], redshift_corr=2.0) == (15600, 15800)


# --------------------------------------------------------------------------
# The B4 fix and the C-filter conditionals
# --------------------------------------------------------------------------


def test_c3_normal_noise_upper_bound_is_now_defined():
    """
    B4: the original normal-C3 branch of calculate_noise misspelt bound_upper
    as `buond_upper`, so the upper bound was never set and leaked in from a
    module global.  The registry gives it the intended value.
    """
    lo, hi = get_filter("C3").noise_bounds(lines=["Halpha"])
    assert (lo, hi) == (20000, 20250)
    assert hi > lo  # the whole point: a real, ordered window


def test_c3_switches_to_sn1_like_windows_when_fitting_oii():
    """OII3726 present -> object near z~0.465 -> C3 behaves like SN1."""
    spec = get_filter("C3")
    assert spec.fit_bounds(lines=["OII3726"]) == (26000, 29000)
    assert spec.noise_bounds(lines=["OII3726"]) == (26000, 26200)
    # ... but the default (no OII3726) stays in the C3 band.
    assert spec.fit_bounds(lines=["Halpha"]) == (18100, 19500)


def test_c4_noise_is_unconditional_now():
    """
    The original C4 noise branch was gated on 'Halpha' in lines; without it the
    code fell through to an error and stale globals.  The registry always
    returns a window.
    """
    spec = get_filter("C4")
    assert spec.noise_bounds(lines=["SII6716"]) == pytest.approx((11800, 12150))


@pytest.mark.parametrize(
    "name,corr,expected",
    [
        ("C4", 1.25, (12150 * 1.25, 12550 * 1.25)),
        ("C2", 1.25, (15990 * 1.25, 17880 * 1.25)),
        ("C1", 1.25, (20408 * 1.25, 25974 * 1.25)),
    ],
)
def test_c_filter_fit_windows_scale_with_redshift(name, corr, expected):
    assert get_filter(name).fit_bounds(redshift_corr=corr) == pytest.approx(expected)


def test_c_filter_windows_are_unscaled_at_zero_redshift():
    """redshift_corr == 1 (obj_redshift == 0) leaves the base numbers intact."""
    assert get_filter("C2").fit_bounds(redshift_corr=1.0) == (15990, 17880)


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------


def test_unknown_filter_raises_instead_of_quitting():
    """
    Library code must never call quit()/exit().  The old reference-spectrum
    reader did exactly that; the registry raises a catchable error instead.
    """
    with pytest.raises(UnsupportedFilterError) as exc:
        get_filter("SN9")
    assert "SN9" in str(exc.value)
    assert exc.value.filter_name == "SN9"


def test_registry_covers_exactly_the_supported_filters():
    assert set(FILTERS) == set(SUPPORTED_FILTERS)


def test_filterspec_is_immutable():
    """Frozen dataclass: no caller can mutate a shared filter definition."""
    spec = get_filter("SN3")
    with pytest.raises(Exception):
        spec.name = "hacked"  # type: ignore[misc]
