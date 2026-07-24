"""
Tests for the fit-quality mask.

The point of this filter is an empty field: most of a SITELLE frame is blank sky, every pixel of
it gets a fit anyway, and those fits scatter across the whole allowed parameter range. So the
tests are built around "a bright source in a sea of noise" and check that the noise goes and the
source stays.
"""

from __future__ import annotations

import numpy as np
import pytest

from luci.analysis.quality import (
    apply_quality_mask,
    broadening_bound_kms,
    fit_quality_mask,
    fit_quality_report,
)


@pytest.fixture
def field():
    """A 10x10 field: a 3x3 real source, the rest noise-like junk."""
    rng = np.random.default_rng(0)
    shape = (10, 10)
    source = np.zeros(shape, dtype=bool)
    source[4:7, 4:7] = True

    flux = np.where(source, 1e-15, rng.uniform(-1e-18, 1e-18, shape))
    flux_err = np.where(source, 5e-17, 8e-18)
    snr = np.where(source, 25.0, rng.uniform(0.5, 2.5, shape))
    velocity = np.where(source, -250.0, rng.uniform(-500, 500, shape))
    broadening = np.where(source, 40.0, 196.6)  # junk fits rail at the sigma bound
    chi2 = np.where(source, 0.4, rng.uniform(2, 20, shape))
    return dict(flux=flux, flux_err=flux_err, snr=snr, velocity=velocity, broadening=broadening, chi2=chi2), source


def test_snr_cut_keeps_the_source_and_drops_the_field(field):
    maps, source = field
    mask = fit_quality_mask(flux=maps["flux"], snr=maps["snr"], snr_min=5.0)
    assert np.array_equal(mask, source)


def test_negative_and_nonfinite_fluxes_are_rejected():
    flux = np.array([[1e-16, -1e-16], [np.nan, np.inf]])
    mask = fit_quality_mask(flux=flux, snr=np.full((2, 2), 50.0))
    assert mask.tolist() == [[True, False], [False, False]]


def test_positive_flux_check_can_be_switched_off():
    flux = np.array([[1e-16, -1e-16]])
    mask = fit_quality_mask(flux=flux, snr=np.full((1, 2), 50.0), require_positive_flux=False)
    assert mask.tolist() == [[True, True]]


def test_relative_flux_error_cut(field):
    maps, source = field
    # Source: 5e-17/1e-15 = 0.05. Junk: 8e-18 over ~1e-18 -> well above 1.
    mask = fit_quality_mask(flux=maps["flux"], flux_err=maps["flux_err"], snr=None, max_flux_err_ratio=0.5)
    assert np.array_equal(mask, source)


def test_velocity_range_catches_what_snr_misses():
    """A noise fit can be high-S/N by accident but land at an absurd velocity."""
    flux = np.full((1, 3), 1e-16)
    snr = np.full((1, 3), 50.0)
    velocity = np.array([[-250.0, 4000.0, -3000.0]])
    mask = fit_quality_mask(flux=flux, snr=snr, velocity=velocity, velocity_range=(-800, 800))
    assert mask.tolist() == [[True, False, False]]


def test_broadening_bound_is_the_sigma_constraint_in_kms():
    """10 cm-1 at the SN4 Halpha position is ~197 km/s -- the value every railed fit reports."""
    assert broadening_bound_kms(15253) == pytest.approx(196.5, abs=0.5)
    assert broadening_bound_kms(15000) == pytest.approx(199.9, abs=0.5)


def test_broadening_cut_removes_fits_pinned_to_the_bound(field):
    maps, source = field
    limit = broadening_bound_kms(15253) - 5  # a little below the bound
    mask = fit_quality_mask(flux=maps["flux"], broadening=maps["broadening"], snr=None, broadening_max=limit)
    assert np.array_equal(mask, source)


def test_chi2_cut(field):
    maps, source = field
    mask = fit_quality_mask(flux=maps["flux"], chi2=maps["chi2"], snr=None, chi2_max=1.0)
    assert np.array_equal(mask, source)


def test_cuts_combine_conjunctively(field):
    maps, source = field
    mask = fit_quality_mask(snr_min=5.0, velocity_range=(-800, 800), chi2_max=1.0, **maps)
    assert np.array_equal(mask, source)


def test_report_attributes_rejections_to_each_cut(field):
    maps, _ = field
    mask, report = fit_quality_report(log=False, snr_min=5.0, chi2_max=1.0, **maps)
    assert report["combined"] == pytest.approx(0.91)  # 9 of 100 pixels survive
    assert report["snr"] == pytest.approx(0.91)
    assert report["chi2"] == pytest.approx(0.91)
    assert np.array_equal(mask, ~np.isnan(np.where(mask, 1.0, np.nan)))


def test_apply_quality_mask_blanks_rejected_pixels(field):
    maps, source = field
    mask = fit_quality_mask(flux=maps["flux"], snr=maps["snr"], snr_min=5.0)
    out = apply_quality_mask({"velocity": maps["velocity"]}, mask)
    assert np.all(np.isnan(out["velocity"][~source]))
    assert np.allclose(out["velocity"][source], -250.0)


def test_needs_at_least_one_map():
    with pytest.raises(ValueError, match="at least one map"):
        fit_quality_mask()
