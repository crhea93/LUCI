"""
Tests for co-adding per-line flux maps into a total-complex map.

The reason to add the lines of a blended complex is more signal, so these tests check that the sum
is exact, that the WCS survives (the map has to overlay the per-line maps and the deep image), and
that a pixel where one line's fit failed does not silently inherit the other lines' flux.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from luci.analysis.combine import combination_summary, combine_line_fluxes, line_flux_path


@pytest.fixture
def fluxes(tmp_path):
    """Three line maps on a 6x6 grid: a bright 2x2 source, faint noise elsewhere, one NaN pixel."""
    rng = np.random.default_rng(0)
    shape = (6, 6)
    source = np.zeros(shape, dtype=bool)
    source[2:4, 2:4] = True

    header = fits.Header({"CTYPE1": "RA---TAN", "CRVAL1": 186.483, "CRPIX1": 3.0})
    maps = {}
    for line, strength in [("Halpha", 1.0), ("NII6548", 0.25), ("NII6583", 0.75)]:
        arr = np.where(source, strength * 1e-15, rng.uniform(-1e-18, 1e-18, shape))
        maps[line] = arr.astype(np.float32)
    maps["NII6548"][0, 0] = np.nan  # one line's fit failed here

    (tmp_path / "Fluxes").mkdir()
    for line, arr in maps.items():
        fits.writeto(line_flux_path(str(tmp_path), "obj_1_sincgauss", line), arr, header)
    return str(tmp_path), maps


def test_sum_is_exact(fluxes):
    output_dir, maps = fluxes
    lines = ["Halpha", "NII6548", "NII6583"]
    total, _ = combine_line_fluxes(output_dir, "obj_1_sincgauss", lines)
    expected = sum(maps[line].astype(np.float64) for line in lines)
    finite = np.isfinite(expected)
    assert np.allclose(total[finite], expected[finite], rtol=1e-6)


def test_nan_in_any_line_propagates(fluxes):
    """A pixel missing one line has no meaningful total, so it must not come out as a number."""
    output_dir, _ = fluxes
    total, _ = combine_line_fluxes(output_dir, "obj_1_sincgauss", ["Halpha", "NII6548", "NII6583"])
    assert np.isnan(total[0, 0])
    assert np.isfinite(total[1:, 1:]).all()


def test_written_map_keeps_wcs_and_records_provenance(fluxes):
    output_dir, _ = fluxes
    lines = ["Halpha", "NII6583"]
    total, path = combine_line_fluxes(output_dir, "obj_1_sincgauss", lines)
    with fits.open(path) as hdul:
        header, data = hdul[0].header, hdul[0].data
    assert np.allclose(data, total.astype(np.float32), equal_nan=True)
    assert header["CTYPE1"] == "RA---TAN" and header["CRVAL1"] == pytest.approx(186.483)
    assert header["NCOMB"] == 2
    assert header["COMBLINE"] == "Halpha,NII6583"
    assert [header["CLINE0"], header["CLINE1"]] == lines
    assert "BUNIT" not in header  # unset unless asked for: an uncalibrated cube is in counts


def test_bunit_written_when_requested(fluxes):
    output_dir, _ = fluxes
    _, path = combine_line_fluxes(output_dir, "obj_1_sincgauss", ["Halpha", "NII6583"], bunit="counts")
    assert fits.getheader(path)["BUNIT"] == "counts"


def test_default_out_name_names_its_lines(fluxes):
    output_dir, _ = fluxes
    _, path = combine_line_fluxes(output_dir, "obj_1_sincgauss", ["Halpha", "NII6583"])
    assert path.endswith("obj_1_sincgauss_Halpha+NII6583_Flux.fits")


def test_summary_reports_the_gain_over_the_brightest_line(fluxes):
    """The source is 1 : 0.25 : 0.75 in Halpha : NII6548 : NII6583, so Halpha is brightest and
    the total should carry about twice its flux."""
    output_dir, _ = fluxes
    lines = ["Halpha", "NII6548", "NII6583"]
    total, _ = combine_line_fluxes(output_dir, "obj_1_sincgauss", lines)
    source = np.zeros((6, 6), dtype=bool)
    source[2:4, 2:4] = True
    report = combination_summary(output_dir, "obj_1_sincgauss", lines, total, mask=source)
    assert report["brightest_line"] == "Halpha"
    assert report["gain_vs_brightest"] == pytest.approx(2.0, rel=0.05)
    assert report["n_detected"] == 4
    assert report["n_nan"] == 1
    assert report["n_pixels"] == 36


def test_summary_measures_the_source_not_the_blank_field(fluxes):
    """Co-added noise partially cancels, so measuring over the whole field understates the gain --
    the default top-percent selection has to find the source on its own."""
    output_dir, _ = fluxes
    lines = ["Halpha", "NII6548", "NII6583"]
    total, _ = combine_line_fluxes(output_dir, "obj_1_sincgauss", lines)
    report = combination_summary(output_dir, "obj_1_sincgauss", lines, total, top_percent=12.0)
    # The 2x2 source, give or take a boundary pixel -- not the 32 blank ones
    assert 4 <= report["n_detected"] <= 6
    assert report["brightest_line"] == "Halpha"
    assert report["gain_vs_brightest"] == pytest.approx(2.0, rel=0.05)


def test_brightest_line_is_not_decided_by_a_pedestal(fluxes):
    """A line on a high positive pedestal with no real emission must not be named the brightest:
    it beats a genuine source on both the median and the summed flux of the full map."""
    output_dir, maps = fluxes
    pedestal = np.full((6, 6), 5e-16, dtype=np.float32)  # high floor everywhere, no source
    fits.writeto(line_flux_path(output_dir, "obj_1_sincgauss", "SII6716"), pedestal)
    lines = ["Halpha", "SII6716"]
    total, _ = combine_line_fluxes(output_dir, "obj_1_sincgauss", lines)
    source = np.zeros((6, 6), dtype=bool)
    source[2:4, 2:4] = True
    report = combination_summary(output_dir, "obj_1_sincgauss", lines, total, mask=source)
    # The pedestal would win on either naive full-field statistic
    assert np.nanmedian(pedestal) > np.nanmedian(maps["Halpha"])
    assert pedestal.sum() > maps["Halpha"][np.isfinite(maps["Halpha"])].sum()
    assert report["brightest_line"] == "Halpha"


def test_summary_rejects_a_mask_that_selects_nothing(fluxes):
    output_dir, _ = fluxes
    lines = ["Halpha", "NII6583"]
    total, _ = combine_line_fluxes(output_dir, "obj_1_sincgauss", lines)
    with pytest.raises(ValueError, match="selects no finite pixel"):
        combination_summary(output_dir, "obj_1_sincgauss", lines, total,
                            mask=np.zeros_like(total, dtype=bool))


def test_mismatched_grids_raise_rather_than_broadcast(fluxes):
    """Maps from runs with different binning must not be silently broadcast together."""
    output_dir, _ = fluxes
    fits.writeto(line_flux_path(output_dir, "obj_1_sincgauss", "SII6716"), np.ones((3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="different grids"):
        combine_line_fluxes(output_dir, "obj_1_sincgauss", ["Halpha", "SII6716"])


def test_missing_line_map_raises(fluxes):
    output_dir, _ = fluxes
    with pytest.raises(FileNotFoundError, match="OIII5007"):
        combine_line_fluxes(output_dir, "obj_1_sincgauss", ["Halpha", "OIII5007"])


def test_single_line_is_not_a_combination(fluxes):
    output_dir, _ = fluxes
    with pytest.raises(ValueError, match="at least two lines"):
        combine_line_fluxes(output_dir, "obj_1_sincgauss", ["Halpha"])
