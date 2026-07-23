"""
Output-contract tests for save_fits.

These pin the *filesystem layout and headers* that a fit produces: which
directories are created, how each map is named, and which WCS keywords land in
the output.  Downstream analysis scripts glob for these exact paths, so the
refactor must not rename or relocate anything without this suite noticing.

Kept separate from the numeric golden tests because this is cheap: it calls
save_fits directly with tiny hand-built arrays and never runs the optimiser.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from astropy.io import fits

from LUCI.LuciUtility import save_fits


@pytest.fixture
def written_maps(tmp_path):
    """Call save_fits once with a two-line fit over a 3x4 field."""
    out = str(tmp_path)
    lines = ["Halpha", "NII6583"]
    nx, ny, nl = 3, 4, len(lines)

    def cube3d(scale):
        return np.arange(nx * ny * nl, dtype=np.float32).reshape(nx, ny, nl) + scale

    def map2d(scale):
        return np.arange(nx * ny, dtype=np.float32).reshape(nx, ny) + scale

    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 24.0
    header["CRVAL2"] = 15.0

    save_fits(
        out,
        "OBJ",
        lines,
        cube3d(1),
        cube3d(2),
        cube3d(3),
        cube3d(4),
        cube3d(5),
        cube3d(6),
        cube3d(7),
        map2d(8),
        map2d(9),
        map2d(10),
        header,
        binning=1,
        fit_function="sincgauss",
    )
    return out, lines


def test_creates_the_four_map_subdirectories(written_maps):
    out, _ = written_maps
    for sub in ("Amplitudes", "Fluxes", "Velocity", "Broadening"):
        assert os.path.isdir(os.path.join(out, sub)), f"missing {sub}/"


def test_writes_the_expected_per_line_files(written_maps):
    out, lines = written_maps
    stem = "OBJ_1_sincgauss"  # object _ binning _ fit_function
    for line in lines:
        expected = [
            f"Amplitudes/{stem}_{line}_Amplitude.fits",
            f"Fluxes/{stem}_{line}_Flux.fits",
            f"Fluxes/{stem}_{line}_Flux_err.fits",
            f"Velocity/{stem}_{line}_velocity.fits",
            f"Velocity/{stem}_{line}_velocity_err.fits",
            f"Broadening/{stem}_{line}_broadening.fits",
            f"Broadening/{stem}_{line}_broadening_err.fits",
        ]
        for rel in expected:
            assert os.path.exists(os.path.join(out, rel)), f"missing {rel}"


def test_writes_the_scalar_maps(written_maps):
    out, _ = written_maps
    stem = "OBJ_1_sincgauss"
    for rel in (f"{stem}_Chi2.fits", f"{stem}_continuum.fits", f"{stem}_continuum_error.fits"):
        assert os.path.exists(os.path.join(out, rel)), f"missing {rel}"


def test_output_filename_encodes_binning_and_fit_function(tmp_path):
    """The stem is object[_suffix]_binning[_fitfunction]; pin that grammar."""
    out = str(tmp_path)
    lines = ["Halpha"]
    a = np.ones((2, 2, 1), dtype=np.float32)
    m = np.ones((2, 2), dtype=np.float32)
    header = fits.Header()
    save_fits(
        out, "OBJ", lines, a, a, a, a, a, a, a, m, m, m, header, binning=3, suffix="_wvt", fit_function="gaussian"
    )
    assert os.path.exists(os.path.join(out, "OBJ_wvt_3_gaussian_Chi2.fits"))
    assert os.path.exists(os.path.join(out, "Amplitudes", "OBJ_wvt_3_gaussian_Halpha_Amplitude.fits"))


def test_duplicate_line_names_currently_overwrite_each_other(tmp_path):
    """
    Pins CURRENT (broken) behaviour -- see B16 in REFACTOR_BUGS.md.

    A two-component fit passes the same line name twice.  save_fits *intends* to
    disambiguate the second component as <line>_2:

        lines_fit = []
        for ct, line_ in enumerate(lines):
            if lines_fit.count(line_) >= 1:
                line_ += '_' + str(lines_fit.count(line_) + 1)
            fits.writeto(... line_ ...)

    but `lines_fit` is never appended to, so `.count()` is always 0 and the
    suffix branch is dead code.  Both components write the same filename and the
    second silently overwrites the first.  Only one amplitude map survives, and
    it holds channel 1, not channel 0.
    """
    out = str(tmp_path)
    lines = ["Halpha", "Halpha"]
    a = np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2)
    m = np.ones((2, 2), dtype=np.float32)
    header = fits.Header()
    save_fits(out, "OBJ", lines, a, a, a, a, a, a, a, m, m, m, header, binning=1, fit_function="sincgauss")
    amps = os.path.join(out, "Amplitudes")
    survivor = os.path.join(amps, "OBJ_1_sincgauss_Halpha_Amplitude.fits")
    assert os.path.exists(survivor)
    # The _2 file that the code intends to write never appears.
    assert not os.path.exists(os.path.join(amps, "OBJ_1_sincgauss_Halpha_2_Amplitude.fits"))
    # The surviving map is the *second* component (channel 1), not the first.
    np.testing.assert_array_equal(fits.open(survivor)[0].data, a[:, :, 1])


@pytest.mark.xfail(strict=True, reason="save_fits never appends to lines_fit (B16); fix during Phase 1/5")
def test_duplicate_line_names_should_get_a_component_suffix(tmp_path):
    """Intended contract: the second component lands in a <line>_2 file."""
    out = str(tmp_path)
    lines = ["Halpha", "Halpha"]
    a = np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2)
    m = np.ones((2, 2), dtype=np.float32)
    header = fits.Header()
    save_fits(out, "OBJ", lines, a, a, a, a, a, a, a, m, m, m, header, binning=1, fit_function="sincgauss")
    amps = os.path.join(out, "Amplitudes")
    assert os.path.exists(os.path.join(amps, "OBJ_1_sincgauss_Halpha_Amplitude.fits"))
    assert os.path.exists(os.path.join(amps, "OBJ_1_sincgauss_Halpha_2_Amplitude.fits"))


def test_map_contents_and_wcs_survive_the_write(written_maps):
    """A saved map must read back with its data intact and its WCS keywords present."""
    out, lines = written_maps
    path = os.path.join(out, "Velocity", f"OBJ_1_sincgauss_{lines[0]}_velocity.fits")
    hdu = fits.open(path)[0]
    assert hdu.data.shape == (3, 4)
    assert hdu.header["CTYPE1"] == "RA---TAN"
    assert hdu.header["CRVAL1"] == pytest.approx(24.0)


def test_line_channel_ordering_is_preserved(written_maps):
    """
    Map for line i must come from channel i of the 3-D input.

    The amplitude cube was built as arange + 1, so channel 0 holds the even
    ravel positions and channel 1 the odd ones; a transposed or swapped write
    would scramble which line each map belongs to.
    """
    out, lines = written_maps
    a0 = fits.open(os.path.join(out, "Amplitudes", f"OBJ_1_sincgauss_{lines[0]}_Amplitude.fits"))[0].data
    a1 = fits.open(os.path.join(out, "Amplitudes", f"OBJ_1_sincgauss_{lines[1]}_Amplitude.fits"))[0].data
    source = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2) + 1
    np.testing.assert_array_equal(a0, source[:, :, 0])
    np.testing.assert_array_equal(a1, source[:, :, 1])
