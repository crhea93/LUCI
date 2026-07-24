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

from luci.LuciUtility import save_fits


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


def test_duplicate_line_names_get_a_component_suffix(tmp_path):
    """
    B16: a multi-component fit passes the same line name more than once, and each
    component must land in its own file (<line>, <line>_2, ...).

    save_fits *intended* this, but never appended to `lines_fit`, so `.count()`
    was always 0 and the suffix branch was dead code -- every component wrote the
    same filename and silently overwrote the previous one, leaving only the last.
    """
    out = str(tmp_path)
    lines = ["Halpha", "Halpha"]
    a = np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2)
    m = np.ones((2, 2), dtype=np.float32)
    header = fits.Header()
    save_fits(out, "OBJ", lines, a, a, a, a, a, a, a, m, m, m, header, binning=1, fit_function="sincgauss")
    amps = os.path.join(out, "Amplitudes")
    first = os.path.join(amps, "OBJ_1_sincgauss_Halpha_Amplitude.fits")
    second = os.path.join(amps, "OBJ_1_sincgauss_Halpha_2_Amplitude.fits")
    assert os.path.exists(first)
    assert os.path.exists(second)
    # Each file holds its own component, in order -- nothing was overwritten.
    np.testing.assert_array_equal(fits.open(first)[0].data, a[:, :, 0])
    np.testing.assert_array_equal(fits.open(second)[0].data, a[:, :, 1])


def test_three_components_are_numbered_sequentially(tmp_path):
    """Naming must generalise past two components."""
    out = str(tmp_path)
    lines = ["Halpha", "Halpha", "Halpha"]
    a = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    m = np.ones((2, 2), dtype=np.float32)
    save_fits(out, "OBJ", lines, a, a, a, a, a, a, a, m, m, m, fits.Header(), binning=1, fit_function="sincgauss")
    amps = os.path.join(out, "Amplitudes")
    for suffix, channel in (("", 0), ("_2", 1), ("_3", 2)):
        path = os.path.join(amps, f"OBJ_1_sincgauss_Halpha{suffix}_Amplitude.fits")
        assert os.path.exists(path), f"missing component {channel}"
        np.testing.assert_array_equal(fits.open(path)[0].data, a[:, :, channel])


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
