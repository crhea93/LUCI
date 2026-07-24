"""
Tests for the *new* SITELLE HDF5 layout (header as root attributes, data in a
single dataset), as opposed to the legacy quadrant format the rest of the suite
uses.

This path had no coverage at all, which is how B15 survived: ``update_header``
dispatched on ``header_type is np.str``, and ``np.str`` was removed in numpy
1.24.  The resulting ``AttributeError`` was swallowed by a bare ``except``, so
every string and boolean keyword silently fell through to the fallback,
``clean_hdr_dict`` stayed empty, and the cube came back with a WCS that had no
axes -- on modern numpy, new-format cubes were simply broken.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.wcs import WCS
from fixtures.make_cube import write_new_format_cube


@pytest.fixture(scope="module")
def new_format_cube(tmp_path_factory, luci_path):
    from LuciBase import Luci

    base = tmp_path_factory.mktemp("new_format")
    truth = write_new_format_cube(str(base / "SN3_new"), filter_name="SN3")
    cube = Luci(
        luci_path,
        truth["path"][: -len(".hdf5")],
        str(base),
        "NEWFMT",
        0.0,
        5000,
        ML_bool=False,
    )
    return cube, truth


def test_new_format_cube_loads_with_the_right_shape(new_format_cube):
    cube, truth = new_format_cube
    assert cube.cube_final.shape == (truth["dimx"], truth["dimy"], truth["dimz"])


def test_string_keywords_survive_type_dispatch(new_format_cube):
    """
    B15: string keywords used to be routed by `header_type is np.str`, which
    raises AttributeError on numpy >= 1.24 and was swallowed silently.
    """
    cube, _ = new_format_cube
    assert cube.hdr_dict["FILTER"] == "SN3"
    assert isinstance(cube.hdr_dict["FILTER"], str)
    assert cube.hdr_dict["CTYPE1"] == "RA---TAN"


def test_numeric_and_boolean_keywords_keep_their_types(new_format_cube):
    cube, truth = new_format_cube
    assert cube.hdr_dict["STEPNB"] == truth["step_nb"]
    assert isinstance(cube.hdr_dict["STEPNB"], int)
    assert isinstance(cube.hdr_dict["CRVAL3"], float)
    # bool must not be captured by the integer branch (bool subclasses int).
    assert cube.hdr_dict["APODIZE"] is False
    assert isinstance(cube.hdr_dict["APODIZE"], bool)


def test_new_format_cube_yields_a_usable_two_axis_wcs(new_format_cube):
    """
    The payoff: with the type dispatch broken, clean_hdr_dict stayed empty and
    WCS(...) produced an axis-less header, so every downstream cutout and saved
    map lost its astrometry.
    """
    cube, _ = new_format_cube
    wcs = WCS(cube.header, naxis=2)
    assert wcs.naxis == 2
    assert wcs.wcs.ctype[0].strip() == "RA---TAN"
    sky = wcs.pixel_to_world(5, 5)
    assert np.isfinite(sky.ra.deg) and np.isfinite(sky.dec.deg)


def test_spectral_axis_is_built_for_the_new_format(new_format_cube):
    cube, truth = new_format_cube
    assert len(cube.spectrum_axis) == truth["step_nb"]
    assert np.all(np.diff(cube.spectrum_axis) > 0)
    assert cube.spectrum_axis[0] == pytest.approx(truth["axis_min"], rel=1e-6)
