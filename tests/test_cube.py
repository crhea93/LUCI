"""
Tests for cube reading, geometry, deep images, binning and spectrum extraction.

Rewritten from the original version, which wrapped its setup in a class named
``Test`` with an ``__init__`` (pytest refuses to collect those), pointed at a
cube on one developer's laptop, and rebuilt the whole cube for every test.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from astropy.io import fits

# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def test_cube_has_expected_shape(sn3_cube_noml, sn3_truth):
    assert sn3_cube_noml.cube_final.shape == (
        sn3_truth["dimx"],
        sn3_truth["dimy"],
        sn3_truth["dimz"],
    )


def test_header_values_survive_the_hdf5_round_trip(sn3_cube_noml, sn3_truth):
    hdr = sn3_cube_noml.hdr_dict
    assert hdr["FILTER"] == sn3_truth["filter"]
    assert hdr["STEPNB"] == sn3_truth["step_nb"]
    assert hdr["ZPDINDEX"] == sn3_truth["zpd_index"]
    assert hdr["STEP"] == pytest.approx(sn3_truth["step"], rel=1e-9)
    assert hdr["CRVAL3"] == pytest.approx(sn3_truth["axis_min"], rel=1e-9)


def test_quadrants_are_reassembled_without_gaps(sn3_cube_noml):
    """
    Every spaxel must carry real flux.

    read_in_cube writes each quadrant into a slice of the output array; an
    off-by-one in the quadrant bounds leaves a stripe of zeros that is easy to
    miss visually. 1e-22 is the sentinel read_in_cube substitutes for values it
    considers unphysical.
    """
    cube = sn3_cube_noml.cube_final
    per_pixel_max = cube.max(axis=2)
    assert np.all(per_pixel_max > 1e-20), "some spaxels are empty or were clamped"


def test_flux_values_are_not_clamped_by_the_sanity_filter(sn3_cube_noml):
    """read_in_cube replaces anything >1e-9 or <-1e-16 with 1e-22."""
    assert not np.any(sn3_cube_noml.cube_final == 1e-22)


def test_spectrum_axis_is_monotonic_and_spans_the_requested_range(sn3_cube_noml, sn3_truth):
    axis = sn3_cube_noml.spectrum_axis
    assert len(axis) == sn3_truth["step_nb"]
    assert np.all(np.diff(axis) > 0)
    assert axis[0] == pytest.approx(sn3_truth["axis_min"], rel=1e-6)


def test_redshift_shifts_the_spectrum_axis(luci_factory, sn3_truth):
    rest = luci_factory(sn3_truth, ML_bool=False)
    shifted = luci_factory(sn3_truth, ML_bool=False, redshift=0.01)
    ratio = shifted.spectrum_axis / rest.spectrum_axis
    np.testing.assert_allclose(ratio, 1.01, rtol=1e-5)
    # The unshifted axis must be left alone -- the transmission filter is
    # interpolated onto it.
    np.testing.assert_allclose(shifted.spectrum_axis_unshifted, rest.spectrum_axis_unshifted, rtol=1e-9)


def test_interferometer_angles_match_the_calibration_map(sn3_cube_noml):
    """theta = arccos(calib_ref / calib_map), so it must be real and modest."""
    theta = sn3_cube_noml.interferometer_theta
    assert theta.shape == sn3_cube_noml.cube_final.shape[:2]
    assert np.all(np.isfinite(theta))
    assert theta.min() >= 0.0
    assert theta.max() < 20.0


def test_reference_spectrum_matches_the_ml_predictor_input_length(sn3_cube_noml):
    """
    The R5000 SN3 predictor takes a 460-channel input.  wavenumbers_syn is what
    spectra get interpolated onto before inference, so a mismatch here means
    every ML prior is garbage.
    """
    assert sn3_cube_noml.wavenumbers_syn.shape == (460,)


def test_transmission_is_interpolated_onto_the_unshifted_axis(sn3_cube_noml):
    trans = sn3_cube_noml.transmission_interpolated
    assert trans.shape == sn3_cube_noml.spectrum_axis_unshifted.shape
    assert np.all(np.isfinite(trans))


# --------------------------------------------------------------------------
# Deep image
# --------------------------------------------------------------------------


def test_create_deep_image_writes_a_2d_fits_file(sn3_cube_noml):
    sn3_cube_noml.create_deep_image()
    deep = sn3_cube_noml.deep_image
    assert deep.ndim == 2
    # deep_image is transposed relative to the cube.
    assert deep.shape == sn3_cube_noml.cube_final.shape[:2][::-1]

    path = os.path.join(sn3_cube_noml.output_dir, sn3_cube_noml.object_name + "_deep.fits")
    assert os.path.exists(path)
    written = fits.open(path)[0].data
    np.testing.assert_allclose(written, deep, rtol=1e-6)


def test_deep_image_equals_the_spectral_sum(sn3_cube_noml):
    """
    Only holds because the fixture's x-dimension is a multiple of 10.

    create_deep_image sums the cube in exactly ten slabs of
    int(shape[0] / 10) rows.  When shape[0] is not divisible by 10 the trailing
    rows are silently left as zeros -- for a real 2048-row cube that is the last
    8 rows.  See test_deep_image_drops_trailing_rows.
    """
    sn3_cube_noml.create_deep_image()
    expected = np.nansum(sn3_cube_noml.cube_final, axis=2).T
    np.testing.assert_allclose(sn3_cube_noml.deep_image, expected, rtol=1e-6)


@pytest.mark.xfail(strict=True, reason="create_deep_image drops rows when dimx % 10 != 0; fix in Phase 5")
def test_deep_image_drops_trailing_rows(luci_factory, cube_truth_factory):
    """
    The intended contract, on a cube whose x-dimension is not a multiple of 10.

    Flips to passing once the ten-slab loop is replaced by a full sum.
    """
    truth = cube_truth_factory("SN3", dimx=24, dimy=20)
    cube = luci_factory(truth, ML_bool=False)
    cube.create_deep_image()
    expected = np.nansum(cube.cube_final, axis=2).T
    np.testing.assert_allclose(cube.deep_image, expected, rtol=1e-6)


# --------------------------------------------------------------------------
# Binning
# --------------------------------------------------------------------------


def test_bin_cube_halves_each_spatial_dimension(sn3_cube_noml, sn3_truth):
    cube = sn3_cube_noml
    cube.bin_cube(cube.cube_final, cube.header, 2, 0, sn3_truth["dimx"], 0, sn3_truth["dimy"])
    assert cube.cube_binned.shape[0] == sn3_truth["dimx"] // 2
    assert cube.cube_binned.shape[1] == sn3_truth["dimy"] // 2
    assert cube.cube_binned.shape[2] == sn3_truth["dimz"]


def test_binning_conserves_total_flux(sn3_cube_noml, sn3_truth):
    cube = sn3_cube_noml
    total_before = cube.cube_final.sum()
    cube.bin_cube(cube.cube_final, cube.header, 2, 0, sn3_truth["dimx"], 0, sn3_truth["dimy"])
    assert cube.cube_binned.sum() == pytest.approx(total_before, rel=1e-9)


# --------------------------------------------------------------------------
# Spectrum extraction
# --------------------------------------------------------------------------


def test_extract_spectrum_sums_the_requested_box(sn3_cube_noml):
    axis, spectrum = sn3_cube_noml.extract_spectrum(4, 8, 4, 8)
    assert spectrum.shape == axis.shape
    expected = np.nansum(sn3_cube_noml.cube_final[4:8, 4:8, :], axis=(0, 1))
    np.testing.assert_allclose(spectrum, expected, rtol=1e-6)


def test_extract_spectrum_mean_is_currently_a_no_op(sn3_cube_noml):
    """
    Pins CURRENT behaviour: `mean=True` does nothing in extract_spectrum.

    The pixel counter is only ever incremented inside its own initialisation
    guard::

        if spec_ct == 0:
            axis = self.spectrum_axis[~np.isnan(sky)]
            spec_ct += 1

    so spec_ct is 1 for the whole loop and `integrated_spectrum /= spec_ct`
    divides by one.  The sibling extract_spectrum_region() increments outside
    the guard and *is* correct, so the two functions disagree.

    This matters: the docstring says the method is "primarily used to extract
    background regions", and a background averaged over N pixels comes back N
    times too large.  Feeding that into fit_cube(bkg=...) over-subtracts by a
    factor of N.
    """
    _, summed = sn3_cube_noml.extract_spectrum(4, 8, 4, 8)
    _, averaged = sn3_cube_noml.extract_spectrum(4, 8, 4, 8, mean=True)
    np.testing.assert_allclose(averaged, summed, rtol=1e-9)


@pytest.mark.xfail(strict=True, reason="extract_spectrum never increments spec_ct; fix in Phase 5")
def test_extract_spectrum_mean_should_divide_by_pixel_count(sn3_cube_noml):
    """The intended contract, matching extract_spectrum_region's behaviour."""
    _, summed = sn3_cube_noml.extract_spectrum(4, 8, 4, 8)
    _, averaged = sn3_cube_noml.extract_spectrum(4, 8, 4, 8, mean=True)
    np.testing.assert_allclose(averaged, summed / 16.0, rtol=1e-6)


def test_extract_spectrum_region_mean_does_divide_by_pixel_count(sn3_cube_noml, sn3_truth):
    """The correct sibling, for contrast with the test above."""
    mask = np.zeros((sn3_truth["dimx"], sn3_truth["dimy"]), dtype=bool)
    mask[4:8, 4:8] = True
    _, summed = sn3_cube_noml.extract_spectrum_region(mask)
    _, averaged = sn3_cube_noml.extract_spectrum_region(mask, mean=True)
    np.testing.assert_allclose(averaged, summed / 16.0, rtol=1e-6)


def test_extracted_spectrum_peaks_near_halpha(sn3_cube_noml, sn3_truth):
    """Sanity check that the fixture really contains the lines we injected."""
    axis, spectrum = sn3_cube_noml.extract_spectrum(4, 8, 4, 8)
    peak_position = axis[int(np.argmax(spectrum))]
    expected = 1e7 / (656.280 * (1.0 + sn3_truth["velocity_kms"] / 299792.0))
    assert peak_position == pytest.approx(expected, abs=5.0)


# --------------------------------------------------------------------------
# Housekeeping
# --------------------------------------------------------------------------


def test_heliocentric_correction_returns_a_velocity(sn3_cube_noml):
    correction = sn3_cube_noml.heliocentric_correction()
    assert np.isfinite(correction.value)
    assert abs(correction.value) < 50.0  # Earth's orbital speed bounds this
