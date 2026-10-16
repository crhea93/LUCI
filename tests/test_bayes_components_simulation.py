"""Deterministic tests for Bayesian helpers, components, and simulations."""

from __future__ import annotations

import os

import numpy as np
import pytest
from astropy.io import fits

from luci.analysis.components import calculate_components_in_region_function, create_component_map_function
from luci.fitting.bayes import log_likelihood_bayes, log_prior_uniform, log_probability, prior_transform
from luci.fitting.components import get_individual_components
from luci.fitting.models import Gaussian
from luci.simulation import Spectrum

LINE_DICT = {"Halpha": 656.280}


def test_bayesian_likelihood_prefers_the_generating_parameters():
    axis = np.linspace(15220, 15260, 101)
    theta = np.array([0.6, 0.0, 0.8, 0.1])
    spectrum = Gaussian().evaluate_bayes(axis, theta[:-1], 1, ["Halpha"], LINE_DICT) + theta[-1]

    best = log_likelihood_bayes(theta, axis, spectrum, 0.05, "gaussian", 1, 1.0, [1], [1], ["Halpha"], LINE_DICT)
    worse = log_likelihood_bayes(
        np.array([0.2, 5.0, 1.5, 0.3]),
        axis,
        spectrum,
        0.05,
        "gaussian",
        1,
        1.0,
        [1],
        [1],
        ["Halpha"],
        LINE_DICT,
    )

    assert best > worse


def test_uniform_bayesian_prior_checks_every_compressed_parameter():
    valid = np.array([0.4, 0.5, 0.0, 1.0, 0.2])
    assert np.isfinite(log_prior_uniform(valid, 2, ["Halpha"] * 2, LINE_DICT))

    for index, invalid_value in ((0, -1), (2, 20), (3, 3), (4, 1)):
        invalid = valid.copy()
        invalid[index] = invalid_value
        assert log_prior_uniform(invalid, 2, ["Halpha"] * 2, LINE_DICT) == -np.inf


def test_prior_transform_matches_the_compressed_parameter_layout():
    transformed = prior_transform(np.array([0.25, 0.75, 0.5, 0.25, 0.5]))
    np.testing.assert_allclose(transformed, [0.2475, 0.7425, 0.0, 0.5, 0.4])


def test_log_probability_rejects_out_of_bounds_parameters():
    axis = np.linspace(15220, 15260, 20)
    result = log_probability(
        np.array([-1.0, 0.0, 1.0, 0.1]),
        axis,
        np.ones_like(axis),
        0.1,
        "gaussian",
        1,
        ["Halpha"],
        LINE_DICT,
        1.0,
        [0, 0, 1, 1],
        [1],
        [1],
        False,
    )
    assert result == -np.inf


@pytest.mark.parametrize("model", ["gaussian", "sinc", "sincgauss"])
def test_component_reconstruction_preserves_lines_and_continuum(model):
    axis = np.linspace(15220, 15260, 101)
    solution = np.array([1.0, 15235.0, 0.8, 0.5, 15245.0, 1.1, 0.2])
    components = get_individual_components(
        {
            "fit_sol": solution,
            "fit_axis": axis,
            "fit_function": model,
            "fit_lines": ["Halpha", "Halpha"],
            "sinc_width": 1.0,
        }
    )

    assert set(components) == {"Halpha", "Halpha_2", "continuum"}
    assert all(vector.shape == axis.shape for vector in components.values())
    np.testing.assert_allclose(components["continuum"], 0.2)


def test_component_reconstruction_numbers_three_repeated_lines():
    axis = np.linspace(15220, 15260, 21)
    solution = np.array([1.0, 15230.0, 1.0] * 3 + [0.1])
    components = get_individual_components(
        {
            "fit_sol": solution,
            "fit_axis": axis,
            "fit_function": "gaussian",
            "fit_lines": ["Halpha"] * 3,
            "sinc_width": 1.0,
        }
    )
    assert {"Halpha", "Halpha_2", "Halpha_3"}.issubset(components)


@pytest.mark.parametrize("model", ["gaussian", "sinc", "sincgauss"])
def test_simulated_spectra_are_finite_and_reproducible(model):
    simulation = Spectrum(["Halpha", "NII6583"], model, [1.0, 0.4], [100.0, 120.0], [30.0, 40.0], "SN3", 5000, 50)
    np.random.seed(12)
    axis, spectrum = simulation.create_spectrum()
    np.random.seed(12)
    axis_again, spectrum_again = simulation.create_spectrum()

    assert len(axis) == simulation.n_steps
    assert np.all(np.diff(axis) > 0)
    assert np.all(np.isfinite(spectrum))
    np.testing.assert_array_equal(axis_again, axis)
    np.testing.assert_array_equal(spectrum_again, spectrum)


def test_component_map_respects_offset_bounds_and_mask(sn3_cube_noml, monkeypatch):
    class FakeModel:
        def __call__(self, spectrum, training=False):
            return np.array([[0.2, 0.8]])

    monkeypatch.setattr("luci.analysis.components._load_components_model", lambda path: FakeModel())
    mask = np.zeros(sn3_cube_noml.cube_final.shape[:2], dtype=bool)
    mask[2, 3] = True

    create_component_map_function(
        sn3_cube_noml.header,
        sn3_cube_noml.hdr_dict,
        sn3_cube_noml.Luci_path,
        sn3_cube_noml.resolution,
        sn3_cube_noml.filter,
        sn3_cube_noml.cube_final,
        sn3_cube_noml.spectrum_axis,
        sn3_cube_noml.wavenumbers_syn_full,
        sn3_cube_noml.output_dir,
        "COMPTEST",
        x_min=2,
        x_max=4,
        y_min=3,
        y_max=5,
        n_threads=1,
        region=mask,
    )

    components = fits.getdata(os.path.join(sn3_cube_noml.output_dir, "COMPTEST_comps.fits"))
    probabilities = fits.getdata(os.path.join(sn3_cube_noml.output_dir, "COMPTEST_comps_probs.fits"))
    assert components[3, 2] == 2
    assert probabilities[3, 2] == pytest.approx(0.8)
    assert np.count_nonzero(components) == 1


def test_region_component_classification_uses_the_integrated_spectrum(sn3_cube_noml, monkeypatch):
    observed = {}

    class FakeModel:
        def __call__(self, spectrum, training=False):
            observed["spectrum"] = spectrum
            return np.array([[0.7, 0.3]])

    monkeypatch.setattr("luci.analysis.components._load_components_model", lambda path: FakeModel())
    mask = np.zeros(sn3_cube_noml.cube_final.shape[:2], dtype=bool)
    mask[2:4, 3:5] = True

    result = calculate_components_in_region_function(
        sn3_cube_noml.header,
        sn3_cube_noml.hdr_dict,
        sn3_cube_noml.Luci_path,
        sn3_cube_noml.resolution,
        sn3_cube_noml.filter,
        sn3_cube_noml.cube_final,
        sn3_cube_noml.spectrum_axis,
        sn3_cube_noml.wavenumbers_syn_full,
        mask,
        bkg=None,
    )

    assert result["components"] == 1
    assert result["component_probability"] == pytest.approx(0.7)
    assert observed["spectrum"].shape == (1, len(sn3_cube_noml.wavenumbers_syn_full[2:-2]), 1)
