"""
Tests for FitConfig.

The point of the object is that a bad request fails at the call site with a
readable message, rather than somewhere inside SLSQP -- and that grouping the
options changes nothing for callers who still pass the individual keywords.
"""

from __future__ import annotations

import numpy as np
import pytest

from LUCI.config import AVAILABLE_MODELS, LINE_DICT, FitConfig, InvalidFitConfig

SN3_LINES = ["Halpha", "NII6548", "NII6583"]


# --------------------------------------------------------------------------
# Defaults and derived values
# --------------------------------------------------------------------------


def test_rel_lists_default_to_one_group_per_line():
    config = FitConfig(lines=SN3_LINES)
    assert config.vel_rel == [1, 1, 1]
    assert config.sigma_rel == [1, 1, 1]
    assert config.n_lines == 3


def test_gauss_is_normalised_to_gaussian():
    """'gauss' has always been accepted as an alias."""
    assert FitConfig(lines=["Halpha"], model_type="gauss").model_type == "gaussian"


def test_freeze_reflects_supplied_initial_values():
    assert FitConfig(lines=["Halpha"]).freeze is False
    assert FitConfig(lines=["Halpha"], initial_values=[100.0, 30.0]).freeze is True


def test_replace_returns_a_revalidated_copy():
    config = FitConfig(lines=SN3_LINES)
    other = config.replace(model_type="gaussian")
    assert other.model_type == "gaussian"
    assert config.model_type == "sincgauss"  # original untouched
    with pytest.raises(InvalidFitConfig):
        config.replace(model_type="nonsense")


# --------------------------------------------------------------------------
# Validation -- the reason the object exists
# --------------------------------------------------------------------------


def test_unknown_line_is_rejected_with_the_available_list():
    with pytest.raises(InvalidFitConfig, match="Halpa"):
        FitConfig(lines=["Halpa"])  # typo


def test_unknown_model_is_rejected():
    with pytest.raises(InvalidFitConfig, match="nonsense"):
        FitConfig(lines=["Halpha"], model_type="nonsense")


@pytest.mark.parametrize("field", ["vel_rel", "sigma_rel"])
def test_mismatched_constraint_list_length_is_rejected(field):
    with pytest.raises(InvalidFitConfig, match=field):
        FitConfig(lines=SN3_LINES, **{field: [1, 1]})


def test_unknown_bayes_method_is_rejected():
    with pytest.raises(InvalidFitConfig, match="bayes_method"):
        FitConfig(lines=["Halpha"], bayes_method="mcmc")


def test_n_stoch_must_be_positive():
    with pytest.raises(InvalidFitConfig, match="n_stoch"):
        FitConfig(lines=["Halpha"], n_stoch=0)


def test_every_advertised_model_is_accepted():
    for model in AVAILABLE_MODELS:
        assert FitConfig(lines=["Halpha"], model_type=model).model_type == model


def test_line_dict_is_the_single_source_of_truth(sn3_cube_noml):
    """The fitter must read its line list from the config module, not its own copy."""
    from LUCI.fitting.spectrum_fitter import SpectrumFitter

    fitter = SpectrumFitter(
        np.copy(sn3_cube_noml.cube_final[9, 9, :]),
        sn3_cube_noml.spectrum_axis,
        sn3_cube_noml.wavenumbers_syn,
        "sincgauss",
        ["Halpha"],
        [1],
        [1],
        ML_bool=False,
        filter="SN3",
        resolution=5000,
        Luci_path=sn3_cube_noml.Luci_path,
    )
    assert fitter.line_dict == LINE_DICT


# --------------------------------------------------------------------------
# Backward compatibility
# --------------------------------------------------------------------------


def test_keyword_and_config_paths_produce_the_same_fit(sn3_cube_noml, sn3_truth):
    """
    Passing a FitConfig must be exactly equivalent to passing the keywords.

    This is what makes the object safe to introduce: existing callers are
    untouched, and new ones can build a config and reuse it.
    """
    from LUCI.fitting.spectrum_fitter import SpectrumFitter

    sky = np.copy(sn3_cube_noml.cube_final[9, 9, :])
    common = dict(
        trans_filter=sn3_cube_noml.transmission_interpolated,
        theta=sn3_cube_noml.interferometer_theta[9, 9],
        delta_x=sn3_cube_noml.hdr_dict["STEP"],
        n_steps=sn3_cube_noml.step_nb,
        zpd_index=sn3_cube_noml.zpd_index,
        filter="SN3",
        resolution=5000,
        Luci_path=sn3_cube_noml.Luci_path,
    )

    by_keyword = SpectrumFitter(
        sky,
        sn3_cube_noml.spectrum_axis,
        sn3_cube_noml.wavenumbers_syn,
        "sincgauss",
        SN3_LINES,
        [1, 1, 1],
        [1, 1, 1],
        ML_bool=False,
        **common,
    ).fit()

    config = FitConfig(lines=SN3_LINES, model_type="sincgauss", ML_bool=False)
    by_config = SpectrumFitter(
        np.copy(sn3_cube_noml.cube_final[9, 9, :]),
        sn3_cube_noml.spectrum_axis,
        sn3_cube_noml.wavenumbers_syn,
        config=config,
        **common,
    ).fit()

    np.testing.assert_allclose(by_config["velocities"], by_keyword["velocities"], rtol=1e-12)
    np.testing.assert_allclose(by_config["sigmas"], by_keyword["sigmas"], rtol=1e-12)
    np.testing.assert_allclose(by_config["fluxes"], by_keyword["fluxes"], rtol=1e-12, atol=0.0)


def test_fitter_exposes_its_config(sn3_cube_noml):
    from LUCI.fitting.spectrum_fitter import SpectrumFitter

    fitter = SpectrumFitter(
        np.copy(sn3_cube_noml.cube_final[9, 9, :]),
        sn3_cube_noml.spectrum_axis,
        sn3_cube_noml.wavenumbers_syn,
        "sincgauss",
        SN3_LINES,
        [1, 1, 1],
        [1, 1, 1],
        ML_bool=False,
        filter="SN3",
        resolution=5000,
        Luci_path=sn3_cube_noml.Luci_path,
    )
    assert fitter.config.lines == SN3_LINES
    assert fitter.config.model_type == "sincgauss"
    assert fitter.config.n_lines == 3
