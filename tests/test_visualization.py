"""Tests for non-interactive plotting and visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits

from luci.viz.plotting import check_units, plot_fit, plot_map, plot_map_no_coords, plot_spectrum, set_style
from luci.viz.visualize import add_lines, visualize


def test_plot_spectrum_converts_units_and_writes_output(tmp_path):
    axis = np.array([10_000.0, 20_000.0])
    spectrum = np.array([1.0, 2.0])
    output = tmp_path / "spectrum.png"

    ax = plot_spectrum(axis, spectrum, units="nm", output_name=output, color="red")

    np.testing.assert_allclose(ax.lines[0].get_xdata(), [1000.0, 500.0])
    assert ax.lines[0].get_color() == "red"
    assert output.exists()
    plt.close("all")


def test_plot_fit_uses_a_supplied_axis_and_angstrom_units(tmp_path):
    _, ax = plt.subplots()
    axis = np.array([10_000.0, 20_000.0])
    output = tmp_path / "fit.png"

    returned = plot_fit(axis, [1, 2], [1.1, 1.9], ax=ax, units="Angstrom", output_name=output)

    assert returned is ax
    assert len(ax.lines) == 2
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [10_000.0, 5_000.0])
    assert output.exists()
    plt.close("all")


@pytest.mark.parametrize("quantity", ["flux", "velocity", "broadening"])
def test_plot_map_writes_each_supported_quantity(tmp_path, quantity):
    values = np.arange(1, 17, dtype=float).reshape(4, 4)
    header = fits.Header(
        {
            "NAXIS": 2,
            "NAXIS1": 4,
            "NAXIS2": 4,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": 2.0,
            "CRPIX2": 2.0,
            "CRVAL1": 10.0,
            "CRVAL2": 20.0,
            "CDELT1": -0.001,
            "CDELT2": 0.001,
        }
    )

    plot_map(values, quantity, output_dir=str(tmp_path), header=header, object_name="OBJ", filter_name="SN3")

    assert (tmp_path / f"OBJ_SN3_{quantity}_map.png").exists()
    plt.close("all")


@pytest.mark.parametrize("quantity", ["flux", "velocity", "broadening", "zscore"])
def test_plot_map_without_coordinates_writes_supported_quantities(tmp_path, quantity):
    values = np.arange(1, 17, dtype=float).reshape(4, 4)

    plot_map_no_coords(values, quantity, output_dir=str(tmp_path), object_name="OBJ", filter_name="SN3", clims=[0, 20])

    assert (tmp_path / f"OBJ_SN3_{quantity}_map.png").exists()
    plt.close("all")


def test_plot_helpers_reject_unknown_options(tmp_path):
    with pytest.raises(Exception, match="unit name"):
        check_units("meters")
    with pytest.raises(ValueError, match="quantity_name"):
        plot_map(np.ones((2, 2)), "temperature", output_dir=str(tmp_path), header=fits.Header())
    with pytest.raises(ValueError, match="quantity_name"):
        plot_map_no_coords(np.ones((2, 2)), "temperature", output_dir=str(tmp_path))
    assert set_style(True) == "./dark.mplstyle"
    assert set_style(False) == "./light.mplstyle"


@pytest.mark.parametrize("filter_name, expected", [("SN1", 1), ("SN2", 3), ("SN3", 5), ("SN4", 3)])
def test_add_lines_marks_the_filter_lines(filter_name, expected):
    plt.figure()
    add_lines({"FILTER": filter_name}, np.array([1.0, 2.0, 1.5]))
    assert len(plt.gca().collections) == expected
    plt.close("all")


def test_visualize_builds_the_interactive_figure_without_blocking(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    deep = np.arange(1, 10, dtype=float).reshape(3, 3)
    cube = np.repeat(deep.T[:, :, None], 8, axis=2)

    result = visualize(deep, np.linspace(14_000, 15_000, 8), cube, {"FILTER": "SN3"})

    assert result is None
    assert len(plt.gcf().axes) >= 2
    plt.close("all")
