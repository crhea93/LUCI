"""Tests that exercise the built wheel rather than the source checkout.

Set ``LUCI_WHEEL`` to the wheel path to enable these tests. They are skipped in
the normal source test suite and run by the dedicated package job in CI.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

WHEEL_ENV = "LUCI_WHEEL"


@pytest.fixture(scope="module")
def wheel() -> Path:
    value = os.environ.get(WHEEL_ENV)
    if not value:
        pytest.skip(f"{WHEEL_ENV} is not set; wheel artifact tests run in the package CI job")
    path = Path(value).resolve()
    assert path.is_file(), f"wheel does not exist: {path}"
    return path


def test_wheel_contains_curated_runtime_assets(wheel):
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())

    required = {
        "luci/_assets/Data/SN3_filter.dat",
        "luci/_assets/Data/SN3_Transmission.dat",
        "luci/_assets/Data/sky_lines.dat",
        "luci/_assets/ML/Reference-Spectrum-R5000-SN3.fits",
        "luci/_assets/ML/onnx/R5000-PREDICTOR-I-SN3.onnx",
        "luci/_assets/ML/onnx/R5000-PREDICTOR-I-MDN-SN3.onnx",
        "luci/_assets/ML/R5000-COMPONENTS-SN3.h5",
    }
    assert required <= names

    # Training checkpoints are repository inputs, not runtime wheel data.
    assert not any("saved_model.pb" in name for name in names)
    assert not any("variables.data" in name for name in names)
    assert not any("keras_metadata.pb" in name for name in names)


def test_installed_wheel_resolves_and_loads_assets_outside_checkout(wheel, tmp_path):
    install_root = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(install_root)

    code = """
from pathlib import Path
import numpy as np

import luci
from luci.io.assets import default_luci_path
from luci.ml.onnx_backend import OnnxPredictor

root = Path(default_luci_path())
assert root.name == "_assets", root
assert root.is_relative_to(Path(luci.__file__).parent)

model = root / "ML" / "onnx" / "R5000-PREDICTOR-I-SN3.onnx"
predictor = OnnxPredictor(str(model))
estimate = predictor.predict(np.ones(460, dtype=np.float32))
assert np.isfinite(estimate.velocity)
assert np.isfinite(estimate.broadening)
print(root)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(install_root)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert str(install_root / "luci" / "_assets") in result.stdout
