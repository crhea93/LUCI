"""
Shared pytest fixtures for the LUCI test suite.

Everything here is built from a synthetic cube generated at test time
(``tests/fixtures/make_cube.py``) rather than from an observation at a hardcoded
path, so the suite runs anywhere -- including CI, which has never been able to
execute these tests before.
"""

from __future__ import annotations

import os
import sys

# Must happen before anything pulls in matplotlib.  LuciFit.estimate_priors_ML()
# calls plt.clf() on every single spectrum, which instantiates a Tk window and
# dies with "no display name and no $DISPLAY" on a headless machine.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# With `uv sync` the package is installed editable, so `from LuciBase import
# Luci` resolves without help.  The repo-root insert below is a belt-and-braces
# fallback for anyone running pytest against an un-installed checkout; it is a
# no-op once the package is on the path.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# tests/ itself must be importable so `from fixtures.make_cube import ...` works.
sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))

from fixtures.make_cube import PROFILES, write_cube  # noqa: E402

# LUCI resolves ML/ and Data/ relative to this and requires a trailing slash.
LUCI_PATH = REPO_ROOT + "/"

# Resolutions that have both a reference spectrum and a trained predictor.
RESOLUTION_FOR_FILTER = {"SN3": 5000, "SN2": 1000, "SN1": 1000}


def pytest_addoption(parser):
    parser.addoption(
        "--record",
        action="store_true",
        default=False,
        help="Overwrite tests/golden/*.json with the values produced by the "
        "current code, instead of comparing against them.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: fitting tests that take minutes to run")
    config.addinivalue_line("markers", "ml: requires the Keras/TensorFlow predictors")


@pytest.fixture(scope="session")
def record_golden(pytestconfig) -> bool:
    return bool(pytestconfig.getoption("--record"))


@pytest.fixture(scope="session")
def repo_root() -> str:
    return REPO_ROOT


@pytest.fixture(scope="session")
def luci_path() -> str:
    return LUCI_PATH


@pytest.fixture(scope="session")
def cube_truth_factory(tmp_path_factory):
    """
    Build (and cache) a synthetic cube per filter, returning its ground truth.

    Session-scoped and memoised because writing a cube costs a few seconds and
    several tests want the same one.
    """
    cache: dict[tuple, dict] = {}
    base = tmp_path_factory.mktemp("luci_cubes")

    def _make(filter_name: str = "SN3", **kwargs) -> dict:
        key = (filter_name, tuple(sorted(kwargs.items())))
        if key not in cache:
            suffix = "_".join(f"{k}{v}" for k, v in sorted(kwargs.items()))
            name = f"{filter_name}_{suffix}" if suffix else filter_name
            path = str(base / name)
            cache[key] = write_cube(path, filter_name=filter_name, **kwargs)
        return cache[key]

    return _make


@pytest.fixture(scope="session")
def sn3_truth(cube_truth_factory) -> dict:
    """Ground truth for the default SN3 cube: v=100 km/s, sigma=30 km/s."""
    return cube_truth_factory("SN3")


@pytest.fixture(scope="session")
def luci_factory(tmp_path_factory):
    """
    Construct a ``Luci`` instance for a given cube truth dict.

    Each call gets its own output directory so tests never observe each other's
    FITS products.  Cached per (cube, resolution, ML flag) because reading the
    cube and loading the reference spectrum is not free.
    """
    from LuciBase import Luci

    cache: dict[tuple, object] = {}
    base = tmp_path_factory.mktemp("luci_outputs")
    counter = {"n": 0}

    def _make(
        truth: dict, resolution: int | None = None, ML_bool: bool = True, mdn: bool = False, redshift: float = 0.0
    ):
        filter_name = truth["filter"]
        resolution = resolution or RESOLUTION_FOR_FILTER[filter_name]
        key = (truth["path"], resolution, ML_bool, mdn, redshift)
        if key not in cache:
            counter["n"] += 1
            out = base / f"run{counter['n']}"
            out.mkdir(exist_ok=True)
            cube_path = truth["path"][: -len(".hdf5")]
            cache[key] = Luci(
                LUCI_PATH,
                cube_path,
                str(out),
                "TESTOBJ",
                redshift,
                resolution,
                ML_bool=ML_bool,
                mdn=mdn,
            )
        return cache[key]

    return _make


@pytest.fixture(scope="session")
def sn3_cube(luci_factory, sn3_truth):
    """A ``Luci`` instance over the default SN3 synthetic cube, ML enabled."""
    return luci_factory(sn3_truth)


@pytest.fixture(scope="session")
def sn3_cube_noml(luci_factory, sn3_truth):
    """Same cube with ML priors disabled -- much faster, no predictor needed."""
    return luci_factory(sn3_truth, ML_bool=False)


@pytest.fixture(scope="session")
def profiles():
    return PROFILES
