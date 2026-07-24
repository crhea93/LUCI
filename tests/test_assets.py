"""
Tests for locating LUCI's ML/ and Data/ directories.

``Luci_path`` was a mandatory positional argument -- a string with a required
trailing slash pointing at someone's checkout -- so every example in the
repository began by hardcoding an absolute path into another person's home
directory.  Now that LUCI installs as a package the location is derivable, and
the argument is optional.
"""

from __future__ import annotations

import os

import pytest

from luci.io.assets import (
    LuciAssetsNotFound,
    check_luci_path,
    default_luci_path,
    resolve_luci_path,
)


class TestResolution:
    def test_default_finds_the_repo_root(self, repo_root):
        assert default_luci_path().rstrip("/") == repo_root.rstrip("/")

    def test_result_always_ends_in_a_slash(self):
        """Callers concatenate directly: Luci_path + "ML/..."."""
        assert default_luci_path().endswith("/")
        assert resolve_luci_path("/tmp/x").endswith("/")
        assert resolve_luci_path("/tmp/x/").endswith("/")

    def test_explicit_path_wins(self):
        assert resolve_luci_path("/explicit/place") == "/explicit/place/"

    def test_explicit_path_is_not_validated(self):
        """Back-compat: the old code never checked, and tests pass fake roots."""
        assert resolve_luci_path("/definitely/not/real") == "/definitely/not/real/"

    def test_check_luci_path_still_appends_the_slash(self):
        assert check_luci_path("/a/b") == "/a/b/"
        assert check_luci_path("/a/b/") == "/a/b/"


class TestEnvironmentOverride:
    def test_env_var_takes_precedence(self, tmp_path, monkeypatch, repo_root):
        (tmp_path / "ML").mkdir()
        (tmp_path / "Data").mkdir()
        monkeypatch.setenv("LUCI_DATA_DIR", str(tmp_path))
        assert default_luci_path().rstrip("/") == str(tmp_path)

    def test_env_var_without_the_asset_dirs_raises(self, tmp_path, monkeypatch):
        """Fail loudly rather than silently falling back and reading the wrong models."""
        monkeypatch.setenv("LUCI_DATA_DIR", str(tmp_path))
        with pytest.raises(LuciAssetsNotFound, match="does not contain"):
            default_luci_path()

    def test_empty_env_var_is_ignored(self, monkeypatch, repo_root):
        monkeypatch.setenv("LUCI_DATA_DIR", "")
        assert default_luci_path().rstrip("/") == repo_root.rstrip("/")


class TestCubeConstruction:
    def test_cube_builds_without_an_explicit_luci_path(self, sn3_truth, tmp_path):
        """The point of the whole exercise: no absolute path in the call."""
        from luci.cube import SitelleCube

        cube = SitelleCube(
            cube_path=sn3_truth["path"][: -len(".hdf5")],
            output_dir=str(tmp_path),
            object_name="TESTOBJ",
            redshift=0.0,
            resolution=5000,
            ML_bool=False,
        )
        assert cube.Luci_path.endswith("/")
        assert os.path.isdir(cube.Luci_path + "ML")
        assert cube.cube_final.shape[2] > 0

    def test_explicit_luci_path_still_works_positionally(self, sn3_truth, tmp_path, luci_path):
        """Every existing notebook passes it as the first positional argument."""
        from luci.cube import SitelleCube

        cube = SitelleCube(
            luci_path,
            sn3_truth["path"][: -len(".hdf5")],
            str(tmp_path),
            "TESTOBJ",
            0.0,
            5000,
            ML_bool=False,
        )
        assert cube.Luci_path == luci_path
