# Releasing LUCI

LUCI publishes the `luci-sitelle` distribution from GitHub Releases through
PyPI Trusted Publishing. No PyPI API token is stored in GitHub.

## One-time configuration

1. In the GitHub repository settings, create an environment named `pypi`.
   Requiring approval from a maintainer before deployment is recommended.
2. In the PyPI account's Publishing settings, add a pending GitHub publisher
   with these exact values:

   - PyPI project name: `luci-sitelle`
   - Owner: `crhea93`
   - Repository: `LUCI`
   - Workflow: `publish.yml`
   - Environment: `pypi`

A pending publisher does not reserve the project name. It creates the project
on the first successful publication and then becomes a normal publisher.

## Preparing a release

1. Choose a PEP 440 version that has never been published.
2. Update `project.version` in `pyproject.toml` and run `uv lock`.
3. Open a pull request and require the complete CI workflow to pass. The
   package job builds the wheel and source distribution, validates their
   metadata and README, enforces PyPI's per-file size limit, and loads the
   runtime assets from the built wheel.
4. From the release commit, confirm the local publication checks if desired:

   ```bash
   uv build
   uvx twine check dist/*
   wheel=$(find dist -maxdepth 1 -name '*.whl' -print -quit)
   LUCI_WHEEL="$wheel" uv run --no-sync pytest tests/test_wheel_artifact.py -q
   ```

## Publishing

1. Merge the release pull request.
2. Create and publish a GitHub Release tagged `vX.Y.Z`, where `X.Y.Z` exactly
   matches `project.version` in `pyproject.toml`.
3. Approve the `pypi` environment deployment if GitHub requests it.
4. Wait for the **Publish to PyPI** workflow. It independently rebuilds and
   validates the distributions, rejects a mismatched tag, uploads the exact
   artifacts to the workflow run, and publishes them with attestations.
5. Verify the release from a clean environment:

   ```bash
   python -m venv /tmp/luci-release-check
   /tmp/luci-release-check/bin/python -m pip install luci-sitelle==X.Y.Z
   /tmp/luci-release-check/bin/python -c "import luci; print(luci.__version__)"
   ```

PyPI files and versions are immutable. If a published release is wrong, yank
it and publish a new version; do not try to reuse the version number.
