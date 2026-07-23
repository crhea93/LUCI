"""
Packaging invariants.

These check the *installed* shape of the project rather than its behaviour, and
they exist because two packaging defects were found that no behavioural test
could catch:

  * **The TensorFlow import chain.** Fitting runs on ONNX Runtime, but a single
    module-level ``from tensorflow import keras`` anywhere in the import graph
    drags the whole of TensorFlow into every ``import LuciBase``. That costs
    seconds of startup, pins the supported Python version, and is invisible to
    functional tests, which pass either way.

  * **Stale top-level modules in an editable install (B18).** ``LUCI/`` is
    linked, but ``LuciBase.py`` / ``LuciAbsorp.py`` are *copied* into
    site-packages by hatchling's ``force-include``. An editable install
    therefore serves a frozen snapshot of them. The rest of the suite cannot see
    this because ``conftest`` puts the repo root first on ``sys.path``, which
    shadows the copy -- so this test deliberately compares file contents instead.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP_LEVEL_MODULES = ["LuciBase.py", "LuciAbsorp.py"]


def _digest(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _installed_copy(name: str) -> str | None:
    """Path of a physical copy of `name` in site-packages, if one exists."""
    for entry in sys.path:
        candidate = os.path.join(entry, name)
        if "site-packages" in entry and os.path.isfile(candidate):
            return candidate
    return None


@pytest.mark.parametrize("module", TOP_LEVEL_MODULES)
def test_installed_top_level_module_is_not_stale(module):
    """
    B18: if site-packages holds a *copy* of a top-level module, it must match the
    repository, or an installed user silently runs code that no longer exists.

    Skipped when there is no copy (the ideal state -- the module is resolved
    live from the checkout).
    """
    copy = _installed_copy(module)
    if copy is None:
        pytest.skip(f"{module} is not copied into site-packages (resolved live)")
    assert _digest(copy) == _digest(os.path.join(REPO_ROOT, module)), (
        f"site-packages/{module} differs from the repository copy. An editable "
        f"install is serving stale code; re-run `uv sync`. See B18."
    )


def test_importing_luci_does_not_load_tensorflow():
    """
    Fitting runs on ONNX Runtime; TensorFlow is an optional extra used only for
    training the background interpolator and counting components.

    Run in a subprocess: by the time this test executes, another test may already
    have imported TensorFlow, so checking `sys.modules` in-process proves nothing.
    """
    code = (
        "import sys; "
        "from LuciBase import Luci; "
        "tf=[m for m in sys.modules if m.split('.')[0] in ('tensorflow','keras','tensorflow_probability')]; "
        "print('TF_COUNT', len(tf))"
    )
    env = dict(os.environ, MPLBACKEND="Agg", TF_CPP_MIN_LOG_LEVEL="3")
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT, env=env
    )
    assert result.returncode == 0, f"importing LuciBase failed:\n{result.stderr[-2000:]}"
    line = [ln for ln in result.stdout.splitlines() if ln.startswith("TF_COUNT")]
    assert line, f"no marker in output:\n{result.stdout[-2000:]}"
    assert line[0] == "TF_COUNT 0", (
        "importing LuciBase pulled in TensorFlow. Some module gained a top-level "
        "`import tensorflow`/`keras`; make it a lazy import inside the function "
        "that needs it."
    )


def test_no_module_level_tensorflow_imports_in_the_library():
    """
    Static counterpart to the test above, with a clearer failure message.

    LuciNetwork is exempt: it only builds MDN architectures for the offline
    conversion tool, is never imported by the runtime library, and runs in the
    throwaway TensorFlow environment that tool creates.
    """
    import re

    offenders = []
    exempt = {"LuciNetwork.py"}
    for directory in (REPO_ROOT, os.path.join(REPO_ROOT, "LUCI")):
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".py") or name in exempt:
                continue
            path = os.path.join(directory, name)
            for number, line in enumerate(open(path), start=1):
                # Anchored at column 0 on purpose: an indented import is inside a
                # function, which is exactly the lazy pattern we want.
                if re.match(r"^(import|from)\s+(tensorflow|keras)\b", line):
                    offenders.append(f"{name}:{number}: {line.strip()}")
    assert not offenders, "module-level TensorFlow imports found:\n" + "\n".join(offenders)
