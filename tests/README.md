# LUCI test suite

This suite is the **safety net** for the ongoing refactor (see the plan and
`REFACTOR_BUGS.md` at the repo root). Its job is to let us restructure LUCI while
*proving* the science does not change.

Before this, `tests/` could not run at all in CI: the fixtures pointed at a data
cube on one laptop and were wrapped in a class named `Test` with an `__init__`,
which pytest silently refuses to collect. Every file here has been rebuilt around
a synthetic cube generated at test time.

## Layout

| File | What it pins |
|---|---|
| `fixtures/make_cube.py` | Generates a small synthetic SITELLE cube (legacy quadrant HDF5) with emission lines at *known* velocity/broadening. The whole suite is built on this. |
| `conftest.py` | Session-scoped fixtures: build a cube, construct a `Luci` over it, cache both. `--record` flag lives here. |
| `test_pure_functions.py` | Side-effect-free maths: velocity/flux conversions, line models, quadrant geometry, binning, numerical Hessian. Fast, no cube, no ML. |
| `test_cube.py` | Cube reading, WCS/geometry, deep image, binning, spectrum extraction. |
| `test_output_contract.py` | The exact filesystem layout and FITS headers `save_fits` writes. |
| `test_ml_priors.py` | How initial estimates reach the optimiser — home of the ML-off defect (B1). |
| `test_golden.py` | Characterization: fit values must match `golden/*.json` bit-for-bit (rtol 1e-6). |
| `golden/*.json` | Recorded fit baselines. **Do not hand-edit.** |

## Running

The environment is described entirely by `pyproject.toml` + `uv.lock`. It pins
Python 3.10 and the frozen legacy stack that the current pre-refactor code needs
— modern numba won't even import it (B11), and TF 2.11 has no wheels past 3.10.

```bash
uv sync                # creates .venv from the lockfile

# Fast tests (seconds):
uv run pytest -m "not slow"

# Everything, including the golden fits (~5 min, needs the Keras predictors):
uv run pytest
```

`conftest.py` sets `MPLBACKEND=Agg` as a fallback (needed because of B12,
`plt.clf()` in the per-spectrum path). Export it yourself if you invoke pytest
in some way that bypasses conftest early.

`tests/requirements-legacy.txt` is kept only as a human-readable record of the
exact versions the goldens were recorded against, with the rationale for the two
load-bearing pins. `uv.lock` is the operational source of truth; the two agree.

## The two markers that matter

- **`xfail(strict=True)`** — a *known bug*, with the test stating the intended
  (correct) behaviour. Strict means the suite **fails if it starts passing**, so
  when you fix the bug the suite tells you to delete the marker. See
  `REFACTOR_BUGS.md` for the catalogue; every `Bxx` there has a test here.
- **`slow`** — runs the optimiser over real pixels (minutes). Deselect with
  `-m "not slow"` for a quick inner loop.

## Golden workflow

```bash
# After an intended, reviewed change to fit values:
MPLBACKEND=Agg .venv-legacy/bin/python -m pytest tests/test_golden.py --record
git diff tests/golden/     # <- the numeric change MUST be reviewed here
```

Re-recording to silence a failure defeats the purpose. Only re-record when a
change to the numbers is *intended*, and do it in the same commit as the change
so the diff is visible.
