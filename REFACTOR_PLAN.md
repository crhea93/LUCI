# LUCI Refactor Plan

> **Living document.** This is the working plan for the ongoing refactor, kept in the repo so it
> travels with the code. Its companion is [REFACTOR_BUGS.md](REFACTOR_BUGS.md), the register of every
> defect found (each linked to the test that pins it), and [tests/README.md](tests/README.md), which
> explains the safety net. Phase sections below are updated in place as work lands.

## Status at a glance

| Phase | State |
|---|---|
| 0 — Safety net (synthetic fixture, goldens, bug register) | ✅ done |
| 1 — Packaging & tooling (conda → uv, installable, CI) | ✅ done |
| 2 — Repo weight (~344 MB untracked) | ✅ done |
| 3 — Filter registry, numba removal, FitResult, `fitting/` package | ✅ done |
| 4 — ONNX migration | ✅ core done (TF-drop from core deps remains) |
| 5 — bug fixes, TF optional, orchestration deduplicated | ✅ done |
| 6 — Compat shim & API surface | 🟡 shims in place; lowercase rename outstanding |
| 7 — Docs & examples | ⬜ not started |

**Every bug in the register is fixed or mitigated** — B1–B18. See [REFACTOR_BUGS.md](REFACTOR_BUGS.md)
for each one, the test that pins it, and (where I got something wrong along the way) what the mistake
was.

**Verification state:** 123 fast tests + 12 golden baselines green; ruff and `uv lock --check` clean.
No `xfail` markers remain: every one flipped to a real passing test as its bug was fixed.

### Phase 5 outcome (bug cluster)
Ten bugs fixed with **zero change to fit results** (12/12 goldens byte-identical):

| Bug | Was |
|---|---|
| B2 | `extract_spectrum(mean=True)` was a silent no-op — backgrounds came back N× too large |
| B3 | `fit_region` forwarded `bkg` but not `bkgType`, so the background was ignored entirely |
| B5 | unbinned PCA in `fit_pixel` referenced undefined `x_pix`/`y_pix` → `NameError` |
| B6 | `fit_pixel`'s own defaults (`binning=None`) → `TypeError` |
| B7 | `create_deep_image` blanked the last rows of any cube whose height wasn't a multiple of 10 |
| B8 | `bin_mask` divided a boolean mask by `binning²`, returning 0.25 instead of True |
| B9 | `fit_region` didn't forward `fit_function`, so its outputs were named unlike `fit_cube`'s |
| B10 | `2048`/`2064` hardcoded in defaults and array shapes |
| B15 | `np.str` (removed in numpy 1.24) inside a bare `except` → new-format cubes had an axis-less WCS |
| B16 | multi-component fits overwrote each other's maps; only the last survived |

Also deduplicated: the PCA scaling window (copy-pasted three times, each calling `quit()` from library
code) now lives in the filter registry as `pca_scale_indices`.

New coverage: `tests/test_new_format_cube.py` exercises the new HDF5 layout, which had **none** — that
absence is precisely how B15 survived.

### FitConfig and PixelSelection
`luci/config.py` groups the fit *options* into one validated `FitConfig`; `SpectrumFitter` still takes
every individual keyword, so nothing breaks, but they are collected into a config internally and
validated in one place. The three ad-hoc `check_*` methods moved onto it, and `LINE_DICT` is now
defined once rather than copied across three modules.

`luci/engine/selection.py::resolve_mask` replaces the region/mask/pixel-list `if` chain that each fit
entry point re-implemented. It surfaced **B23**: the pixel-list branch started from `np.ones`, so
`pixel_list=True` fitted the entire cube.

A test asserts the keyword path and the config path produce bit-identical fits.

### The package restructure
The flat `luci/Luci*.py` modules are gone — every one is now a 3–25 line re-export shim over a module
with an actual home:

```
luci/
  cube.py            SitelleCube (was Luci, in LuciBase.py)   1009 lines
  simulation.py      log.py
  instrument/        filters.py  header.py
  io/                hdf5.py  reference.py  outputs.py  assets.py
  fitting/           spectrum_fitter.py (was Fit)  constraints.py  models.py
                     parameters.py  bayes.py  result.py  components.py  uncertainties.py
  engine/            runner.py  maps.py  selection.py  binning.py
  background/        detection.py  pca.py  subtraction.py
  analysis/          wvt.py  snr.py  slicing.py  skylines.py  components.py
  viz/               plotting.py  visualize.py
  ml/                base.py  onnx_backend.py  registry.py  mdn_architecture.py
```

`LuciBase.py` went from **2008 lines to a 7-line shim**; `LuciFit.py` from 1092 to 8. Nothing
downstream had to change — every historical import still resolves.

Bugs this surfaced: **B21** (sigma bounds on only the last line — ~65 km/s error on 2 of 5 lines),
**B22** (`fit_absorption` always raised NameError), **B20** (`LuciMask` ran a hardcoded-path script at
import), and it closed **B18** properly. Three extraction slips of my own were caught by ruff's `F821`
undefined-name check, which is the concrete argument for widening lint onto moved code.

### Phase 5e outcome — the duplicated orchestration is gone
`fit_cube`, `fit_region` and `fit_wvt` no longer hand-roll the same twelve-array allocation, parallel
fan-out, thirteen-tuple unpack and `save_fits` call. That work lives in `luci/engine/`:

- `FitMaps` — allocates the output maps, scatters a slice of results, and saves them.
- `run_fit` — the parallel per-slice runner; `**fit_kwargs` pass straight through to `fit_calc`, so a
  new fit option is threaded once rather than three times.
- `resolve_initial_values`, `deep_image_cutout` — the other shared preamble.

`LuciBase.py` 2008 → 1876 lines; duplicated allocation blocks 16 → 0. 12/12 goldens byte-identical.

This is the root cause of the B3/B5/B6/B9 family finally removed: those bugs all existed because a fix
or parameter added to one entry point never reached the others. **B19** was found while doing it — the
WVT scatter wrote the continuum error into the continuum map.

Still deferred: the full `SitelleCube` / `FitRunner` / `PixelSelection` class redesign. The duplication
it was meant to eliminate is now gone, so what remains is naming and the god-class split — genuinely
cosmetic, and best paired with Phase 6.

### Phase 5f outcome — TensorFlow is now optional
`import LuciBase` no longer loads TensorFlow at all. The two remaining consumers import it lazily and
raise a message pointing at the extra: the PCA background interpolator (`interpolation='nn'`) and
component counting. `LuciNetwork` is untouched — it is only used by the offline conversion tool.

- `requires-python` is now **>=3.10** (was `>=3.10,<3.11`); CI runs **3.10, 3.11, 3.12, 3.13**.
- Verified: full core install on **Python 3.13 with numpy 2.5.1**, 123 tests passing, zero TF.
- numpy is capped `<2` only on 3.10, where the `[ml-legacy]` extra needs it.
- Two `tests/test_packaging.py` guards keep it that way: a subprocess check that importing LuciBase
  loads no TF, and a static check for module-level TF imports.
- The nightly golden job pins 3.10, since the baselines were recorded there and numpy-2 bit-equality
  is unverified.
- **B18** found and mitigated along the way — see the register.

### Phase 4 outcome
**All 39 models converted to ONNX and validated — 39/39, worst deviation 6.7e-4 (float32 precision).**
`luci/ml/` (predictor protocol, ONNX backend with cached sessions, registry) is wired into `Fit`,
which now contains **no TensorFlow references at all**.

- **B13 fixed:** predictors and ONNX sessions are cached per process, so a model loads once instead of
  once per pixel. 5.7 s/pixel → 3.0 s/pixel; the residual is the actual SLSQP fit.
- **B12 fixed:** the per-spectrum `plt.clf()` is gone.
- **SN4 works** (both standard and MDN). Its conversion was always fine; only the *validator* was
  broken — `keras.models.load_model` can't read that SavedModel (newer Keras: `Unrecognized keyword
  arguments: ['optional']`). Validation now uses `tf.saved_model.load` + serving signature, which is
  what tf2onnx itself consumes.
- **B17 found and fixed:** MDN `stddev` needs `softplus(scale)`, not identity. Identity happened to be
  exact for R5000-MDN-SN3 (all large positive scales) but produced *negative* sigmas for the SN2 MDNs.
  The `--validate` gate caught it and refused to ship the two failing models.
- **ML goldens re-recorded** against ONNX; re-recording produced a **zero diff**, confirming the ONNX
  path is deterministic and the committed baselines were already ONNX.

**Done in Phase 5f:** TensorFlow is now an optional `[ml-legacy]` extra. See below.

> **Known fixture-quality issue (not an ONNX regression):** the synthetic SN2/SN1 cubes don't resemble
> real SITELLE data closely enough for the R1000 predictors, so their priors are poor (broad ≈ 226 vs
> truth 30) and those fits don't recover truth — the pre-ONNX Keras goldens showed the same. Those
> baselines pin *reproducibility*, not physics. SN3 is the meaningful ML golden (recovers 99.5/30.2).

## Context

LUCI is a ~7,000-line emission-line fitting pipeline for SITELLE IFU data cubes, written ~5 years ago
and grown organically since. It works and is published, but it has accumulated the failure modes of a
long-lived research code: a 1,989-line god-class, a 1,137-line fitting class with ~50 mutable
attributes, five copy-pasted `if filter == 'SN3'/elif...` wavelength-bound tables, four near-identical
fit-orchestration blocks, a broken `setup.py`, a test suite that cannot run in CI, and 300 MB of
*generated output* committed to git.

The goal is a codebase that is installable, testable, and safe to modify — without changing the
science. Numerical output must be provably identical (or knowingly, deliberately different) at every
step.

Decisions made with the user:
- **API**: clean break to a proper `luci` package, with a deprecation shim so existing notebooks keep working for one release.
- **ML**: convert the 137 tracked Keras artifacts to ONNX; drop TensorFlow from the runtime dependency set.
- **Scope**: packaging + tooling, core architecture, test suite, and repo weight — all four.
- **Sequencing**: safety net first. Characterization tests before any refactoring.
- **Environments**: replace conda with **uv** — `pyproject.toml` + `uv.lock` as the single source of truth for dependencies, environments, and CI.

---

## What's actually wrong

### Structural

| Problem | Where |
|---|---|
| `Luci` god-class: IO, deep images, binning, fit orchestration, SNR, WVT, PCA background, *and* inline Keras training | [LuciBase.py](LuciBase.py) — 1,989 lines |
| `Fit` class: ~50 attributes assigned in `__init__`, ordering-dependent, mutated throughout | [luci/LuciFit.py:52-176](luci/LuciFit.py#L52-L176) |
| Four near-identical orchestration blocks (allocate 12 arrays → `Parallel(fit_calc)` → unpack 13-tuple → `save_fits`) | `fit_cube` [L412](LuciBase.py#L412), `fit_region` [L556](LuciBase.py#L556), `create_snr_map` [L1054](LuciBase.py#L1054), `wvt_fit_region` [L1482](LuciBase.py#L1482) |
| Filter→bounds `if/elif` chain duplicated 5× with divergent values | [LuciFit.py:236](luci/LuciFit.py#L236) (fit bounds), [LuciFit.py:298](luci/LuciFit.py#L298) (noise), [LuciUtility.py:275](luci/LuciUtility.py#L275) (ref-spectrum clip), [LuciBase.py:332](LuciBase.py#L332) + [L781](LuciBase.py#L781) + [L1809](LuciBase.py#L1809) (PCA scale) |
| `fit_calc` takes 30 positional/keyword params and returns a 13-tuple | [LuciBase.py:259](LuciBase.py#L259) |
| Fit results returned as an 18-key untyped dict | [LuciFit.py:844](luci/LuciFit.py#L844) |
| `from LUCI.LuciWVT import *` — star import pollutes `LuciBase` namespace (this is where `np` and `fits` actually come from) | [LuciBase.py:19](LuciBase.py#L19) |
| `quit()` / `exit()` called from library code on unsupported filter | [LuciBase.py:345](LuciBase.py#L345), [L793](LuciBase.py#L793), [LuciUtility.py:303](luci/LuciUtility.py#L303) |
| 91 `print()` calls, 10 bare `except:`, 4 module-level `global` statements | throughout |

### Real bugs found while reading (fix during the refactor, each with a regression test)

1. **`buond_upper` typo** — [LuciFit.py:321](luci/LuciFit.py#L321). For the normal-C3 branch of `calculate_noise`, `bound_upper` is never assigned. Because `bound_lower`/`bound_upper` are **module-level globals**, C3 silently inherits the noise window from whatever spectrum was fit previously — results depend on fit order. This is the single strongest argument for the filter registry.
2. **`NameError` in `fit_pixel` PCA path** — [LuciBase.py:801](LuciBase.py#L801) references `x_pix`/`y_pix`, which don't exist in that scope. Unbinned PCA background subtraction on a single pixel always crashes.
3. **`fit_region` never subtracts the background** — [LuciBase.py:694-707](LuciBase.py#L694-L707) passes `bkg=bkg` but not `bkgType`, so `fit_calc` sees `bkgType is None` and skips subtraction entirely. Silent wrong answer.
4. **`fit_region` drops `corr`/`step`** and doesn't forward `fit_function` to `save_fits`, so its outputs are named differently from `fit_cube`'s — [LuciBase.py:709-724](LuciBase.py#L709-L724).
5. **`fit_pixel` with `binning=None`** — [LuciBase.py:779](LuciBase.py#L779) computes `bkg * (binning) ** 2` → `TypeError`.
6. **Hardcoded `2048`/`2064`** in default args and in the PCA coefficient array — [LuciBase.py:1054](LuciBase.py#L1054), [L1913](LuciBase.py#L1913), [L630](LuciBase.py#L630). Breaks on any non-standard cube shape.

### Packaging / infrastructure

- [setup.py](setup.py) declares `packages=['distutils', 'distutils.command']` — the package has never been installable. Every user and every example does `sys.path.insert(0, '/home/carterrhea/Documents/LUCI/')`.
- `Luci_path` is threaded through **every** call as a string with a mandatory trailing `/` ([check_luci_path](luci/LuciUtility.py#L10) exists solely to paper over this) just to locate `ML/` and `Data/`.
- CI pins Python 3.9.5 + `tensorflow==2.11` via a 700-line frozen conda `luci.yml` (plus a stale `luci_macOS-11.6.yml`) with hashed build strings like `libgcc-ng=13.2.0=h807b86a_3` — unreproducible across platforms and impossible to update incrementally. You are on Python 3.13. `requirements.txt` separately lists the long-dead `sklearn` package and contradicts `luci.yml`.
- [tests/test_cube.py](tests/test_cube.py) hardcodes `/home/carterrhea/Documents/Luci_test/NGC6946_SN3.hdf5` and wraps fixtures in a `Test` class with an `__init__`, which pytest refuses to collect. CI has never actually exercised the code.
- **300 MB of generated output is tracked**: `Data/ExampleData/Luci_outputs/` — nine 33 MB FITS files that are *products* of running the example, not inputs. Plus 171 MB of ML artifacts. `.git` is 470 MB.

---

## Target architecture

```
luci/
  __init__.py              # public API: SitelleCube, FitConfig, FitResult, __version__
  config.py                # frozen dataclasses: FitConfig, BayesConfig, BackgroundConfig, ParallelConfig
  instrument/
    filters.py             # ★ FilterSpec registry — the 5 duplicated tables collapse to one
    header.py              # update_header, spectrum_axis_func, get_interferometer_angles, get_quadrant_dims
  io/
    hdf5.py                # quadrant assembly + new-format branch (from read_in_cube)
    reference.py           # reference spectrum + transmission loading
    outputs.py             # save_fits → FitMapWriter
    assets.py              # ★ replaces Luci_path: resolve ML/ and Data/ via package data + env var
  cube.py                  # SitelleCube: data, WCS, deep image, binning, spectrum extraction
  fitting/
    models.py              # Gaussian / Sinc / SincGauss (moved from LuciFunctions.py)
    parameters.py          # calculate_vel/broad/flux + errors (moved from LuciFitParameters.py)
    constraints.py         # ★ extracted from Fit: sigma/vel/NII/amplitude/multi-component constraints
    spectrum_fitter.py     # ★ replaces Fit — pure numerics, no I/O, no keras, no globals
    bayes.py               # emcee + dynesty (from LuciBayesian.py)
    result.py              # ★ FitResult dataclass replaces the 18-key dict
  engine/
    selection.py           # ★ PixelSelection: rectangle | ds9 region | mask | pixel list | WVT bins
    runner.py              # ★ ONE orchestrator — replaces all four duplicated blocks
  background/
    subtraction.py         # ★ standard + PCA subtraction, deduped from its 3 copies
    pca.py                 # create_background_subspace, split from the NN interpolator
    detection.py           # from LuciBackground.py
    interpolators.py       # griddata + NN interpolation (the only place keras may remain, optional)
  ml/
    base.py                # ★ ParameterPredictor protocol: predict(spectrum) -> PriorEstimate
    onnx_backend.py        # ★ onnxruntime inference (default)
    keras_backend.py       # legacy fallback, lazy import, only if TF present
    registry.py            # resolve (resolution, filter, mdn) -> model artifact
  analysis/                # snr.py, components.py, wvt.py, detection.py, slicing.py, skylines.py
  viz/                     # plotting.py, visualize.py
  compat.py                # Luci shim: old signatures → new API, emits DeprecationWarning
LuciBase.py                # 3-line shim: `from luci.compat import Luci`
```

Two abstractions carry most of the win:

**`FilterSpec`** ([luci/instrument/filters.py]) — one frozen dataclass per filter holding every
band-dependent number now scattered across five `if/elif` chains:

```python
@dataclass(frozen=True)
class FilterSpec:
    name: str
    fit_bounds: tuple[float, float]        # restrict_wavelength
    noise_bounds: tuple[float, float]      # calculate_noise
    reference_bounds: tuple[float, float]  # read_in_reference_spectrum
    pca_scale_bounds: tuple[float, float] | None
    redshift_scaled: bool                  # C1/C2/C4 multiply bounds by (1 + obj_redshift)

FILTERS: dict[str, FilterSpec] = {...}   # SN1-SN4, C1-C4
```

Unknown filter raises `UnsupportedFilterError` instead of `quit()`. Adding a filter becomes one dict
entry — which is exactly what [docs/source/newFilter.rst](docs/source/newFilter.rst) currently
documents as a multi-file surgery.

**`FitRunner`** ([luci/engine/runner.py]) — the single orchestration path:

```python
maps = FitRunner(cube, config).run(selection)   # selection: Rectangle | Region | Mask | PixelList | WVTBins
```

`fit_cube`, `fit_region`, `fit_entire_cube`, `wvt_fit_region`, and `create_snr_map` all become
~10-line methods that build a selection and delegate. The 12-array allocation, the joblib fan-out,
the 13-tuple unpack, and the `save_fits` call each exist exactly once.

---

## Phase 0 outcome (completed 2026-07-23)

The safety net is built and green: **75 passing tests** (63 fast + 12 golden) where CI previously ran
zero, plus 6 `xfail(strict=True)` markers pinning known bugs. Full catalogue in
[REFACTOR_BUGS.md](REFACTOR_BUGS.md); test-suite guide in [tests/README.md](tests/README.md).

What Phase 0 established, and how it reshapes the rest of the plan:

- **The legacy code will not run on a modern stack at all.** Bare `@jit` decorators fail to import on
  numba ≥0.59 (B11), and the new-format HDF5 path uses `np.str`/`np.bool_`, removed in numpy 1.24
  (B15). Goldens are therefore recorded in a pinned Python 3.10 / TF 2.11 / numba 0.58.1 env
  ([tests/requirements-legacy.txt](tests/requirements-legacy.txt)), built with `uv venv`. This env is
  disposable — Phase 1's `uv.lock` replaces it.
- **`tensorflow-probability` must be `0.19`, not `0.14`.** `requirements.txt`'s `0.14` is inconsistent
  with its own `tensorflow==2.11`. Correct the extras accordingly in Phase 1/4.
- **`ML_bool=False` is not a safe fallback — it fabricates zero kinematics (B1).** This is the most
  serious finding and it is promoted into Phase 3: the extracted `SpectrumFitter` must ship a real
  non-ML prior (data-driven line position + default broadening), because the code actively tells
  users to set `ML_bool=False` on unsupported filters. The golden matrix is deliberately **ML-on
  dominant** for this reason; ML-off baselines exist only to document the defect and must be
  re-recorded when B1 is fixed.
- **Nine further bugs were found and pinned** beyond the six in the original plan: B7 (deep image
  drops trailing rows), B8 (`bin_mask` ÷ binning²), B12 (`plt.clf()` per spectrum breaks headless),
  B13 (ML model reloaded from disk *per pixel* — 5.7 s/pixel), B14 (sinc-Gauss unstable at σ→0), B16
  (multi-component fits overwrite each other's output maps), plus B2 (`extract_spectrum(mean=True)` is
  a no-op) upgraded from suspicion to confirmed. These slot into existing phases; none needs a new
  phase.
- **Convention decisions surfaced** (C1–C4 in the register) that the `FilterSpec` work in Phase 3 must
  resolve deliberately rather than silently: the Gaussian-flux sinc factor, three divergent
  `line_dict` copies, six copies of the filter table, and SN1's noise window sitting inside its fit
  window.
- **B13 makes ML fits ~5 s/pixel**, so the golden matrix uses tiny (2×2–8×8) boxes. Fixing B13 in
  Phase 4 (predictor injected once, not per `Fit`) is now a stated success criterion, not a nice-to-have.

## Phases

Each phase is a separate PR onto a long-lived `refactor` branch. **The characterization suite must be
green at the end of every phase.**

### Phase 0 — Safety net (do this first, nothing else until it's done)

Nothing here changes production code.

1. **Synthetic cube fixture** — `tests/fixtures/make_cube.py` writes a small (64×64×~840) HDF5 file in
   old SITELLE quadrant format: `quad_nb`/`dimx`/`dimy`/`dimz` attrs, `quad00N/data` datasets, a
   `header` structured array, and a `calib_map`. Seed known Gaussian/sinc lines at known velocities
   and broadenings so tests assert on recoverable truth, not just "it ran". Required header keys,
   from [update_header](luci/LuciUtility.py#L168) and [spectrum_axis_func](luci/LuciUtility.py#L143):
   `STEPNB`, `ZPDINDEX`, `FILTER`, `STEP`, `ORDER`, `CRVAL3`, `CDELT3`, `CRVAL1`, `CRVAL2`,
   `DATE-OBS`, `CALIBNM`. Commit the generator, not the cube; build it in a session-scoped fixture.
2. **Golden-value tests** — run the *current* code over the fixture for the cross product of
   {`gaussian`, `sinc`, `sincgauss`} × {SN1, SN2, SN3} × {`ML_bool=True/False`} × {binning 1, 2} and
   snapshot `fit_sol`, `fluxes`, `velocities`, `sigmas`, `chi2`, `continuum` to JSON under
   `tests/golden/`. Assert with `np.testing.assert_allclose(rtol=1e-6)`.
3. **Output-contract tests** — assert the exact set of files `save_fits` writes and their header
   keys, so directory layout can't silently drift.
4. **Pure-function tests** — direct unit tests for `LuciFitParameters`, `LuciFunctions`,
   `get_quadrant_dims`, `bin_cube_function`, `bin_mask`, `hessianComp`. These have no I/O and are the
   cheapest coverage available.
5. Delete the uncollectable `Test` class in [tests/test_cube.py](tests/test_cube.py); rewrite as
   pytest fixtures.

> Golden values are recorded from code containing the bugs listed above. Record them anyway, then
> update each affected golden file **in the same commit that fixes the bug**, with the diff visible
> in review. That is the whole point of the safety net.

### Phase 1 — Packaging and tooling (conda → uv) — DONE (2026-07-23)

Outcome: LUCI is pip-installable (`uv sync`); imports resolve from anywhere with **zero `sys.path`
hacks** (verified from `/tmp`). CI rewritten from miniconda to `astral-sh/setup-uv`. 63 fast tests
green under the uv-managed `.venv`, whose pins are byte-identical to the golden-recording env, so the
goldens hold unchanged.

Delivered: [pyproject.toml](pyproject.toml) (hatchling, packages `LUCI` + top-level `LuciBase`/
`LuciAbsorp`), committed [uv.lock](uv.lock), new [luci/__init__.py](luci/__init__.py), rewritten
[.github/workflows/python-app.yml](.github/workflows/python-app.yml), [.pre-commit-config.yaml](.pre-commit-config.yaml),
ruff config. Deleted `setup.py`, `requirements.txt`, `luci.yml`, `luci_macOS-11.6.yml`, `pytest.ini`.

Honest deviations from the plan as originally written, all forced by the same root cause — the code
**hard-imports keras/TF at module load** (5 sites), and TF 2.11 has no wheels past Python 3.10:

- **CI runs Python 3.10 only** for now (plus a lint job on any Python via `uvx ruff`, and a nightly
  golden-fit job). The 3.11/3.12/3.13 matrix moves to **Phase 4**, when dropping the TF hard-import
  unblocks it. Adding those rows now would just be permanently-red entries.
- **TF/tensorflow-probability stay core deps**, not an `[ml-legacy]` extra — the code can't import
  without them until Phase 4 makes ML lazy. `requires-python = ">=3.10,<3.11"` reflects this and
  widens in Phase 4.
- **`tensorflow-probability` pinned to `0.19`**, not the `requirements.txt` value of `0.14` (which was
  inconsistent with `tensorflow==2.11`).
- **Ruff scope is the refactored code only** (tests today). The legacy modules are excluded via
  `extend-exclude` and re-included one at a time as each phase rewrites them — a ratchet, not a
  190-finding wall on day one.
- **B11's `numba==0.58.1` and B15's `numpy<2` are pinned in `pyproject.toml`**, with comments tying
  each to its bug. Phase 3 removes the `@jit` decorators and drops the numba pin.

Original plan detail (still the intended shape), retained for reference:

**uv becomes the only environment manager.** `pyproject.toml` declares dependencies; `uv.lock`
(committed) pins the exact resolution for every platform. Conda is removed entirely.

- `pyproject.toml` with hatchling backend, flat `luci/` layout, `requires-python = ">=3.10"`, and
  real `[project.dependencies]` — reconciling the currently-contradictory `requirements.txt` and
  `luci.yml`.
- Dependency groups so the heavy stack is opt-in:
  ```toml
  [project.optional-dependencies]
  ml-legacy = ["tensorflow", "tensorflow-probability"]   # dropped from core in Phase 4
  viz       = ["seaborn", "corner", "photutils"]
  [dependency-groups]
  dev = ["pytest", "pytest-cov", "ruff", "nbval"]
  ```
- Core deps: `numpy`, `scipy`, `astropy`, `h5py`, `joblib`, `tqdm`, `pandas`, `scikit-learn`,
  `emcee`, `dynesty`, `pyregion`, `matplotlib`, `numba`. Drop `sklearn` (dead shim package) for
  `scikit-learn`; unpin `emcee==3` and `tensorflow==2.11`. `onnxruntime` joins core in Phase 4.
- **Delete** `setup.py`, `requirements.txt`, `luci.yml`, `luci_macOS-11.6.yml`.
- `uv lock` → commit `uv.lock`. Developer onboarding collapses to:
  ```bash
  uv sync            # creates .venv, installs luci editable + dev group
  uv run pytest
  ```
  This replaces the current README instructions to build a conda env from a 700-line YAML.
- `.gitignore`: `Luci_outputs/`, `__pycache__/`, `*.pkl`, `.venv/`.
- Ruff (lint + format) + `pre-commit`, both run via `uv run`. Start permissive; tighten per phase.
- **CI rewrite** — [.github/workflows/python-app.yml](.github/workflows/python-app.yml) drops
  `conda-incubator/setup-miniconda` for `astral-sh/setup-uv` with caching:
  ```yaml
  strategy:
    matrix:
      python-version: ["3.10", "3.11", "3.12", "3.13"]
  steps:
    - uses: actions/checkout@v4
    - uses: astral-sh/setup-uv@v5
      with: { enable-cache: true }
    - run: uv sync --locked --all-extras --dev
    - run: uv run ruff check luci/
    - run: uv run pytest -v
  ```
  Add `uv run --locked` so CI fails loudly if `uv.lock` drifts from `pyproject.toml`.
  Python 3.13 is included from the start — it's what you develop on, and TF (the only thing that
  blocked it) leaves the core in Phase 4. If any 3.13 wheel is missing during Phases 1–3, mark that
  matrix entry `continue-on-error` rather than capping the ceiling.
- The Phase 4 ONNX conversion needs the *old* TF stack, which won't resolve on modern Python. uv
  handles this without a second tool — the conversion script gets an inline
  [PEP 723](https://peps.python.org/pep-0723/) header and runs in its own ephemeral env:
  ```python
  # /// script
  # requires-python = "==3.10.*"
  # dependencies = ["tensorflow==2.11", "tensorflow-probability==0.14", "tf2onnx", "onnxruntime"]
  # ///
  ```
  then `uv run tools/convert_models_to_onnx.py` — no conda, nothing installed into the project env.
- `luci/io/assets.py` replaces `Luci_path`: models and filter data resolve from package data, then
  `$LUCI_DATA_DIR`, then a cached download. `Luci_path` remains an accepted-but-deprecated argument.

### Phase 2 — Repo weight — DONE (2026-07-23)

Outcome: **~344 MB untracked** (working-tree checkout 550 → 208 MB), fully non-destructive — every
file stays on disk, nothing rewritten. 539 files removed from the index, all gitignored so they can't
creep back. Tests still green (63/63). `.git` stays ~470 MB (history rewrite deferred, as planned) —
so fresh *checkouts* shrink, `.git` does not.

What was removed and why it was safe:
- **`Data/ExampleData/Luci_outputs/` — 310 MB** of fit products (velocity/flux/broadening/amplitude
  maps, slices, deep frame). Pure output, regenerated by running the example; the input cube was
  already gitignored (`*.hdf5`). The single largest win.
- **6 generated map PNGs** at the ExampleData root (~0.8 MB), regenerable, referenced by nothing.
- **`docs/build/` (19 MB) + `docs/_build/` (14 MB)** — two stale copies of Sphinx output. Confirmed
  with the user that GitHub Pages serves the *flattened* HTML at the `docs/` root
  (`docs/index.html` + `docs/*.html` + `docs/_static`), and verified no served page links into either
  build tree, so both are dead weight. `docs/source/` and the served docs are untouched.

gitignore additions: `Luci_outputs/`, `Data/ExampleData/*_map.png`, `*.pkl`, `docs/build/`,
`docs/_build/`, `.pytest_cache/`, `*.egg-info/`, `build/`.

**ML artifacts (171 MB) deliberately deferred to Phase 4, not externalized here.** Phase 4 replaces
the Keras models with ONNX (~17 MB, 10×), so building a downloader + checksum manifest + Release
uploads for the Keras models now is throwaway work — and untracking them before Releases exist would
break every fresh clone. At ~17 MB, ONNX may be small enough to ship in-repo as package data, which
dissolves the download-on-demand question entirely. The `luci/io/assets.py` resolver and any Release
externalization move to Phase 4, decided against the *final* artifacts. Kept-tracked inputs verified
intact: `ML/` predictors, `Data/*.dat` filter/transmission tables, `regions/*.reg`.

Original plan detail retained for reference:

- **`git rm -r --cached Data/ExampleData/Luci_outputs/`** and gitignore it — ~300 MB of regenerated
  output that never belonged in git. This alone is the largest single win and is non-destructive.
- Move `Data/ExampleData/` inputs and the `ML/` artifacts to GitHub Release assets, fetched on demand
  by `luci/io/assets.py` (pooch, or a small stdlib downloader with checksums). Ship only the small
  `Data/*_filter.dat` / `*_Transmission.dat` files as package data.
- Audit `docs/` (60 MB, 762 files) for unreferenced images.

> **History rewrite is explicitly deferred.** Untracking shrinks new clones' working tree but not
> `.git` (still ~470 MB). A `git filter-repo` pass would fix that but invalidates every fork, clone,
> and commit SHA — including any cited in papers. Decide separately, after everything else lands.

### Phase 3 — Filter registry and the extracted `SpectrumFitter`

**Progress (2026-07-23): 3a and 3b DONE, golden-verified. 3c (Fit split + B1/B14) remains.**

- **3a — FilterSpec registry — DONE.** [luci/instrument/filters.py](luci/instrument/filters.py) replaces
  the five duplicated filter→bounds chains. `restrict_wavelength`, `calculate_noise`, and
  `read_in_reference_spectrum` route through it; all four `global bound_lower, bound_upper` statements
  deleted; `exit()`/`print()` on unknown filter replaced by `UnsupportedFilterError`. **Fixes B4.**
  23 registry unit tests + 12/12 goldens byte-identical. (Staged under `luci/instrument/` not `luci/`
  to avoid the macOS `LUCI`-vs-`luci` case collision; the lowercase rename happens atomically at the end.)
- **3b — numba removed — DONE.** All 25 bare `@jit(fastmath=True)` object-mode no-ops deleted; numba +
  llvmlite dropped from deps and lock. **Fixes B11.** 12/12 goldens byte-identical; fast suite ~2.5×
  faster, warnings 126→46. numpy<2 remains the only load-bearing pin (B15).
- **3c-i — B1 fix — DONE.** `estimate_priors_data()` seeds the ML-off / non-frozen path from the
  brightest peak + `DEFAULT_BROADENING_KMS=50`, so the optimiser never starts at the singular σ=0.
  ML-off now recovers physics (sincgauss 99.5/30.2, gaussian 99.5/30.3, were 0/0). The two
  `*_broken_zeros` goldens were renamed `*_noml` and re-recorded to the recovered values; the 10 ML-on
  baselines stayed byte-identical (12/12). The B1 xfail flipped to a passing test.
- **3c-ii — FitResult + B14 — DONE.** [luci/fitting/result.py](luci/fitting/result.py): `FitResult`
  dataclass replaces the 22-key untyped dict `fit()` returned. It stays read-compatible
  (`result['velocities']` still works via `__getitem__`), so no caller changed; `fit()` returns it and
  12/12 goldens stayed byte-identical. **B14** turned out to be a Phase-0 mischaracterization —
  the sinc-Gauss is finite for all non-zero σ; only σ==0 exactly gives NaN (the downstream mechanism of
  B1). Guarded with `SINCGAUSS_SIGMA_FLOOR`; fires only at exact zero, goldens byte-identical. Register
  corrected.
- **3c-iii — structural rename — PARTIALLY DONE (maintainability only).**
  - **DONE:** Gaussian/Sinc/SincGauss → [luci/fitting/models.py](luci/fitting/models.py) and
    calculate_* → [luci/fitting/parameters.py](luci/fitting/parameters.py), both with re-export shims
    ([LuciFunctions.py](luci/LuciFunctions.py), [LuciFitParameters.py](luci/LuciFitParameters.py) now
    3-line shims). Verified byte-identical (12 goldens + 31 pure-function tests). Both new modules are
    now lint-clean (bare `except:` → `except Exception:`, stray semicolon removed).
  - **DEFERRED (optional, low value):** extracting the constraint methods to `fitting/constraints.py`,
    `FitConfig` to replace the 25-arg `Fit.__init__`, and the `Fit` → `SpectrumFitter` rename. These
    are the most invasive and least valuable pieces (pure cosmetics; the compat alias means both names
    coexist anyway). Best folded into **Phase 6** (compat layer), where the API surface is redesigned
    regardless — doing the rename there avoids double-churn. No bug fixes ride on them.

Original detail follows:


- Build `luci/instrument/filters.py` and route `restrict_wavelength`, `calculate_noise`,
  `read_in_reference_spectrum`, and the PCA scaling windows through it. **Fixes bug 1.** Delete all
  four `global bound_lower, bound_upper` statements.
- Split `Fit` into `SpectrumFitter` + `constraints.py` + `bayes.py`. `SpectrumFitter.__init__` takes
  a `FitConfig` dataclass instead of 25 keyword arguments, does no work beyond validation, and holds
  no keras reference — the ML prior arrives as an injected `ParameterPredictor`.
- **Fix B1 here**: give the non-ML path a working prior. When no predictor is supplied, estimate the
  line position from the data (argmax near the rest wavelength) and default the broadening to a
  sensible instrumental value, so the optimiser never starts from the singular σ=0 point. The freeze
  path already does exactly this and works (`test_freezing_supplies_priors_without_the_ml_model`) —
  reuse it. Re-record the `*_broken_zeros` golden baselines in this commit and flip the
  `test_ml_disabled_should_still_recover_the_injected_velocity` xfail.
- Fix B14 (guard or reformulate the sinc-Gauss at small σ) while the line models move to
  `fitting/models.py`; it compounds B1.
- Introduce `FitResult` and return it from `fit()`. `FitResult.as_dict()` preserves the old 18-key
  dict for the compat layer and for [LuciConvenience.get_individual_components](luci/LuciConvenience.py#L10).
- Remove the `@jit(fastmath=True)` decorators on `apply_transmission`, `calculate_correction`,
  `calc_sinc_width`, `interpolate_spectrum`. These are numba object-mode fallbacks on pure-Python
  list comprehensions and scipy calls — they add import cost and warnings and compile nothing useful.
  Benchmark before/after to confirm.

### Phase 4 — ONNX migration

The two model families need different treatment:

- **Standard predictors** (`R*-PREDICTOR-I-{SN1..SN4,C1..C4}`) are TF SavedModels →
  `python -m tf2onnx.convert --saved-model <dir> --output <name>.onnx --opset 17`. Direct.
- **MDN predictors** (`R*-PREDICTOR-I-MDN-*`) are **weights-only checkpoints**
  (`.index` + `.data-00000-of-00001`), not SavedModels, and their head is a
  `tfp.layers.IndependentNormal(2)` on top of `Dense(units=4)`
  ([LuciNetwork.py:47-48](luci/LuciNetwork.py#L47-L48)). TFP distribution layers have no ONNX
  equivalent. **The fix is straightforward**: rebuild the architecture via `create_MDN_model`, load
  the checkpoint, then export the model *truncated at the `Dense(4)` layer* — that layer's output is
  exactly `[loc_vel, loc_broad, scale_raw_vel, scale_raw_broad]`. Reproduce the distribution in numpy:
  `mean = loc`, `stddev = softplus(scale_raw) + 1e-5`, matching `IndependentNormal.new()`. The
  `estimate_priors_ML` contract ([LuciFit.py:347](luci/LuciFit.py#L347)) needs only mean and stddev,
  so nothing is lost.
- Conversion lives in a **one-shot script** (`tools/convert_models_to_onnx.py`), never in the
  runtime. It carries a PEP 723 inline-dependency header pinning `python==3.10` + `tensorflow==2.11`
  + `tensorflow-probability==0.14` + `tf2onnx`, so `uv run tools/convert_models_to_onnx.py` builds
  that legacy environment on the fly and discards it. The project's own `uv.lock` never sees TF.
- **Validation gate**: for every one of the 137 artifacts, feed ≥100 spectra from the fixture through
  both the Keras and ONNX paths and require `assert_allclose(rtol=1e-5)`. Any model that fails is
  reported, not silently shipped.
- `luci/ml/base.py` defines `ParameterPredictor`; `onnx_backend.py` is the default,
  `keras_backend.py` a lazy-imported fallback so old artifacts still work if someone has TF.
  `ML_bool=False` maps to a null predictor rather than the current `self.ML_model = ''` sentinel
  ([LuciFit.py:811](luci/LuciFit.py#L811)).
- Move `ML/TrainPredictor.py` to `tools/` — it's a training script, not library code.
- **Only after the validation gate passes**, drop `tensorflow` / `tensorflow-probability` /
  `keras` from core dependencies into an optional `[ml-legacy]` extra.

### Phase 5 — `SitelleCube` and the unified `FitRunner`

- Split `Luci` → `SitelleCube` (data + geometry + extraction) and `FitRunner` (orchestration).
- Implement `PixelSelection` and collapse `fit_cube` / `fit_region` / `fit_entire_cube` /
  `wvt_fit_region` / `create_snr_map` onto the single runner. **Fixes bugs 3 and 4.**
- Extract background handling into `luci/background/subtraction.py`, deduplicating the PCA block
  currently copied at [LuciBase.py:331](LuciBase.py#L331), [L780](LuciBase.py#L780), and
  [L1809](LuciBase.py#L1809). **Fixes bugs 2 and 5.**
- Replace hardcoded `2048`/`2064` with `cube.shape`. **Fixes bug 6.**
- Lift the inline Keras training out of `create_background_subspace`
  ([LuciBase.py:1908-1950](LuciBase.py#L1908-L1950)) into
  `luci/background/interpolators.py` behind a strategy interface, so `'nn'` sits alongside
  `'linear'`/`'nearest'` rather than being 40 lines of hyperparameters inside a 200-line method.
- Split the god-class remainder into `luci/analysis/` (SNR, components, WVT, detection map, slicing,
  skyline calibration) and `luci/viz/`.

### Phase 6 — Compat shim and API surface

- `luci/compat.py` provides `Luci` with the exact old signatures, translating to the new API and
  emitting a `DeprecationWarning` naming the replacement. Top-level `LuciBase.py` re-exports it.
- **Test the shim against the Phase 0 goldens** — that is what makes "clean break" safe.
- Replace all 91 `print()` calls with a `luci` logger; keep `tqdm` for progress.
- Replace the 10 bare `except:` clauses with specific exceptions.
- Public API gets type hints and a documented `__all__`.

### Phase 6 — package rename, compat alias, and API surface — DONE (2026-07-23)

The package is now lowercase `luci/`. A shim *package* named `LUCI` is impossible — macOS and Windows
cannot hold both spellings — so [LUCI.py](LUCI.py) registers the alias in `sys.modules` via a
meta-path finder inserted at position 0, ahead of the path finder. That ordering matters: without it
the path finder would import `luci/cube.py` a second time as `LUCI.cube`, giving two module objects
with independent state. Tests assert `sys.modules["LUCI.cube"] is sys.modules["luci.cube"]` and that
mutating one is visible through the other.

`luci/__init__.py` gained a real public API (`SitelleCube`, `FitConfig`, `FitResult`, `FILTERS`, ...)
resolved lazily through PEP 562 `__getattr__`, so `import luci` does not drag in scipy/onnxruntime.

**61 `print()` calls became logger calls** through [luci/log.py](luci/log.py), converted with an AST
script rather than regex so multi-line and multi-argument calls kept their meaning (`print(a, b)`
became `logger.info("%s %s", a, b)`, not `logger.info(a, b)` — which would have treated `b` as a
format argument). The logger attaches a handler on first use *unless* the application already
configured logging, so interactive output still appears by default but never fights a host app.

**All 8 remaining `exit()`/`quit()` calls in library code became exceptions.** Several of those sites
were worse than uncatchable: `calculate_flux_err` evaluated its error message as a bare expression
(never printed) before exiting, and three sites logged a message then *continued* with a variable
left unbound, converting a clear input error into a confusing `NameError` further downstream.

The last 5 bare `except:` clauses were narrowed. One of them was hiding **B24**, an infinite loop.
**B25** (unimportable `luci.simulation`) surfaced while writing the tests.

### Phase 7 — Docs and examples — DONE (2026-07-23)

- **`Luci_path` is now optional.** [luci/io/assets.py](luci/io/assets.py) resolves it from
  `$LUCI_DATA_DIR`, else from the installed package's own location. This was the single most
  copy-pasted line in the project — every example opened by hardcoding an absolute path into
  someone else's home directory. Passing it positionally still works, so nothing breaks.
- **All 23 notebooks and 17 `.rst` pages updated**: `sys.path.insert` removed everywhere (it was in
  every single notebook), `Luci` → `SitelleCube`, `LUCI.LuciFit` → `luci.fitting.spectrum_fitter`,
  `LUCI.LuciSim` → `luci.simulation`. Every notebook re-verified as valid JSON with every code cell
  parsing.
- **[docs/source/migration.rst](docs/source/migration.rst)** — new: full old→new mapping plus a
  "what changed in the results" section naming B21, B1, B3 and B23, since those move published
  numbers.
- **[newFilter.rst](docs/source/newFilter.rst) Step 3 rewritten** around the single `FILTERS` entry.
  It previously listed **nine** functions that each had to be edited to add a filter, and warned
  that missing any one of them caused a silent fallthrough. That table is now one dataclass.
- Docs build verified with Sphinx; `migration` and `newFilter` build without warnings.

**Not done:** the `nbval` CI smoke-run of two notebooks. It needs an example cube (~900 MB from
CADC), which is the same prerequisite as the end-to-end validation below — worth doing once, together.

### Phase 7 — Docs and examples

- Update the ~25 `docs/source/*.rst` files and 25 notebooks in [Examples/](Examples/) to
  `uv add luci` / `pip install luci` + `from luci import SitelleCube`, dropping every
  `sys.path.insert`.
- Rewrite the [README.md](README.md) install section: the conda-env instructions become
  `uv sync` (contributors) or `uv add luci` / `pip install luci` (users). Notebooks run under
  `uv run jupyter lab`.
- Rewrite [newFilter.rst](docs/source/newFilter.rst) around the single `FILTERS` dict entry.
- Add a migration guide mapping every old method to its replacement.
- Add `nbval` or `jupyter nbconvert --execute` smoke runs for two representative notebooks in CI,
  using the downloaded example cube.

---

## Verification

**Per phase (must be green before merge):**
```bash
uv run pytest tests/ -v                       # incl. golden comparisons
uv run ruff check luci/ && uv run ruff format --check luci/
uv lock --check                               # lockfile matches pyproject.toml
```

**Phase 0 baseline capture** (before uv lands — plain `python` against the existing env):
```bash
python tests/fixtures/make_cube.py --out /tmp/luci_fixture.hdf5
pytest tests/test_golden.py --record          # writes tests/golden/*.json from current code
```

**Phase 4 ONNX gate** (legacy TF env is built and torn down by uv from the script's PEP 723 header):
```bash
uv run tools/convert_models_to_onnx.py --all --validate --rtol 1e-5
# fails loudly per-model; prints a conversion report table
```

**Phase 6 shim equivalence:**
```bash
uv run pytest tests/test_compat.py -v    # old API on new internals vs. Phase 0 goldens
```

**End-to-end on real data** (the honest final check — the fixture is synthetic):
run `Examples/Fit-Single-Spectrum.ipynb` and `Examples/BasicExample.ipynb` against the NGC628 SN3
cube on both `main` and `refactor`, and diff the output FITS maps pixel-by-pixel. Velocity and
broadening maps should agree to floating-point noise except where a listed bug fix changes them —
and each of those differences should be explainable by name.

**Install check — the clean-clone test** (this is what a new collaborator experiences):
```bash
git clone <repo> /tmp/luci-clean && cd /tmp/luci-clean
uv sync                                       # must succeed with no conda, no manual steps
uv run python -c "import luci; print(luci.__version__)"
uv run python -c "import luci, sys; assert 'tensorflow' not in sys.modules"   # after Phase 4
```

---

## Risks and non-goals

**Risks**
- *ONNX numerical drift.* Mitigated by the per-model `rtol=1e-5` gate. If a model fails, it stays on
  the Keras backend and is flagged rather than shipped broken.
- *The synthetic fixture won't exercise everything.* Real SITELLE cubes have instrument quirks
  (calibration maps, quadrant seams, NaN patterns) the fixture won't reproduce. The real-data diff in
  the verification section is the backstop, not the fixture.
- *Long-lived branch drift.* Merge each phase to `main` as it lands; don't let `refactor` run for
  months.
- *Conda→uv transition friction.* Existing collaborators (and any Compute Canada jobs — see
  [ComputeCanada/run_SN3.py](ComputeCanada/run_SN3.py)) have conda envs and module-load scripts built
  around `luci.yml`. Mitigation: land uv in Phase 1 while the old code still works unchanged, so the
  environment change and the API change are never debugged at the same time. On HPC, uv installs as a
  single static binary with no admin rights and honours `$UV_CACHE_DIR` for shared filesystems —
  verify on the actual cluster before deleting `luci.yml`.
- *Bug fixes change published results.* Every fix is isolated to its own commit with a golden-file
  diff, so any downstream user can see exactly what moved and by how much.

**Explicit non-goals**
- No git history rewrite (deferred; destructive to forks and citations).
- No retraining of ML models.
- No change to the fitting algorithms, likelihood, constraints, or Bayesian machinery beyond
  relocation — modulo the six listed bugs.
- No performance work beyond removing the useless `@jit` decorators. Optimization comes after
  correctness is pinned.
