# Bug register

Defects found while building the Phase 0 safety net for the refactor.

Every entry is **pinned by a test**. The pattern is:

- a test asserting *current* behaviour, so it cannot silently get worse; and
- where the intended behaviour is clear, a `@pytest.mark.xfail(strict=True)` test
  stating it. `strict=True` means that test **fails the suite if it starts
  passing** — so when someone fixes the bug, the suite tells them to delete the
  xfail marker. Nothing gets fixed by accident and nothing stays broken quietly.

When you fix one: remove the xfail, update the "pins current behaviour" test to
assert the corrected result, and re-record any affected golden baseline **in the
same commit**, so the numeric change is visible in review.

Severity is about the risk of a *wrong scientific result reaching a paper*, not
about how hard it is to fix.

---

## S1 — Silently wrong science

### B1. `ML_bool=False` returns all-zero velocities and broadenings — FIXED (Phase 3c)
**Where (was):** `Fit` prior initialisers + `line_vals_estimate`.
**Fixed by:** `LuciFit.estimate_priors_data()`, called from `fit()` on the ML-off / non-frozen path.
It locates the brightest peak in the fit window, derives a coarse velocity from it, and defaults the
broadening to `DEFAULT_BROADENING_KMS = 50.0` — so the optimiser never starts from the singular σ=0.
**Tests:** `tests/test_ml_priors.py::test_ml_disabled_now_recovers_kinematics` (+ siblings); the two
`sn3_*_noml` golden baselines were re-recorded from all-zeros to recovered physics in this commit, and
the 10 ML-on baselines stayed byte-identical (12/12 green).
**Recovered on the fixture:** sincgauss vel 99.5 / broad 30.2, gaussian 99.5 / 30.3 (were 0.0 / 0.0).

`Fit.__init__` set `vel_ml = broad_ml = 0.0` and nothing updated them without ML, so
`line_broad_est = line_pos * broad_ml / c` was exactly 0.0 — a singular point of the Gaussian and
sinc-Gauss profiles. SLSQP could not move, so every pixel reported velocity 0.0 and broadening 0.0
while amplitude/continuum still fit, making the output look normal. The code actively steered users
here (it printed *"Please set ML_bool=False"* for filters without a trained predictor), so the silent
failure was easy to hit. `sinc` was unaffected for velocity (its width is the fixed instrumental
`sinc_width`); the fix covers all three models.

### B2. `extract_spectrum(mean=True)` is a no-op — FIXED (Phase 5)
**Where (was):** `Luci.extract_spectrum`
**Fixed by:** counting every contributing spaxel instead of incrementing only inside the axis-init
guard. `mean=True` now genuinely averages.
**Tests:** `tests/test_cube.py::test_extract_spectrum_mean_divides_by_pixel_count` and
`::test_extract_spectrum_and_region_agree_on_the_mean`.

```python
integrated_spectrum += sky[~np.isnan(sky)]
if spec_ct == 0:
    axis = self.spectrum_axis[~np.isnan(sky)]
    spec_ct += 1          # only ever incremented inside its own init guard
...
if mean:
    integrated_spectrum /= spec_ct   # always divides by 1
```

The counter never advances past 1, so the "mean" spectrum equals the sum. The
docstring says the method is *"primarily used to extract background regions"* —
so a background averaged over N pixels comes back N times too large, and feeding
it to `fit_cube(bkg=...)` over-subtracts by a factor of N.

The sibling `extract_spectrum_region` increments correctly. The two functions
disagree, which is pinned by
`test_extract_spectrum_region_mean_does_divide_by_pixel_count`.

### B3. `fit_region` never subtracts the background — FIXED (Phase 5)
**Where (was):** `Luci.fit_region`
**Fixed by:** adding a `bkgType='standard'` parameter to `fit_region` and forwarding it to
`fit_calc`. The standard path is also now guarded on `bkg is not None`, so `bkgType` can be passed
unconditionally without tripping over `sky -= None`.
**Test:** `tests/test_background_and_regions.py::test_fit_region_subtracts_the_background`.

`fit_region` forwarded `bkg` but never `bkgType`, and `fit_calc` gates subtraction on `bkgType`, so
the background was silently ignored — an unsubtracted fit with no warning. Measured on the fixture,
the fix takes the fitted Halpha flux from 1.14e-16 to 4.27e-18 when a background is supplied.

> Note for anyone writing a test like this: the fluxes are ~1e-16, and `np.allclose` defaults to
> `atol=1e-8`, which calls *any* two such arrays equal. The first version of this test passed
> vacuously against the fixed code. Use `atol=0`.

### B4. C3 noise window leaks across fits via a module global — FIXED (Phase 3)
**Where (was):** `LuciFit.calculate_noise` — `buond_upper` typo
**Fixed by:** the filter registry [LUCI/instrument/filters.py](LUCI/instrument/filters.py). C3 normal
noise is now `(20000, 20250)`; all four `global bound_lower, bound_upper` statements are deleted and
`restrict_wavelength` / `calculate_noise` / `read_in_reference_spectrum` route through the registry.
**Tests:** `tests/test_filters.py::test_c3_normal_noise_upper_bound_is_now_defined` plus the SN
byte-identical checks; goldens re-verified green (12/12).

The normal-C3 branch of `calculate_noise` misspelt `bound_upper` as `buond_upper`, so it was never
assigned. Because the bounds were **module-level globals**, C3 inherited the noise window from
whichever spectrum was fitted previously — order-dependent results, and an undefined bound on a
single-spectrum fit. Collapsing the five duplicated filter→bounds chains into one registry removed
the whole class of bug.

---

## S2 — Crashes and wrong output on valid input

### B5. `fit_pixel` PCA path raises `NameError` — FIXED (Phase 5)
**Where (was):** the unbinned branch of `Luci.fit_pixel`'s PCA path
**Fixed by:** using `pixel_x` / `pixel_y`, the coordinates the method actually receives.
**Test:** covered by the `fit_pixel` tests in `tests/test_background_and_regions.py`.

It referenced `x_pix` / `y_pix`, which do not exist in that scope, so unbinned PCA background
subtraction on a single pixel always died with `NameError`.

### B6. `fit_pixel` crashes with `binning=None` — FIXED (Phase 5)
**Where (was):** `Luci.fit_pixel`
**Fixed by:** treating an unbinned pixel as one spaxel and only subtracting when a background was
supplied. An unknown `bkgType` now raises `ValueError` instead of printing and continuing.
**Test:** `tests/test_background_and_regions.py::test_fit_pixel_works_with_default_arguments`.

`sky -= bkg * (binning) ** 2` with the documented default `binning=None` raised `TypeError`, so the
function's own defaults could not be used together.

### B7. `create_deep_image` silently drops trailing rows — FIXED (Phase 5)
**Where (was):** `Luci.create_deep_image`
**Fixed by:** deriving the slab count from the step size so the loop covers the whole cube.
**Test:** `tests/test_cube.py::test_deep_image_covers_cubes_whose_height_is_not_a_multiple_of_ten`
(parametrised over dimx = 24, 25, 33).

The cube was summed in exactly ten slabs of `int(shape[0] / 10)` rows, so when the x-dimension was not
divisible by 10 the remainder stayed zero — on a real 2048-row cube, **the last 8 rows of every deep
image were blank**.

### B8. `bin_mask` divides a boolean mask by `binning ** 2` — FIXED (Phase 5)
**Where (was):** `LuciUtility.bin_mask`
**Fixed by:** dropping the stray division and returning a genuine boolean array.
**Tests:** `tests/test_pure_functions.py::test_bin_mask_returns_a_boolean_mask` (which also checks it
is usable for boolean indexing, which the float version was not).

It built a correct boolean mask, then ran `binned_mask = binned_mask / (binning ** 2)` — copy-pasted
from the flux-averaging path in `bin_cube_function` — turning `True` into `0.25` and the result into a
float array. It survived only because its one consumer tests `if mask[x, y]:` and 0.25 is truthy.

### B9. `fit_region` outputs are named differently from `fit_cube`'s — FIXED (Phase 5)
**Where (was):** `Luci.fit_region`'s `save_fits` call
**Fixed by:** forwarding `fit_function`, so `fit_region` and `fit_cube` name their products
identically.
**Test:** `tests/test_background_and_regions.py::test_fit_region_names_outputs_like_fit_cube`.

Without it the two entry points wrote different filenames for the same fit, and downstream scripts
globbing for outputs silently missed them.

### B16. Multi-component fits overwrite each other's output maps — FIXED (Phase 5)
**Where (was):** `LuciUtility.save_fits`
**Fixed by:** actually appending to `lines_fit`, so repeated line names get `_2`, `_3`, ... suffixes.
**Tests:** `tests/test_output_contract.py::test_duplicate_line_names_get_a_component_suffix` and
`::test_three_components_are_numbered_sequentially` (each asserts the right component landed in the
right file, not merely that the files exist).

`save_fits` intended to disambiguate repeated line names, but `lines_fit` was never appended to, so
`.count()` was always 0 and the renaming branch was dead code. Every component wrote the same
filename and the second silently overwrote the first, leaving only the last. This is the concrete
failure behind the `# TODO: Only works for 2 components` note in
[LuciConvenience.py:32](LUCI/LuciConvenience.py#L32).

### B10. Hardcoded `2048` / `2064` cube dimensions — FIXED (Phase 5)
**Where (was):** `create_snr_map` / `calculate_component_map` default arguments, three `header.set`
blocks, the Voronoi bin map, and the PCA coefficient array.
**Fixed by:** taking every extent from `self.cube_final.shape`. The two default arguments became
`None` and resolve to the cube's extent.
**Tests:** `tests/test_background_and_regions.py::test_snr_map_defaults_to_this_cubes_extent` and
`::test_no_hardcoded_detector_dimensions_remain`, which greps `LuciBase.py` so the pattern cannot
quietly return.

Defaults and array shapes assumed the standard SITELLE detector, so anything else — a trimmed cube, a
test fixture, a future detector — silently mis-indexed or truncated.

### B18. Editable installs serve a stale copy of the top-level modules — MITIGATED (Phase 5f)
**Where:** `[tool.hatch.build.targets.wheel.force-include]` in `pyproject.toml`
**Test:** `tests/test_packaging.py::test_installed_top_level_module_is_not_stale`

Introduced by my own Phase 1 packaging. `LUCI/` is linked editable and picks up edits, but
`LuciBase.py` / `LuciAbsorp.py` are **copied** into site-packages by `force-include`, and site-packages
shadows the project root — so an editable install serves a frozen snapshot of them. The rest of the
suite could not see this, because `conftest` puts the repo root first on `sys.path`.

Found while tracing why `import LuciBase` still pulled in TensorFlow after the lazy-import work: the
import was resolving from a months-old copy, not the file I had just edited.

`dev-mode-dirs = ["."]` now puts the project root on `sys.path`, and the documented workflow
(`uv run pytest`, which syncs first) refreshes the copy, so the failure mode only appears if you
bypass sync with `--no-sync`. The test catches it either way. The real fix lands in Phase 6, when
`LuciBase.py` becomes a thin compat shim that essentially never changes.

---

## S3 — Environment and robustness

### B11. The code cannot run on modern numba — FIXED (Phase 3b)
**Where (was):** bare `@jit(fastmath=True)` on 25 methods across LuciFit, LuciFunctions,
LuciFitParameters, LuciUtility.
**Fixed by:** deleting all 25 decorators and every `numba` import. numba was a direct dependency used
for nothing else, so it (and llvmlite) came out of `pyproject.toml`/`uv.lock` entirely.
**Verification:** fast suite + 12/12 goldens byte-identical after removal; warnings dropped 126→46 and
the fast suite got ~2.5× faster (no import/compile overhead).

numba ≥0.59 removed object-mode fallback, so a bare `@jit` meant `nopython=True` and every decorated
method raised `TypingError`. The decorators never compiled anything useful — they wrapped methods full
of Python lists, `scipy` calls and `self` access, always falling back to object mode (which is why
`fastmath` did nothing and the deprecation warnings fired). Removing them changed no numerics and lifted
the `numba==0.58.1` pin.

### B13. The ML model is reloaded from disk for every pixel — FIXED (Phase 4)
**Where (was):** `get_ML_model()` called from `Fit.__init__`
**Fixed by:** [LUCI/ml/registry.py](LUCI/ml/registry.py) caches predictors per
`(resolution, filter, mdn)` and [LUCI/ml/onnx_backend.py](LUCI/ml/onnx_backend.py) caches ONNX
sessions per path, both at module scope — so each process loads a model at most once instead of once
per spectrum.
**Measured:** 5.7 s/pixel → **3.0 s/pixel** on the fixture; the ~2.7 s/pixel of pure model loading is
gone and the remainder is the actual SLSQP optimisation.
**Test:** `tests/test_ml_predictors.py::test_registry_resolves_and_caches_a_predictor`.

Every `Fit` instance used to call `keras.models.load_model(...)`, so a full 2048×2064 cube was
dominated by model loading rather than fitting.

### B12. `plt.clf()` runs on every spectrum — FIXED (Phase 4)
**Where (was):** top of `estimate_priors_ML`
**Fixed by:** removed when `estimate_priors_ML` was rewired to the injected predictor. It was a debug
leftover that instantiated a Tk window per spectrum, breaking headless runs (`TclError: no display
name`) unless `MPLBACKEND=Agg` was set, and costing a matplotlib figure teardown per pixel.

`LuciFit` now contains **no TensorFlow references at all** — no `import keras`, no
`TF_CPP_MIN_LOG_LEVEL`, no `logging.getLogger('tensorflow')`.

### B17. MDN sigmas needed softplus, not identity — FIXED during Phase 4 (introduced and caught here)
**Where:** `mdn_split` in [tools/convert_models_to_onnx.py](tools/convert_models_to_onnx.py) and
`OnnxMDNPredictor.predict` in [LUCI/ml/onnx_backend.py](LUCI/ml/onnx_backend.py)
**Tests:** `tests/test_ml_predictors.py::test_mdn_sigmas_are_strictly_positive`,
`::test_mdn_applies_softplus_to_the_scale_half`,
`::test_mdn_softplus_differs_from_identity_where_raw_scale_is_negative`

The MDN head is `IndependentNormal(2)` over a `Dense(4)` emitting
`[loc_v, loc_b, scale_v, scale_b]`. Reproducing it in numpy, I concluded from the first model
inspected (R5000-MDN-SN3) that `stddev == scale` with no transform — the empirical match was exactly
0.0, which looked conclusive.

It was wrong. That model's raw scales are all large and positive (range ~[77, 430]), and
`softplus(x) == x` in float32 for such values. The SN2 MDNs emit **negative** raw scales, where
identity produces a **negative standard deviation** — nonsense, and off by up to 35 km/s. The correct
transform is `stddev = softplus(scale)`, implemented as the numerically stable `np.logaddexp(0, x)`.

Worth recording as a process point: the per-model `--validate` gate is what caught this. Two SN2 MDNs
failed with ~26 and ~42 km/s divergence and were **not shipped**; the gate turned a silently-wrong
generalisation into a loud, localised failure. After the fix, **39/39 models validate** (worst
deviation 6.7e-4, i.e. float32 precision). The regression test asserts the *invariant* — a standard
deviation must be positive — rather than just the formula, and deliberately exercises a model with
negative raw scales, since a test using only SN3 would pass under either transform.

### B14. `sincgauss` returns NaN at σ exactly 0 — FIXED (Phase 3c), and re-scoped
**Where (was):** `SincGauss.function` in [LUCI/LuciFunctions.py](LUCI/LuciFunctions.py)
**Fixed by:** a guard that substitutes `SINCGAUSS_SIGMA_FLOOR = 1e-8` when `sigma == 0` (or non-finite),
so the profile is finite at the singular point. Fires only at exact zero; every ordinary evaluation is
bit-for-bit unchanged (12/12 goldens byte-identical).
**Tests:** `tests/test_pure_functions.py::test_sincgauss_at_exactly_zero_sigma_is_guarded` and
`test_sincgauss_is_finite_and_bounded_for_small_nonzero_sigma`.

**Correction of the original Phase-0 finding.** This was first logged as "numerically unstable at small
σ — becomes uncorrelated noise". That was wrong, and worth recording as a lesson: the original test
compared `SincGauss` against LUCI's separate `Sinc` model, which uses a *different* width convention
(`p2 = sinc_width/(π·FWHM_SINC_COEFF)` vs `sinc_width`), so a constant ~0.2 offset between two
genuinely different functions was misread as divergence. Empirically the sinc-Gauss is finite and
bounded by its amplitude for every non-zero σ down to 1e-8. The only genuinely singular input is
`σ == 0` exactly (a 0/0 division → NaN for all channels), which was the *downstream mechanism* of B1:
a zero initial broadening fed σ=0 into the model. With B1 fixed the degenerate input no longer occurs
in practice; the guard is defence in depth. Checking empirically before "fixing" avoided perturbing a
function that was fine.

The Dawson form evaluates `dawsn((channel - p1) / (sqrt(2) * sigma))`, whose
argument diverges as σ→0. Instead of tending to a pure sinc, the profile becomes
uncorrelated noise. Directly compounds B1, whose failure mode is σ=0.

### B15. New-format HDF5 cubes are broken on numpy ≥1.24 — FIXED (Phase 5)
**Where (was):** the new-format branch of `LuciUtility.update_header`
**Fixed by:** dispatching with `issubclass` against `np.floating` / `np.integer` / `np.str_` / `str` /
`bytes` / `np.bool_` instead of `is np.str`, and narrowing the bare `except:` to `except Exception:`.
`np.bool_` is checked *before* the integer branch, since `bool` subclasses `int` in Python.
**Tests:** the new `tests/test_new_format_cube.py` (5 tests), backed by a
`write_new_format_cube` fixture generator — the new-format path previously had **no coverage at all**,
which is exactly how this survived.

`np.str` was removed in numpy 1.24, so the type check raised `AttributeError`, which the surrounding
bare `except:` swallowed. Every string and boolean keyword fell through to the fallback,
`clean_hdr_dict` stayed empty, and the cube came back with an axis-less WCS — losing astrometry on
every cutout and saved map.

---

## Conventions worth a decision (not bugs)

### C1. Gaussian flux carries a sinc normalisation factor
**Where:** [LUCI/LuciFitParameters.py:135](LUCI/LuciFitParameters.py#L135)
**Test:** `tests/test_pure_functions.py::test_gaussian_flux_uses_lucis_sinc_scaled_normalisation`

`calculate_flux` for `model_type='gaussian'` returns
`(1.20671 / FWHM_COEFF) * sqrt(2π) * amp * sigma` — about **0.512×** the textbook
Gaussian integral. `1.20671` is the sinc FWHM coefficient, which has no obvious
reason to appear in a pure-Gaussian flux.

This is pinned rather than changed: it is published behaviour and altering it
would shift every previously reported Gaussian flux by a factor of two. Flagging
for a deliberate decision, not silently "fixing" it.

### C2. `line_dict` is duplicated three times, divergently
- [LuciFit.py:93](LUCI/LuciFit.py#L93) — 26 lines including SN4 and C-filter entries
- [LuciFunctions.py:30](LUCI/LuciFunctions.py#L30) — 10 lines, stale
- [LuciSim.py:31](LUCI/LuciSim.py#L31) — third copy

`frozen_values()` uses the stale copy, so freezing a fit on any SN4 or C-filter
line raises `KeyError`.

### C3. The filter→bounds table is duplicated six times
`LuciFit.restrict_wavelength`, `LuciFit.calculate_noise`,
`LuciUtility.read_in_reference_spectrum`, the PCA scaling windows in `LuciBase`
(×3), and `LuciSim.Spectrum`'s `delta_x`/`order` table. B4 is a direct consequence.

### C4. SN1's noise window sits inside its fit window
[LuciFit.py:304-306](LUCI/LuciFit.py#L304-L306) puts the SN1 noise estimate at
26000–26200 cm⁻¹, which is inside the SN1 fit window of 26000–28000. Every other
filter measures noise *outside* the fit range. Either intentional or a
copy-paste slip; needs a decision when building the `FilterSpec` registry.
