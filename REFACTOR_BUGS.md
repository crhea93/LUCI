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

### B2. `extract_spectrum(mean=True)` is a no-op
**Where:** [LuciBase.py:893-898](LuciBase.py#L893-L898)
**Tests:** `tests/test_cube.py::test_extract_spectrum_mean_is_currently_a_no_op`
(+ xfail `..._should_divide_by_pixel_count`)

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

### B3. `fit_region` never subtracts the background
**Where:** [LuciBase.py:694-707](LuciBase.py#L694-L707)
**Test:** not yet written — needs the Phase 5 selection refactor to test cleanly.

`fit_region` forwards `bkg=bkg` to `fit_calc` but never passes `bkgType`. Inside
`fit_calc` the subtraction is gated on `if bkgType is not None`, so the background
is silently ignored. Callers get an unsubtracted fit with no warning.

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

### B5. `fit_pixel` PCA path raises `NameError`
**Where:** [LuciBase.py:801](LuciBase.py#L801)

References `x_pix` / `y_pix`, which do not exist in that scope. Unbinned PCA
background subtraction on a single pixel always crashes.

### B6. `fit_pixel` crashes with `binning=None`
**Where:** [LuciBase.py:779](LuciBase.py#L779)

`sky -= bkg * (binning) ** 2` with the default `binning=None` raises `TypeError`.
The default arguments of the function cannot be used together.

### B7. `create_deep_image` silently drops trailing rows
**Where:** [LuciBase.py:173-177](LuciBase.py#L173-L177)
**Test:** `tests/test_cube.py::test_deep_image_drops_trailing_rows` (xfail)

The cube is summed in exactly ten slabs of `int(shape[0] / 10)` rows. When the
x-dimension is not divisible by 10 the remainder is left as zeros — for a real
2048-row cube, **the last 8 rows of every deep image are blank**.

### B8. `bin_mask` divides a boolean mask by `binning ** 2`
**Where:** [LUCI/LuciUtility.py:389](LUCI/LuciUtility.py#L389)
**Test:** `tests/test_pure_functions.py::test_bin_mask_*` (+ xfail)

Builds a correct boolean mask, then runs `binned_mask = binned_mask / (binning ** 2)`
— a line copy-pasted from the flux-averaging path in `bin_cube_function`. `True`
becomes `0.25`, and the return type is float, not bool.

Survives only because its one consumer tests `if mask[x, y]:` and 0.25 is truthy.
Any caller that sums the mask, counts non-zeros, or uses it for boolean indexing
gets wrong answers.

### B9. `fit_region` outputs are named differently from `fit_cube`'s
**Where:** [LuciBase.py:709-724](LuciBase.py#L709-L724)

`fit_region` never assigns `corr_fits` / `step_fits`, and does not forward
`fit_function` to `save_fits`, so its FITS products land under different filenames
than the equivalent `fit_cube` call. Downstream scripts that glob for outputs
silently miss them.

### B16. Multi-component fits overwrite each other's output maps
**Where:** [LUCI/LuciUtility.py:63-68](LUCI/LuciUtility.py#L63-L68)
**Test:** `tests/test_output_contract.py::test_duplicate_line_names_*` (+ xfail)

`save_fits` intends to disambiguate repeated line names (a two-component fit
passes `['Halpha', 'Halpha']`) by appending `_2` to the second. But `lines_fit`
is initialised to `[]` and **never appended to**, so `lines_fit.count(line_)` is
always 0 and the renaming branch is dead code. Both components write the same
filename; the second silently overwrites the first. Only one map survives, and
it holds the *second* component. This is the concrete failure behind the
`# TODO: Only works for 2 components` note in
[LuciConvenience.py:32](LUCI/LuciConvenience.py#L32).

### B10. Hardcoded `2048` / `2064` cube dimensions
**Where:** [LuciBase.py:1054](LuciBase.py#L1054), [L630](LuciBase.py#L630), [L1913](LuciBase.py#L1913)

Default arguments and the PCA coefficient array assume the standard SITELLE
detector size. Anything else — a trimmed cube, a test fixture, a future detector —
silently mis-indexes or truncates.

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

### B15. New-format HDF5 cubes are broken on numpy ≥1.24
**Where:** [LUCI/LuciUtility.py:218](LUCI/LuciUtility.py#L218), [L224](LUCI/LuciUtility.py#L224)

The new-format branch of `update_header` tests `header_type is np.str` and
`np.bool_`. `np.str` was **removed** in numpy 1.24, so it raises `AttributeError`
— caught by the surrounding bare `except:`, which leaves `clean_hdr_dict` empty
and produces a WCS with no axes. Only the legacy quadrant format is exercised by
the current test fixture.

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
