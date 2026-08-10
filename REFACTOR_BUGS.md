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

### B21. Sigma bounds applied to only the last line — FIXED (restructure)
**Where (was):** `Fit.sigma_constraints`
**Now:** `luci/fitting/constraints.py::sigma_bounds`
**Tests:** `tests/test_constraints.py::test_late_binding_bug_left_all_but_the_last_line_unconstrained`
and `::test_sigma_bounds_constrain_every_line`; plus a new golden guard,
`test_velocity_tied_lines_agree_with_each_other`.

The per-line sigma bounds were built like this:

```python
for i in range(len(self.sigma_rel)):
    sigma_dict_list.append({'type': 'ineq', 'fun': lambda x: x[3*i+2]})
    sigma_dict_list.append({'type': 'ineq', 'fun': lambda x: -x[3*i+2]+10})
```

Every *other* loop in that class binds its index through a default argument
(`ind_unique=ind_unique`). This one does not, so each lambda reads `i` when SLSQP
**calls** it — by which time the loop has ended and `i` is its final value. All 2N
constraints therefore bounded the *last* line's sigma, N times over, and every other
line's sigma was completely unconstrained.

**This was silently wrong science.** On the SN3 fixture, with all five lines velocity-tied:

| line | before | after | truth |
|---|---|---|---|
| Halpha | 99.61 | 99.53 | 100 |
| NII6548 | 101.97 | 101.92 | 100 |
| **NII6583** | **164.36** | **101.55** | 100 |
| SII6716 | 101.63 | 101.55 | 100 |
| **SII6731** | **167.10** | **101.55** | 100 |

Two of five lines were ~65 km/s wrong, and the spread across lines that are *explicitly tied
together* was 67.5 km/s (now 2.4). Halpha — the line anyone would eyeball — looked perfect
throughout, which is why nothing caught it.

The safety net missed it too: `test_golden_baselines_recover_the_injected_physics` only checked
line index 0. It now checks every line, and a second guard asserts that velocity-tied lines actually
agree. Goldens were re-recorded in the same change; all ten ML/no-ML baselines moved.

### B23. `pixel_list=True` fitted the whole cube — FIXED (restructure)
**Where (was):** the pixel-list branch of `Luci.fit_region`
**Now:** `luci/engine/selection.py::mask_from_pixel_list`
**Test:** `tests/test_background_and_regions.py::test_pixel_list_selects_only_the_listed_pixels`

```python
mask = np.ones((shape), dtype=bool)   # every pixel already selected
for pair in region:
    mask[pair] = True                 # ...then set the listed ones True again
```

Starting from `np.ones` means the mask was already all-True, so setting the requested pixels changed
nothing: `pixel_list=True` fitted **every pixel in the cube** instead of the handful asked for. Slow
and silently wrong rather than an error. `resolve_mask` now handles all four region forms in one
place, and an unrecognised value raises instead of printing a message and continuing with no mask.

### B29. A group of pixels' PCA coefficients were combined by sum, not mean — FIXED
**Where (was):** the binned branches of `fit_calc` and `fit_pixel`
**Now:** [luci/background/subtraction.py](luci/background/subtraction.py)`::combine_pca_coefficients`,
used by both and by the new region path
**Tests:** `tests/test_background_and_regions.py::test_combined_coefficients_reproduce_the_summed_background_exactly`
and `::test_combined_coefficients_are_a_no_op_for_a_single_pixel`

A pixel's PCA background is `pca_mean + sum_i c_i v_i`. For a binned spectrum — which
`bin_cube_function` builds by `nansum`, so it is a *sum* of `binning²` spaxels — the background is
`N * pca_mean + sum_i (sum_p c_ip) v_i`. Both binned branches formed the coefficients with `nansum`
and handed them to `pca_background`, which adds `pca_mean` exactly once: the mean term came out
**under-weighted by N** relative to the components. That is an error in the background's *shape*, so
`subtract_pca`'s rescaling onto the observed continuum cannot absorb it, and it grows with the bin —
4× at `binning=2`, 16× at `binning=4`.

Rebuilding from the *mean* coefficients reproduces the summed background's shape exactly (the test
asserts `N * pca_background(mean(c)) == sum_p pca_background(c_p)` identically), and is a no-op for a
single pixel, so unbinned fits are unchanged. No golden covers a binned PCA fit, so none moved —
which is also why this survived.

### B30. A mean region spectrum had its background subtracted N times — FIXED
**Where:** `extract_region_for_fit`
**Test:** `tests/test_background_and_regions.py::test_a_mean_region_spectrum_subtracts_the_background_once`

`integrated_spectrum -= bkg * spec_ct` ran regardless of `mean`, so `fit_spectrum_region(mean=True,
bkg=...)` over-subtracted the background by the region's pixel count. The same mistake as B2, fixed
there in `extract_spectrum` and left standing here. Now goes through `subtract_standard` with a
spaxel count of 1 when the mean was taken.

### B28. `output_name` reached `save_fits` from nowhere — FIXED
**Where (was):** `fit_entire_cube`, `fit_region`, and `save_fits`, which had no such parameter
**Now:** [luci/io/outputs.py](luci/io/outputs.py) takes `output_name` as a base-name override;
`fit_cube` and `fit_region` forward it
**Tests:** `tests/test_background_and_regions.py::test_fit_entire_cube_forwards_only_what_fit_cube_accepts`
and `::test_output_name_renames_the_products`

Two halves of one hole, both in the B3/B9/B27 wrapper-drift family:

* **`fit_entire_cube` was dead outright.** It passed `output_name=output_name` to `fit_cube`, which
  had no such parameter, so *every* call raised `TypeError` — not just calls that supplied a name.
  Nothing in the repo, tests or examples calls it, which is how a completely non-functional public
  entry point survived. Now pinned by a signature comparison rather than a fit, so the next instance
  of this class is caught in milliseconds.
* **`fit_region` silently discarded it.** It accepted `output_name`, and if none was given built a
  default from the object name and the region's stem — then never passed it anywhere, because
  `save_fits` derived every filename from `object_name` alone. So a caller's name was ignored, and a
  region fit's maps overwrote a whole-cube fit's. The default it computed was a full path and could
  not have been used even if forwarded: `save_fits` joins the name onto `output_dir/<product>/`, so
  it would have nested one path inside another.

Default filenames are deliberately unchanged: `output_name` applies only when the caller passes one,
so no existing script's globs break. The `binning`/`fit_function` decorations still apply on top.

### B27. The absorption hook was unreachable from four of six fit entry points — FIXED
**Where (was):** `absorp` existed on `fit_calc`, `fit_cube` and `fit_pixel` only
**Now:** [luci/background/absorption.py](luci/background/absorption.py), reached from every entry point
**Tests:** `tests/test_absorption.py` (16), notably
`::test_fit_region_applies_the_absorption_template`

Same family as B3 and B9 — an option threaded through one entry point by hand and never the others.
`fit_region`, `fit_spectrum_region`, `fit_wvt` and `wvt_fit_region` had no `absorp` parameter at all,
so a region or WVT fit could not have its stellar continuum removed while a rectangular fit could.
Passing `absorp=` to any of them raised `TypeError`, so this one was loud rather than silently wrong.

The subtraction itself was copy-pasted verbatim between `fit_calc` and `fit_pixel`; it now lives once
in `subtract_absorption`, which also range-checks the template. That check is the point: a template on
a *rebinned* axis of the right length would otherwise subtract the wrong wavelength from every
channel and produce a plausible, wrong fit — the same failure mode as the transmission-curve
misalignment in `fit_calc`.

The other half had no home at all. `build_absorption_template` is the revived `LuciAbsorp.py`, whose
module-level functions referenced `self` and so raised `NameError` however they were called — which is
why the builder and the hook were never connected in the first place. Three things about the original
were wrong and are not reproduced: it indexed `vel_map[x, y]` while every map LUCI writes is `[y, x]`
(now validated, with the orientation named in the message), it concatenated `n_pixels x n_channels`
arrays before averaging, and it returned a rebinned wavelength axis — which the hook it was written
for cannot consume. The rewrite returns the template on `cube.spectrum_axis`.

The one behaviour deliberately kept: the stack is left at the region's **mean velocity**, not at rest.
That is what the original's `beta_avg` bin range did, and it is what `subtract_absorption` needs,
since it works channel by channel against spectra observed at roughly that velocity.

**Follow-up in the same branch — the template has to be built behind the same background as the fit.**
`build_absorption_template` first stacked raw `cube_final` spectra, so the template carried the sky as
well as the stellar continuum. The consumer then subtracted that sky a second time, from a spectrum
`fit_calc` had already cleaned — and on the `bkgType='pca'` path there is no cancellation to hope for,
since the doubled term is a *per-pixel* background averaged over a region. The builder now takes the
same `bkg`/`bkgType`/`pca_*` arguments as the fit and removes the background per spaxel **before** the
Doppler shift, because the sky sits at fixed observed wavenumbers. Pinned by
`test_the_sky_is_removed_before_the_shift_not_after`, which is exact: a sky line common to spaxels of
differing velocity cancels completely when removed in the observed frame, and survives as one spike
per spaxel when removed after the shift.

Still not covered: `extract_region_for_fit` — and so `fit_spectrum_region`, `fit_wvt` and
`wvt_fit_region` — supports only a `standard` background, never `pca`. Those paths can therefore
apply a template but cannot be run behind the PCA background it was built for. `fit_cube`, `fit_region`
and `fit_pixel` are complete.

### B24. `reassign_pixels` looped forever instead of failing — FIXED (Phase 6)
**Where:** `reassign_pixels` in [luci/analysis/wvt.py](luci/analysis/wvt.py)
**Test:** `tests/test_errors_and_logging.py::TestWvtReassignTerminates`

```python
while pixel.assigned_to_bin == False:
    try:
        dists = [dist(...) for (centx, centy) in potential_centroids]
        closest_bin_index = dists.index(min(dists))
        ...
    except:
        pass
```

The loop exits only once the pixel is assigned, and the bare `except: pass` swallowed the failure
that prevented assignment. With no candidate bins, `min([])` raises `ValueError` on every pass, so
the function span forever at 100% CPU rather than raising — the worst failure mode for a batch job.
It also swallowed `KeyboardInterrupt`, so it could not be interrupted. Now guards the empty case,
narrows to `(IndexError, ValueError)`, and breaks with a warning.

### B25. `luci.simulation` was unimportable on a clean install — FIXED (Phase 6)
**Where:** module-level `import ppxf.sps_util as lib` in [luci/simulation.py](luci/simulation.py)
**Test:** `tests/test_errors_and_logging.py::TestRaisesInsteadOfExiting::test_unsupported_filter_in_simulation_raises`

`ppxf` is not a declared dependency, but it was imported at module scope, so
`from LUCI.LuciSim import Spectrum` — used by `Examples/Create-Mock-Spectrum.ipynb` and three doc
pages — raised `ModuleNotFoundError` on any install that did not happen to have ppxf. Only
`abs_template` actually uses it. The import is now lazy and raises a message naming the package,
so `Spectrum` works without it.

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
**Fixed by:** the filter registry [luci/instrument/filters.py](luci/instrument/filters.py). C3 normal
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
[LuciConvenience.py:32](luci/LuciConvenience.py#L32).

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

### B18. Editable installs serve a stale copy of the top-level modules — FIXED (restructure)
**Where:** `[tool.hatch.build.targets.wheel.force-include]` in `pyproject.toml`
**Test:** `tests/test_packaging.py::test_installed_top_level_module_is_not_stale`

Introduced by my own Phase 1 packaging. `luci/` is linked editable and picks up edits, but
`LuciBase.py` / `LuciAbsorp.py` are **copied** into site-packages by `force-include`, and site-packages
shadows the project root — so an editable install serves a frozen snapshot of them. The rest of the
suite could not see this, because `conftest` puts the repo root first on `sys.path`.

Found while tracing why `import LuciBase` still pulled in TensorFlow after the lazy-import work: the
import was resolving from a months-old copy, not the file I had just edited.

`dev-mode-dirs = ["."]` now puts the project root on `sys.path`, but the copy still shadows it, and
plain `uv sync` does **not** refresh it — uv only reinstalls when package *metadata* changes, not when
a `.py` file does. After editing a top-level module you need:

```bash
uv sync --reinstall-package luci-sitelle
```

The test is the real protection: it fails whenever the copy drifts, so the staleness cannot be silent.
**Now fixed properly:** the cube class moved to `luci/cube.py` (inside the editable-linked package)
and `LuciBase.py` is a 7-line re-export shim. A shim that never changes cannot go stale, so the copy
in site-packages stays correct on its own.

### B19. WVT fits wrote the continuum error into the continuum map — FIXED (Phase 5e)
**Where (was):** the per-pixel scatter loop in `Luci.fit_wvt`

Two consecutive lines both assigned to `continuum_fits`:

```python
continuum_fits[a, b] = bin_fit_dict['continuum']
continuum_fits[a, b] = bin_fit_dict['continuum_error']   # meant continuum_error_fits
```

So every Voronoi-binned fit produced a continuum map holding the *error*, and a continuum-error map
that stayed all zeros. Found while collapsing this loop onto `FitMaps` — the kind of copy-paste slip
that a hand-maintained twelve-array scatter invites, and that having one shared implementation
prevents.

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
**Fixed by:** [luci/ml/registry.py](luci/ml/registry.py) caches predictors per
`(resolution, filter, mdn)` and [luci/ml/onnx_backend.py](luci/ml/onnx_backend.py) caches ONNX
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
`OnnxMDNPredictor.predict` in [luci/ml/onnx_backend.py](luci/ml/onnx_backend.py)
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
**Where (was):** `SincGauss.function` in [luci/LuciFunctions.py](luci/LuciFunctions.py)
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

### B20. `luci/LuciMask.py` ran an analysis script at import time — FIXED (restructure)
**Where (was):** `luci/LuciMask.py`, module level
**Now:** `scripts/mask_background_exploration.py`

It was never an importable module. At module level it opened a hardcoded absolute path
(`/mnt/carterrhea/.../NGC1275_lowres_deep.fits`), ran a background fit, and set analysis parameters —
so `import LUCI.LuciMask` raised `FileNotFoundError` on every machine but its author's. Nothing
imported it, which is why it went unnoticed. Found while bisecting the TensorFlow import chain.

Moved to `scripts/` rather than deleted: it records how the background masking was explored, and is
honest about being a script now instead of shipping inside the package.

### B22. `fit_absorption` raised `NameError` on every call — NOW A REAL FEATURE
**Where (was):** `Fit.fit_absorption`, now `luci/fitting/spectrum_fitter.py`
**Now:** [luci/fitting/absorption.py](luci/fitting/absorption.py) does the measuring;
`SpectrumFitter.fit_absorption` drives it behind `absorption_bool`
**Tests:** `tests/test_absorption.py` — six unit tests plus three end-to-end, including
`::test_fitted_absorption_recovers_an_injected_trough_end_to_end`

The dead method is gone and stellar absorption is now a fitted component: `absorption_bool=True`
measures the trough, fills it in, refits the lines on the corrected spectrum, and reports
`absorption_depth` / `_velocity` / `_broadening` on the result and as three extra maps. The
synthetic fixture can inject a trough (`write_cube(absorption_depth=...)`), which is what makes any
of this checkable.

**The width is an input, not a measurement, and this was the whole difficulty.** The trough is
centred on the emission line, so the channels where it is deepest are exactly the ones the emission
occupies and the mask removes. Its central depth is never observed — only extrapolated from the
flanks — and there it trades off almost exactly against width. Four designs were tried against the
fixture before this was clear:

| Attempt | Injected 0.30 came back as |
|---|---|
| Constant continuum over the whole fit window | 0.12, sigma pinned at bound (fitting filter curvature) |
| Local window, linear continuum, free sigma | 0.46, sigma at floor (wings absorbed into the slope) |
| Emission model subtracted, iterated to convergence | 0.465 — the loop *amplifies*: filling an over-deep trough brightens the next emission fit, which deepens the next trough |
| Sidebands for continuum, free sigma | 0.454 at every depth — a clean 1.5x, sigma pinned |
| **Sidebands, sigma supplied** | **0.254** |

So the stellar velocity dispersion is supplied (`absorption_broadening_kms`, default 200) and only
depth and velocity are fitted. Continuum comes from sidebands beyond three sigmas rather than being
fitted alongside the trough, because a Gaussian's wings over a finite window look like a slope.

**What it is worth, measured — and a correction to my first reading of it.** I initially recorded a
systematic shortfall (~15% at depth 0.3, ~45% at 0.1) and attributed it to emission filling the
masked core. That was wrong on both counts. Every fixture cube uses the same noise seed, so the same
noise realisation appeared at every depth and produced the same *absolute* error each time; I read
that constancy as a bias. A Monte Carlo over 300 realisations shows the estimator is unbiased — mean
recovered depth matches injected to within 0.005 from 0.1 to 1.0, and is exact with no noise. And
emission brightness turns out to be irrelevant: the recovered depth does not move with lines from 0
to 30× the continuum.

The real limit is **precision, and it is set by depth over noise rather than depth**:
`sigma_depth ≈ 0.7 × (continuum noise / continuum)`, measured to hold from 1% to 10% noise. A trough
within ~1.5σ of that is both noisy and *biased high*, since a depth cannot be negative and the
distribution piles against zero — injected 0.10 at continuum S/N 10 returns 0.125 ± 0.201. Clear of it,
the bias vanishes: the same trough at S/N 20 returns 0.098, at S/N 50 returns 0.099 ± 0.015. So per
spaxel, a 3σ detection of depth d needs continuum S/N ≳ 2/d and 20% precision needs ≳ 3.5/d, both
relaxing as √N when averaged — a WVT bin of 100 spaxels turns S/N 10 into an effective 100.

*(Corrected twice. I first read a constant absolute deficit as bias — it was one noise realisation seen
at five depths, since every fixture cube shares a seed. Then I measured the scatter with an empty line
list, i.e. no emission mask, which understated it by 2×. The numbers above are with the mask the
pipeline actually applies.)*

The supplied width is the one input that does bias the answer: a 310 km/s trough read at the 200 km/s
default gives 0.41 for a true 0.30, against 0.26 with the width measured from a template.

The four-design table above records single realisations, so its numbers carry that same ±0.03; the
conclusions do not depend on them, since the failures that ruled those designs out were qualitative
(sigma pinned at a bound, a loop that diverges).

### B22-old. The original defect, for the record
**Where (was):** `Fit.fit_absorption`, now `luci/fitting/spectrum_fitter.py`

It built four initial-guess scalars (`ampl_init`, `pos_init`, `pos_sigma`, `cont_init`), never
assembled them, then called `minimize(nll, initial, ...)` with `initial` undefined — so the method
raised `NameError` however it was invoked. It also assigned `parameters = soln.x` and returned
nothing.

Now assembles `initial = [ampl_init, pos_init, pos_sigma, cont_init]` and returns the solution.
Nothing in the repo calls it, so this could not regress anything — but it also means the fix is
**untested**: the method is now runnable rather than verified.

---

## Conventions worth a decision (not bugs)

### C1. Gaussian flux carries a sinc normalisation factor
**Where:** [luci/LuciFitParameters.py:135](luci/LuciFitParameters.py#L135)
**Test:** `tests/test_pure_functions.py::test_gaussian_flux_uses_lucis_sinc_scaled_normalisation`

`calculate_flux` for `model_type='gaussian'` returns
`(1.20671 / FWHM_COEFF) * sqrt(2π) * amp * sigma` — about **0.512×** the textbook
Gaussian integral. `1.20671` is the sinc FWHM coefficient, which has no obvious
reason to appear in a pure-Gaussian flux.

This is pinned rather than changed: it is published behaviour and altering it
would shift every previously reported Gaussian flux by a factor of two. Flagging
for a deliberate decision, not silently "fixing" it.

### C2. `line_dict` is duplicated three times, divergently
- [LuciFit.py:93](luci/LuciFit.py#L93) — 26 lines including SN4 and C-filter entries
- [LuciFunctions.py:30](luci/LuciFunctions.py#L30) — 10 lines, stale
- [LuciSim.py:31](luci/LuciSim.py#L31) — third copy

`frozen_values()` uses the stale copy, so freezing a fit on any SN4 or C-filter
line raises `KeyError`.

### C3. The filter→bounds table is duplicated six times
`LuciFit.restrict_wavelength`, `LuciFit.calculate_noise`,
`LuciUtility.read_in_reference_spectrum`, the PCA scaling windows in `LuciBase`
(×3), and `LuciSim.Spectrum`'s `delta_x`/`order` table. B4 is a direct consequence.

### C4. SN1's noise window sits inside its fit window
[LuciFit.py:304-306](luci/LuciFit.py#L304-L306) puts the SN1 noise estimate at
26000–26200 cm⁻¹, which is inside the SN1 fit window of 26000–28000. Every other
filter measures noise *outside* the fit range. Either intentional or a
copy-paste slip; needs a decision when building the `FilterSpec` registry.
