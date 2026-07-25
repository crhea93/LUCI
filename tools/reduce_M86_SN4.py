"""
LUCI reduction of the M86 SN4 cube.

Two modes:

    --method cube   fit_cube over a rectangle (fast, use for the whole field)
    --method wvt    weighted Voronoi tessellation to a target S/N (slow, small regions only)

SN4 is the narrow Halpha filter (652.5 - 664.0 nm), so only Halpha and the NII doublet are in
band -- the SII doublet at 671.6/673.1 nm falls outside it. The three are blended at this
resolution, so they are always fit together with a tied velocity and broadening; fitting Halpha
alone makes the fitter centre between it and NII6583 and rail sigma at its bound.

COORDINATES. `cube_final` is indexed [i0, i1] and `fit_cube(x_min, x_max, y_min, y_max)` slices
`cube_final[x_min:x_max, y_min:y_max]`. The saved FITS maps and any DS9 display are TRANSPOSED
relative to that: a DS9 readout of (x, y) is `cube_final[y, x]`. So this script's x arguments are
DS9's y. Measured Halpha emission at DS9 (333, 157) = cube_final[157, 333] confirms it.

The Halpha emission (v = +43 km/s) is a filament over roughly cube_final[50:500, 250:1000],
i.e. DS9 x 250-1000, y 50-500.

    uv run python tools/reduce_M86_SN4.py --method cube --region full --binning 2
    uv run python tools/reduce_M86_SN4.py --method cube --region emission --binning 1
    uv run python tools/reduce_M86_SN4.py --method wvt  --region emission --stn 20
"""
import argparse
import os
import time

# Set before matplotlib is imported anywhere, and via the environment so that the joblib
# workers -- which re-import everything -- pick it up too
os.environ.setdefault("MPLBACKEND", "Agg")

# One fit is one spectrum, so every worker wants a single thread. Without this the BLAS
# underneath scipy spins up a full thread pool per worker and they fight each other.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from luci import SitelleCube
from luci.analysis.combine import combination_summary, combine_line_fluxes
from luci.analysis.quality import apply_quality_mask, broadening_bound_kms, fit_quality_report

LUCI_PATH = "/home/crhea/Projects/LUCI/"
CUBE_DIR = "/home/crhea/Documents/LUCI_DATA"
CUBE_NAME = "M86_SN4"
OBJECT_NAME = "M86"
FILTER_NAME = "SN4"
REDSHIFT = -0.0010201857421032745
RESOLUTION = 4800
BKG_REGION = os.path.join(CUBE_DIR, "M86_bkg.reg")
OUTPUT_DIR = os.path.join(CUBE_DIR, "Luci_outputs")

# All three in-band lines, fit together. A single-line fit cannot work here: Halpha and
# NII6583 are only 940 km/s apart and blended at this resolution, so one Gaussian centres
# between them (near the brighter NII6583) and rails sigma at its upper bound.
LINES = ["Halpha", "NII6548", "NII6583"]
VEL_REL = [1, 1, 1]  # One shared velocity
SIGMA_REL = [1, 1, 1]  # ... and one shared broadening
FIT_FUNCTION = "sincgauss"

# Regions as (x_min, x_max, y_min, y_max) in cube_final index order -- see COORDINATES above.
# EMISSION is where the Halpha is; the earlier (250, 560, 500, 860) was the transposed guess and
# clipped only the filament's edge, which is why most of those bins were noise.
REGIONS = {
    "emission": (50, 500, 250, 1000),
    "full": None,  # filled in from the cube shape
    "test": (120, 220, 300, 400),  # a small box on the brightest part
}


def extract_background(cube, region):
    """
    Mean background spectrum over a ds9 region, ignoring NaN channels.

    `extract_spectrum_region` sums the spectra outright, so the result inherits the *union* of
    the NaN channels over every pixel in the region -- and ORB leaves a good fraction of each
    SN4 spectrum NaN. Subtracting such a background knocks those channels out of every spectrum
    in the cube, which is enough to empty a fit or noise window. Averaging over the finite
    values instead keeps the background itself finite.
    """
    from luci.engine.selection import reg_to_mask

    header = cube.header
    header.set("NAXIS1", cube.cube_final.shape[1])
    header.set("NAXIS2", cube.cube_final.shape[0])
    mask = reg_to_mask(region, header)
    xs, ys = np.where(mask)
    block = cube.cube_final[xs.min() : xs.max() + 1, ys.min() : ys.max() + 1, :]
    selected = mask[xs.min() : xs.max() + 1, ys.min() : ys.max() + 1]
    spectra = block[selected]
    print("#    %i pixels, x %i-%i, y %i-%i" % (spectra.shape[0], xs.min(), xs.max(), ys.min(), ys.max()))
    bkg = np.nanmean(spectra, axis=0)
    print("#    background is finite in %i / %i channels" % (np.isfinite(bkg).sum(), bkg.size))
    return cube.spectrum_axis, bkg


def plot_background(axis, spectrum, units):
    """Save a plot of the extracted background spectrum so it can be eyeballed."""
    plt.figure(figsize=(10, 5))
    plt.plot(1e7 / axis, spectrum, color="k", lw=1)
    for wavelength, name, color in [
        (654.803, "NII6548", "forestgreen"),
        (656.280, r"H$\alpha$", "coral"),
        (658.341, "NII6583", "seagreen"),
    ]:
        plt.axvline(wavelength * (1 + REDSHIFT), ls="--", color=color, label=name)
    plt.xlim(650, 667)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Flux [%s]" % units)
    plt.title("%s %s -- mean background spectrum" % (OBJECT_NAME, FILTER_NAME))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "%s_%s_background.png" % (OBJECT_NAME, FILTER_NAME)), dpi=120)
    plt.close()


def aperture_mask(shape, x, y, radius):
    """Boolean mask of a circular aperture centred on (x, y)."""
    xx, yy = np.ogrid[: shape[0], : shape[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 <= radius**2


def pick_bright_spots(flux_map, bounds, n, cube_shape, binning=1, min_separation=25):
    """
    Pick the n brightest, well-separated positions, returned in cube_final coordinates.

    The two fit paths save maps on different grids: `fit_cube` writes only the region it fit
    (optionally binned), while `wvt_fit_region` writes a full-field map. Detect which we have
    rather than assuming, or the lookup silently finds nothing.
    """
    x_min, x_max, y_min, y_max = bounds
    work = np.where(np.isfinite(flux_map), flux_map, -np.inf).astype(float)
    full_field = flux_map.shape == (cube_shape[1], cube_shape[0])
    if full_field:
        keep = np.zeros_like(work, dtype=bool)
        keep[y_min:y_max, x_min:x_max] = True
        work[~keep] = -np.inf
        to_cube = lambda i, j: (i, j)  # noqa: E731
    else:
        # Region-local map, indexed [ (y - y_min)//binning, (x - x_min)//binning ]
        to_cube = lambda i, j: (x_min + i * binning, y_min + j * binning)  # noqa: E731

    spots = []
    for _ in range(n):
        if not np.isfinite(work).any() or np.nanmax(work) == -np.inf:
            break
        j, i = np.unravel_index(np.nanargmax(work), work.shape)  # j -> y axis, i -> x axis
        x, y = to_cube(int(i), int(j))
        spots.append((x, y, float(flux_map[j, i])))
        yy, xx = np.ogrid[: work.shape[0], : work.shape[1]]
        sep = max(1, min_separation // binning)
        work[(xx - i) ** 2 + (yy - j) ** 2 <= sep**2] = -np.inf
    return spots


def transmission_corrected(cube, axis, sky):
    """
    Reproduce the division by the filter curve that `Fit` does internally.

    `fit_spectrum_region` hands back the *raw* summed spectrum, but `Fit.apply_transmission`
    divides by the transmission (where it is above 0.5) before fitting. Plotting the raw
    spectrum against the model therefore compares two different things -- across the filter
    edge the difference is large enough to look like a broken fit.
    """
    good = np.isin(cube.spectrum_axis, axis)
    trans = cube.transmission_interpolated[good]
    return np.where(trans > 0.5, sky / np.where(trans > 0.5, trans, 1.0), sky)


def save_fit(axis, data, model, fit_dict, x, y, radius):
    """Write one example fit to a FITS table so the numbers are inspectable, not just the plot."""
    cols = fits.ColDefs([
        fits.Column(name="Wavenumber", format="E", unit="cm-1", array=axis),
        fits.Column(name="Wavelength", format="E", unit="nm", array=1e7 / axis),
        fits.Column(name="Flux", format="E", array=data),
        fits.Column(name="Model", format="E", array=model),
        fits.Column(name="Residual", format="E", array=data - model),
    ])
    hdr = fits.Header()
    hdr["OBJECT"] = OBJECT_NAME
    hdr["FILTER"] = FILTER_NAME
    hdr["XPIX"], hdr["YPIX"], hdr["APRAD"] = x, y, radius
    hdr["FITFUNC"] = FIT_FUNCTION
    hdr["LINES"] = ",".join(LINES)
    for i, line in enumerate(LINES):
        hdr["VEL%d" % i] = (float(fit_dict["velocities"][i]), "%s velocity [km/s]" % line)
        hdr["SIG%d" % i] = (float(fit_dict["sigmas"][i]), "%s broadening [km/s]" % line)
        hdr["FLUX%d" % i] = (float(fit_dict["fluxes"][i]), "%s flux" % line)
    hdr["CHI2"] = float(fit_dict["chi2"])
    hdr["CONTINUU"] = float(fit_dict["continuum"])
    path = os.path.join(OUTPUT_DIR, "Spectra", "%s_%s_x%d_y%d_fit.fits" % (OBJECT_NAME, FILTER_NAME, x, y))
    fits.HDUList([fits.PrimaryHDU(header=hdr), fits.BinTableHDU.from_columns(cols)]).writeto(path, overwrite=True)
    return path


def plot_example_spectra(cube, bkg, spots, units, radius=4):
    """
    Fit and plot a handful of individual spectra so the fit quality is visible.

    Each panel is a circular aperture summed over `radius` pixels, background subtracted, with
    the fitted model overlaid and the residual underneath. Each fit is also written to
    Luci_outputs/Spectra/ as a FITS table.
    """
    n = len(spots)
    if n == 0:
        print("#    no spots to plot")
        return
    os.makedirs(os.path.join(OUTPUT_DIR, "Spectra"), exist_ok=True)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]}, squeeze=False)
    for col, (x, y, _flux) in enumerate(spots):
        mask = aperture_mask(cube.cube_final.shape[:2], x, y, radius)
        axis, raw, fit_dict = cube.fit_spectrum_region(
            LINES, FIT_FUNCTION, VEL_REL, SIGMA_REL, mask, bkg=bkg,
        )
        # Compare like with like: the model was fit to the transmission-corrected spectrum
        sky = transmission_corrected(cube, axis, raw)
        wavelength = 1e7 / axis
        model = fit_dict["fit_vector"]
        print("#    wrote %s" % save_fit(axis, sky, model, fit_dict, x, y, radius))
        top, bot = axes[0][col], axes[1][col]
        top.plot(wavelength, sky, color="0.35", lw=1, label="data")
        top.plot(wavelength, model, color="crimson", lw=1.6, label="fit")
        for rest, name, color in [
            (654.803, "NII6548", "forestgreen"),
            (656.280, r"H$\alpha$", "coral"),
            (658.341, "NII6583", "seagreen"),
        ]:
            top.axvline(rest * (1 + REDSHIFT), ls=":", color=color, lw=1, label=name)
        top.set_title(
            "x=%d y=%d   v = %.0f km/s   $\\sigma$ = %.0f km/s\nflux = %.2e   $\\chi^2$ = %.2f"
            % (x, y, fit_dict["velocities"][0], fit_dict["sigmas"][0],
               fit_dict["fluxes"][0], fit_dict["chi2"]),
            fontsize=10,
        )
        top.set_xlim(651, 666)
        finite = np.isfinite(sky) & (wavelength > 651) & (wavelength < 666)
        if finite.any():
            lo, hi = np.nanpercentile(sky[finite], [1, 99])
            pad = 0.25 * (hi - lo) if hi > lo else 1.0
            top.set_ylim(lo - pad, hi + pad)
        if col == 0:
            top.set_ylabel("Flux [%s]" % units)
            top.legend(fontsize=8, ncol=2)
            bot.set_ylabel("residual")
        bot.plot(wavelength, sky - model, color="0.35", lw=1)
        bot.axhline(0, color="crimson", lw=1)
        bot.set_xlabel("Wavelength [nm]")
    fig.suptitle(
        "%s %s -- example spectra (r=%d px apertures), %s fit"
        % (OBJECT_NAME, FILTER_NAME, radius, FIT_FUNCTION), fontsize=13,
    )
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "%s_%s_example_spectra.png" % (OBJECT_NAME, FILTER_NAME))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("#    wrote %s" % out)


def write_masked_maps(prefix):
    """
    Blank the untrustworthy bins and write *_masked.fits beside the raw maps.

    Most of a SITELLE field is empty, and a fit to noise still returns a velocity, a broadening
    and a flux. Those land all over the allowed parameter range, so they dominate any percentile
    colour scale and bury the real structure. The raw maps are left untouched; the masked copies
    are the ones to look at.
    """
    def _open(subdir, suffix):
        path = os.path.join(OUTPUT_DIR, subdir, prefix + suffix) if subdir else os.path.join(OUTPUT_DIR, prefix + suffix)
        return (fits.open(path)[0].data, fits.open(path)[0].header, path) if os.path.exists(path) else (None, None, path)

    line = LINES[0]
    flux, header, flux_path = _open("Fluxes", "_%s_Flux.fits" % line)
    if flux is None:
        print("#    no flux map at %s -- skipping" % flux_path)
        return
    flux_err, _, _ = _open("Fluxes", "_%s_Flux_err.fits" % line)
    vel, _, _ = _open("Velocity", "_%s_velocity.fits" % line)
    broad, _, _ = _open("Broadening", "_%s_broadening.fits" % line)
    chi2, _, _ = _open("", "_Chi2.fits")
    snr_path = os.path.join(OUTPUT_DIR, "SNR", "%s_SNR.fits" % OBJECT_NAME)
    snr = fits.open(snr_path)[0].data if os.path.exists(snr_path) else None
    if snr is not None and snr.shape != flux.shape:
        snr = None  # SNR map is cut to the fitted sub-region; skip rather than mis-align

    # Halpha sits near 15253 cm-1 in SN4, so the sigma bound is ~197 km/s. Anything within
    # 5 km/s of it never converged.
    sigma_ceiling = broadening_bound_kms(15253) - 5.0
    mask, report = fit_quality_report(
        flux=flux, flux_err=flux_err, velocity=vel, broadening=broad, chi2=chi2, snr=snr,
        snr_min=5.0 if snr is not None else None,
        velocity_range=(-1500, 1500),
        broadening_max=sigma_ceiling,
        max_flux_err_ratio=0.5,
        log=False,
    )
    print("#    rejected %.1f%% of the fitted area" % (100 * report["combined"]))
    for name, fraction in sorted(report.items()):
        if name != "combined":
            print("#      %-11s rejects %5.1f%%" % (name, 100 * fraction))

    maps = {"Fluxes/%s_%s_Flux" % (prefix, line): flux,
            "Velocity/%s_%s_velocity" % (prefix, line): vel,
            "Broadening/%s_%s_broadening" % (prefix, line): broad}
    for name, array in apply_quality_mask({k: v for k, v in maps.items() if v is not None}, mask).items():
        out = os.path.join(OUTPUT_DIR, name + "_masked.fits")
        fits.writeto(out, array.astype(np.float32), header, overwrite=True)
        print("#    wrote %s" % out)
    fits.writeto(os.path.join(OUTPUT_DIR, "%s_quality_mask.fits" % prefix),
                 mask.astype(np.uint8), header, overwrite=True)


def write_combined_flux(prefix):
    """
    Co-add the three fitted lines into one total-complex flux map.

    All of Halpha and the NII doublet are in the SN4 band and were fit together with a tied
    velocity and broadening, so their sum is the total flux of the complex -- brighter than any
    single line, which brings out filament structure the individual maps only hint at.
    """
    try:
        total, path = combine_line_fluxes(OUTPUT_DIR, prefix, LINES)
    except FileNotFoundError as err:
        print("#    %s -- skipping" % err)
        return
    # Summarise over the pixels the quality mask kept, if it was written -- over the whole field the
    # blank sky dominates and the co-added noise cancels, which understates the gain.
    mask_path = os.path.join(OUTPUT_DIR, "%s_quality_mask.fits" % prefix)
    mask = fits.open(mask_path)[0].data.astype(bool) if os.path.exists(mask_path) else None
    if mask is not None and mask.shape != total.shape:
        mask = None
    report = combination_summary(OUTPUT_DIR, prefix, LINES, total, mask=mask)
    print("#    over %d %s pixels, combined map carries %.2fx the flux of %s"
          % (report["n_detected"], "masked" if mask is not None else "brightest-1%",
             report["gain_vs_brightest"], report["brightest_line"]))
    print("#    wrote %s" % path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=("cube", "wvt"), default="cube",
                        help="cube = fit_cube over a rectangle (fast); wvt = Voronoi bins (slow)")
    parser.add_argument("--region", choices=tuple(REGIONS), default="emission",
                        help="Which part of the field to fit (default: emission)")
    parser.add_argument("--binning", type=int, default=1, help="Spatial binning, --method cube only (default 1)")
    parser.add_argument("--stn", type=int, default=20, help="Target S/N per bin, --method wvt only (default 20)")
    parser.add_argument("--snr-floor", type=float, default=None,
                        help="Only bin pixels above this S/N, --method wvt only. Pixels below are left "
                             "unfitted rather than cropped, so the maps stay full-field.")
    parser.add_argument("--snr-percentile", type=float, default=None,
                        help="Set the S/N floor from this percentile of the S/N map instead of an absolute "
                             "value (e.x. 85 bins the brightest 15%% of the field). Method 1 is not a "
                             "calibrated S/N, so a percentile travels better than a number.")
    parser.add_argument("--snr-method", type=int, default=1, choices=(1, 2),
                        help="S/N estimator: 1 is LUCI's default, 2 is flux-in-window over the noise "
                             "standard deviation (default 1)")
    parser.add_argument("--n-threads", type=int, default=24,
                        help="Threads for joblib (default 24; leave headroom or the desktop lags)")
    parser.add_argument("--n-spectra", type=int, default=4, help="Example spectra to plot (default 4)")
    parser.add_argument("--no-ml", action="store_true", help="Turn off the ML initial guess")
    args = parser.parse_args()

    t0 = time.time()
    cube = SitelleCube(LUCI_PATH, os.path.join(CUBE_DIR, CUBE_NAME), CUBE_DIR, OBJECT_NAME,
                       REDSHIFT, RESOLUTION, ML_bool=not args.no_ml, mdn=False)
    dimx, dimy, dimz = cube.cube_final.shape
    units = "ergs/s/cm$^2$/A" if cube.flux_calibrated else "counts"
    print("# Cube read in %.1f s -- %s, filter %s, %i x %i x %i, units: %s"
          % (time.time() - t0, CUBE_NAME, cube.hdr_dict["FILTER"], dimx, dimy, dimz, units))

    print("# -- Deep image -- #")
    cube.create_deep_image()

    print("# -- Background spectrum from %s -- #" % os.path.basename(BKG_REGION))
    bkg_axis, bkg_sky = extract_background(cube, BKG_REGION)
    np.save(os.path.join(OUTPUT_DIR, "%s_%s_bkg_spectrum.npy" % (OBJECT_NAME, FILTER_NAME)), bkg_sky)
    plot_background(bkg_axis, bkg_sky, units)

    bounds = REGIONS[args.region] or (0, dimx, 0, dimy)
    x_min, x_max, y_min, y_max = bounds
    npx = (x_max - x_min) * (y_max - y_min)
    print("# -- %s fit, cube_final[%d:%d, %d:%d] = %d px, %d threads -- #"
          % (args.method.upper(), x_min, x_max, y_min, y_max, npx, args.n_threads))

    t_fit = time.time()
    if args.method == "wvt":
        # Accretion and refinement both scale with the number of pixels binned, and on a full field
        # most of them are blank sky whose bins never reach the target and are discarded anyway. The
        # S/N floor drops those up front; the pixels below it are left unfitted, not cropped, so the
        # maps stay full-field. For SN4 the S/N flux window spans Halpha and both NII lines, so this
        # is a cut on the whole complex rather than on Halpha alone.
        if args.snr_percentile is None and args.snr_floor is None and npx > 1000000:
            print("# NOTE: binning all %d px. Pass --snr-percentile to skip the blank sky." % npx)
        cube.wvt_fit_region(x_min, x_max, y_min, y_max, LINES, FIT_FUNCTION, VEL_REL, SIGMA_REL,
                            stn_target=args.stn, bkg=bkg_sky, bayes_bool=False,
                            uncertainty_bool=False, n_threads=args.n_threads,
                            snr_floor=args.snr_floor, snr_percentile=args.snr_percentile,
                            snr_method=args.snr_method)
        prefix = "%s_wvt_%i_1" % (OBJECT_NAME, args.stn)
    else:
        cube.fit_cube(LINES, FIT_FUNCTION, VEL_REL, SIGMA_REL, x_min, x_max, y_min, y_max,
                      bkg=bkg_sky, bkgType="standard", binning=args.binning, n_threads=args.n_threads)
        prefix = "%s_%i_%s" % (OBJECT_NAME, args.binning, FIT_FUNCTION)
    print("# Fit done in %.1f s" % (time.time() - t_fit))

    print("# -- Quality mask -- #")
    write_masked_maps(prefix)

    print("# -- Combined Halpha + NII flux map -- #")
    write_combined_flux(prefix)

    print("# -- Example spectra -- #")
    flux_path = os.path.join(OUTPUT_DIR, "Fluxes", "%s_%s_Flux.fits" % (prefix, LINES[0]))
    if os.path.exists(flux_path):
        flux_map = fits.open(flux_path)[0].data
        # flux_map is [i1, i0]; pick_bright_spots takes bounds in cube_final order
        binning = args.binning if args.method == 'cube' else 1
        spots = pick_bright_spots(flux_map, bounds, args.n_spectra,
                                  cube.cube_final.shape[:2], binning)
        for x, y, f in spots:
            print("#    cube_final[%d, %d]  flux=%.3e" % (x, y, f))
        plot_example_spectra(cube, bkg_sky, spots, units)
    else:
        print("#    no flux map at %s" % flux_path)

    print("# Done in %.1f s. Output in %s" % (time.time() - t0, OUTPUT_DIR))


if __name__ == "__main__":
    main()
