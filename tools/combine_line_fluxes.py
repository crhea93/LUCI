"""
Co-add per-line flux maps from a finished fit into one total-complex flux map.

Works on maps already on disk, so there is no need to refit. For M86 SN4 the complex is Halpha
plus the NII doublet -- all three are in band and were fit together with a tied velocity and
broadening, so their sum is the total flux of the complex.

    uv run python tools/combine_line_fluxes.py
    uv run python tools/combine_line_fluxes.py --prefix M86_wvt_20_1
    uv run python tools/combine_line_fluxes.py --lines Halpha NII6583 --out M86_Ha_NII6583_Flux

The default prefix follows `luci.io.outputs.save_fits`: '<object>_<binning>_<fit function>' for a
`fit_cube` run, '<object>_wvt_<stn>_1' for a WVT run.
"""

import argparse
import os

import numpy as np

from luci.analysis.combine import combination_summary, combine_line_fluxes

OUTPUT_DIR = "/home/crhea/Documents/LUCI_DATA/Luci_outputs"
DEFAULT_PREFIX = "M86_2_sincgauss"
DEFAULT_LINES = ["Halpha", "NII6548", "NII6583"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Cube output dir holding Fluxes/ (default: %s)" % OUTPUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Fit output-name prefix (default: %s)" % DEFAULT_PREFIX)
    parser.add_argument("--lines", nargs="+", default=DEFAULT_LINES, help="Lines to add (default: %s)" % " ".join(DEFAULT_LINES))
    parser.add_argument("--out", default=None, help="Output basename without .fits (default: <prefix>_<lines>_Flux)")
    parser.add_argument("--bunit", default=None,
                        help="BUNIT card for the output. Omit it unless the cube was flux calibrated -- "
                             "M86_SN4 was not, so its fluxes are in counts.")
    parser.add_argument("--top-percent", type=float, default=1.0,
                        help="Percentage of the brightest pixels the summary is measured over (default 1.0). "
                             "The rest of a SITELLE field is blank sky, where the co-added noise cancels and "
                             "makes the comparison meaningless.")
    args = parser.parse_args()

    print("# Co-adding %s from %s" % (" + ".join(args.lines), os.path.join(args.output_dir, "Fluxes")))
    total, path = combine_line_fluxes(args.output_dir, args.prefix, args.lines,
                                      out_name=args.out, bunit=args.bunit)

    report = combination_summary(args.output_dir, args.prefix, args.lines, total,
                                 top_percent=args.top_percent)
    print("#    over the %d detected pixels (brightest %.1f%% of the combined map):"
          % (report["n_detected"], args.top_percent))
    for line, flux in report["detected_flux"].items():
        print("#      %-8s %.3e" % (line, flux))
    print("#    brightest line is %s; combined map carries %.2fx its flux"
          % (report["brightest_line"], report["gain_vs_brightest"]))
    print("#    %d pixels, %d NaN (%d beyond what %s already lost), %d negative"
          % (report["n_pixels"], report["n_nan"], report["n_nan_beyond_brightest"],
             report["brightest_line"], report["n_negative"]))
    print("#    total map: median positive %.3e, max %.3e"
          % (np.nanmedian(total[total > 0]), np.nanmax(total)))
    print("# wrote %s" % path)


if __name__ == "__main__":
    main()
