"""Writing fit products to FITS."""

import os

from astropy.io import fits


def save_fits(
    output_dir,
    object_name,
    lines,
    ampls_fits,
    flux_fits,
    flux_errors_fits,
    velocities_fits,
    broadenings_fits,
    velocities_errors_fits,
    broadenings_errors_fits,
    chi2_fits,
    continuum_fits,
    continuum_error_fits,
    header,
    binning=1,
    suffix="",
    fit_function=None,
    output_name=None,
    absorption_maps=None,
):
    """
    Function to save the fits files returned from the fitting routine. We save the velocity, broadening,
    amplitude, flux, and chi-squared maps with the appropriate headers in the output directory
    defined when the cube is initiated.

    Args:
        lines: Lines to fit (e.x. ['Halpha', 'NII6583'])
        ampls_fits: 3D Numpy array of amplitude values
        flux_fis: 3D Numpy array of flux values
        flux_errors_fits 3D numpy array of flux errors
        velocities_fits: 3D Numpy array of velocity values
        broadenings_fits: 3D Numpy array of broadening values
        velocities_errors_fits: 3D Numpy array of velocity errors
        broadenings_errors_fits: 3D Numpy array of broadening errors
        chi2_fits: 2D Numpy array of chi-squared values
        continuum_fits: 2D Numpy array of continuum value
        continuum_error_fits: 2D numpy array of continuum errors
        header: Header object (either binned or unbinned)
        binning: Value by which to bin (default None)
        suffix: Additional suffix to add (e.x. '_wvt')
        fit_function: Fit function, appended to every filename (default None)
        output_name: Base name for the products, replacing `object_name` (default None). A name,
            not a path: the products still go in the `Amplitudes`/`Fluxes`/`Velocity`/`Broadening`
            subdirectories of `output_dir`. Use it to keep a region fit's maps from overwriting a
            whole-cube fit's, since both otherwise derive their names from `object_name` alone.
        absorption_maps: ``{name: 2D array}`` of stellar-absorption products to write alongside
            the rest, or None to write none (default None). Passed only when the fit actually
            measured absorption, so an ordinary fit's output layout is byte-for-byte unchanged --
            downstream scripts glob these directories.

    """
    # Make sure output dirs exist for amps, flux, vel, and broad
    if not os.path.exists(output_dir + "/Amplitudes"):
        os.mkdir(output_dir + "/Amplitudes")
    if not os.path.exists(output_dir + "/Fluxes"):
        os.mkdir(output_dir + "/Fluxes")
    if not os.path.exists(output_dir + "/Velocity"):
        os.mkdir(output_dir + "/Velocity")
    if not os.path.exists(output_dir + "/Broadening"):
        os.mkdir(output_dir + "/Broadening")
    # `output_name` overrides the object name as the base; the decorations still apply, so a
    # caller-named run is still distinguishable by binning and fit function.
    output_name = (output_name if output_name else object_name) + suffix
    if binning is not None:
        output_name += "_" + str(binning)
    if fit_function is not None:
        output_name += "_" + fit_function
    lines_fit = []  # Line names already written, so repeats can be disambiguated
    for ct, line_ in enumerate(lines):  # Step through each line to save their individual amplitudes
        # Repeat line names (multi-component fits) become <line>_2, <line>_3...
        # lines_fit was never appended to, so every component overwrote the last (B16).
        seen = lines_fit.count(line_)
        lines_fit.append(line_)
        if seen >= 1:
            line_ += "_" + str(seen + 1)
        fits.writeto(
            output_dir + "/Amplitudes/" + output_name + "_" + line_ + "_Amplitude.fits",
            ampls_fits[:, :, ct],
            header,
            overwrite=True,
        )
        fits.writeto(
            output_dir + "/Fluxes/" + output_name + "_" + line_ + "_Flux.fits",
            flux_fits[:, :, ct],
            header,
            overwrite=True,
        )
        fits.writeto(
            output_dir + "/Fluxes/" + output_name + "_" + line_ + "_Flux_err.fits",
            flux_errors_fits[:, :, ct],
            header,
            overwrite=True,
        )
        fits.writeto(
            output_dir + "/Velocity/" + output_name + "_" + line_ + "_velocity.fits",
            velocities_fits[:, :, ct],
            header,
            overwrite=True,
        )
        fits.writeto(
            output_dir + "/Broadening/" + output_name + "_" + line_ + "_broadening.fits",
            broadenings_fits[:, :, ct],
            header,
            overwrite=True,
        )
        fits.writeto(
            output_dir + "/Velocity/" + output_name + "_" + line_ + "_velocity_err.fits",
            velocities_errors_fits[:, :, ct],
            header,
            overwrite=True,
        )
        fits.writeto(
            output_dir + "/Broadening/" + output_name + "_" + line_ + "_broadening_err.fits",
            broadenings_errors_fits[:, :, ct],
            header,
            overwrite=True,
        )
    if absorption_maps:
        # Alongside the continuum products rather than in their own directory: like the
        # continuum, these are one scalar per pixel describing the light the lines sit on.
        for name, data in absorption_maps.items():
            fits.writeto(output_dir + "/" + output_name + "_" + name + ".fits", data, header, overwrite=True)
    fits.writeto(output_dir + "/" + output_name + "_Chi2.fits", chi2_fits, header, overwrite=True)
    fits.writeto(output_dir + "/" + output_name + "_continuum.fits", continuum_fits, header, overwrite=True)
    fits.writeto(output_dir + "/" + output_name + "_continuum_error.fits", continuum_error_fits, header, overwrite=True)
