.. _fit_options:

Fit Options
===========

`LUCI` has several options that can either be included or not when fitting. The details of these
options is explained in the `howLuciWorks` page, but we wanted to reiterate them here.

The fitting methods in `LUCI` are: `cube.fit_cube(), cube.fit_entire_cube(), cube.fit_region(), cube.fit_spectrum_region(), cube.fit_pixel()`.
With the exception of `cube.fit_spectrum_region()`, these methods fit each pixel in a predefined region of the cube.

In addition to the required arguments, we have the following set of optional arguments:

- bkg: This is where you define the background region that will be subtracted from each spectrum. This is technically optional, but you really should do this...
- binning: This will bin your cube into larger pixels of nxn size where n is the integer you supply (i.e. `binning=2` will create 2x2 bins)
- bayes_bool: This activates the Bayesian inference calculations using EMCEE (`bayes_bool=True`). Might as well grab a beer (or two) while you wait for it to run because it's going to be a while...
- output_name: If you want a different output suffix, please add that information here. (i.e. `output_name='SuperCoolName'`)
- uncertainty_bool: This is if you don't want to wait for the full Bayesian inference to run. Instead, it calculates the inverse Hessian of the fit matrix and takes its diagonals as the errors on the fit parameters.
- n_threads: This sets the number of threads that will be used in the fits. Choose this based on the number of threads available to you. I like `n_threads=4`. **Note that this frequently doesn't work in a jupyter notebook setting**.
- mdn: This boolean determines whether or not a mixture density model is used to obtain the initial guesses. Details on this algorithm can be found at (https://arxiv.org/abs/2111.12755). The default value is False.
- nii_cons: This boolean determines whether or not we constrain the NII doublet emission to the 1/3rds rule. By default, this is included.
- n_stoch: This integer sets the number of attempts to fit the model to the spectrum. Instead of taking the ML estimates as
the initial values in the fit, we will randomly select a value from a region around the ML estimate. This introduces some stochasticity
to the fit, and, in some cases, can guide the optimization algorithm to a better final fit. The fit which minimizes the likelihood
out of all the n_stoch fits will be taken as the best fit.
- spec_min: Minimal wavelength value over which the fit will be computed
- spec_max: Maximal wavelength value over which the fit will be computed.


`LUCI` has the abilitiy to restrict the wavelength region over which the fit is performed. So, for instance, if you only wish to fit
Halpha and the NII-doublet, you can set restrict the wavelength to be between 15000 and 15250 cm^-1^ by adding two
arguments to your fit function: `spec_min` and `spec_max`. Note that these values need to be in wavenumber. We have seen, in
some cases of low signal-to-noise, that adding this restriction helps the fit algorithm.
