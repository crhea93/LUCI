.. _uncertainties:

Uncertainties
=============

In order to calculate uncertainties, set `uncertainty_bool=True` as an argument
of a fitting function (i.e. `cube.fit_cube()` or `cube.fit_region()`).

Before we describe how we calculate uncertainties using the standard vs Bayesian approach, let's look at the theoretical
calculation of uncertainties using the Cramer-Rao lower bound. To calculate the uncertainty of a Gaussian line, :math:`\delta \mu`, we can use the following equation:

.. math::
    \delta \mu = \frac{\sqrt{2}\cdot{}\text{FWHM}}{\text{SNR}\cdot{} \sqrt{N}}


where FWHM is the full-width half-max defined as :math:`\text{FWHM}=2 \sqrt{2\ln{2}} \cdot{} \sigma` where $\sigma$ is the standard deviation of the Gaussian line.
SNR is the signal-to-noise ratio and $N$ is the number of independent data points across the line profile. If you have multiple lines
that are independent you can approximate the uncertainty as

.. math::
    \delta \mu_{multi} \approx \delta \mu_{single} \cdot{} \frac{1}{\sqrt{M}}

where $M$ is the number of lines.

We can estimate $N$ as

.. math::
    N = \frac{\text{FWHM}}{\Delta \nu}

where :math:`\Delta \nu` is the sampling step size.

Let's walk through an example. Assume that the SNR is 100,:math:`$\sigma=10` (a typical value -- keep in mind this is in units of :math:`cm^{-1}`),
and we are fitting H\alpha. For a resolution 5000 observation, this is typical of SN3 for the SIGNALS program, :math:`\Delta \nu \approx 2.254 cm^{-1}`
Thus :math:`N = \frac{2.335 \cdot{} 10 \text{cm}^{-1} }{2.254 \text{cm}^{-1}} \approx 10`.

We can then calculate the theoretical uncertainty in terms of km/s. Note that the velocity resolution would be

.. math::
     \Delta v = c \cdot{} \frac{\Delta \nu}{\nu_0}  = 299,792 \text{km/s} * \Bigg(\frac{2.524 \text{cm$^{-1}$}}{15244.4\text{cm$^{-1}$}}\Bigg) \approx 49.6 \text{km/s}

which is more or less the FWHM.

So

.. math::
    \delta \mu = \frac{ \sqrt{2} \cdot{} 49.64 \text{km/s}^{-1} }{ 100 \cdot{} \sqrt{10} } \approx 0.222 \text{km/s}


So, for this particular case, the best possible uncertainty on the velocity is around 0.2 km/s.


For an illustration of this on synthetic data in `LUCI` check out the example entitled `Fit-Synthetic`.


----

Since uncertainty estimates are often crucial in astrophysical calculations, we apply
a full Bayesian MCMC approach (using the python module *emcee*). The likelihood function
is defined as a standard Gaussian function. Additionally, we employ the same priors described
above for the fitting function bounds.

Likelihood Function
-------------------
We assume a standard Gaussian likelihood function. Note that :math:`\sigma` is the
error on the measurement (also noted as noise -- see next section).

.. math::
    LL = -0.5 * \Sigma((y - model) ^ 2 / \sigma^2 + log(2 * \pi * \sigma^2))


.. toctree::
   :maxdepth: 2


Noise Calculation
-----------------

In order to estimate the uncertainties (and complete the fits), we assume a homogenous
noise level associated with the the instrument. This is then accepted as the noise
over the entire spectrum in the filter. This is calculated for each individual spectrum
by considering a region outside of the filter (i.e. where the transmission is zero).
We then take the noise as the standard deviation of the region. This is typically
on the order of 1% of the flux in high SNR regions. We take the following wavelength regions:

SN1: 25300 - 25700
^^^^^^^^^^^^^^^^^^

.. image:: SN1_noise.png
    :alt: Luci Initialization Output

SN1 filter of example background (M33 Field 7). The noise region is bounded
by the magenta box.

SN2: 18600 - 19000
^^^^^^^^^^^^^^^^^^

.. image:: SN2_noise.png
    :alt: Luci Initialization Output

SN2 filter of example background (M33 Field 7). The noise region is bounded
by the magenta box.

SN3: 16000 - 16400
^^^^^^^^^^^^^^^^^^



Hessian Approach
----------------

The calculation of fit uncertainties is a very important consideration. As previously
discussed, we already have a methodology to calculate the uncertainties using an MCMC Bayesian
approach. However, this method can be extremely time-consuming. Thus, we offer a default
uncertainty estimate measurement based solely off the best-fit parameters.

The algorithm is as follows:
 - Calculate the best-fit parameters as previously discussed
 - Calculate the Hessian matrix of the likelihood function given the best-fit parameters.
 - Calculate the negative inverse of the Hessian -- this yield the covariance matrix
 - Calculate the square root of the diagonals of the covariance matrix

In this manner, we calculate the 1-sigma uncertainties on our  fit parameters. We
further propagate these to the velocity and broadening by calculating the relative error.


We calculate the Hessian matrix manually using finite differences -- the implementation can be found in `LuciUtility.py/hessianComp()`. 
 Previously, we used other packages; however, these introduced
unnecessary overhead and served as a bottleneck in our fitting scheme.