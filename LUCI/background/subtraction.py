"""
Removing the sky background from a spectrum.

Two schemes are supported:

* ``standard`` -- subtract a single background spectrum the caller measured,
  scaled by the number of spaxels that went into the target spectrum.
* ``pca`` -- rebuild a per-pixel background from the PCA model
  (``create_background_subspace``) and scale it onto the observed continuum.

Both used to be written out longhand in ``fit_calc`` and again in ``fit_pixel``,
which is how the two drifted apart (B3, B5, B6).
"""

from __future__ import annotations

import numpy as np

from LUCI.instrument.filters import pca_scale_indices


def subtract_standard(sky, bkg, binning=None):
    """Subtract a measured background, scaled by the spaxel count of the bin."""
    if bkg is None:
        return sky
    n_spaxels = binning**2 if binning else 1
    return sky - bkg * n_spaxels


def pca_background(coefficients, pca_vectors, pca_mean):
    """Rebuild a background spectrum from its PCA coefficients."""
    return pca_mean + np.sum([coefficients[i] * pca_vectors[i] for i in range(len(pca_vectors))], axis=0)


def subtract_pca(sky, background, spectrum_axis, filter_name):
    """
    Scale a PCA background onto this spectrum's continuum and subtract it.

    The scale factor is the peak of the observed spectrum inside the filter's
    line-free window, so the background matches the local continuum level.
    """
    lower, upper = pca_scale_indices(filter_name, spectrum_axis)
    scale = np.nanmax(sky[lower:upper])
    return sky - scale * background
