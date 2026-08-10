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

from luci.instrument.filters import pca_scale_indices


def subtract_standard(sky, bkg, binning=None):
    """Subtract a measured background, scaled by the spaxel count of the bin."""
    if bkg is None:
        return sky
    n_spaxels = binning**2 if binning else 1
    return sky - bkg * n_spaxels


def pca_background(coefficients, pca_vectors, pca_mean):
    """
    Rebuild a background spectrum from its PCA coefficients.

    Only as many components are summed as the pixel has coefficients: `create_background_subspace`
    keeps `n_components_keep` coefficients per pixel but the PCA object still carries all
    `n_components` eigenspectra, so iterating over the eigenspectra (there can be more of them) ran
    off the end of the coefficient vector.
    """
    n = min(len(coefficients), len(pca_vectors))
    return pca_mean + np.sum([coefficients[i] * pca_vectors[i] for i in range(n)], axis=0)


def combine_pca_coefficients(coefficients):
    """
    Collapse a group of pixels' PCA coefficients into one set, for a spectrum that
    is the *sum* of those pixels (a bin, or a region).

    The mean, not the sum. A pixel's background is ``pca_mean + sum_i c_i v_i``, so
    the background of N summed pixels is ``N * pca_mean + sum_i (sum_p c_ip) v_i``.
    Rebuilding from the mean coefficients gives exactly that, divided by N -- and
    `subtract_pca` rescales onto the observed continuum anyway, so the 1/N is
    immaterial while the *shape* is right. Rebuilding from the summed coefficients
    instead leaves `pca_mean` under-weighted by N relative to the components, an
    error in shape that grows with the group and that no rescaling can undo (B29).

    Args:
        coefficients: Coefficients for the group, any shape ending in the
            component axis (e.x. ``(binning, binning, n_components)``)

    Return:
        1D array of combined coefficients
    """
    coefficients = np.asarray(coefficients)
    return np.nanmean(coefficients.reshape(-1, coefficients.shape[-1]), axis=0)


def subtract_pca(sky, background, spectrum_axis, filter_name):
    """
    Scale a PCA background onto this spectrum's continuum and subtract it.

    The scale factor is the peak of the observed spectrum inside the filter's
    line-free window, so the background matches the local continuum level.
    """
    lower, upper = pca_scale_indices(filter_name, spectrum_axis)
    scale = np.nanmax(sky[lower:upper])
    return sky - scale * background
