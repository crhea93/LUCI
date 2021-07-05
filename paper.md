---
title: 'Luci: A Python package for SITELLE spectral analysis'
tags:
  - Python
  - astronomy
  - kinematics
  - SITELLE
  - IFU
authors:
  - name: Carter L. Rhea
    orcid: 0000-0003-0872-7098
    affiliation: "1, 2"
  - name: Julie Hlavacek-Larrondo
    affiliation: 1
affiliations:
 - name: Département de Physique, Université de Montréal, Succ. Centre-Ville, Montréal, Québec, H3C 3J7, Canada
   index: 1
 - name: Centre de Recherche en Astrophysique du Québec (CRAQ), Québec, QC, G1V 0A6, Canada
   index: 2
date: 5 July 2021
bibliography: paper.bib

---

# Summary

High-resolution optical integral field units (IFUs) are rapidly expanding our knowledge
of extragalactic emission nebulae in galaxies and galaxy clusters. By studying the spectra
of these objects -- which include classic HII regions, supernova remnants, planetary nebulae,
and cluster filaments -- we are able to constrain their kinematics (velocity and velocity dispersion).
In conjunction with additional tools, such as the BPT diagram (REF), we can further classify
emission regions based on strong emission-line flux ratios. `LUCI` is a simple-to-use python module
intended to facilitate the rapid analysis of IFU spectra. `LUCI` does this by integrating
well-developed pre-existing python tools such as `astropy` and `scipy` with new
machine learning tools for spectral analysis (Rhea et al. 2020a).

Recent advances in the science and technology of IFUs have resulted in the creation
of the high-resolution, wide field-of-view (11 arcmin x 11 arcmin) instrument SITELLE (REF)
at the Canada-France-Hawaii Telescope. Due to the large field-of-view and the small
angular resolution of the pixels (0.32 arcseconds), the resulting data cubes contain
over 4 million spectra. Therefore, a simple, fast, and adaptable fitting code is
paramount -- it is with this in mind that we created `LUCI`.


# Functionality
At her heart, `LUCI` is nothing more than a wrapper for a `scipy.optimize.minimize`
function call. Since SITELLE data cubes are available as **HDF5** files, `LUCI` was built
to parse the original file and create an instance of a `LUCI` cube which contains the 2D header information and
a 3D numpy array (Spatial X, Spatial Y, Spectral). Once the data cube has been successfully converted
to a `LUCI` cube, there are several options for fitting different regions of the cube
(e.g., `fit_cube`, `fit_entire_cube`, `fit_region`) or fitting single spectra (e.g., `fit_spectrum_region`).
The primary use case of `LUCI` is to fit a region of a cube defined either as a box (in this case, the
user would employ the `fit_cube` method and pass the limits of the bounding box) or to
fit a region of the cube defined by a standard *ds9* file (in this case, the user would pass the name of the
region file to `fit_region`). Regardless of the region being fit, the user need only specify the
lines they wish to fit and the fitting function. We currently support all standard lines in
the SN1 filter ([OII3626] & [OII3629]), SN2 filter ([OIII4959], [OIII5007], & Hbeta), and SN3 filter ([SII6716], [SII6731], [NII6548], [NII6583], & Halpha).    
The user also must chose between three fitting functions: a pure Gaussian, and pure sinc function, or a sinc function convolved with a Gaussian.
In either case, `LUCI` will solve for the three primary quantities of interest which are the **amplitude** of the line, the **position** of the line (often described as the velocity and quoted in km/s), and the **broadening** of the line (often described as the velocity dispersion and quoted in units of km/s

The three fitting functions are mathematically described below where p0 corresponds to the **amplitude**, p1 corresponds to the **position**, and p2 corresponds of the **broadening**.

The pure Gaussian function is expressed as
\begin{equation}
    f(x) = p0*exp(-(x-p1)^2/(2*p2^2))
\end{equation}

The pure since function is expressed as
\begin{equation}
    p0*(\frac{(x-p1)/p2}{(x-p1)/p2})
\end{equation}

The convolved sincgauss function is expressed as
\begin{equation}
    p0*exp(-b*^2)*((sps.erf(a-i*b)+erf(a+i*b))/(2*erf(a)))
\end{equation}

where *x* represents a given spectral channel, `a = p2/(\sqrt{2}*\sigma)`, `b=(x-p1)/(\sqrt(2)*\sigma)`, where `\sigma` is the
pre-defined width of the sinc function (see paper for details).

In each case, after solving for these values, the velocity and velocity dispersion are calculated using the following equations.



# Other Software


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures




# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
