.. LUCI documentation master file, created by
   sphinx-quickstart on Sun Jun 20 15:48:57 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LUCI's documentation!
================================

.. image:: luci-basket.jpg
    :width: 800
    :alt: Alternative text

LUCI is a general purpose fitting pipeline built specifically with SITELLE IFU
data cubes in mind; however, if you need to fit any emission line spectra, LUCI
will be able to help!

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5385351.svg
   :target: https://doi.org/10.5281/zenodo.5385351

You can download the example data using the following command:


.. code-block:: bash

    wget -O NGC6946_SN3.hdf5 https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/2307000z.hdf5?RUNID=xc9le6u8llecp7fp

This will download the hdf5 file for SN3 (R~400) NGC 6946. The file is just under 900 Mb,
so the download may take a while.
Note you may need to change the name of the HDF5 file to `NGC6946_SN3.merged.cm1.1.0.hdf5`.

The region files used in the examples can be
found in the 'Examples/regions' folder. To run the examples, place these region files in the same
directory as the hdf5 file.


If you are new to SITELLE, I suggest you start with the Basic Example. If you
already know how the fitting procedure works, you can go straight to Basic Example Lite.


Please note that this documentation is not exhaustive. If you have any questions or
wish to see any new implementation, please send a message to carter.rhea@umontreal.ca.


Lines Available
---------------
+-----------+-----------------+
|Line Name  | Wavelength (nm) |
|           |                 |
+-----------+-----------------+
|Halpha     | 656.280         |
+-----------+-----------------+
|NII6548    | 654.803         |
+-----------+-----------------+
|NII6583    | 658.341         |
+-----------+-----------------+
|SII6716    | 671.647         |
+-----------+-----------------+
|SII6731    | 673.085         |
+-----------+-----------------+
|OII3726    | 372.603         |
+-----------+-----------------+
|OII3729    | 372.882         |
+-----------+-----------------+
|OIII4959   | 495.891         |
+-----------+-----------------+
|OII5007    | 500.684         |
+-----------+-----------------+
|Hbeta      | 486.133         |
+-----------+-----------------+


The available fitting functions are as follows:
'gaussian', 'sinc', 'sincgauss'


The output files will be in the following structure:

.. image:: LuciOutput.svg
    :width: 800
    :alt: LUCI Output Structure


Citing LUCI
-----------
If you cite LUCI, please use both the following citations.

Software Citation: Carter Lee Rhea, Laurie Rousseau-Nepton, Jessie Covington, Leo Alcorn, Benjamin Vigneron, Julie Hlavacek-Larrondo, & Louis-Simon Guité. (2021). crhea93/LUCI: Luci Updates (v1.1). Zenodo. https://doi.org/10.5281/zenodo.5730149

Paper Citation:  Carter Rhea et al 2021 Res. Notes AAS 5 208

If you use the mixture density network (MDN) implementation, please include the following citation:
Carter Rhea et al 2021 Res. Notes AAS 12 276

Prerequisites
^^^^^^^^^^^^^
    .. toctree::
       :maxdepth: 2
       :caption: Prerequisites:

       howLuciWorks
       uncertainties
       fit_options
       wavelength_corrections
       license

Examples
^^^^^^^^
    .. toctree::
       :maxdepth: 2
       :caption: Example Modules:

       example_basic
       example_basic_lite
       example_fit_ngc6888
       example_sn1_sn2
       example_fit_region
       example_fit_single_spectrum
       example_fit_mask
       example_fit_snr
       example_pixel
       example_synthetic_spectrum
       example_double_fit
       example_visualize
       example_wvt



Pipeline
^^^^^^^^
   .. toctree::
      :maxdepth: 1
      :caption: Python Modules:

      fits
      luci
      params
      plot
      bayes


FAQ & Errors
^^^^^^^^^^^^
  .. toctree::
     :maxdepth: 1
     :caption: FAQ:

     FAQ


.. image:: Luci-Prancing.jpg
    :width: 400
    :alt: Alternative text




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

The software is protected under the :ref:`license`.
