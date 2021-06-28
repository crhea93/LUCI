.. LUCI documentation master file, created by
   sphinx-quickstart on Sun Jun 20 15:48:57 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LUCI's documentation!
================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   howLuciWorks
   license



LUCI is a general purpose fitting pipeline built specifically with SITELLE IFU
data cubes in mind; however, if you need to fit any emission line spectra, LUCI
will be able to help!


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

Examples
^^^^^^^^
    .. toctree::
       :maxdepth: 1
       :caption: Example Modules:

       example_basic

Pipeline
^^^^^^^^
   .. toctree::
      :maxdepth: 1
      :caption: Python Modules:

      fits


.. image:: Luci-Prancing.jpg
    :width: 400
    :alt: Alternative text


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

The software is protected under the :ref:`license`.
