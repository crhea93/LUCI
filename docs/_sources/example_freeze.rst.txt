.. _example_freeze:

How to use LUCI with frozen parameters
======================================

In this example we will go through how to fit a spectrum with frozen velocity and broadening parameters. This may
be useful if you want to say fit the Halpha + NII complex to calculate the velocity and broadening of the gas -- then you
can use these values to constrain a fit on the SII doublet which is frequently weaker.

For this particular example, we will be using the Field 7 of M33 and fitting twice: with and without frozen parameters.

This example can be found as a jupyter notebook in "LUCI/Examples/SN3_freeze.ipynb".

.. code-block:: python

    # Imports
    import sys
    import numpy as np
    sys.path.insert(0, '/home/carterrhea/Documents/LUCI/')  # Location of Luci
    from LuciBase import Luci


Inputs
------
We simply need to load our LUCI cube like usual :) So we start by defining the path, the name of the cube,
the name of the object, the resolution (which isn't actually used here), and the redshift.



.. code-block:: python

    Luci_path = '/home/carterrhea/Documents/LUCI/'
    cube_dir = '/mnt/carterrhea/carterrhea/M33'  # Path to data cube
    cube_name = 'M33_Field7_SN3.merged.cm1.1.0'  # don't add .hdf5 extension
    object_name = 'M33_Field7'
    redshift = -0.0006  # Redshift of M33
    resolution = 5000


As usual we will extract a background

.. code-block:: python

    bkg_axis, bkg_sky = cube.extract_spectrum_region(cube_dir+'/bkg.reg', mean=True)  # We use mean=True to take the mean of the emission in the region instead of the sum
    lplt.plot_spectrum(bkg_axis, bkg_sky)
