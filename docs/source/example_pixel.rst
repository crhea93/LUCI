.. _example_basic:

Fit Pixel
=============

In this notebook we will fit a single pixel in a data cube for NGC628.


This is also available as a jupyter notebook (complete with output) under *Exmples/Fit-Pixel.ipynb* in the main Luci repository.

You can download the example data using the following command:


.. code-block:: bash

    wget -O NGC6946_SN3.hdf5 https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/2307000z.hdf5?RUNID=xc9le6u8llecp7fp


This will download the hdf5 file for SN3 (R~400) NGC 6946. The file is just under 900 Mb,
so the download may take a while.
Note you may need to change the name of the HDF5 file to `NGC6946_SN3.merged.cm1.1.0`.

The region files used in the examples can be
found in the 'Examples/regions' folder. To run the examples, place these region files in the same
directory as the hdf5 file.

We should start by import the appropriate modules.

.. code-block:: python

    import os
    import sys

    import numpy as np
    import matplotlib.pyplot as plt

    # Get location of LUCI
    path = os.path.abspath(os.path.pardir)
    sys.path.insert(0, path)  # add LUCI to the available paths

    from LuciBase import Luci
    import LUCI.LuciPlotting as lplt
    %config Completer.use_jedi=False  # enable autocompletion when typing in Jupyter notebooks





For example:

.. code-block:: python

    # Initialize paths and set parameters
    Luci_path = path+'/'
    cube_dir = '/home/carterrhea/Documents/Luci_test'  # Full path to data cube
    cube_name = 'NGC6946_SN3'  # don't add .hdf5 extension
    object_name = 'NGC6949'
    redshift = 0.00015  # Redshift of object
    resolution = 1000



With these parameters set, we can invoke `LUCI` with the following command:

.. code-block:: python

    cube = Luci(luci_path, cube_dir+'/'+cube_name, cube_dir, object_name, redshift, resolution, ML_bool)

Now we should get our background region.

.. code-block:: python

    # We use 'mean = True' to take the mean of the emission in the region instead of the sum
    bkg_axis, bkg_sky = cube.extract_spectrum_region(cube_dir+'/bkg.reg', mean=True)
    lplt.plot_spectrum(bkg_axis, bkg_sky)

We will now fit a single pixel and take a look at the fit. This fit commands has all the same options as all the other commands except for binning :)

.. code-block:: python

    axis, sky, fit_dict = cube.fit_pixel(
        ['Halpha', 'NII6548', 'NII6583'],  # lines
        'gaussian',   # fit function
        [1,1,1],  # velocity relationship
        [1,1,1],  # sigma relationship
        1250, 1045,    # x & y coordinate
        bkg=bkg_sky, uncertainty_bool=True
    )

We can plot the fit with the following:

.. code-block:: python

    lplt.plot_fit(axis, sky, fit_dict['fit_vector'])

