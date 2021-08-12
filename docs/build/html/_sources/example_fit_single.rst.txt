.. _example_fit_single:

Fit Single Spectrum
===================
In this example, we are going to fit a single region of the science verification Abell 426 data cube (found at: https://www.cfht.hawaii.edu/Instruments/Sitelle/SITELLE_sv.php).

We will read in the data as usual using a LUCI cube object. We then will extract a background region and plot it.
We will then extract a spectrum from a square region around 1357<x<1367 and 608<y<618.
These values were chosen to correspond with the brightest region of the horseshoe. Finally, we use the LuciFit Fit object to fit the region.



.. code-block:: python

    # Imports
    import sys
    sys.path.insert(0, '/media/carterrhea/carterrhea/SIGNALS/LUCI/')  # Location of Luci
    from LuciBase import Luci
    import LUCI.LuciPlotting as lplt
    import matplotlib.pyplot as plt
    import LUCI.LuciFit as lfit
    from astropy.io import fits
    import numpy as np
    import keras

We now will set the required parameters. We are also going to be using our machine learning algorithm to get the initial guesses.

.. code-block:: python

    #Set Parameters
    # Using Machine Learning Algorithm for Initial Guess
    cube_dir = '/media/carterrhea/carterrhea/Benjamin'  # Path to data cube
    cube_name = 'A0426_SN3.merged.cm1.1.0'  # don't add .hdf5 extension
    object_name = 'NGC1275'
    redshift = 0.017284  # Redshift of NGC 1275
    ML_ref = '/media/carterrhea/carterrhea/SIGNALS/LUCI/ML/Reference-Spectrum-R1800'
    ML_model = '/media/carterrhea/carterrhea/SIGNALS/LUCI/ML/R1800-PREDICTOR-I'


We intialize our LUCI object

.. code-block:: python

    # Create Luci object
    cube = Luci(cube_dir+'/'+cube_name, cube_dir, object_name, redshift, ML_ref, ML_model)

Let's extract and visualize a background region we defined in ds9:

.. code-block:: python

    # Extract and visualize background
    bkg_axis, bkg_sky = cube.extract_spectrum_region(cube_dir+'/bkg.reg', mean=True)  # We use mean=True to take the mean of the emission in the region instead of the sum
    plt.plot(bkg_axis, bkg_sky)

    .. image:: example-single-fit-background.png
        :alt: Background output



We now fit our region

.. code-block:: python

    # fit region
    velocity_map, broadening_map, flux_map, chi2_map, mask = cube.fit_region(['OII3726', 'OII3729'], 'gaussian', [1,1], [1,1],
            region=cube_dir+'/reg1.reg', bkg=bkg_sky)


And let's check out what this looks like.

.. code-block:: python

    lplt.plot_map(np.log10(flux_map[:,:,0]), 'flux', cube_dir, cube.header, clims=[-17, -15])


.. image:: example-single-fit-fit.png
    :alt: Fit
