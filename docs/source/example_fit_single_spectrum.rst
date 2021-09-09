.. _example_fit_single_spectrum:

Fit Single Spectrum
==================

In this example, we will extract a region of M33 field 7 and fit is using  `LUCI`.
This is a very basic example.

.. code-block:: python

    # Imports
    import sys
    sys.path.insert(0, '/media/carterrhea/carterrhea/SIGNALS/LUCI/')  # Location of Luci
    from LuciBase import Luci
    import LUCI.LuciPlotting as lplt

We now will set the required parameters. We are also going to be using our machine learning algorithm to get the initial guesses.

.. code-block:: python

    #Set Parameters
    # Using Machine Learning Algorithm for Initial Guess
    Luci_path = '/media/carterrhea/carterrhea/SIGNALS/LUCI/'
    cube_dir = '/media/carterrhea/carterrhea/M33'  # Path to data cube
    cube_name = 'M33_Field7_SN3.merged.cm1.1.0'  # don't add .hdf5 extension
    object_name = 'M33_Field7'
    redshift = -0.0006  # Redshift of M33
    resolution = 5000


We intialize our LUCI object

.. code-block:: python

    # Create Luci object
    cube = Luci(Luci_path, cube_dir+'/'+cube_name, cube_dir, object_name, redshift, resolution)

The output will look something like this:

.. image:: ReadingIn.png
    :alt: Luci Initialization Output


Let's extract a background region and take a look at it. The background region is defined in a ds9 region file called `bkg.reg`.

.. code-block:: python

  bkg_axis, bkg_sky = cube.extract_spectrum_region(cube_dir+'/bkg.reg', mean=True)  # We use mean=True to take the mean of the emission in the region instead of the sum


Now we can define our fit region and fit it!

.. code-block:: python

    axis, sky, fit_dict = cube.fit_spectrum_region(['NII6548', 'Halpha', 'NII6583', 'SII6716', 'SII6731'],
                                    'sincgauss', [1,1,1,1,1], [1,1,1,1,1],
                                    region='../Data/reg1.reg', bkg=bkg_sky)
    plt.plot(axis, sky, color='blue', label='spectrum')
    plt.plot(fit_dict['fit_axis'], fit_dict['fit_vector'], color='coral', label='fit')
    plt.xlim((15100, 15400))
    plt.legend()



And let's see how this looks

.. image:: example-single-spectrum-fit.png
    :alt: Single Spectrum Fit