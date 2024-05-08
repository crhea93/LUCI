.. _example_basic_lite:

Basic Example Lite
==================

This example is the much condensed version of our basic example for those of us
already familiar with the parameters and how `LUCI` works. Let's get started!

Let's start with importing `LUCI`. Just as descsribed on the *readthedocs* page, we need to point our system path to our install location of the **LUCI** folder.
We will additionally import the Luci plotting functions :)

You can download the example data using the following command:


.. code-block:: bash

    !wget -O M33.hdf5  https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/raven/files/cadc:CFHT/2309128z.hdf5

This will save a file called `M33.hdf5` to the directory where you run this command. Do not be concerned if it takes a few minutes (or more than a few) to download.

The region files used in the examples can be
found in the 'Examples/regions' folder. To run the examples, place these region files in the same
directory as the hdf5 file.


.. code-block:: python

    # Imports
    import sys
    sys.path.insert(0, '/media/carterrhea/carterrhea/SIGNALS/LUCI/')  # Location of Luci
    from LuciBase import Luci
    import LUCI.LuciPlotting as lplt

Set the required parameters. We will be using our machine learning algorithm to get the initial guesses - this happens under the hood in `LuciFit`, so that the user is not required to think about the initial guess

.. code-block:: python

    # Initialize paths and set parameters
    Luci_path = '/home/carterrhea/Documents/LUCI/'
    cube_dir = '/export/home/carterrhea/M33'  # Path to data cube
    cube_name = 'M33_SN3'  # don't add .hdf5 extension
    object_name = 'M33'
    filter_name = 'SN3'
    redshift = -0.0006  # Redshift of object
    resolution = 5000


We intialize our LUCI object

.. code-block:: python

    # Create Luci object
    cube = Luci(Luci_path, cube_dir+'/'+cube_name, cube_dir, object_name, redshift, resolution)

The output will look something like this:

.. image:: ReadingIn.png
    :alt: Luci Initialization Output


Let's quickly create a deep frame

.. code-block:: python

    # Create Deep Image
    cube.create_deep_image()

Let's extract a background region and take a look at it. The background region is defined in a ds9 region file in the `Examples/regions` folder in the LUCI directory.

.. code-block:: python

  bkg_axis, bkg_sky = cube.extract_spectrum_region(Luci_path+'Examples/regions/bkg_M33.reg', mean=True)  # We use mean=True to take the mean of the emission in the region instead of the sum
  lplt.plot_spectrum(bkg_axis, bkg_sky)
  
.. image:: M33_BKG.png
    :alt: M33_BKG

We now fit part of our cube defined by the bounding box 1200 < x < 1350 and 1700 < y < 1950 with a SincGauss on the Halpha line, the NII-doublet, and the SII-doublet with a binning of 2.
We are also going to constrain our velocities and sigmas.  We can also run multiple threads with the `n_threads` argument.


For this example, we do not calculate the errors because it slows down calculations, but note that it can easily be done by adding the argument `uncertainty_bool=True`. If you want the full Bayesian calculation you can add `bayes_bool=True`.


.. code-block:: python

    # Fit!
    vel_map, broad_map, flux_map, ampls_map = cube.fit_cube(
        ['Halpha', 'NII6548', 'NII6583', 'SII6716', 'SII6731'], 
        'sincgauss', 
        [1,1,1,1,1], [1,1,1,1,1], 
        1200, 1350,
        1700, 1950, 
        bkg=bkg_sky, binning=2, 
        n_threads=2)



Let's take a look at the velocity map. We can play with the colorbar limits with the `clims` argument. Please note that the
flux plot is automatically scaled by log10. However, the velocity and broadening maps are not scaled automatically.

.. code-block:: python

    lplt.plot_map(flux_map[:,:,0], 'flux', object_name=object_name, filter_name=filter_name, output_dir=cube_dir, header=cube.header, clims=[-18, -14])

And let's see what this looks like!

.. image:: M33_Flux.png
    :alt: M33_Flux

We can also plot the velocity and broadening.

.. code-block:: python

    lplt.plot_map(vel_map[:,:,0], 'velocity', object_name=object_name, filter_name=filter_name, output_dir=cube_dir, header=cube.header, clims=[0, 100])

.. image:: M33_velocity.png
    :alt: M33_Velocity
   
.. code-block:: python

    lplt.plot_map(broad_map[:,:,0], 'broadening', object_name=object_name, filter_name=filter_name, output_dir=cube_dir, header=cube.header, clims=[0, 40])
    
.. image:: M33_Broadening.png
    :alt: M33_Broadening

The resulting data maps will be placed in a folder called *Luci_outputs*. Inside there, you
will find additional folders containing the Flux, Amplitude, Velocity, and Broadening maps
for each line and their uncertainties (if calculated).
