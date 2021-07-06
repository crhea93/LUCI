.. _example_basic:

Basic Example
=============

The most fundamental use for `LUCI` is to read in an HDF5 file and fit a
region of the data cube. In this tutorial, we will outline how to do just that.

If you want to see the complete example code, please see the bottom of the page.
This is also available as a jupyter notebook (complete with output) under *Exmples/BasicExamples.ipynb* in the main Luci repository.

I am working on making the import easier (BTW).

We should start by import the appropriate modules.

.. code-block:: python

    import sys
    sys.path.insert(0, '/the/path/to/LUCI/')
    import LuciBase as Luci
    import LUCI.LuciPlotting as lplt


Remember that '/the/path/to/LUCI/' is the full path to the directory where you cloned
LUCI from github. For example, yours may look like '/home/carterrhea/LUCI/'.
We will also be highlighting the built in plotting functions found in `LUCI/LuciPlotting.py`.

The next step is to load/read the HDF5 data cube. To do this we **invoke** LUCI
by initiating an instance of her along with the proper parameters. First we
define the correct parameters:

1. cube_dir = '/path/to/data/cube'  # Path to data cube
2. cube_name = 'name_of_data_cube'  # don't add .hdf5 extension
3. object_name = 'name_of_object'
4. redshift = 0.01  # Redshift of object
5. ML_ref = '/the/path/to/LUCI/ML/Reference-Spectrum-R5000'  # Relative path to reference spectrum
6. ML_model = '/the/path/to/LUCI/ML/R5000-PREDICTOR-I'  # Relative path to train ML algorithm


For example:

.. code-block:: python

    # Using Machine Learning Algorithm for Initial Guess
    cube_dir = '/home/carterrhea/Documents'  # Path to data cube
    cube_name = 'A0426_SN3.merged.cm1.1.0'  # don't add .hdf5 extension
    object_name = 'NGC1275'
    redshift = 0.017284  # Redshift of NGC 1275
    ML_ref = '/the/path/to/LUCI/ML/Reference-Spectrum-R1800'
    ML_model = '/the/path/to/LUCI/ML/R1800-PREDICTOR-I'

Although the first three arguments are rather self explanatory, it is worth discussing the others.
The redshift is provided so that we can shift x-axis of the spectra to the rest-frame.
As discussed in `howLuciWorks`_, this enables better fitting. The redshift of an object
can be found at `http://cdsportal.u-strasbg.fr/ <http://cdsportal.u-strasbg.fr/>`_ .

The `ML_ref` argument contains the path to the reference spectrum with the appropriate resolution
that will be used for the initial fit estimates. Although this was previously described, I'll remind
you that you can find more information at `https://sitelle-signals.github.io/Pamplemousse/index.html <https://sitelle-signals.github.io/Pamplemousse/index.html>`_.
The Luci directory contains already a handful differing resolutions (R ~ 1800, 2000, 2500, 3000, 3500, 4000 ,4500, 5000, 7000).
If you require a different resolution for your work, please send me an email at carter.rhea@umontreal.ca.
Similarly, the `ML_model` argument contains the path to the trained network corresponding to the same
resolutions available for the `ML_ref` argument. Note that the naming conventions
follow the same structure as is indicated in the example. Therefore, for a resolution 2000
cube, we would set `ML_ref='ML/Reference-Spectrum-R2000'` and `ML_model='ML/R5000-PREDICTOR-I'`.

If you do not wish to use the machine learning methodology to estimate the initial values for
the velocity, broadening, and amplitude of the line, please simply set both parameters equal to **None**.

.. code-block:: python

    # Not Using Machine Learning Algorithm for Initial Guess
    cube_dir = '/home/carterrhea/Documents'  # Path to data cube
    cube_name = 'NGC1275-LowRes'  # don't add .hdf5 extension
    object_name = 'NGC1275'
    redshift = 0.017284  # Redshift of NGC 1275
    ML_ref = None
    ML_model = None


With these parameters set, we can invoke `LUCI` with the following command:

.. code-block:: python

    cube = Luci(cube_dir+'/'+cube_name, cube_dir, object_name, redshift, ML_ref, ML_model)

This reads the HDF5 file, transforms the data cube into a 3d numpy array, and updates the header to be of an appropriate form.
It also reads in the machine learning reference spectrum (we need the x-axis for interpolation purposes) and
creates the x-axis for the uninterpolated cube. Note that the first argument is the full path to the cube
and the second argument is the full path to the output directory (i.e. the output files will be located at cube_dir+'/Luci/'; the 'Luci' at the end is appended by the code itself).


.. code-block:: python

    cube.create_deep_image()

We can quickly make a *deep image* by collapsing (summing) the spectral axis.


At last, we can fit a region of the cube. There are three functions for fitting the cube: `fit_cube`, `fit_entire_cube`, and `fit_region`.
The first option, `fit_cube`, fits a rectangular region of the cube and is invoked by calling:

.. code-block:: python

    vel_map, broad_map, flux_map, chi2_fits = cube.fit_cube(line_list, fit_function, x_min, x_max, y_min, y_max)

line_list is a list of lines to fit (e.x. ['Halpha']), fit function is the fitting function to be used (e.x. 'gaussian'), and the remaining
arguments are the x and y bounds (respectively) of the bounding box.

For example:

.. code-block:: python

        vel_map, broad_map, flux_map, chi2_fits = cube.fit_cube(['Halpha', 'NII6548', 'NII6583', 'SII6716', 'SII6731'], 'gaussian', 1300, 1400, 550, 650)

This final command fits the regions and saves the velocity, velocity dispersion (broadening), amplitude, flux, and fit statistic (chi-squared)
maps in the output directory defined above. Additionally, it returns the velocity, velocity dispersion, flux, and fit statistics maps for plotting purposes.

To fit the entire cube, we would simply run the following instead:

.. code-block:: python

    vel_map, broad_map, flux_map, chi2_fits = cube.fit_entire_cube(line_list, fit_function)


Or we can fit an entire region

.. code-block:: python

    vel_map, broad_map, flux_map, chi2_fits = cube.fit_region(line_list, fit_function, region_file)

where `region_file` is the path to the ds9 region file save in **fk5** coordinates.

If you wish to activate the Bayesian MCMC implementation, simply add `bayes_bool=True` to any of the fit functions described above.

Additionally, **binning** can be applied by adding the `binning` argument to any of the above fit functions. For example, we
can bin 2x2 regions as such:

.. code-block:: python

    vel_map, broad_map, flux_map, chi2_fits = cube.fit_cube(['Halpha'], 'gaussian', 1300, 1400, 550, 650, binning=2)

And with those few lines, we have read in our data cube, created a *deep image* and fit the cube.

We can now visualize our fits with our specialized plotting functionality:

.. code-block:: python

    lplt.plot_map(vel_map, 'velocity', cube_dir)


The `LUCI.LuciPlotting.plot_map` function takes the map of interest, the name of the map (either 'velocity', 'broadening', or 'flux'),
and the output directory as arguments. Of course, we can also use simply `matplotlib` plotting functionality as well.

For clarity, we reproduce the commands required to obtain fits here:

.. code-block:: python

    cube_dir = '/home/carterrhea/Documents'  # Path to data cube
    cube_name = 'A0426_SN3.merged.cm1.1.0'  # don't add .hdf5 extension
    object_name = 'NGC1275'
    redshift = 0.017284  # Redshift of NGC 1275
    ML_ref = '/the/path/to/LUCI/ML/Reference-Spectrum-R1800'
    ML_model = '/the/path/to/LUCI/ML/R1800-PREDICTOR-I'

    cube = Luci(cube_dir+'/'+cube_name, cube_dir, object_name, redshift, ML_ref, ML_model)

    cube.create_deep_image()

    vel_map, broad_map, flux_map, chi2_fits = cube.fit_cube(['Halpha', 'NII6548', 'NII6583', 'SII6716', 'SII6731'], 'gaussian', 1300, 1400, 550, 650)
