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

1. luci_path = /the/path/to/LUCI  # Path to Luci
2. cube_dir = '/path/to/data/cube'  # Path to data cube
3. cube_name = 'name_of_data_cube'  # don't add .hdf5 extension
4. object_name = 'name_of_object'
5. redshift = 0.01  # Redshift of object
6. resolution = 5000  # Resolution of the ML reference spectrum


For example:

.. code-block:: python

    # Using Machine Learning Algorithm for Initial Guess
    Luci_path = '/media/carterrhea/carterrhea/SIGNALS/LUCI/'
    cube_dir = '/media/carterrhea/carterrhea/M33'  # Path to data cube
    cube_name = 'M33_Field7_SN3.merged.cm1.1.0'  # don't add .hdf5 extension
    object_name = 'M33_Field7'
    redshift = -0.0006  # Redshift of M33
    resolution = 5000
    ML_bool = True

Although the first three arguments are rather self explanatory, it is worth discussing the others.
The redshift is provided so that we can shift x-axis of the spectra to the rest-frame.
As discussed, this enables better fitting. The redshift of an object
can be found at `http://cdsportal.u-strasbg.fr/ <http://cdsportal.u-strasbg.fr/>`_ .

The `resolution` is the resolution of the reference spectrum
that will be used for the initial fit estimates. Although this was previously described, I'll remind
you that you can find more information at `https://sitelle-signals.github.io/Pamplemousse/index.html <https://sitelle-signals.github.io/Pamplemousse/index.html>`_.
The Luci directory contains already a handful differing resolutions (R ~ 1000, 1800, 2000, 2500, 3000, 3500, 4000 ,4500, 5000, 7000).
If you require a different resolution for your work, please send me an email at carter.rhea@umontreal.ca.

Note that `ML_bool=True` by default.


If you do not wish to use the machine learning methodology to estimate the initial values for
the velocity, broadening, and amplitude of the line, please simply include the argument **ML_bool=False.

.. code-block:: python

    # Not Using Machine Learning Algorithm for Initial Guess
    Luci_path = '/media/carterrhea/carterrhea/SIGNALS/LUCI/'
    cube_dir = '/media/carterrhea/carterrhea/M33'  # Path to data cube
    cube_name = 'M33_Field7_SN3.merged.cm1.1.0'  # don't add .hdf5 extension
    object_name = 'M33_Field7'
    redshift = -0.0006  # Redshift of M33
    resolution = 5000
    ML_bool = False


With these parameters set, we can invoke `LUCI` with the following command:

.. code-block:: python

    cube = Luci(luci_path, cube_dir+'/'+cube_name, cube_dir, object_name, redshift, resolution, ML_bool)

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

    vel_map, broad_map, flux_map, chi2_fits = cube.fit_cube(line_list, fit_function, vel_rel, sigma_rel, x_min, x_max, y_min, y_max)

line_list is a list of lines to fit (e.x. ['Halpha']), fit function is the fitting function to be used (e.x. 'gaussian'), and the remaining
arguments are the x and y bounds (respectively) of the bounding box. The vel_rel parameter describes the relational constraints between the lines. For example,
if we are fitting three lines and we want the velocities of the second and third lines to be tied to one another, we would simply set vel_rel=[1,2,2].
If we wanted all the lines tied, then we would put [1,1,1]. The sigma_rel parameter functions in the exact same way except it affects the broadening (velocity dispersion) of the lines.

For example if we want to fit the three lines in SN3 with all their parameters tied together in a small region, we would do:

.. code-block:: python

        vel_map, broad_map, flux_map, chi2_fits = cube.fit_cube(['Halpha', 'NII6548', 'NII6583', 'SII6716', 'SII6731'], 'gaussian', [1,1,1,1,1], [1,1,1,1,1], 500, 1100, 700, 1300)

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

    Luci_path = '/media/carterrhea/carterrhea/SIGNALS/LUCI/'  # Path to Luci
    cube_dir = '/media/carterrhea/carterrhea/M33'  # Path to data cube
    cube_name = 'M33_Field7_SN3.merged.cm1.1.0'  # don't add .hdf5 extension
    object_name = 'M33_Field7'
    redshift = -0.0006  # Redshift of M33
    resolution = 5000

    cube = Luci(cube_dir+'/'+cube_name, cube_dir, object_name, redshift, ML_ref, ML_model)

    cube.create_deep_image()

    vel_map, broad_map, flux_map, chi2_fits = cube.fit_cube(['Halpha', 'NII6548', 'NII6583', 'SII6716', 'SII6731'], 'gaussian', [1,1,1,1,1], [1,1,1,1,1], 500, 1100, 700, 1300)
