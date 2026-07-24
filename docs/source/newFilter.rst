.. _newFilter:

Adding a New Filter
===================

LUCI knows about a fixed set of SITELLE filters. Every one of them needs four things before a
cube taken with it can be fit:

1. **Instrument parameters** -- the folding order and step size, which set the spectral axis.
2. **A transmission curve** -- ``Data/<FILTER>_filter.dat``.
3. **A handful of hard-coded wavenumber windows** -- where to fit, where to measure the noise,
   where to measure the continuum.
4. **A reference spectrum and a trained network** -- ``ML/Reference-Spectrum-R<res>-<FILTER>.fits``
   and ``ML/R<res>-PREDICTOR-I-<FILTER>/``, which supply the machine-learning initial guess.

This page walks through all four using **SN4** (the narrow Halpha filter, offered since 24B) as
the worked example. SN4 is already done -- everything below is committed -- so use it as the
template for SN5, SN6, or anything else that comes along.

.. note::
   Steps 1-3 are enough to fit a cube with ``ML_bool=False``. Step 4 is only needed for the
   machine-learning initial guess.


Step 1: Get the instrument parameters
-------------------------------------

The folding order and step size are not on the CFHT web pages, but they are stored in ORB's
filter files. Grab the one for your filter from
`thomasorb/orb <https://github.com/thomasorb/orb/tree/master/orb/data>`_ and read its attributes:

.. code-block:: python

    import h5py
    f = h5py.File('filter_SN4.hdf5', 'r')
    print(dict(f.attrs))
    # {'filter': 'SN4', 'order': 15, 'step': 5304.168,
    #  'bandpass_min_nm': 652.5, 'bandpass_max_nm': 664.0, ...}

The spectral axis of a cube spans the free spectral range of that order,

.. math::

    \sigma_{min} = \frac{\text{order}}{2\,\text{step}}\times 10^7 \times \frac{1}{\cos\theta}
    \qquad
    \sigma_{max} = \frac{\text{order}+1}{2\,\text{step}}\times 10^7 \times \frac{1}{\cos\theta}

with :math:`\theta = 11.96^\circ` the interferometer angle. For SN4 that is
**14454 - 15417 cm-1 (648.6 - 691.9 nm)**. Note how much wider this is than the 652 - 665 nm
pass band -- that turns out to be very convenient, because it means there is a large stretch of
the axis that the filter blocks completely and where the noise can be measured cleanly.

Add the order and step to ``Spectrum.__init__`` in ``LUCI/LuciSim.py``:

.. code-block:: python

    elif self.filter == 'SN4':  # Narrow Halpha filter (652 - 665 nm)
        self.delta_x = 5304.168
        self.order = 15

Here are the values for every SITELLE filter, for reference:

+--------+-------+----------+-------------------+---------------------+
|Filter  | Order | Step (nm)| Band pass (nm)    | Free spec. range    |
|        |       |          |                   | (cm-1)              |
+========+=======+==========+===================+=====================+
|SN1     | 8     | 1647.0   | 362.6 - 385.6     | 24825 - 27929       |
+--------+-------+----------+-------------------+---------------------+
|SN2     | 6     | 1680.0   | 479.8 - 513.2     | 18253 - 21296       |
+--------+-------+----------+-------------------+---------------------+
|SN3     | 8     | 2943.0   | 647.7 - 685.0     | 13893 - 15630       |
+--------+-------+----------+-------------------+---------------------+
|SN4     | 15    | 5304.168 | 652.5 - 664.0     | 14454 - 15417       |
+--------+-------+----------+-------------------+---------------------+
|SN5     | 13    | 3510.309 | 499.5 - 505.5     | 18928 - 20384       |
+--------+-------+----------+-------------------+---------------------+
|SN6     | 8     | 2258.922 | 499.0 - 502.5     | 18100 - 20363       |
+--------+-------+----------+-------------------+---------------------+
|C3      | 6     | 1785.0   | 511.0 - 555.4     | 17180 - 20043       |
+--------+-------+----------+-------------------+---------------------+
|C4      | 12    | 5271.5   | 798.5 - 823.4     | 11635 - 12604       |
+--------+-------+----------+-------------------+---------------------+


Step 2: Add the transmission curve
----------------------------------

LUCI divides each spectrum by the filter transmission before fitting
(``LuciUtility.read_in_transmission``), which needs ``Data/<FILTER>_filter.dat``: two columns,
the axis in cm-1 and the transmission in **percent**.

Download the raw curve from the
`SITELLE filters page <https://www.cfht.hawaii.edu/Instruments/Sitelle/SITELLE_filters.php>`_
(SN4 is ``cfh3602.dat``), save it as ``Data/<FILTER>_Transmission.dat``, add the filter to the
``FILTERS`` list at the top of ``Data/create_filters.py``, and run it:

.. code-block:: bash

    wget -O Data/SN4_Transmission.dat http://www.cfht.hawaii.edu/Instruments/Filters/curves/cfh3602.dat
    cd Data && python create_filters.py

``create_filters.py`` converts nm to cm-1 and averages over however many measurement columns the
CFHT file has (7 for SN1/SN2/SN3, 2 for SN4).

.. note::
   ``apply_transmission`` only divides through where the transmission is above 50%, so a curve
   that does not cover the entire free spectral range is fine -- the ends extrapolate to ~0 and
   are simply left alone.


Step 3: Add the wavenumber windows
----------------------------------

This is the part that is easy to forget. Several functions branch on the filter name and will
either crash or silently print "not supported" if yours is missing. All of them want a window in
**cm-1**:

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Function
     - What the window is for
     - SN4 value (cm-1)
   * - ``LuciFit.Fit.restrict_wavelength``
     - The region actually fit. Keep it inside the pass band so the continuum is sane.
     - 15040 - 15330
   * - ``LuciFit.Fit.calculate_noise``
     - A line-free region used for the noise estimate. Best taken **outside** the pass band.
     - 14600 - 14900
   * - ``LuciFit.Fit.calculate_continuum``
     - A line-free region **inside** the pass band.
     - 15060 - 15150
   * - ``LuciUtility.read_in_reference_spectrum``
     - The clip applied to the reference axis. This sets the input length of the network, so it
       must be a little wider than the fit region.
     - 15000 - 15350
   * - ``LuciBase.Luci.create_snr_map``
     - The line region and a noise region.
     - 15150 - 15300 / 14600 - 14900
   * - ``LuciBase.Luci.slicing``
     - ``filter_line``: which lines live in the filter.
     - Halpha, NII6583, NII6548
   * - ``LuciBase.Luci.create_background_subspace``, ``LuciBase.Luci.fit_calc``,
       ``LuciBase.Luci.fit_pixel``
     - PCA background: the spectral range plus a line-free region used to rescale the
       eigenspectra. **All three must agree.**
     - 664.5 / 661 nm
   * - ``LuciFit.Fit.get_ML_model``
     - Add the filter to the list of filters that have a trained network (and to the MDN list if
       you train one).
     - --
   * - ``LuciVisualize.add_lines``
     - Which lines to draw on the interactive plot.
     - --

For SN4 the lines sit at 15190 (NII6583), 15238 (Halpha) and 15272 cm-1 (NII6548), which is why
the continuum window is placed just redward of them at 15060 - 15150 cm-1.

A quick way to sanity check your choices:

.. code-block:: python

    import numpy as np
    from LUCI.LuciUtility import read_in_transmission
    axis = np.linspace(14454, 15417, 470)          # your filter's free spectral range
    t = read_in_transmission('.', {'FILTER': 'SN4'}, axis)
    # The fit/continuum windows should sit where t is high, the noise window where t is ~0
    for lo, hi in [(15040, 15330), (14600, 14900), (15060, 15150)]:
        m = (axis >= lo) & (axis <= hi)
        print(lo, hi, t[m].min(), t[m].max())


Step 4: Build the reference spectrum and train the network
----------------------------------------------------------

LUCI's initial guess for the velocity and broadening comes from a convolutional neural network
(`Rhea et al. 2020a <https://arxiv.org/abs/2008.08093>`_). At fit time
``LuciFit.Fit.interpolate_spectrum`` interpolates the observed spectrum onto the axis of
``ML/Reference-Spectrum-R<res>-<FILTER>.fits``, normalises it by its maximum, and hands it to
``ML/R<res>-PREDICTOR-I-<FILTER>/``. So the reference spectrum and the network are a matched
pair: **the reference axis defines the input length of the network**, and both are specific to a
single resolution.

The networks shipped with LUCI were built with the notebooks in
`sitelle-signals/Pamplemousse <https://github.com/sitelle-signals/Pamplemousse>`_, which needed
ORBS to synthesise spectra and a connection to the 3MdB database for line ratios. That is no
longer necessary: ``LUCI.LuciSim.Spectrum`` can generate the spectra itself, and
``ML/TrainPredictor.py`` wraps the whole recipe up:

.. code-block:: bash

    python ML/TrainPredictor.py --filter SN4 --resolution 5000 --n-steps 470 \
                                --num-spectra 50000 --epochs 10 --mdn

That does three things:

1. **Creates the reference spectrum.** Only its *axis* matters; the fluxes are there so you can
   eyeball the file. It is written to ``ML/Reference-Spectrum-R<res>-<FILTER>.fits`` as a two
   column (``Wavenumber``, ``Flux``) binary table, which is what
   ``read_in_reference_spectrum`` expects.
2. **Generates the synthetic training set.** Each spectrum gets a random velocity in
   [-500, 500] km/s, a random broadening in [10, 200] km/s (the range LUCI advertises for the
   network), random line ratios, a random SNR, and a slightly perturbed resolution. It is then
   interpolated onto the clipped reference axis and normalised -- exactly the preprocessing real
   spectra get.
3. **Trains and saves the network**, reporting the residual scatter on a held-out test set. With
   ``--mdn`` it also trains the mixture density network (``LUCI/LuciNetwork.py``) used when
   ``mdn=True``, saving weights under the doubled-up path convention the loader expects.

.. important::
   **Convert the trained network to ONNX before LUCI will use it.** Fitting loads predictors from
   ``ML/onnx/``, not the Keras artifacts, so a freshly trained network is invisible until you run::

       uv run tools/convert_models_to_onnx.py --models R5000-PREDICTOR-I-SN4 --validate

   (or ``--all`` to redo everything). ``--validate`` checks the converted model against its Keras
   original on 200 spectra and refuses to publish it if they disagree beyond float32 tolerance, so a
   silently broken conversion cannot reach your fits. The script pins its own legacy-TensorFlow
   environment through a PEP 723 header, so ``uv run`` builds what it needs and discards it
   afterwards.

   If you skip this step LUCI does not crash -- it reports that no predictor was found for that
   filter/resolution and falls back to data-driven initial guesses.

To support a genuinely new filter you need to add it to ``FILTER_LINES`` and
``sample_amplitudes`` at the top of ``ML/TrainPredictor.py``, which say which lines fall in the
filter and how to sample their relative amplitudes, and add a ``FilterSpec`` entry in
``LUCI/instrument/filters.py`` giving the fit, noise and reference windows.

.. warning::
   **Pick ``--n-steps`` from your cube's header.** ``LuciSim.Spectrum`` derives the number of
   steps from the resolution as :math:`1.20671\,R/(\text{order}+0.5)`. That is about 20% below
   both CFHT's published relation for SITELLE (:math:`1.20671^2\,R/(\text{order}+0.5)`, which
   reproduces their quoted :math:`N_{steps} = 0.094R` for SN4, :math:`0.108R` for SN5 and
   :math:`0.173R` for SN6) and the channel counts of the reference spectra shipped with LUCI.
   Reading ``STEPNB`` out of the header of the cube you actually want to fit and passing it as
   ``--n-steps`` avoids the question altogether and guarantees your reference axis is sampled the
   same way your data are.

Train one network per resolution you care about. If you do not train one, everything still works
with ``ML_bool=False`` -- LUCI just falls back to its non-machine-learning initial guess.


Step 5: Fit a cube
------------------

Nothing filter-specific is needed at the call site; LUCI reads the filter out of the header.

.. code-block:: python

    cube = SitelleCube(luci_path, cube_dir + '/' + cube_name, cube_dir, object_name,
                redshift, resolution, ML_bool=True)
    cube.fit_cube(['Halpha', 'NII6583', 'NII6548'], 'sincgauss', [1, 1, 1], [1, 1, 1],
                  x_min, x_max, y_min, y_max)
