Migration guide
===============

LUCI was restructured so it can be installed, tested, and modified safely. This
page maps every old name to its replacement.

**Nothing in this guide is urgent.** Every old spelling still works — the old
module names are re-export shims and the uppercase ``LUCI`` package name is
aliased onto the new lowercase ``luci``. Existing notebooks and scripts run
unchanged. Migrate when convenient.

The one thing that *did* change is numerical output, and only where a bug was
fixed. See `What changed in the results`_ — read that section even if you never
touch the new API.


Installation
------------

The conda environment is gone. LUCI is a normal Python package:

.. code-block:: bash

    # users
    pip install luci-sitelle

    # contributors
    git clone https://github.com/crhea93/LUCI && cd LUCI
    uv sync
    uv run pytest

Because the package installs properly now, **the ``sys.path.insert`` line at the
top of every example is no longer needed** and should be deleted:

.. code-block:: python

    # before
    import sys
    sys.path.insert(0, '/home/carterrhea/Documents/LUCI/')
    from LuciBase import Luci

    # after
    from luci import SitelleCube

``Luci_path`` is now optional
-----------------------------

``Luci_path`` — the absolute path to the checkout, with its mandatory trailing
slash — is how LUCI found its ``ML/`` and ``Data/`` directories. It is why every
example began by hardcoding a path into somebody else's home directory.

It is now resolved automatically: from ``$LUCI_DATA_DIR`` if set, otherwise from
the installed package's own location. Pass one only to override.

.. code-block:: python

    # before
    Luci_path = '/home/carterrhea/Documents/LUCI/'
    cube = Luci(Luci_path, cube_path, output_dir, name, redshift, resolution)

    # after
    cube = SitelleCube(cube_path=cube_path, output_dir=output_dir,
                       object_name=name, redshift=redshift, resolution=resolution)

Passing it positionally still works exactly as before, so no existing call needs
to change.


Package and class names
-----------------------

============================== ===============================================
Old                            New
============================== ===============================================
``LUCI`` (package)             ``luci`` — ``LUCI`` still resolves, see below
``from LuciBase import Luci``  ``from luci import SitelleCube``
``Luci``                       ``SitelleCube`` (``Luci`` remains an alias)
``Fit``                        ``SpectrumFitter`` (``Fit`` remains an alias)
the 22-key dict from ``fit()`` ``FitResult`` — indexes exactly like the dict
============================== ===============================================

.. note::

   A lowercase ``luci`` package and an uppercase ``LUCI`` package cannot coexist
   on macOS or Windows, whose filesystems are case-insensitive. So ``LUCI`` is
   not a package but a module (``LUCI.py``) that registers the alias in
   ``sys.modules``. ``LUCI.cube`` and ``luci.cube`` are therefore *the same
   module object*, not two copies — no duplicated state.


Module layout
-------------

The flat ``LUCI/Luci*.py`` files were split by responsibility. Each old module
still exists as a re-export shim, so old imports resolve.

================================== ===========================================
Old module                         New home
================================== ===========================================
``LUCI.LuciFit``                   ``luci.fitting.spectrum_fitter``,
                                   ``luci.fitting.constraints``
``LUCI.LuciFunctions``             ``luci.fitting.models``
``LUCI.LuciFitParameters``         ``luci.fitting.parameters``
``LUCI.LuciBayesian``              ``luci.fitting.bayes``
``LUCI.LuciUtility``               ``luci.instrument.header``, ``luci.io.*``,
                                   ``luci.engine.binning``,
                                   ``luci.fitting.uncertainties``
``LUCI.LuciWVT``                   ``luci.analysis.wvt``
``LUCI.LuciComponentCalculations`` ``luci.analysis.components``
``LUCI.LuciBackground``            ``luci.background.detection``
``LUCI.LuciConvenience``           ``luci.engine.selection``,
                                   ``luci.fitting.components``
``LUCI.LuciPlotting``              ``luci.viz.plotting``
``LUCI.LuciVisualize``             ``luci.viz.visualize``
``LUCI.LuciSim``                   ``luci.simulation``
``LUCI.LuciLog``                   ``luci.log``
``LuciBase``                       ``luci.cube``
================================== ===========================================

Two modules are new rather than moved:

``luci.instrument.filters``
    One ``FilterSpec`` per filter, holding every band-dependent number that used
    to be copy-pasted across five ``if filter == 'SN3'`` chains. Adding a filter
    is now one dict entry — see :doc:`newFilter`.

``luci.engine.runner`` and ``luci.engine.selection``
    A single fit orchestrator and a single pixel-selection resolver, replacing
    four near-identical blocks in the old ``Luci`` class.


Configuring a fit
-----------------

``SpectrumFitter`` still takes all of its original keyword arguments. They can
now also be grouped into a ``FitConfig``, which validates them in one place:

.. code-block:: python

    from luci import FitConfig, SpectrumFitter

    config = FitConfig(
        lines=["Halpha", "NII6583"],
        model="sincgauss",
        vel_rel=[1, 1],
        sigma_rel=[1, 1],
    )
    fitter = SpectrumFitter(spectrum, axis, wavenumbers_syn, config=config)

Both spellings produce bit-identical fits; a test asserts it.


Errors instead of ``exit()``
----------------------------

Library code no longer calls ``quit()``/``exit()`` on bad input, which used to
kill the host interpreter — including a Jupyter kernel — with no traceback.

======================================= =====================================
Situation                                Now raises
======================================= =====================================
Unrecognised filter                      ``UnsupportedFilterError``
PCA background on an unsupported filter  ``PCABackgroundUnsupportedError``
Invalid fit configuration                ``InvalidFitConfig``
Unrecognised region/mask argument        ``ValueError``
======================================= =====================================


Output messages
---------------

Progress and diagnostic messages go through the ``luci`` logger instead of
``print()``, so they can be filtered or silenced:

.. code-block:: python

    import logging
    logging.getLogger("luci").setLevel(logging.WARNING)   # quieter

    from luci.log import silence
    silence()                                             # off entirely

Output is on by default at ``INFO``, so nothing disappears unless you ask.


What changed in the results
---------------------------

Bug fixes changed numerical output. The full register is in
``REFACTOR_BUGS.md``; these are the ones that move published numbers.

**Line broadening was wrong for every line except the last** (B21). The
per-line σ bounds were built in a loop of ``lambda`` closures over the loop
variable, so all of them read the *final* value — every line was constrained by
the last line's bounds. On the test cube, ``NII6583`` moved 164 → 101 km/s and
``SII6731`` 167 → 101 km/s, and the spread across velocity-tied lines dropped
from 67.5 to 2.4 km/s.

.. warning::

   If you published broadening or velocity maps from a multi-line fit, they were
   affected by B21. Re-running is the only way to know by how much.

**``ML_bool=False`` fabricated zero kinematics** (B1). The non-ML path started
the optimiser at σ = 0, a singular point of the sinc-Gauss, and returned
velocity = broadening = 0 rather than failing. The documentation actively
recommended ``ML_bool=False`` for unsupported filters. It now seeds from the
brightest peak with a default broadening and recovers the injected physics.

**``fit_region`` ignored the background** (B3). It passed ``bkg`` but not
``bkgType``, so subtraction was silently skipped — results were as if no
background had been given.

**``pixel_list=True`` fitted the whole cube** (B23). The mask started as
all-``True``, so the pixel list narrowed nothing.

Smaller fixes — a deep image dropping its trailing rows (B7), multi-component
fits overwriting each other's output maps (B16), MDN broadening priors read
without their softplus transform (B17) — are described in the register with the
test that pins each one.
