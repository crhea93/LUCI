.. _luci:

FAQ and Errors
==============

This section contains a discussion on frequently asked questions and errors that sometimes pop up.



.. toctree::
   :maxdepth: 2


FAQ:
----


Common Errors
-------------

ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject
############################################################################################################################

This error occasionally comes up when you have newly installed the luci conda environment and try to run
`extract_spectrum_region`. This happens because your numpy version is disscordant with the
environment. This can be simply remedied by reinstalling numpy.

`pip uninstall numpy`

`pip install numpy`
