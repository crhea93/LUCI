.. _luci:

FAQ and Errors
==============

This section contains a discussion on frequently asked questions and errors that sometimes pop up.



.. toctree::
   :maxdepth: 2


FAQ:
----

How is the interferometric angle taken into account?
####################################################
We calculate the interferometric angle to calculate a correction factor `(1/cos(theta))` which is applied to the broadening.
We do not apply this correction factor to the spectral axis of each spaxel since this
is already done in ORB -- therefore, the data product downloaded from the CADC already has this
correction applied! Normally, we would need to calculate the wavelength axis of each spaxel individually
and then interpolate it (carefully!!!) onto a standard axis. 


Common Errors
-------------

ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject
############################################################################################################################

This error occasionally comes up when you have newly installed the luci conda environment and try to run
`extract_spectrum_region`. This happens because your numpy version is disscordant with the
environment. This can be simply remedied by reinstalling numpy.

`pip uninstall numpy`

`pip install numpy`
