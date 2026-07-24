"""
Backward-compatibility shim.

This grab-bag module was split by concern:

  * header/WCS handling   -> ``LUCI.instrument.header``
  * HDF5 layout           -> ``LUCI.io.hdf5``
  * reference/transmission-> ``LUCI.io.reference``
  * writing FITS products -> ``LUCI.io.outputs``
  * asset paths           -> ``LUCI.io.assets``
  * spatial binning       -> ``LUCI.engine.binning``
  * numerical Hessian     -> ``LUCI.fitting.uncertainties``
"""

from LUCI.engine.binning import bin_cube_function, bin_mask  # noqa: F401
from LUCI.fitting.uncertainties import hessian, hessianComp  # noqa: F401
from LUCI.instrument.header import (  # noqa: F401
    get_interferometer_angles,
    spectrum_axis_func,
    update_header,
)
from LUCI.io.assets import check_luci_path  # noqa: F401
from LUCI.io.hdf5 import get_quadrant_dims  # noqa: F401
from LUCI.io.outputs import save_fits  # noqa: F401
from LUCI.io.reference import read_in_reference_spectrum, read_in_transmission  # noqa: F401
