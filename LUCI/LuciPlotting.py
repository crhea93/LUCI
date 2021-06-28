"""
Collection of plotting functions
"""
import matplotlib.pyplot as plt
def plot_spectrum(axis, spectrum, units='cm-1', output_name = None, fig_size=(10,8), **kwargs):
    """
    Plot Spectrum with Luci format. If output name is supplied, the plot will be saved
    Args:
        axis: X axis of spectrum (1d numpy array)
        spectrum: Y axis of spectrum (1d numpy array)
        units: Wavelength units (e.x. 'cm')
        output_name: Path to output file (default None)
    """
    f, ax = plt.subplots(figsize=fig_size)
    check_units(units)  # Check that user supplied appropriate wavelength option
    if units == 'nm':
        axis = [1e7/axis_val for axis_val in axis]
        ax.set_xlim(635, 675)
    else:
        ax.set_xlim(14750, 15750)
    ax.plot(axis, spectrum, **kwargs)
    ax.set_xlabel(r"Wavelength [%s]"%units, fontsize=20, fontweight='bold')
    ax.set_ylabel(r'Flux [ergs s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]', fontsize=20, fontweight='bold')
    ax.tick_params(labelsize=14)
    if output_name is not None:
        plt.savefig(output_name)
    return ax


def check_units(unit):
    """
    This function checks to see that the unit provided is in the available options
    Args:
        unit: User supplied unit
    Return:
        Nothing if the user provides an appropriate unit
        Else it will throw an error
    """
    unit_options = ['cm-1', 'nm']
    if unit in unit_options:
        pass
    else:
        raise Exception('Please submit a unit name in the available list: \n {}'.format(unit_options))
