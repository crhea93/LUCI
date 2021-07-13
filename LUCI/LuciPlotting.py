"""
Collection of plotting functions
"""


import matplotlib.pyplot as plt
from astropy.wcs import WCS
import numpy as np


def plot_spectrum(axis, spectrum, ax=None, units='cm-1', output_name = None, fig_size=(10,8), **kwargs):
    """
    Plot Spectrum with Luci format. If output name is supplied, the plot will be saved
    Args:
        axis: X axis of spectrum (1d numpy array)
        spectrum: Y axis of spectrum (1d numpy array)
        units: Wavelength units (e.x. 'cm')
        output_name: Path to output file (default None)
    """
    if ax is None:
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


def plot_map(quantity_map, quantity_name, output_dir, header, clims=None, fig_size=(10,8), **kwargs):
    """
    Function to plot fit map
    Args:
        quantity_map: 2d numpy array from fit
        quantity_name: Name of quantity (e.x. 'flux')
        output_dit: Path (absolute or partial) to output directory
        clims: List containing lower and upper limits of colorbar (e.x. [-500, 500])
    """
    #if quantity_name == 'broadening' or quantity_name == 'velocity':
    #    quantity_name = 'velocity'  # The quantities are the same
    if quantity_name != 'flux':
        quantity_map = np.log10(quantity_map)
        print('Please enter either flux, velocity, or broadening')
    units = {'flux':'ergs/s/cm^2/A','velocity':'km/s', 'broadening':'km/s'}
    if clims is None:
        c_min = np.nanpercentile(quantity_map, 5)
        c_max = np.nanpercentile(quantity_map, 95)
    else:
        c_min = clims[0]
        c_max = clims[1]
    #Plot
    #hdu = fits.open(Name+'_SN3.1.0.ORCS/MAPS/'+Name+'_SN3.1.0.LineMaps.map.all.'+Bin+'.rchi2.fits')[0]
    wcs = WCS(header)
    fig = plt.figure(figsize=fig_size)
    ax = plt.subplot(projection=wcs)
    ax.coords[0].set_major_formatter('hh:mm:ss')
    ax.coords[1].set_major_formatter('dd:mm:ss')
    plt.imshow(quantity_map, cmap='magma', **kwargs)
    plt.title((quantity_name +' map').upper(), fontsize=26, fontweight='bold')
    plt.xlabel("RA", fontsize=20, fontweight='bold')
    plt.ylabel("DEC", fontsize=20, fontweight='bold')
    plt.xlim(0,quantity_map.shape[0])
    plt.ylim(0,quantity_map.shape[1])
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    plt.clim(c_min, c_max)
    cbar.ax.set_ylabel(units[quantity_name], rotation=270, labelpad=25, fontsize=20, fontweight='bold')
    plt.savefig(output_dir+'/'+quantity_name+'_map.png')


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
