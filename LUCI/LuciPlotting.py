"""
Collection of plotting functions
"""

import matplotlib.pyplot as plt
from astropy.wcs import WCS
import numpy as np
import seaborn as sns


def set_style(dark):
    """
    Set style as light or dark
    """
    if dark:
        return './dark.mplstyle'
    else:
        return './light.mplstyle'


def plot_spectrum(axis, spectrum, ax=None, units='cm-1', output_name=None, fig_size=(10, 8), dark=False, **kwargs):
    """
    Plot Spectrum with Luci format. If output name is supplied, the plot will be saved
    Args:
        axis: X axis of spectrum (1d numpy array)
        spectrum: Y axis of spectrum (1d numpy array)
        units: Wavelength units (e.x. 'cm')
        output_name: Path to output file (default None)
        fig_size: Size of figure (default (10,8))
        dark: Boolean to turn on dark mode (default False)
    """

    if ax is None:
        f, ax = plt.subplots(figsize=fig_size)
    check_units(units)  # Check that user supplied appropriate wavelength option
    if units == 'nm':
        axis = [1e7 / axis_val for axis_val in axis]
    else:
        pass
    ax.plot(axis, spectrum, **kwargs)
    ax.set_xlabel(r"Wavelength [%s]" % units, fontsize=20, fontweight='bold')
    ax.set_ylabel(r'Flux [ergs s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]', fontsize=20, fontweight='bold')
    ax.tick_params(labelsize=14)
    if output_name is not None:
        plt.savefig(output_name)
    return ax

def plot_fit(axis, spectrum, fit, ax=None, units='cm-1', output_name=None, fig_size=(10, 8), dark=False, **kwargs):
    """
    Plot Spectrum and fit with Luci format. If output name is supplied, the plot will be saved
    Args:
        axis: X axis of spectrum (1d numpy array)
        spectrum: Y axis of spectrum (1d numpy array)
        fit: Fit vector (1d numpy array)
        ax: Current axis (defaults to None)
        units: Wavelength units (e.x. 'cm')
        output_name: Path to output file (default None)
        fig_size: Size of figure (default (10,8))
        dark: Boolean to turn on dark mode (default False)
    """

    if ax is None:
        f, ax = plt.subplots(figsize=fig_size)
    check_units(units)  # Check that user supplied appropriate wavelength option
    if units == 'nm':
        axis = [1e7 / axis_val for axis_val in axis]
    else:
        pass
    ax.plot(axis, spectrum, label='Spectrum', **kwargs)
    ax.plot(axis, fit, linestyle='--', linewidth=2, label='Fit Vector', **kwargs)
    ax.set_xlabel(r"Wavelength [%s]" % units, fontsize=20, fontweight='bold')
    ax.set_ylabel(r'Flux [ergs s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]', fontsize=20, fontweight='bold')
    ax.tick_params(labelsize=14)
    plt.legend()
    if output_name is not None:
        plt.savefig(output_name)
    return ax


def plot_map(quantity_map, quantity_name, output_dir='', header=None, object_name='', filter_name='', clims=None, fig_size=(10, 8), dark=False, **kwargs):
    """
    Function to plot fit map. The three options are 'flux', 'velocity', and 'broadening'.
    The flux map is automatically scaled by log10. The velocity and broadening are not.
    The plot is saved here: `output_dir+'/'+quantity_name+'_map.png'`

    Args:
        quantity_map: 2d numpy array from fit
        quantity_name: Name of quantity (e.x. 'flux')
        output_dir: Path (absolute or partial) to output directory
        header: Header information for plot
        clims: List containing lower and upper limits of colorbar (e.x. [-500, 500])
        fig_size: Size of figure (default (10,8))
        dark: Boolean to turn on dark mode (default False)

    Example:
        We can plot the flux assuming a min and max value of 1e-18 and 1e-16.

        >>> lplt.plot_map(flux_map[:,:,0], 'flux', cube_dir, cube.header, clims=[-18, -16])

        Similarly, we can plot the velocity and broadening:

        >>> lplt.plot_map(vel_map[:,:,0], 'veloity', cube_dir, cube.header, clims=[-200, -200])

        >>> lplt.plot_map(broad_map[:,:,0], 'broadening', cube_dir, cube.header, clims=[10, 50])

    """
    if quantity_name == 'broadening' or quantity_name == 'velocity':
        pass
    elif quantity_name == 'flux':
        quantity_map = np.log10(quantity_map)  # NaNs to extremely small number
    else:
        print('Please enter either flux, velocity, or broadening')
    units = {'flux': r'log[ergs/s/cm$^2$/A]', 'velocity': 'km/s', 'broadening': 'km/s'}
    if clims is None:
        c_min = np.nanpercentile(quantity_map, 5)
        c_max = np.nanpercentile(quantity_map, 99.5)
    else:
        c_min = clims[0]
        c_max = clims[1]
    # Plot
    wcs = WCS(header)
    plot_style = set_style(dark)
    fig = plt.figure(figsize=fig_size)
    try:
        ax = plt.subplot(projection=wcs)
        ax.coords[0].set_major_formatter('hh:mm:ss')
        ax.coords[1].set_major_formatter('dd:mm:ss')
    except (AttributeError, ValueError):
        pass
    plt.imshow(quantity_map, cmap='mako', **kwargs)
    plt.title((quantity_name + ' map').upper(), fontsize=26, fontweight='bold')
    plt.xlabel("RA", fontsize=20, fontweight='bold')
    plt.ylabel("DEC", fontsize=20, fontweight='bold')
    plt.xlim(0, quantity_map.shape[1])
    plt.ylim(0, quantity_map.shape[0])
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    plt.clim(c_min, c_max)
    cbar.ax.set_ylabel(units[quantity_name], rotation=270, labelpad=25, fontsize=20, fontweight='bold')
    if object_name != '':
        object_name += '_'
    if filter_name != '':
        filter_name += '_'
    plt.savefig(output_dir + '/' +object_name+filter_name+ quantity_name + '_map.png')
    return None


def plot_map_no_coords(quantity_map, quantity_name, object_name='', filter_name='', output_dir='', clims=None, fig_size=(10, 8), dark=False, **kwargs):
    """
    Function to plot fit map. The four options are 'flux', 'velocity', 'broadening', and 'zscore'.
    The flux map is automatically scaled by log10. The velocity and broadening are not.
    The plot is saved here: `output_dir+'/'+quantity_name+'_map.png'`

    Args:
        quantity_map: 2d numpy array from fit
        quantity_name: Name of quantity (e.x. 'flux')
        output_dir: Path (absolute or partial) to output directory
        clims: List containing lower and upper limits of colorbar (e.x. [-500, 500])
        fig_size: Size of figure (default (10,8))
        dark: Boolean to turn on dark mode (default False)

    Example:
        We can plot the flux.

        >>> lplt.plot_map(flux_map[:,:,0], 'flux')



    """
    if quantity_name == 'broadening' or quantity_name == 'velocity' or quantity_name == 'zscore':
        pass
    elif quantity_name == 'flux':
        quantity_map = np.log10(quantity_map)  # NaNs to extremely small number
    else:
        print('Please enter either flux, velocity, broadening, or zscore')
    units = {'flux': r'log[ergs/s/cm$^2$/A]', 'velocity': 'km/s', 'broadening': 'km/s', 'zscore': ''}
    if clims is None:
        c_min = np.nanpercentile(quantity_map, 5)
        c_max = np.nanpercentile(quantity_map, 99)
    else:
        c_min = clims[0]
        c_max = clims[1]
    # Plot
    plot_style = set_style(dark)
    fig = plt.figure(figsize=fig_size)
    plt.imshow(quantity_map, cmap='magma', **kwargs)
    plt.title((quantity_name + ' map').upper(), fontsize=26, fontweight='bold')
    plt.xlabel("RA", fontsize=20, fontweight='bold')
    plt.ylabel("DEC", fontsize=20, fontweight='bold')
    plt.xlim(0, quantity_map.shape[1])
    plt.ylim(0, quantity_map.shape[0])
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    plt.clim(c_min, c_max)
    cbar.ax.set_ylabel(units[quantity_name], rotation=270, labelpad=25, fontsize=20, fontweight='bold')
    plt.savefig(output_dir + '/' +object_name+'_'+filter_name+'_'+ quantity_name + '_map.png')
    return None

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
