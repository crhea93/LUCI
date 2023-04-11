"""
Luci visualization tools
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets  import RectangleSelector, Slider
import seaborn as sns


def add_lines(header, integrated_spectrum):
    """
    This function will add the emission lines depending on the filter used
    """
    filter = header['FILTER']
    min_y = np.min(integrated_spectrum)
    max_y = np.max(integrated_spectrum)
    if filter == 'SN1':
        plt.vlines(372.6, 0.8*min_y, 1.2*max_y, linestyles='dashed', colors='coral', linewidths=(2,), label='OII')
    if filter == 'SN2':
        plt.vlines(486.1, 0.8*min_y, 1.2*max_y, linestyles='dashed', colors='coral', linewidths=(2,), label=r'H$\beta$')
        plt.vlines(495.9, 0.8*min_y, 1.2*max_y, linestyles='dashed', colors='seagreen', linewidths=(2,), label='OIII4959')
        plt.vlines(500.7, 0.8*min_y, 1.2*max_y, linestyles='dashed', colors='forestgreen', linewidths=(2,), label='OIII5007')
    if filter == 'SN3':
        plt.vlines(656.3, 0.8*min_y, 1.2*max_y, linestyles='dashed', colors='coral', linewidths=(2,), label=r'H$\alpha$')
        plt.vlines(658.3, 0.8*min_y, 1.2*max_y, linestyles='dashed', colors='seagreen', linewidths=(2,), label='NII6583')
        plt.vlines(654.8, 0.8*min_y, 1.2*max_y, linestyles='dashed', colors='forestgreen', linewidths=(2,), label='NII6548')
        plt.vlines(673.1, 0.8*min_y, 1.2*max_y, linestyles='dashed', colors='lightpink', linewidths=(2,), label='NII6583')
        plt.vlines(671.6, 0.8*min_y, 1.2*max_y, linestyles='dashed', colors='plum', linewidths=(2,), label='NII6548')
    return None

def visualize(deep_image, spectrum_axis, cube_final, header):
    """
    Function that allows you to visualize the deep frame, click on a pixel, and
    then see the sepctrum. This is under development at the moment (4.8.22 -- Carter)
    """
    fig,axes = plt.subplots(2,1,figsize=(15,15))
    plt.style.use('fivethirtyeight')
    spectrum_axis = [1e7/val for val in spectrum_axis]

    shift_ct = 0
    point1 = []

    rectangles = []
    deep_image = deep_image

    def line_select_callback(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        rect = plt.Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2))
        if len(rectangles) > 0:
            rectangles[-1].remove()
        rectangles.append(rect)
        axes[0].add_patch(rect)
        integrated_spectrum = np.zeros(cube_final.shape[2])
        for i in range(int(y2 - y1)):
            y_pix = int(y1 + i)
            for j in range(int(x2 - x1)):
                x_pix = int(x1 + j)
                # Check if pixel is in the mask or not
                integrated_spectrum += cube_final[x_pix, y_pix, :]
        axes[1].cla()
        axes[1].set_title('Spectrum of region %i<x<%i %i<y%i'%(int(x1), int(x2), int(y1), int(y2)))
        plt.plot(spectrum_axis, integrated_spectrum, linewidth=2)
        add_lines(header, integrated_spectrum)
        axes[1].set_xlabel('Wavelength [nm]', fontweight='bold')
        axes[1].set_ylabel(r'Intensity (Ergs/cm$^2$/s/$\AA$)', fontweight='bold')
        plt.legend()


    def onclick(event):
        plt.subplot(212)
        global shift_ct
        global point1
        if event.key == 'control':
            shift_is_held = True
        else:
            shift_is_held = False
            point1 = []
            shift_ct = 0
        if shift_is_held is True:
            print('SHIFT %i %i'%(event.xdata, event.ydata))
            #axes[0].plot(int(event.xdata), int(event.ydata), 'o', color='y')
            if shift_ct != 1:
                point1 = [int(event.xdata), int(event.ydata)]
                shift_ct = 1
            if len(point1) == 1:
                dist = np.sqrt((int(event.xdata)-point1[0])**2+(int(event.ydata)-point1[1])**2)
                circle = plt.Circle((point1[0], point1[1]), dist, color='b', fill=False)
                point1 = []
        else:
            X_coordinate = int(event.xdata)
            Y_coordinate = int(event.ydata)
            axes[1].cla()
            plt.title('Spectrum of point (%i,%i)'%(X_coordinate, Y_coordinate))
            plt.plot(spectrum_axis, cube_final[X_coordinate, Y_coordinate], linewidth=2)
            axes[1].set_xlabel('Wavelength [nm]', fontweight='bold')
            axes[1].set_ylabel(r'Intensity (Ergs/cm$^2$/s/$\AA$)', fontweight='bold')
            add_lines(header, cube_final[X_coordinate, Y_coordinate])
            plt.legend()
            plt.show()

    def update_min(min):
        axes[0].clf()
        axes[0].set_title('Scaled Deep Image')
        scaled_deep_image = np.nan_to_num(np.log10(deep_image), 0)
        axes[0].imshow(scaled_deep_image, vmin=float(min))
        plt.show()


    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.subplot(211)
    axes[0].set_title('Scaled Deep Image')
    scaled_deep_image = np.nan_to_num(np.log10(deep_image), 0)
    plt.imshow(scaled_deep_image, origin='lower', vmin=np.percentile(scaled_deep_image, 5), vmax=np.percentile(scaled_deep_image, 99))
    rs = RectangleSelector(axes[0], line_select_callback,
                           drawtype='box', useblit=False, button=[1],
                           minspanx=2, minspany=2, spancoords='pixels',
                           interactive=False)
    plt.show()
