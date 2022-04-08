"""
Luci visualization tools
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets  import RectangleSelector

def visualize(deep_image, spectrum_axis, cube_final):
    """
    Function that allows you to visualize the deep frame, click on a pixel, and
    then see the sepctrum. This is under development at the moment (4.8.22 -- Carter)
    """
    fig,axes = plt.subplots(2,1,figsize=(15,15))
    plt.style.use('fivethirtyeight')

    shift_ct = 0
    point1 = []

    rectangles = []


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
        plt.plot(1e7 / spectrum_axis, integrated_spectrum, linewidth=1)
        axes[1].set_xlabel('Wavelength [nm]', fontweight='bold')
        axes[1].set_ylabel(r'Intensity (Ergs/cm$^2$/s/$\AA$)', fontweight='bold')


    #deep_image = fits.open('Luci_outputs/NGC628_deep.fits')[0].data
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
            plt.plot(1e7/spectrum_axis,cube_final[X_coordinate, Y_coordinate], linewidth=1)
            axes[1].set_xlabel('Wavelength [nm]', fontweight='bold')
            axes[1].set_ylabel(r'Intensity (Ergs/cm$^2$/s/$\AA$)', fontweight='bold')
            plt.show()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.subplot(211)
    plt.imshow(np.log10(deep_image))
    rs = RectangleSelector(axes[0], line_select_callback,
                           drawtype='box', useblit=False, button=[1],
                           minspanx=5, minspany=5, spancoords='pixels',
                           interactive=False)
    plt.show()
