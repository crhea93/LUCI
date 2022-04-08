"""
Luci visualization tools
"""

import matplotlib.pyplot as plt
import numpy as np

def visualize(deep_image, spectrum_axis, cube_final):
    """
    Function that allows you to visualize the deep frame, click on a pixel, and
    then see the sepctrum. This is under development at the moment (4.8.22 -- Carter)
    """
    fig,axes = plt.subplots(2,1,figsize=(15,15))
    plt.style.use('fivethirtyeight')
    shift_is_held = False
    shift_ct = 0

    def on_key_press(event):
        print(event.key)
        if event.key == 'shift':
            shift_is_held = True
            return True
        else:
            return False


    def on_key_release(event):
        if event.key == 'shift':
            shift_is_held = False
        return False

    shift_is_held = fig.canvas.mpl_connect('key_press_event', on_key_press)
    shift_is_held = fig.canvas.mpl_connect('key_release_event', on_key_release)

    point1 = []

    #deep_image = fits.open('Luci_outputs/NGC628_deep.fits')[0].data
    def onclick(event):
        plt.subplot(212)

        print(shift_is_held)
        if shift_is_held is True:
            print('SHIFT %i %i'%(event.xdata, event.ydata))
            axes[0].plot(int(event.xdata), int(event.ydata), 'o', color='y')
            if shift_ct == 0:
                point1 = [int(event.xdata), int(event.ydata)]
                shift_ct += 1
            if shift_ct == 1:
                dist = np.sqrt((int(event.xdata)-point1[0])**2+(int(event.ydata)-point[1])**2)
                circle = plt.Circle((point1[0], point1[1]), dist, color='b', fill=False)
                shift_ct = 0
        else:
            X_coordinate = int(event.xdata)
            Y_coordinate = int(event.ydata)
            axes[1].cla()
            plt.title('Spectrum of point (%i,%i)'%(X_coordinate, Y_coordinate))
            plt.plot(1e7/spectrum_axis,cube_final[X_coordinate, Y_coordinate], linewidth=1)
            axes[1].set_xlabel('Wavelength [nm]', fontweight='bold')
            axes[1].set_ylabel(r'Intensity (Ergs/cm$^2$/s/$\AA$)', fontweight='bold')
            plt.show()
            shift_ct = 0


    plt.subplot(211)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.imshow(np.log10(deep_image))
    plt.get_current_fig_manager().toolbar.zoom()

    plt.show()
