"""
This file contains an assortment of functions used for background-related calculations.

"""
import numpy as np
from astropy.convolution import convolve
from photutils.segmentation import detect_sources
from photutils.segmentation import make_2dgaussian_kernel
from photutils.background import Background2D, MedianBackground
import matplotlib.pyplot as plt
import seaborn
import os
from astropy.wcs import WCS
import matplotlib.colors


def find_background_pixels(deep_image, outputDir='', sigma_threshold=.1, plot_mask=True, npixels=10, bkg_algo='detect_source', filter_='SN3'):
    """
    This algorithm uses a sigma thresholding algorithm on the deep image to determine which pixels belong to the background.

    Args:
        deep_image: Deep image in 2D
        sigma_threshold: Threshold parameter for determining the background (default 0.1)
        plot_mask: Boolean to make/save background plot (default True)
        npixels: Minimum number of connected pixels in a detected group (default 10)
        bkg_algo: Background algorithm to use (default 'sourece_detect'; options: 'source_detect', 'threshold')
        filter_: SITELLE Filter used (default 'SN3')
        
    Return:
        idx_bkg: List of background pixel positions (list of tuples)]
        idx_source: List of source pixel positions
    """
    #cmap = plt.get_cmap('rainbow')
    #cmap = matplotlib.colors.ListedColormap(['white', 'black'])
    bkg_estimator = MedianBackground()
    bkg = Background2D(deep_image, (50, 50), filter_size=(3, 3),
                       bkg_estimator=bkg_estimator)  # Experimentally found this values to be good
    deep_image_sub = deep_image - bkg.background  # subtract the background
    threshold = sigma_threshold * bkg.background_rms
    kernel = make_2dgaussian_kernel(3.0, size=3)  # FWHM = 3.0
    convolved_data = convolve(deep_image_sub, kernel)  # Convolve data
    if bkg_algo == 'detect_source':
        segment_map = detect_sources(convolved_data, threshold, npixels=npixels)  # Apply detection
        idx_bkg = np.argwhere(np.abs(segment_map) == 0)  # Get only background pixels
        idx_source = np.argwhere(np.abs(segment_map) > 0)  # Get source pixels
        if plot_mask:
            plt.imshow(segment_map.data.T, origin='lower', cmap=segment_map.cmap, interpolation='nearest')
            ##plt.ylim(0, len(deep_image.T))
            #plt.xlim(0, len(deep_image))
            plt.xlabel('Physical Coordinates', fontsize=24)
            plt.ylabel('Physical Coordinates', fontsize=24)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            '''if header is not None:
                wcs = WCS(header)
                ax = plt.subplot(projection=wcs)
                ax.coords[0].set_major_formatter('hh:mm:ss')
                ax.coords[1].set_major_formatter('dd:mm:ss')'''
            plt.savefig(os.path.join(outputDir, 'BackgroundPixelMap_%s.png'%filter_))
            
    elif bkg_algo == 'threshold':
        idx_bkg = np.argwhere(deep_image < 200000)
        idx_source = np.argwhere(deep_image > 200000)

    else:
        print('You need to pass an appropriate method: detect_source or threshold')
        quit()
    return idx_bkg, idx_source
