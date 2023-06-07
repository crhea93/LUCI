"""
This file contains an assortment of functions used for background-related calculations.

"""
import numpy as np
from astropy.convolution import convolve
from photutils.segmentation import detect_sources
from photutils.segmentation import make_2dgaussian_kernel
from photutils.background import Background2D, MedianBackground
import matplotlib.pyplot as plt
import os


def find_background_pixels(deep_image, outputDir='', sigma_threshold=2, plot_mask=True):
    """
    This algorithm uses a sigma thresholding algorithm on the deep image to determine which pixels belong to the background.

    Args:
        deep_image: Deep image in 2D
        sigma_threshold: Threshold parameter for determining the background (default 0.1)
        plot_mask: Boolean to make/save background plot (default True)

    Return:
        idx_bkg: List of background pixel positions (list of tuples)]
        idx_source: List of source pixel positions
    """
    bkg_estimator = MedianBackground()
    bkg = Background2D(deep_image, (50, 50), filter_size=(3, 3),
                       bkg_estimator=bkg_estimator)  # Experimentally found this values to be good
    deep_image -= bkg.background  # subtract the background
    threshold = sigma_threshold * bkg.background_rms
    kernel = make_2dgaussian_kernel(3.0, size=3)  # FWHM = 3.0
    convolved_data = convolve(deep_image, kernel)  # Convolve data
    segment_map = detect_sources(convolved_data, threshold, npixels=10)  # Apply detection
    idx_bkg = np.argwhere(np.abs(segment_map) < 1)  # Get only background pixels
    idx_source = np.argwhere(np.abs(segment_map) > 0)  # Get source pixels
    if plot_mask:
        plt.imshow(segment_map, origin='lower', cmap=segment_map.cmap, interpolation='nearest')
        plt.ylim(0, len(deep_image.T))
        plt.xlim(0, len(deep_image))
        plt.xlabel('Right Ascension (physical)', fontsize=24)
        plt.ylabel('Declination (physical)', fontsize=24)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(os.path.join(outputDir, 'BackgroundPixelMap.png'))
    return idx_bkg, idx_source
