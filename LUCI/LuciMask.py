"""
Code for masking out the background pixels in LUCI
"""
from astropy.io import fits
# Imports
import os
import sys
from sklearn import decomposition
import numpy as np
import matplotlib.pyplot as plt
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import detect_sources
# Get location of LUCI
path = os.path.abspath('/home/carterrhea/Documents/LUCI')
sys.path.insert(0, path)  # add LUCI to the available paths

from LuciBase import Luci
import LUCI.LuciPlotting as lplt

def find_background_pixels(deep_image, sigma_threshold=1, plot_mask=True):
    """
    This algorithm uses a sigma thresholding algorithm on the deep image to determine which pixels belong to the background.
    """
    bkg_estimator = MedianBackground()
    bkg = Background2D(deep_image, (50, 50), filter_size=(3, 3),
                       bkg_estimator=bkg_estimator)
    deep_image -= bkg.background  # subtract the background
    threshold = 0.1 * bkg.background_rms
    kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
    convolved_data = convolve(deep_image, kernel)
    segment_map = detect_sources(convolved_data, threshold, npixels=8)
    print(segment_map)
    idx = np.argwhere(np.abs(segment_map) < 1)
    #print(idx)
    #zscore = (deep_image - deep_image.mean())/deep_image.std()  # Calculate z-score for each pixel
    #idx = np.argwhere(np.abs(zscore) < sigma_threshold)  # Get indices in array where z-score is above sigma_threshold
    if plot_mask == True:
        #zscore_masked = np.where(np.abs(zscore) < sigma_threshold, zscore, np.nan)
        #plt.imshow(zscore_masked)
        plt.imshow(segment_map, origin='lower', cmap=segment_map.cmap, interpolation='nearest')
        plt.ylim(0, 2048)
        plt.xlim(0, 2048)
        plt.show()
        #plot_map_no_coords(zscore_masked, '/home/carterrhea/Downloads/zscore')
        #plt.savefig('Background.png')
    return idx
deep = fits.open('/mnt/carterrhea/carterrhea/Perseus/SITELLE/Luci_outputs/NGC1275_lowres_deep.fits')[0].data
#deep = fits.open('/mnt/carterrhea/carterrhea/NGC4449/Luci_outputs/NGC4449_deep.fits')[0].data
idx = find_background_pixels(deep)  # List of background pixels
#Set Parameters
# Using Machine Learning Algorithm for Initial Guess
Luci_path = '/home/carterrhea/Documents/LUCI/'
cube_dir = '/mnt/carterrhea/carterrhea/Perseus/SITELLE'  # Path to data cube
cube_name = 'A0426_SN3.merged.cm1.1.0'  # don't add .hdf5 extension
#cube_dir = '/mnt/carterrhea/carterrhea/NGC4449/'
#cube_name = 'NGC4449_SN3'
object_name = 'NGC1275'
redshift = 0.0179  # Redshift
resolution = 1800

cube = Luci(Luci_path, cube_dir+'/'+cube_name, cube_dir, object_name, redshift, resolution, mdn=False)

bkg_spectra = [cube.cube_final[index[0], index[1]] for index in idx if index[0]<2048 and index[1]<2064]
print(len(bkg_spectra))
n_components = 50
pca = decomposition.IncrementalPCA(n_components=n_components)
pca.fit(bkg_spectra)
X = pca.transform(bkg_spectra)

# Plot the primary components
plt.figure(figsize=(18, 16))
# plot the mean first
l = plt.plot(1e7/cube.spectrum_axis, pca.mean_/np.max(pca.mean_)-1,linewidth=3)
c = l[0].get_color()
plt.text(6, -0.9, 'mean emission', color=c,fontsize='xx-large')
shift = 0.6
for i in range(5):
    l = plt.plot(1e7/cube.spectrum_axis, pca.components_[i] + (i*shift), linewidth=3)
    c = l[0].get_color()
    plt.text(6, i*shift+0.1, "component %i" % (i + 1), color=c, fontsize='xx-large')
plt.xlabel('nm', fontsize=24)
plt.ylabel('Normalized Emission + offset',  fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
#plt.savefig('./PCA.png')
#plt.clf()
plt.show()

#nComp = 5
projectedPCAValues = pca.transform(bkg_spectra)[:,:n_components]  # PCA Coefficient values in PCA space
Xhat = np.dot(projectedPCAValues, pca.components_[:n_components,:])
Xhat += pca.mean_

lplt.plot_spectrum(1e7/cube.spectrum_axis, np.median(Xhat, axis=0))
plt.show()



PC_values = np.arange(pca.n_components_)[:n_components] + 1
plt.plot(PC_values, pca.explained_variance_ratio_[:n_components], 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

interpolatedPoints = np.empty((2048, 2064, n_components))  # Initialize cube with third parameter has n_components
for bkg_ct, index in enumerate(idx):  # Step through background points
    interpolatedPoints[index[0], index[1]] = projectedPCAValues[bkg_ct]
print(interpolatedPoints.shape)